import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import open_clip
from tqdm import tqdm
import clip
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
from model_zoo.clip_models import CLIPWrapper

import os

CACHE_DIR = "./model_cache"
import multiprocessing
#multiprocessing.set_start_method('spawn', force=True)



def config():
    parser = argparse.ArgumentParser()
    # Default to 'cuda' if available
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use: 'cpu', 'cuda' if available, or 'mps' if available on MacOS")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="NegCLIP", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight-decay", default=0.2, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if it doesn't exist")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for analysis")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--spatial-attention", default=True, action="store_true", help="Use spatial attention module")
    return parser.parse_args()

def get_device(device_arg):
    """
    Safe device selection for different platforms
    """
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print(f"Requested device '{device_arg}' not available, using CPU instead")
        return torch.device("cpu")

class SpatialAttentionModule(nn.Module):
    def __init__(self, feature_dim=512, reduction=8):
        super(SpatialAttentionModule, self).__init__()

        # dim reduction for efficiency
        self.channel_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )

        # projection to original dimension with residual connection
        self.spatial_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

    def forward(self, x):
        channel_weights = self.channel_attention(x)
        x_attended = x * channel_weights

        # spatial MLP with residual connection
        x_out = x + self.spatial_mlp(x_attended)

        x_out = F.normalize(x_out, dim=-1)

        return x_out


class CLIPWithSpatialAttention(nn.Module):
    def __init__(self, clip_model, feature_dim=512):
        super(CLIPWithSpatialAttention, self).__init__()
        self.clip_model = clip_model
        self.spatial_attention = SpatialAttentionModule(feature_dim)

    def encode_image(self, image):
        # image features from base CLIP model
        image_features = self.clip_model.encode_image(image)

        # spatial attention to enhance features
        enhanced_features = self.spatial_attention(image_features)

        return enhanced_features

    def encode_text(self, text):
        # Use original text encoding
        return self.clip_model.encode_text(text)


def load_negclip_model(device_str, use_spatial_attention=False, root_dir=CACHE_DIR):
    from model_zoo.clip_models import CLIPWrapper
    os.makedirs(root_dir, exist_ok=True)
    device = get_device(device_str)
    path = os.path.join(root_dir, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegCLIP model...")
        import gdown
        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
    print("Loading NegCLIP weights...")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    base_model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device="cpu")
    base_model.load_state_dict(state_dict, strict=False)

    if use_spatial_attention:
        print("Adding spatial attention module...")
        # Determine feature dimension from model
        feature_dim = base_model.text_projection.shape[1]
        model = CLIPWithSpatialAttention(base_model, feature_dim=feature_dim)

        # Freeze base model parameters
        for name, param in model.clip_model.named_parameters():
            param.requires_grad = False

        # Enable training for specific layers
        print("Enabling training for specific layers...")
        for name, param in model.named_parameters():
            if 'spatial_attention' in name:
                param.requires_grad = True
                print(f"Training enabled for: {name}")
            elif 'clip_model.ln_final' in name or 'clip_model.text_projection' in name:
                param.requires_grad = True
                print(f"Training enabled for: {name}")
    else:
        model = base_model
        # Freeze most parameters
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Only train specific layers
        print("Enabling training for specific layers...")
        for name, param in model.named_parameters():
            if 'ln_final' in name or 'text_projection' in name:
                param.requires_grad = True
                print(f"Training enabled for: {name}")

    model = model.to(device)
    model = model.train()
    print(f"Model loaded and moved to {device}")

    if use_spatial_attention:
        class EnhancedCLIPWrapper:
            def __init__(self, model, device):
                self.model = model
                self.device = device

            def get_retrieval_scores_batched(self, dataloader):
                scores = []
                self.model.eval()
                with torch.no_grad():
                    for batch in tqdm(dataloader):
                        image = batch["image_options"][0].to(self.device)
                        caption_options = batch["caption_options"]

                        # Process with enhanced model
                        image_features = self.model.encode_image(image)
                        caption_tokenized = torch.cat([clip.tokenize(c) for c in caption_options])
                        text_features = self.model.encode_text(caption_tokenized.to(self.device))

                        # Compute similarity
                        similarity = 100 * (image_features @ text_features.T)
                        scores.append(similarity.cpu().numpy())
                return scores

        clip_model = EnhancedCLIPWrapper(model, device)
    else:
        clip_model = CLIPWrapper(model, device)

    return clip_model, image_preprocess, device


def train_negclip_on_controlled_images(model, train_loader, optimizer, device, epochs, output_dir, args):

    def contrastive_loss(logits, targets, margin=0.2):
        """
        Contrastive loss for 1D logits tensor (for batch size of 1)
        """
        positive_idx = targets[0]
        positive_score = logits[positive_idx]

        margins = margin - (positive_score - logits)
        margins[positive_idx] = 0
        margins = torch.clamp(margins, min=0)

        return margins.sum()

    best_accuracy = 0

    for epoch in range(epochs):
        model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            # In Controlled_Images, we have multiple images and for each image, we have 4 caption options
            # The correct caption is always at index 0
            image = batch["image_options"][0]
            batch_size = 1
            total += batch_size

            if args.spatial_attention:
                # Using enhanced model with spatial attention
                image_features = model.model.encode_image(image.to(device))
                caption_options = batch["caption_options"]

                caption_tokenized = torch.cat([clip.tokenize(c) for c in caption_options])
                text_features = model.model.encode_text(caption_tokenized.to(device))

                # Features are already normalized by the enhanced model
                logits = 100 * (image_features @ text_features.T)
            else:
                # Using standard CLIP model
                image_features = model.model.encode_image(image.to(device))
                image_features = F.normalize(image_features, dim=1)
                caption_options = batch["caption_options"]

                caption_tokenized = torch.cat([clip.tokenize(c) for c in caption_options])
                text_features = model.model.encode_text(caption_tokenized.to(device))
                text_features = F.normalize(text_features, dim=1)

                logits = 100 * (image_features @ text_features.T)

            # Correct caption is always at index 0
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = contrastive_loss(logits[0], targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'accuracy': 100 * correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.output_dir, f"checkpoint_epoch_{epoch + 1}.pt"))

    print("Training completed!")

    return model


def evaluate_model(model, test_loader, device, dataset, args):
    model.model.eval()
    print("Evaluating model...")

    scores = model.get_retrieval_scores_batched(test_loader)
    result_records = dataset.evaluate_scores(scores)

    for record in result_records:
        model_name = f"{args.model_name}_spatial" if args.spatial_attention else f"{args.model_name}_finetuned"
        record.update({"Model": model_name, "Dataset": args.dataset, "Seed": args.seed})

    output_file = os.path.join(args.output_dir, f"{args.dataset}_spatial_attention_results_finetuned.csv")
    df = pd.DataFrame(result_records)
    print(f"Saving results to {output_file}")
    if os.path.exists(output_file):
        all_df = pd.read_csv(output_file, index_col=0)
        all_df = pd.concat([all_df, df])
        all_df.to_csv(output_file)
    else:
        df.to_csv(output_file)

    if args.save_scores:
        save_scores(scores, args)

    print("\nEvaluation Results:")
    for record in result_records:
        print(f"Preposition: {record['Preposition']}, Accuracy: {record['Accuracy']:.4f}")

    return result_records


def main():
    args = config()
    seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Use spatial attention if requested
    model, image_preprocess, device = load_negclip_model(args.device, use_spatial_attention=args.spatial_attention)

    train_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Use batch size of 1 for this dataset structure
        shuffle=True,
        num_workers=args.num_workers
    )

    collate_fn = _default_collate if image_preprocess is None else None

    test_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=False,
        split="test"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    if args.spatial_attention:
        # Use different learning rates for spatial attention module
        attention_params = []
        base_params = []

        for name, param in model.model.named_parameters():
            if param.requires_grad:
                if 'spatial_attention' in name:
                    attention_params.append(param)
                else:
                    base_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': args.lr},
            {'params': attention_params, 'lr': args.lr * 5}  # Higher learning rate for attention
        ], weight_decay=args.weight_decay)
    else:
        # Standard optimizer for regular model
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    if not args.evaluate_only:
        print(f"Starting training for {args.epochs} epochs...")
        model = train_negclip_on_controlled_images(
            model,
            train_loader,
            optimizer,
            device,
            args.epochs,
            args.output_dir,
            args
        )

    evaluate_model(model, test_loader, device, test_dataset, args)


if __name__ == "__main__":
    main()