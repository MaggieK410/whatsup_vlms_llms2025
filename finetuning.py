import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import numpy as np
import json

from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores

def custom_collate(batch):
    collated = {}
    for key in batch[0]:
        if key == "caption_options":
            collated[key] = [sample[key] for sample in batch]
        else:
            collated[key] = torch.utils.data.default_collate([sample[key] for sample in batch])
    return collated


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use: 'cpu', 'cuda', or 'mps'")
    parser.add_argument("--batch-size", default=1, type=int,
                        help="Batch size for training (set to 1 if dataset structure requires it)")
    parser.add_argument("--eval-batch-size", default=1, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="xvlm-pretrained-16m", type=str,
                        help="Model name to load from model_zoo")
    parser.add_argument("--dataset", default="On_Under_Images", type=str,
                        help="Name of the dataset to fine-tune on (should point to your on/under dataset)")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight-decay", default=0.2, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--download", action="store_true",
                        help="Download the dataset if it doesn't exist")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--save-scores", action="store_true",
                        help="Whether to save the scores for analysis")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Skip training and only evaluate")
    return parser.parse_args()


def get_device(device_arg):
    if device_arg == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    elif device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print(f"Requested device '{device_arg}' not available, using CPU instead")
        return torch.device("cpu")


def load_xvlm_model(device):
    print("Loading xvlm-pretrained-16m model...")
    model, image_preprocess = get_model("xvlm-pretrained-16m", device=device)
    model.model.train()
    print(f"Model loaded and moved to {device}")
    return model, image_preprocess



def contrastive_loss(logits, target_idx, margin=0.2):
    """
    Computes a simple margin-based contrastive loss.
    Assumes that the correct caption is at target_idx.
    """
    positive_score = logits[target_idx]
    margins = margin - (positive_score - logits)
    margins[target_idx] = 0
    margins = torch.clamp(margins, min=0)
    return margins.sum()


def train_model(model, train_loader, optimizer, device, epochs, output_dir):
    best_accuracy = 0.0

    for epoch in range(epochs):
        model.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            optimizer.zero_grad()

            image = batch["image_options"][0].to(device)
            caption_options = batch["caption_options"]
            if isinstance(caption_options[0], (list, tuple)):
                caption_options = caption_options[0]

            if total == 0:
                print("Caption options:", caption_options)

            tokenized_text_data = model.tokenizer(
                caption_options,
                padding='max_length',
                truncation=True,
                max_length=model.config["max_tokens"],
                return_tensors="pt"
            ).to(device)

            if total == 0:
                decoded = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in tokenized_text_data.input_ids]
                print("Decoded tokenized captions:", decoded)

            # get image features
            image_features = model.model.vision_encoder(image)
            image_features = model.model.vision_proj(image_features[:, 0, :])
            image_features = F.normalize(image_features, dim=1)

            # get text features
            text_output = model.model.text_encoder(
                tokenized_text_data.input_ids,
                attention_mask=tokenized_text_data.attention_mask,
                mode='text'
            )
            text_features = F.normalize(
                model.model.text_proj(text_output.last_hidden_state[:, 0, :]),
                dim=1
            )

            # compute logits (scaled cosine similarity)
            logits = 100 * (image_features @ text_features.T)

            # for debugging
            if total == 0:
                print("Logits:", logits.detach().cpu().numpy())
                _, predicted = torch.max(logits, dim=1)
                print("Predicted index:", predicted.item(), "Target index:", 0)

            target = 0
            loss = contrastive_loss(logits[0], target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += 1
            _, predicted = torch.max(logits, dim=1)
            correct += (predicted == target).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / total,
                'accuracy': 100 * correct / total
            })


        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} -- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        # save checkpoint if improvement
        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            checkpoint_path = os.path.join(output_dir, f"xvlm_checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    print("Training completed!")
    return model


def evaluate_model(model, test_loader, device, dataset_name, output_dir, args):
    model.model.eval()

    print("Evaluating model...")

    scores = model.get_retrieval_scores_batched(test_loader)
    result_records = test_loader.dataset.evaluate_scores(scores)

    for record in result_records:
        record.update({"Model": f"{args.model_name}_finetuned", "Dataset": dataset_name, "Seed": args.seed})

    output_file = os.path.join(output_dir, f"{dataset_name}_results_finetuned.csv")
    df = pd.DataFrame(result_records)
    print(f"Saving evaluation results to {output_file}")
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
    device = get_device(args.device)
    model, image_preprocess = load_xvlm_model(device)

    train_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
        split="train"
    )
    test_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=False,
        split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_default_collate
    )

    if args.evaluate_only:
        evaluate_model(model, test_loader, device, args.dataset, args.output_dir, args)
        return

    optimizer = torch.optim.AdamW(model.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Pre-training evaluation:")
    evaluate_model(model, test_loader, device, args.dataset, args.output_dir, args)

    model = train_model(model, train_loader, optimizer, device, args.epochs, args.output_dir)

    print("Post-training evaluation:")
    evaluate_model(model, test_loader, device, args.dataset, args.output_dir, args)


if __name__ == "__main__":
    main()
