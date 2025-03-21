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
import random
import numpy as np
import json
import clip
from model_zoo import get_model
from dataset_zoo import get_dataset
from misc import seed_all, _default_collate, save_scores
from model_zoo.xvlm_models import XVLMWrapper

CACHE_DIR = "./model_cache"
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


def config():
    parser = argparse.ArgumentParser()
    # Default to 'cuda' if available
    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to use: 'cpu', 'cuda' if available, or 'mps' if available on MacOS")
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--eval-batch-size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--model-name", default="xvlm-pretrained-16m", type=str)
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--lr", default=5e-6, type=float)
    parser.add_argument("--weight-decay", default=0.2, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--download", action="store_true", help="Download the dataset if it doesn't exist")
    parser.add_argument("--output-dir", default="./outputs", type=str)
    parser.add_argument("--save-scores", action="store_true", help="Whether to save the scores for analysis")
    parser.add_argument("--evaluate-only", action="store_true", help="Skip training and only evaluate")
    return parser.parse_args()


def load_xvlm_model():
    """
    Load the xvlm model for fine-tuning, safely handling device mapping
    """
    model, image_preprocess = get_model("xvlm-pretrained-16m", "cpu")
    return model, image_preprocess


def train_xvlm_on_controlled_images(model, train_loader, optimizer, device, epochs, output_dir, args):
    import torch.nn.functional as F

    def cross_entropy_loss(logits_i2t, logits_t2i, targets):

        # Apply temperature scaling to make logits more discriminative
        temperature = 0.07
        scaled_i2t = logits_i2t / temperature
        scaled_t2i = logits_t2i / temperature

        # Compute cross-entropy loss for both directions
        i2t_loss = F.cross_entropy(scaled_i2t.unsqueeze(0), targets)
        t2i_loss = F.cross_entropy(scaled_t2i.unsqueeze(0), targets)

        # Combine both losses
        return i2t_loss + t2i_loss


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

            # Process image features with XVLM
            image_feat = model.model.vision_encoder(image.to(device))
            image_embed = model.model.vision_proj(image_feat[:, 0, :])  # Take [CLS] token
            image_embed = F.normalize(image_embed, dim=1)

            # Process captions with XVLM
            caption_options = batch["caption_options"]
            text_embeds = []
            text_features = []
            text_attention_masks = []

            for caption in caption_options:
                # Tokenize and encode text
                text_input = model.tokenizer(caption, padding='max_length', truncation=True,
                                             max_length=model.config["max_tokens"], return_tensors="pt").to(device)

                # Get text embeddings
                text_output = model.model.text_encoder(text_input.input_ids,
                                                       attention_mask=text_input.attention_mask,
                                                       mode='text')
                text_embed = model.model.text_proj(text_output.last_hidden_state[:, 0, :])  # Get [CLS] token
                text_embed = F.normalize(text_embed, dim=1)

                text_embeds.append(text_embed)
                text_features.append(text_input.input_ids)
                text_attention_masks.append(text_input.attention_mask)

            # Compute image-to-text similarity scores
            text_embeds_cat = torch.cat(text_embeds, dim=0)
            i2t_scores = 100 * (image_embed @ text_embeds_cat.T)

            # Compute text-to-image similarity scores using XVLM's cross-attention
            t2i_scores = torch.zeros_like(i2t_scores)

            for i, (text_ids, text_atts) in enumerate(zip(text_features, text_attention_masks)):
                # Use the image as encoder hidden states for cross-attention
                encoder_output = image_feat.repeat(text_ids.size(0), 1, 1)
                encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

                # Run cross-attention
                output = model.model.text_encoder(text_ids,
                                                  attention_mask=text_atts,
                                                  encoder_hidden_states=encoder_output,
                                                  encoder_attention_mask=encoder_att,
                                                  return_dict=True,
                                                  mode='fusion')

                # Get image-text matching score
                itm_score = model.model.itm_head(output.last_hidden_state[:, 0, :])
                t2i_scores[0, i] = itm_score[0, 1] * 100  # Scale to match i2t_scores

            # Correct caption is always at index 0
            targets = torch.zeros(batch_size, dtype=torch.long, device=device)
            loss = cross_entropy_loss(i2t_scores[0], t2i_scores[0], targets)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy based on combined scores for better alignment
            combined_scores = i2t_scores + t2i_scores
            _, predicted = torch.max(combined_scores, 1)
            correct += (predicted == targets).sum().item()

            # Track individual accuracies for analysis
            _, i2t_predicted = torch.max(i2t_scores, 1)
            _, t2i_predicted = torch.max(t2i_scores, 1)
            i2t_correct = (i2t_predicted == targets).sum().item()
            t2i_correct = (t2i_predicted == targets).sum().item()

            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100 * correct / total,
                'i2t_acc': 100 * i2t_correct / total,
                't2i_acc': 100 * t2i_correct / total
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

        if epoch_acc > best_accuracy:
            best_accuracy = epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"{output_dir}/checkpoint_epoch_{epoch + 1}.pt")

    print("Training completed!")

    return model


def evaluate_model(model, test_loader, device, dataset, args):
    """
    Evaluate the model on the test dataset
    """
    model.model.eval()
    print("Evaluating model...")

    scores = model.get_retrieval_scores_batched(test_loader)
    result_records = dataset.evaluate_scores(scores)

    for record in result_records:
        record.update({"Model": f"{args.model_name}_finetuned", "Dataset": args.dataset, "Seed": args.seed})

    output_file = os.path.join(args.output_dir, f"{args.dataset}_xvlm_results_finetuned.csv")
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
    device = "cpu"

    model, image_preprocess = load_xvlm_model()

    train_dataset = get_dataset(
        args.dataset,
        image_preprocess=image_preprocess,
        download=args.download,
        split="train"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Use batch size of 1 for this dataset structure
        shuffle=False,
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

 #   evaluate_model(model, test_loader, device, test_dataset, args)

    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print(f"Starting training for {args.epochs} epochs...")
    model = train_xvlm_on_controlled_images(
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