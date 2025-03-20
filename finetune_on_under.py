import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from tqdm import tqdm
import random
import numpy as np
import json
import clip

from torchvision import transforms
from PIL import Image


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])

def save_scores(scores, args):

    output_path = os.path.join(args.output_dir, f"{args.dataset}_scores.npy")
    np.save(output_path, scores)
    print(f"Saved raw scores to {output_path}")

class OnUnderDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, split="train", transform=None):
        super().__init__()
        with open(json_file, "r") as f:
            data = json.load(f)
        split_idx = int(0.8 * len(data))
        if split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

        self.transform = transform

        self.samples = []
        for entry in self.data:
            self.samples.append((entry["image_path"], entry["caption_options"][0], 1))
            self.samples.append((entry["image_path"], entry["caption_options"][1], 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "caption": caption,
            "label": label
        }


def load_negclip_model_unfrozen(device, root_dir="model_cache"):

    os.makedirs(root_dir, exist_ok=True)

    path = os.path.join(root_dir, "negclip.pth")
    if not os.path.exists(path):
        print("Downloading the NegCLIP model...")
        import gdown
        gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)

    model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None, device="cpu")
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)


    for name, param in model.named_parameters():
        param.requires_grad = False
        if name.startswith("token_embedding") or name.startswith("transformer") or "text_projection" in name:
            param.requires_grad = True

    model = model.to(device)
    return model

class SimilarityClassifier(nn.Module):

    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip_model = clip_model 
        self.classifier = nn.Linear(1, 2) 

    def forward(self, images, texts):

        image_features = self.clip_model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)

        text_tokens = clip.tokenize(texts).to(image_features.device)
        text_features = self.clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

        similarity = (image_features * text_features).sum(dim=-1)

        similarity = similarity.unsqueeze(1)

        logits = self.classifier(similarity)
        return logits


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        captions = batch["caption"]
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, captions) 
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0, 0, 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        captions = batch["caption"]
        labels = batch["label"].to(device)

        logits = model(images, captions)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train-json", default="data/on_under_images.json", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output-dir", default="./outputs_ce", type=str)
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clip_model = load_negclip_model_unfrozen(device)

    model = SimilarityClassifier(clip_model).to(device)

    train_dataset = OnUnderDataset(json_file=args.train_json, split="train", transform=train_transforms)
    test_dataset = OnUnderDataset(json_file=args.train_json, split="test", transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        print(f"  [Train] loss={train_loss:.4f}, acc={train_acc*100:.2f}%")

        val_loss, val_acc = evaluate(model, test_loader, device)
        print(f"  [Test]  loss={val_loss:.4f}, acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt_path = os.path.join(args.output_dir, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved at {ckpt_path}")

    print(f"\nTraining complete. Best test accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
