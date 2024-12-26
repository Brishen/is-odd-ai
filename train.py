#!/usr/bin/env python3
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional: for splitting train/test
from sklearn.model_selection import train_test_split

from DigitTransformer import DigitTransformer
from EvenOddDataset import EvenOddDataset

# ------------------------
# 3) Training and Evaluation
# ------------------------
def train_transformer(model, dataloader, optimizer, criterion, device="cpu"):
    model.train()
    total_loss = 0
    for batch in dataloader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_transformer(model, dataloader, criterion, device="cpu"):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == y).sum().item()
            total += len(y)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total if total > 0 else 0
    return avg_loss, accuracy


# ------------------------
# 4) Putting it All Together
# ------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate random numbers for training
    n_samples = 2000000
    min_val, max_val = -500000, 500000
    random_numbers = [random.randint(min_val, max_val) for _ in range(n_samples)]

    # Train/Test split
    train_nums, test_nums = train_test_split(random_numbers, test_size=0.2, random_state=42)

    # Create Datasets
    train_dataset = EvenOddDataset(train_nums, max_length=7)
    test_dataset = EvenOddDataset(test_nums, max_length=7)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=10240, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10240, shuffle=False)

    mult = 1
    # Build our Transformer model
    model = DigitTransformer(
        d_model=mult*32,
        nhead=mult*4,
        num_layers=mult*2,
        dim_feedforward=mult*64,
        max_length=7,
        vocab_size=12,  # 0..9, PAD=10, MINUS=11 => 12 tokens
        num_classes=2
    ).to(device)

    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train
    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss = train_transformer(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate_transformer(model, test_loader, criterion, device)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # ------------------------
    # 5) Save the Trained Model
    # ------------------------
    # Save just the state_dict (recommended practice)
    model_save_path = Path("my_transformer_model.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model weights saved to {model_save_path.resolve()}")

    # ------------------------
    # 6) Testing on new numbers
    # ------------------------
    test_numbers = [-42, -55, 0, 9998, -10003, 31415926]
    model.eval()
    with torch.no_grad():
        for num in test_numbers:
            abs_num = abs(num)
            label_truth = "Odd" if abs_num % 2 == 1 else "Even"

            # Tokenize
            digit_str = str(abs_num)
            digits = [int(d) for d in digit_str] if digit_str != "0" else [0]
            if num < 0:
                digits = [11] + digits  # minus token

            # Pad/truncate to max_length=7
            if len(digits) < 7:
                digits += [10] * (7 - len(digits))  # pad
            else:
                digits = digits[:7]

            x_tensor = torch.tensor([digits], dtype=torch.long).to(device)
            logits = model(x_tensor)
            pred_idx = torch.argmax(logits, dim=-1).item()
            label_pred = "Odd" if pred_idx == 1 else "Even"

            print(f"Number: {num:8d} => Predicted: {label_pred}, Actual: {label_truth}")



if __name__ == "__main__":
    main()
