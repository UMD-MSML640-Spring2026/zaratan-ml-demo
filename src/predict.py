import argparse
import os

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


class DigitMLP(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare_test_split(seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    X_train, _, _, _ = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_val,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_test = (X_test - mean) / std
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Run one prediction using the saved model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default="outputs/prediction.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DigitMLP(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test, y_test = prepare_test_split(seed=args.seed)

    idx = args.index % len(X_test)
    sample = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sample)
        pred = torch.argmax(logits, dim=1).item()

    true_label = int(y_test[idx])

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"Sample index: {idx}\n")
        f.write(f"Predicted label: {pred}\n")
        f.write(f"True label: {true_label}\n")

    print(f"Sample index: {idx}")
    print(f"Predicted label: {pred}")
    print(f"True label: {true_label}")
    print(f"Saved prediction to {args.output_file}")


if __name__ == "__main__":
    main()
import argparse
import os

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


class DigitMLP(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def prepare_test_split(seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )

    X_train, _, _, _ = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.2,
        random_state=seed,
        stratify=y_train_val,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0

    X_test = (X_test - mean) / std
    return X_test, y_test


def main():
    parser = argparse.ArgumentParser(description="Run one prediction using the saved model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-file", type=str, default="outputs/prediction.txt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DigitMLP(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X_test, y_test = prepare_test_split(seed=args.seed)

    idx = args.index % len(X_test)
    sample = torch.tensor(X_test[idx], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(sample)
        pred = torch.argmax(logits, dim=1).item()

    true_label = int(y_test[idx])

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(f"Sample index: {idx}\n")
        f.write(f"Predicted label: {pred}\n")
        f.write(f"True label: {true_label}\n")

    print(f"Sample index: {idx}")
    print(f"Predicted label: {pred}")
    print(f"True label: {true_label}")
    print(f"Saved prediction to {args.output_file}")


if __name__ == "__main__":
    main()