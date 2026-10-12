"""Frozen Point-JEPA max+mean feature evaluation with the paper's linear SVM."""

import argparse
from pathlib import Path

import numpy as np
import torch

import stable_pretraining as spt
from sklearn.svm import SVC
from torch.utils.data import DataLoader


@torch.no_grad()
def features(
    model: spt.PointJEPA, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pooled student features and labels from a ModelNet40 split."""
    embeddings, labels = [], []
    model.eval()
    for batch in loader:
        output = model({"points": batch["points"].to(device)}, "test")
        embeddings.append(output["embedding"].cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(embeddings), np.concatenate(labels)


def main() -> None:
    """Fit linear SVC with C=0.012 on frozen train features and score test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    parameters = checkpoint["hyper_parameters"]
    model = spt.PointJEPA(
        **{
            key: parameters[key]
            for key in (
                "num_groups",
                "group_size",
                "embed_dim",
                "encoder_depth",
                "encoder_heads",
                "predictor_dim",
                "predictor_depth",
                "predictor_heads",
                "lr",
                "weight_decay",
                "ema_decay_start",
                "ema_decay_end",
                "accumulate_steps",
            )
        }
    )
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    transform = spt.data.PointCloudTransform(training=False)
    train = DataLoader(
        spt.data.ModelNet40H5(args.root, "train"),
        batch_size=args.batch_size,
        collate_fn=transform.collate,
    )
    test = DataLoader(
        spt.data.ModelNet40H5(args.root, "test"),
        batch_size=args.batch_size,
        collate_fn=transform.collate,
    )
    x_train, y_train = features(model, train, device)
    x_test, y_test = features(model, test, device)
    classifier = SVC(kernel="linear", C=0.012).fit(x_train, y_train)
    print(f"ModelNet40 linear SVM accuracy: {classifier.score(x_test, y_test):.4f}")


if __name__ == "__main__":
    main()
