import os

# Prevent Transformers from importing TensorFlow.
# This project uses PyTorch.
os.environ["USE_TF"] = "0"

import torch
import numpy as np

from pathlib import Path
from PIL import Image, UnidentifiedImageError

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from transformers import ViTConfig, ViTForImageClassification

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "models"
    / "vit"
    / "checkpoints"
    / "vit_best.pt"
)

DATASET_PATH = PROJECT_ROOT / "dataset"

VALIDATION_DIR = DATASET_PATH / "validation"
TEST_DIR = DATASET_PATH / "test"

IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 0

CLASS_NAMES = ["REAL", "FAKE"]

CLASS_TO_IDX = {
    "real": 0,
    "fake": 1,
}

IDX_TO_CLASS = {
    0: "REAL",
    1: "FAKE",
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("FALSEFACET - VISION TRANSFORMER VALIDATION")
print("=" * 65)

print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# IMAGE TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

class ViTValidationDataset(Dataset):

    def __init__(self, dataset_dir, transform=None):

        self.dataset_dir = Path(dataset_dir)
        self.transform = transform
        self.samples = []

        if not self.dataset_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_dir}"
            )

        for class_name, label in CLASS_TO_IDX.items():

            class_dir = self.dataset_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Class directory not found: {class_dir}"
                )

            for image_path in class_dir.rglob("*"):

                if image_path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                }:
                    self.samples.append((image_path, label))

        if not self.samples:
            raise RuntimeError(
                f"No images found in {self.dataset_dir}"
            )

        print(f"\nDataset: {self.dataset_dir}")
        print(f"Total images: {len(self.samples)}")

        for class_name, label in CLASS_TO_IDX.items():

            count = sum(
                1 for _, sample_label in self.samples
                if sample_label == label
            )

            print(f"{class_name.upper()}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:
            image = Image.open(image_path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label

        except (OSError, UnidentifiedImageError) as error:
            raise RuntimeError(
                f"Failed to load image: {image_path}\n"
                f"Error: {error}"
            )


# ============================================================
# CHECKPOINT HANDLING
# ============================================================

def extract_state_dict(checkpoint):

    """
    Supports:
    1. checkpoint["model_state_dict"]
    2. checkpoint["state_dict"]
    3. Direct model state_dict
    """

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Invalid checkpoint format."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        # Check whether this is a direct state dictionary.
        if not checkpoint or not all(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            raise KeyError(
                "Could not find model_state_dict or state_dict."
            )

        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    if all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    # Support ViTBinary checkpoints saved using wrapper.state_dict().
    if all(key.startswith("model.") for key in state_dict):
        state_dict = {
            key.removeprefix("model."): value
            for key, value in state_dict.items()
        }

    return state_dict


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\nLoading existing ViT checkpoint...")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load the existing checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    # Use the saved configuration if the checkpoint contains one.
    saved_config = None

    if isinstance(checkpoint, dict):

        saved_config = (
            checkpoint.get("config")
            or checkpoint.get("model_config")
        )

    if isinstance(saved_config, dict):

        config = ViTConfig.from_dict(saved_config)

        print("Using configuration saved in checkpoint.")

    else:

        # Local architecture initialization.
        # Does not download pretrained weights.
        config = ViTConfig(
            image_size=IMAGE_SIZE,
            num_channels=3,
            num_labels=2,
        )

        print(
            "No saved ViT configuration found."
        )
        print(
            "Using default ViT-Base Patch16-224 configuration."
        )

    # Ensure the classifier has two output classes.
    if config.num_labels != 2:
        raise ValueError(
            f"Expected 2 classes, got {config.num_labels}."
        )

    model = ViTForImageClassification(config)

    # Load trained weights.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("\nViT checkpoint loaded successfully.")
    print("Architecture: ViTForImageClassification")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Number of classes: {config.num_labels}")

    if isinstance(checkpoint, dict):

        if "epoch" in checkpoint:
            print(f"Checkpoint epoch: {checkpoint['epoch']}")

        if "loss" in checkpoint:
            print(f"Checkpoint loss: {checkpoint['loss']}")

    print("Checkpoint weights loaded without retraining.")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, dataloader, split_name):

    print("\n" + "=" * 65)
    print(f"EVALUATING {split_name.upper()} DATASET")
    print("=" * 65)

    all_labels = []
    all_predictions = []
    all_probabilities = []

    total_loss = 0.0
    total_samples = 0

    criterion = torch.nn.CrossEntropyLoss()

    model.eval()

    with torch.inference_mode():

        for batch_idx, (images, labels) in enumerate(dataloader):

            images = images.to(
                DEVICE,
                non_blocking=True,
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(pixel_values=images)

            logits = outputs.logits

            loss = criterion(logits, labels)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            print(
                f"Batch [{batch_idx + 1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )

    # Convert results to NumPy arrays.
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    average_loss = total_loss / total_samples

    # Metrics.
    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=1,
        zero_division=0,
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    # Display results.
    print("\nRESULTS")
    print("-" * 65)

    print(f"Average Loss : {average_loss:.4f}")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Precision    : {precision * 100:.2f}%")
    print(f"Recall       : {recall * 100:.2f}%")
    print(f"F1-Score     : {f1 * 100:.2f}%")

    print("\nCONFUSION MATRIX")
    print("-" * 65)

    print("                 Predicted")
    print("                 REAL    FAKE")

    print(
        f"Actual REAL     {cm[0][0]:5d}   {cm[0][1]:5d}"
    )

    print(
        f"Actual FAKE     {cm[1][0]:5d}   {cm[1][1]:5d}"
    )

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "labels": all_labels,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


# ============================================================
# CREATE DATALOADER
# ============================================================

def create_dataloader(dataset_dir):

    dataset = ViTValidationDataset(
        dataset_dir,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # Load the existing trained model.
    model = load_model()

    results = {}

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    if VALIDATION_DIR.exists():

        validation_loader = create_dataloader(
            VALIDATION_DIR
        )

        results["validation"] = evaluate_model(
            model,
            validation_loader,
            "Validation",
        )

    else:

        print(
            f"\nValidation directory not found: {VALIDATION_DIR}"
        )

    # --------------------------------------------------------
    # TEST DATASET
    # --------------------------------------------------------

    if TEST_DIR.exists():

        test_loader = create_dataloader(
            TEST_DIR
        )

        results["test"] = evaluate_model(
            model,
            test_loader,
            "Test",
        )

    else:

        print(
            f"\nTest directory not found: {TEST_DIR}"
        )

    # --------------------------------------------------------
    # COMPLETION
    # --------------------------------------------------------

    if not results:
        raise RuntimeError(
            "No validation or test dataset was evaluated."
        )

    print("\n" + "=" * 65)
    print("FALSEFACET ViT EVALUATION COMPLETED")
    print("=" * 65)

    print("\nThe existing ViT checkpoint was evaluated.")
    print("No training or weight updates were performed.")


if __name__ == "__main__":
    main()