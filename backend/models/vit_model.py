# backend/models/vit_model.py

from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification


class ViTModel:
    """
    FALSEFACET Vision Transformer inference model.

    Class mapping:
        0 = REAL
        1 = FAKE
    """

    IMAGE_SIZE = 224

    CLASS_NAMES = {
        0: "REAL",
        1: "FAKE"
    }

    def __init__(self, checkpoint_path=None, device=None):

        # Project root
        self.project_root = Path(__file__).resolve().parents[2]

        # Checkpoint location
        if checkpoint_path is None:
            checkpoint_path = (
                self.project_root
                / "models"
                / "vit"
                / "checkpoints"
                / "vit_best.pt"
            )

        self.checkpoint_path = Path(checkpoint_path)

        # Select device
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Image preprocessing: must match ViT training validation transform
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = None

        self.load_model()

    def load_model(self):
        """Load the trained ViT checkpoint."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"ViT checkpoint not found: {self.checkpoint_path}"
            )

        print("Loading FALSEFACET ViT model...")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")

        # Load checkpoint safely
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        # Extract model weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove common prefixes if present
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[len("module."):]

            if new_key.startswith("model."):
                new_key = new_key[len("model."):]

            cleaned_state_dict[new_key] = value

        # Recreate the architecture used during training
        config = ViTConfig(
            image_size=224,
            num_channels=3,
            num_labels=2
        )

        self.model = ViTForImageClassification(config)

        # Strict loading ensures the checkpoint matches the architecture
        self.model.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("ViT model loaded successfully!")

        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                print(f"Checkpoint epoch: {checkpoint['epoch']}")

            if "loss" in checkpoint:
                print(f"Checkpoint loss: {checkpoint['loss']}")

    def preprocess_image(self, image):
        """Convert an image into a normalized ViT input tensor."""

        if isinstance(image, (str, Path)):
            try:
                image = Image.open(image)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(
                    f"Unable to open image: {image}"
                ) from error

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Input must be an image path or PIL Image."
            )

        image = image.convert("RGB")

        image_tensor = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor.to(self.device)

    @torch.inference_mode()
    def predict(self, image):
        """Predict whether an image is REAL or FAKE."""

        if self.model is None:
            raise RuntimeError("ViT model has not been loaded.")

        image_tensor = self.preprocess_image(image)

        outputs = self.model(pixel_values=image_tensor)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )[0]

        predicted_class = int(torch.argmax(probabilities).item())

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        return {
            "model": "ViT",
            "prediction": self.CLASS_NAMES[predicted_class],
            "class_id": predicted_class,
            "real_probability": round(real_probability, 4),
            "fake_probability": round(fake_probability, 4),
            "real_percentage": round(real_probability * 100, 2),
            "fake_percentage": round(fake_probability * 100, 2)
        }
