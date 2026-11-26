# backend/models/cnn_model.py

from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image, UnidentifiedImageError
from torchvision import transforms


class CNNNetwork(nn.Module):
    """
    Custom FALSEFACET CNN architecture.

    Input:  RGB image, 128x128
    Output: 2 classes
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 2)
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x


class CNNModel:
    """
    FALSEFACET CNN inference wrapper.

    Class mapping:
        0 = REAL
        1 = FAKE
    """

    IMAGE_SIZE = 128

    CLASS_NAMES = {
        0: "REAL",
        1: "FAKE"
    }

    def __init__(self, checkpoint_path=None, device=None):

        self.project_root = Path(__file__).resolve().parents[2]

        if checkpoint_path is None:
            checkpoint_path = (
                self.project_root
                / "models"
                / "cnn"
                / "checkpoints"
                / "best_model.pt"
            )

        self.checkpoint_path = Path(checkpoint_path)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Basic image preprocessing.
        # Confirm normalization against the original CNN training transform.
        self.transform = transforms.Compose([
            transforms.Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        self.model = None

        self.load_model()

    def load_model(self):
        """Load the trained CNN checkpoint."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"CNN checkpoint not found: {self.checkpoint_path}"
            )

        print("Loading FALSEFACET CNN...")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Device: {self.device}")

        state_dict = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self.model = CNNNetwork()

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        print("CNN checkpoint loaded successfully!")

    def preprocess_image(self, image):
        """Prepare an image for CNN inference."""

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

        tensor = self.transform(image)

        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image):
        """Predict REAL or FAKE."""

        if self.model is None:
            raise RuntimeError("CNN model is not loaded.")

        image_tensor = self.preprocess_image(image)

        logits = self.model(image_tensor)

        probabilities = torch.softmax(logits, dim=1)[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        return {
            "model": "CNN",
            "prediction": self.CLASS_NAMES[predicted_class],
            "class_id": predicted_class,
            "real_probability": round(real_probability, 4),
            "fake_probability": round(fake_probability, 4),
            "real_percentage": round(real_probability * 100, 2),
            "fake_percentage": round(fake_probability * 100, 2)
        }
