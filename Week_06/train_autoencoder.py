from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class Config:
    data_dir: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    noise_factor: float
    validation_split: float
    seed: int
    num_workers: int


class MnistPngDataset(Dataset):
    """Loads MNIST-style PNG images arranged in class folders."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.image_paths = sorted(root_dir.glob("*/*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found under {root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("L").resize((28, 28))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)


class DenoisingAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def add_noise(images: torch.Tensor, noise_factor: float) -> torch.Tensor:
    noise = noise_factor * torch.randn_like(images)
    return torch.clamp(images + noise, 0.0, 1.0)


def psnr(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 20 * math.log10(1.0 / math.sqrt(mse))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    noise_factor: float,
) -> float:
    model.train()
    total_loss = 0.0

    for clean_images in loader:
        clean_images = clean_images.to(device)
        noisy_images = add_noise(clean_images, noise_factor)

        optimizer.zero_grad()
        denoised_images = model(noisy_images)
        loss = criterion(denoised_images, clean_images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * clean_images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    noise_factor: float,
) -> float:
    model.eval()
    total_loss = 0.0

    for clean_images in loader:
        clean_images = clean_images.to(device)
        noisy_images = add_noise(clean_images, noise_factor)
        denoised_images = model(noisy_images)
        loss = criterion(denoised_images, clean_images)
        total_loss += loss.item() * clean_images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def save_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: Path,
    noise_factor: float,
) -> None:
    model.eval()
    clean_images = next(iter(loader))[:8].to(device)
    noisy_images = add_noise(clean_images, noise_factor)
    denoised_images = model(noisy_images)

    rows = [
        ("Clean", clean_images.cpu()),
        ("Noisy", noisy_images.cpu()),
        ("Denoised", denoised_images.cpu()),
    ]

    fig, axes = plt.subplots(3, clean_images.size(0), figsize=(12, 4.8))
    for row_index, (label, images) in enumerate(rows):
        for col_index in range(clean_images.size(0)):
            ax = axes[row_index, col_index]
            ax.imshow(images[col_index].squeeze(0), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if col_index == 0:
                ax.set_ylabel(label, fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def create_loaders(config: Config) -> tuple[DataLoader, DataLoader, int]:
    dataset = MnistPngDataset(config.data_dir)
    validation_size = max(1, int(len(dataset) * config.validation_split))
    train_size = len(dataset) - validation_size

    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return train_loader, validation_loader, len(dataset)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Train a denoising autoencoder on MNIST PNG images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("mnist_png/testing"),
        help="Folder containing digit subfolders with PNG images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Where model checkpoints and sample images are saved.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--noise-factor", type=float, default=0.30)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    return Config(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        noise_factor=args.noise_factor,
        validation_split=args.validation_split,
        seed=args.seed,
        num_workers=args.num_workers,
    )


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, total_images = create_loaders(config)

    model = DenoisingAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print(f"Loaded {total_images} images from {config.data_dir}")
    print(f"Training on {device} with noise_factor={config.noise_factor}")

    best_validation_loss = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.noise_factor,
        )
        validation_loss = evaluate(
            model,
            validation_loader,
            criterion,
            device,
            config.noise_factor,
        )

        print(
            f"Epoch {epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.5f} "
            f"val_loss={validation_loss:.5f} "
            f"val_psnr={psnr(validation_loss):.2f}dB"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "validation_loss": validation_loss,
                },
                config.output_dir / "best_denoising_autoencoder.pth",
            )

    save_reconstruction_grid(
        model,
        validation_loader,
        device,
        config.output_dir / "denoising_examples.png",
        config.noise_factor,
    )
    print(f"Saved model to {config.output_dir / 'best_denoising_autoencoder.pth'}")
    print(f"Saved examples to {config.output_dir / 'denoising_examples.png'}")


if __name__ == "__main__":
    main()
