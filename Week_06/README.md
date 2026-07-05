# Week_06 Project 1: Image Denoising Autoencoder on MNIST

This project trains a deep learning autoencoder to remove random noise from MNIST handwritten digit images.

## Dataset

The data is stored as PNG files:

```text
mnist_png/
  testing/
    0/
    1/
    2/
    3/
    4/
```

Each subfolder name is the digit label. For denoising, labels are not used directly. The model receives a noisy image as input and learns to reconstruct the original clean image.

## Model

The model is a convolutional denoising autoencoder:

- Encoder compresses the image into a smaller latent representation.
- Decoder reconstructs a clean image from that compressed representation.
- Mean squared error is used to compare the denoised output with the clean target image.

## Run

From the `Week_06` folder:

```bash
python train_autoencoder.py
```

Optional shorter test run:

```bash
python train_autoencoder.py --epochs 1 --batch-size 64
```

## Outputs

After training, files are saved in `outputs/`:

- `best_denoising_autoencoder.pth`: best model checkpoint
- `denoising_examples.png`: clean, noisy, and denoised sample comparison

## Notes

The current dataset contains digits `0` to `4`. That is enough for the denoising task because the network learns image reconstruction, not digit classification.
