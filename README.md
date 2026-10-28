# Adversarial Image Perturbation Demo

A Python-based demonstration of adversarial attacks against a pretrained **ResNet-18 ImageNet classifier** using PyTorch and TorchVision.

This project implements four gradient-based adversarial attack methods: FGSM, PGD, DeepFool, and a simplified Carlini–Wagner (C&W) L2 attack. It evaluates the resulting adversarial images, compares classification predictions, calculates perturbation metrics, and saves the outputs for visual analysis.

## Features

- Pretrained TorchVision ResNet-18 ImageNet classifier.
- Four adversarial attack implementations:
  - **FGSM** — Fast Gradient Sign Method
  - **PGD** — Projected Gradient Descent
  - **DeepFool** — iterative decision-boundary attack
  - **C&W L2** — simplified Carlini–Wagner L2 optimization
- Automatic selection of CUDA or CPU.
- Original and adversarial prediction comparison.
- Perturbation metrics: L∞ distance, L2 distance, and pixel RMSE.
- Automatic saving of generated images.
- Side-by-side visualization of the original image and all four adversarial outputs.

## Technologies Used

- Python
- PyTorch
- TorchVision
- Pillow
- Matplotlib

## Project Structure

```text
adversarial-image-perturbation/
├── adversarial_demo.py
├── test.png
├── README.md
└── adversarial_results/
    ├── original.png
    ├── fgsm.png
    ├── pgd.png
    ├── deepfool.png
    ├── cw_l2.png
    └── comparison.png
```

`test.png` is the example input image. The script creates the `adversarial_results/` directory automatically. The output files appear after the script runs successfully.

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd adversarial-image-perturbation
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Install dependencies

Python 3.10 or newer is a reasonable starting point, subject to compatibility with your selected PyTorch and TorchVision versions.

```bash
python -m pip install --upgrade pip
pip install torch torchvision pillow matplotlib
```

For GPU acceleration, install a PyTorch version compatible with your system and CUDA environment. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Input Image Configuration

The script currently uses the following configuration:

```python
IMAGE_PATH = r"..\test.png"
```

Place your input image at the specified path, or modify `IMAGE_PATH` to point to your image.

For example, if the image is in the same directory as `adversarial_demo.py`, use:

```python
IMAGE_PATH = "test.png"
```

The script converts the image to RGB and preprocesses it for the ResNet-18 classifier by resizing, center cropping to 224 × 224 pixels, and converting it to a tensor.

## Run the Project

Execute the Python script:

```bash
python adversarial_demo.py
```

On the first run, TorchVision downloads the pretrained ResNet-18 ImageNet weights. An internet connection is required for this initial download unless the weights are already cached locally.

The script automatically selects CUDA when available and otherwise uses the CPU.

## Adversarial Attack Methods

### 1. FGSM — Fast Gradient Sign Method

FGSM generates an adversarial image in one gradient-based step by perturbing the input in the direction that increases the classification loss.

Configured perturbation magnitude:

```python
EPSILON = 8 / 255
```

### 2. PGD — Projected Gradient Descent

PGD iteratively updates the input image using gradient ascent and projects each update back into the permitted perturbation region.

Configuration:

```python
PGD_ALPHA = 2 / 255
PGD_STEPS = 20
```

The implementation starts from a random perturbation within the specified L∞ constraint.

### 3. DeepFool

DeepFool iteratively estimates the nearest decision boundary among a subset of competing classes and adjusts the image to cross the classifier's boundary.

Configuration:

```python
DEEPFOOL_STEPS = 30
DEEPFOOL_CLASSES = 10
```

The implementation considers the highest-scoring classes at each iteration and stops early if the predicted class changes.

### 4. Carlini–Wagner L2

The simplified C&W L2 implementation optimizes a classification margin objective alongside the squared L2 distance between the original and adversarial images.

Configuration:

```python
CW_STEPS = 300
CW_LEARNING_RATE = 0.01
CW_C = 1.0
CW_KAPPA = 0.0
```

The script tracks the lowest-L2 successful adversarial candidate found during optimization. If no successful candidate is found, it returns the final optimization candidate and reports the unsuccessful search.

**Note:** This is a simplified implementation, not a complete reproduction of the original C&W attack with its full optimization and binary-search procedure.

## Evaluation Metrics

For each generated image, the script reports:

| Metric | Description |
|---|---|
| Prediction | ImageNet class predicted for the adversarial image |
| Confidence | Softmax probability of the predicted class |
| Prediction changed | Whether the predicted class differs from the original prediction |
| L∞ difference | Maximum absolute pixel difference |
| L2 difference | Euclidean norm of the pixel perturbation |
| Pixel RMSE | Root mean squared pixel difference |

The script also saves a comparison figure displaying the original and adversarial images with their predicted labels and confidence scores.

These metrics quantify changes to the classifier input; they do not, by themselves, establish perceptual similarity or attack success against other models.

## Output Files

After a successful execution, the `adversarial_results/` directory contains:

| File | Description |
|---|---|
| `original.png` | Preprocessed original image |
| `fgsm.png` | FGSM-generated image |
| `pgd.png` | PGD-generated image |
| `deepfool.png` | DeepFool-generated image |
| `cw_l2.png` | C&W L2-generated image |
| `comparison.png` | Side-by-side visual comparison |

The console also prints the predicted class, confidence, prediction-change status, and perturbation metrics for each attack.

## Limitations

- The classifier is pretrained on ImageNet and is not fine-tuned by this script.
- FGSM and PGD use the original predicted class as the optimization label. They are intended as untargeted attacks, but a changed prediction is not guaranteed.
- DeepFool searches a limited set of high-scoring classes rather than all possible ImageNet classes.
- C&W L2 uses a fixed `C` value and does not implement the complete original attack procedure.
- The script does not calculate a formal aggregate attack success rate across a dataset; it evaluates one input image per run.
- The image is resized and center-cropped before evaluation, so the saved outputs represent the processed 224 × 224 input rather than the original full-resolution photograph.
- Runtime depends on the attack, hardware, and selected parameters. C&W L2 may be particularly slow on CPU.

## Responsible Use

This project is intended for educational purposes, controlled experiments, and authorized research into machine-learning robustness. Use it only with images and models you own or are authorized to test.

## References

1. Goodfellow, I. J., Shlens, J., and Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR.
2. Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.
3. Moosavi-Dezfooli, S.-M. et al. (2016). *DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks*. CVPR.
4. Carlini, N., and Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks*. IEEE Symposium on Security and Privacy.

