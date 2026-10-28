# Adversarial Image Perturbation using PyTorch

A Python-based image processing project for generating and evaluating adversarial examples against a pretrained **ResNet-18 ImageNet classifier** using PyTorch and TorchVision.

The project implements four adversarial attack methods—Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), DeepFool, and Carlini–Wagner (C&W) L2—to investigate how small, controlled perturbations can affect image classification predictions.

## Overview

Deep learning image classifiers can be sensitive to carefully crafted changes in their input images. Adversarial attacks exploit this property by generating perturbed images that may cause a model to misclassify an image.

This project provides a local Python workflow to:

- Load an input image and preprocess it for ResNet-18.
- Generate adversarial examples using four attack algorithms.
- Compare the original and adversarial model predictions.
- Measure perturbation magnitude and image distortion.
- Save generated adversarial images for visual inspection and further analysis.

The Python implementation is intended to be developed and validated independently before potential integration into a native Android camera application.

## Algorithms

| Algorithm | Description | Primary objective |
|---|---|---|
| FGSM | Fast Gradient Sign Method | Generate a perturbation using the sign of the loss gradient in one step. |
| PGD | Projected Gradient Descent | Iteratively update the input while constraining the perturbation. |
| DeepFool | Iterative boundary-based attack | Find a small perturbation that crosses a classifier's decision boundary. |
| C&W L2 | Carlini–Wagner L2 attack | Optimize a classification objective while minimizing L2 distortion. |

The effectiveness of each attack depends on the implementation, hyperparameters, input preprocessing, and model predictions.

## Technology Stack

- **Python** — implementation and experimentation
- **PyTorch** — model inference, gradient computation, and optimization
- **TorchVision** — pretrained ResNet-18 and ImageNet preprocessing utilities
- **Pillow (PIL)** — image loading and saving
- **NumPy** — numerical operations and metric calculations
- **Matplotlib** — optional visualization of original and adversarial images

## Model

The prototype uses a pretrained ResNet-18 model trained on ImageNet.

The model is used to obtain predictions for the original image and its adversarial variants. During gradient-based attacks, the model's parameters remain fixed while gradients are computed with respect to the input image.

The model's expected input preprocessing must be applied consistently. For the standard TorchVision ImageNet ResNet-18 weights, this includes resizing and cropping, conversion to a tensor, and ImageNet normalization as specified by the selected weights.

## Project Structure

A suggested repository structure is:

```text
adversarial-image-perturbation/
├── README.md
├── requirements.txt
├── main.py
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py
│   ├── pgd.py
│   ├── deepfool.py
│   └── cw_l2.py
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   ├── metrics.py
│   └── model_utils.py
├── data/
│   └── input/
└── results/
    └── .gitkeep
```

*Note: This is a proposed structure. Adjust the filenames and directories to match the actual implementation.*

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd adversarial-image-perturbation
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on your operating system.

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Create a `requirements.txt` file containing the dependencies required by your implementation:

```text
torch
torchvision
Pillow
numpy
matplotlib
```

Install them with:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA acceleration, install a PyTorch build compatible with your installed NVIDIA driver and CUDA environment. Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Usage

Place an input image in the input directory, then run the Python entry point:

```bash
python main.py --image data/input/example.jpg
```

The command above is an example interface. The actual command-line arguments must match those implemented in `main.py`.

Depending on the implementation, you may be able to select an attack method, configure its parameters, and choose an output directory.

For example, a future command-line interface could support:

```bash
python main.py \
    --image data/input/example.jpg \
    --attack fgsm \
    --output-dir results/
```

Do not use these example commands until the corresponding arguments have been implemented.

## Experimental Workflow

1. **Load the image:** Read the input image and preserve a copy of the original.
2. **Preprocess the input:** Apply the preprocessing expected by the selected ResNet-18 weights.
3. **Run baseline inference:** Record the original predicted class and confidence score.
4. **Generate perturbations:** Apply FGSM, PGD, DeepFool, or C&W L2.
5. **Run adversarial inference:** Evaluate the perturbed image using the same classifier and preprocessing pipeline.
6. **Calculate metrics:** Measure the difference between the original and adversarial images.
7. **Save the results:** Export the adversarial images and record the corresponding predictions and metrics.

For reproducible comparisons, use the same input image, model weights, preprocessing configuration, and evaluation procedure across attacks.

## Evaluation Metrics

The following metrics can be used to evaluate the generated adversarial examples.

| Metric | Purpose |
|---|---|
| Original prediction | Records the classifier's predicted class before perturbation. |
| Adversarial prediction | Records the predicted class after perturbation. |
| Prediction confidence | Compares the model's confidence scores before and after perturbation. |
| L∞ distance | Measures the maximum absolute pixel-wise difference. |
| L2 distance | Measures the Euclidean norm of the perturbation. |
| Mean Squared Error (MSE) | Measures the average squared pixel-wise difference. |
| Peak Signal-to-Noise Ratio (PSNR) | Provides a signal-based measure of image distortion. |
| Attack success rate | Measures the proportion of evaluated inputs for which the attack meets its defined success criterion. |

For targeted attacks, success means reaching the specified target class. For untargeted attacks, success is commonly defined as changing the original correct prediction to another class. The evaluation should explicitly state which definition is used.

Calculate image-distortion metrics in a consistent pixel-value range, such as `[0, 1]`. If metrics are calculated on normalized model inputs, account for the normalization before interpreting the results as pixel-space distortion.

## Output

The results directory can contain:

- Generated adversarial images.
- Original and adversarial predicted class labels.
- Confidence scores.
- Perturbation magnitudes.
- Image-quality metrics.
- Experimental configuration and execution logs.

Example output organization:

```text
results/
├── fgsm/
│   └── example_fgsm.png
├── pgd/
│   └── example_pgd.png
├── deepfool/
│   └── example_deepfool.png
├── cw_l2/
│   └── example_cw_l2.png
└── metrics.csv
```

The output filenames and structure are illustrative and should be adapted to the actual implementation.

## Implementation Considerations

- **Input normalization:** Apply the preprocessing associated with the exact pretrained weights used.
- **Gradient computation:** FGSM, PGD, DeepFool, and C&W L2 require gradient-based optimization or related gradient calculations.
- **Model evaluation:** Use evaluation mode for inference while preserving gradient computation during attacks.
- **Perturbation constraints:** Define the permitted perturbation magnitude and input bounds for each algorithm.
- **Attack configuration:** Record algorithm-specific parameters, including step size, iteration count, and optimization settings.
- **Image saving:** Convert tensors back into the correct pixel range before saving images.
- **Reproducibility:** Record dependency versions, model weights, random seeds where applicable, and experimental settings.
- **Performance:** Attack execution time and memory consumption can vary significantly with the algorithm and hardware.

## Limitations

- Results depend on the pretrained classifier and its ImageNet label space.
- A successful attack against ResNet-18 does not automatically transfer to other models.
- Classification changes do not necessarily imply that a human observer will perceive a meaningful difference.
- Pixel-space perturbation metrics do not fully characterize perceptual similarity.
- White-box gradient-based attacks may be computationally expensive, particularly iterative optimization methods such as PGD and C&W L2.
- The Python prototype does not itself provide native Android camera access or guarantee that adversarial perturbations will survive image resizing, JPEG compression, or other camera-processing operations.

## Future Work

- Compare attack success rates and perturbation magnitudes across the four algorithms.
- Evaluate additional image classifiers and investigate transferability.
- Study robustness under resizing, compression, and other image transformations.
- Profile execution time and memory requirements.
- Validate a consistent preprocessing and image-saving pipeline for potential Android integration.
- Explore exporting compatible model components to an on-device inference runtime, subject to the requirements of each attack algorithm.

## Responsible Use

This project is intended for educational research, controlled experimentation, and the evaluation of machine-learning model robustness. Use it with images and systems you are authorized to test.

## References

- Goodfellow, I. J., Shlens, J., and Szegedy, C. (2015). *Explaining and Harnessing Adversarial Examples*. ICLR.
- Madry, A. et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.
- Moosavi-Dezfooli, S.-M. et al. (2016). *DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks*. CVPR.
- Carlini, N., and Wagner, D. (2017). *Towards Evaluating the Robustness of Neural Networks*. IEEE Symposium on Security and Privacy.

## License

Choose and add a license appropriate for your project before distributing the repository. If the project will be open source, see [Choose a License](https://choosealicense.com/).
