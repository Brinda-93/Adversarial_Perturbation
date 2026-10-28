"""
Adversarial perturbation demo: FGSM, PGD, DeepFool, and simplified C&W L2.

Use only on models and images you own or are authorized to test.

Install:
    pip install torch torchvision pillow matplotlib

Run:
    python adversarial_demo.py

Place input.jpg next to this script, or change IMAGE_PATH below.
The script downloads pretrained TorchVision ResNet-18 ImageNet weights on first run.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------- Configuration ---------------------------

IMAGE_PATH = r"E:\Projects\Python\CV\Adverserial_perturbation\test.png"
OUTPUT_DIR = Path("adversarial_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 20

DEEPFOOL_STEPS = 30
DEEPFOOL_CLASSES = 10

CW_STEPS = 300
CW_LEARNING_RATE = 0.01
CW_C = 1.0
CW_KAPPA = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------- Model and input ---------------------------

weights = models.ResNet18_Weights.DEFAULT
base_model = models.resnet18(weights=weights).to(device).eval()

mean = torch.tensor(
    [0.485, 0.456, 0.406], device=device
).view(1, 3, 1, 1)
std = torch.tensor(
    [0.229, 0.224, 0.225], device=device
).view(1, 3, 1, 1)


class NormalizedModel(nn.Module):
    """Accept RGB tensors in [0, 1], then apply ImageNet normalization."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model((x - mean) / std)


classifier = NormalizedModel(base_model).eval()
categories = weights.meta["categories"]

if not Path(IMAGE_PATH).is_file():
    raise FileNotFoundError(
        f"Could not find {IMAGE_PATH!r}. Put an image at that path "
        "or edit IMAGE_PATH at the top of the script."
    )

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

pil_image = Image.open(IMAGE_PATH).convert("RGB")
x = preprocess(pil_image).unsqueeze(0).to(device).detach()


# --------------------------- Helpers ---------------------------

def predict(image_tensor):
    with torch.no_grad():
        logits = classifier(image_tensor)
        probabilities = logits.softmax(dim=1)
        class_id = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, class_id].item()
    return class_id, confidence


def save_image(image_tensor, filename):
    tensor = image_tensor.detach().cpu().squeeze(0).clamp(0, 1)
    array = (tensor * 255).round().byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(OUTPUT_DIR / filename)


def distortion(original, adversarial):
    difference = adversarial.detach() - original.detach()
    return {
        "linf": difference.abs().max().item(),
        "l2": difference.flatten().norm(p=2).item(),
        "rmse": difference.square().mean().sqrt().item(),
    }


def evaluate(name, adversarial):
    class_id, confidence = predict(adversarial)
    metrics = distortion(x, adversarial)

    print(f"\n{name}")
    print(f"  Prediction: {categories[class_id]} (class {class_id})")
    print(f"  Confidence: {confidence:.4f}")
    print(f"  Prediction changed: {class_id != original_class}")
    print(f"  L-inf difference: {metrics['linf']:.6f}")
    print(f"  L2 difference:    {metrics['l2']:.6f}")
    print(f"  Pixel RMSE:       {metrics['rmse']:.6f}")

    filename = name.lower().replace(" ", "_") + ".png"
    save_image(adversarial, filename)
    print(f"  Saved: {OUTPUT_DIR / filename}")


original_class, original_confidence = predict(x)
print(
    f"Original prediction: {categories[original_class]} "
    f"(class {original_class}, confidence {original_confidence:.4f})"
)
save_image(x, "original.png")


# --------------------------- FGSM ---------------------------

def fgsm_attack(image, epsilon):
    adversarial_input = image.detach().clone().requires_grad_(True)
    logits = classifier(adversarial_input)
    label = torch.tensor([original_class], device=device)
    loss = F.cross_entropy(logits, label)
    gradient = torch.autograd.grad(loss, adversarial_input)[0]

    adversarial = image + epsilon * gradient.sign()
    return adversarial.detach().clamp(0, 1)


fgsm = fgsm_attack(x, EPSILON)
evaluate("FGSM", fgsm)


# --------------------------- PGD ---------------------------

def pgd_attack(image, epsilon, alpha, steps):
    original = image.detach()
    adversarial = original + torch.empty_like(original).uniform_(
        -epsilon, epsilon
    )
    adversarial = adversarial.clamp(0, 1)
    label = torch.tensor([original_class], device=device)

    for _ in range(steps):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = classifier(adversarial)
        loss = F.cross_entropy(logits, label)
        gradient = torch.autograd.grad(loss, adversarial)[0]

        adversarial = adversarial.detach() + alpha * gradient.sign()
        delta = torch.clamp(
            adversarial - original, min=-epsilon, max=epsilon
        )
        adversarial = (original + delta).clamp(0, 1)

    return adversarial.detach()


pgd = pgd_attack(x, EPSILON, PGD_ALPHA, PGD_STEPS)
evaluate("PGD", pgd)


# --------------------------- DeepFool ---------------------------

def deepfool_attack(image, steps=30, overshoot=0.02, top_k=10):
    original = image.detach()
    current = original.clone()

    with torch.no_grad():
        initial_logits = classifier(original)
        original_label = initial_logits.argmax(dim=1).item()

    for _ in range(steps):
        current = current.detach().requires_grad_(True)
        logits = classifier(current)[0]
        current_label = logits.argmax().item()

        if current_label != original_label:
            break

        candidate_ids = logits.topk(
            min(top_k + 1, logits.numel())
        ).indices.tolist()

        grad_original = torch.autograd.grad(
            logits[original_label], current, retain_graph=True
        )[0]

        best_distance = float("inf")
        best_step = None

        for class_id in candidate_ids:
            if class_id == original_label:
                continue

            grad_other = torch.autograd.grad(
                logits[class_id], current, retain_graph=True
            )[0]
            w = (grad_other - grad_original).detach()
            score_gap = (
                logits[original_label] - logits[class_id]
            ).detach().item()
            norm_squared = w.flatten().dot(w.flatten()).item()

            if norm_squared < 1e-12:
                continue

            distance = abs(score_gap) / (norm_squared ** 0.5)
            if distance < best_distance:
                best_distance = distance
                best_step = (score_gap / norm_squared) * w

        if best_step is None:
            break

        current = (
            current.detach() + (1 + overshoot) * best_step
        ).clamp(0, 1)

    return current.detach()


deepfool = deepfool_attack(
    x, steps=DEEPFOOL_STEPS, top_k=DEEPFOOL_CLASSES
)
evaluate("DeepFool", deepfool)


# --------------------------- Simplified C&W L2 ---------------------------

def cw_l2_attack(
    image, steps=300, learning_rate=0.01, c=1.0, kappa=0.0
):
    original = image.detach()
    safe_image = original.clamp(1e-6, 1 - 1e-6)
    w_original = torch.atanh(2 * safe_image - 1)

    w = nn.Parameter(w_original.clone())
    optimizer = torch.optim.Adam([w], lr=learning_rate)
    label = torch.tensor([original_class], device=device)

    best_adversarial = original.clone()
    best_l2 = float("inf")
    found_success = False

    for step in range(steps):
        optimizer.zero_grad()
        adversarial = 0.5 * (torch.tanh(w) + 1)
        logits = classifier(adversarial)

        real = logits.gather(1, label[:, None]).squeeze(1)
        one_hot = F.one_hot(
            label, num_classes=logits.shape[1]
        ).bool()
        other = logits.masked_fill(
            one_hot, float("-inf")
        ).max(dim=1).values

        margin_loss = torch.clamp(real - other + kappa, min=0)
        l2_squared = (
            adversarial - original
        ).flatten(1).square().sum(dim=1)

        loss = (l2_squared + c * margin_loss).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate = 0.5 * (torch.tanh(w) + 1)
            candidate_logits = classifier(candidate)
            candidate_class = candidate_logits.argmax(dim=1).item()
            candidate_l2 = (
                candidate - original
            ).flatten().norm(p=2).item()

            success = candidate_class != original_class
            if success and candidate_l2 < best_l2:
                best_adversarial = candidate.detach().clone()
                best_l2 = candidate_l2
                found_success = True

        if (step + 1) % 100 == 0:
            best_text = f"{best_l2:.6f}" if found_success else "not found"
            print(
                f"C&W iteration {step + 1}/{steps}; "
                f"best successful L2: {best_text}"
            )

    if not found_success:
        print(
            "C&W did not find a successful example within these "
            "iterations and c value; returning the last candidate."
        )
        with torch.no_grad():
            best_adversarial = 0.5 * (torch.tanh(w) + 1)

    return best_adversarial.detach().clamp(0, 1)


cw = cw_l2_attack(
    x,
    steps=CW_STEPS,
    learning_rate=CW_LEARNING_RATE,
    c=CW_C,
    kappa=CW_KAPPA,
)
evaluate("CW L2", cw)


# --------------------------- Comparison figure ---------------------------

examples = [
    ("Original", x),
    ("FGSM", fgsm),
    ("PGD", pgd),
    ("DeepFool", deepfool),
    ("CW L2", cw),
]

fig, axes = plt.subplots(1, len(examples), figsize=(17, 4))
for axis, (name, example) in zip(axes, examples):
    array = (
        example.detach().cpu().squeeze(0)
        .permute(1, 2, 0).clamp(0, 1).numpy()
    )
    class_id, confidence = predict(example)
    axis.imshow(array)
    axis.set_title(
        f"{name}\n{categories[class_id]}\n{confidence:.2%}"
    )
    axis.axis("off")

plt.tight_layout()
comparison_path = OUTPUT_DIR / "comparison.png"
plt.savefig(comparison_path, dpi=160, bbox_inches="tight")
plt.show()

print(f"\nAll results saved in: {OUTPUT_DIR.resolve()}")
