import torch

from models.pattern.classes import CLASS_NAMES
from models.pattern.model import EfficientNetB0Classifier, MobileNetV3SmallClassifier
from models.pattern.transforms import PadToSquare, val_transform
from PIL import Image


def test_v3_small_head_outputs_six_logits() -> None:
    model = MobileNetV3SmallClassifier(num_classes=len(CLASS_NAMES), pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 64))
    assert logits.shape == (2, 6)


def test_pad_to_square_makes_equal_sides() -> None:
    image = Image.new("RGB", (24, 37), color=(8, 8, 8))
    padded = PadToSquare()(image)
    assert padded.size[0] == padded.size[1] == 37


def test_efficientnet_b0_head_outputs_six_logits() -> None:
    model = EfficientNetB0Classifier(num_classes=len(CLASS_NAMES), pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 96, 96))
    assert logits.shape == (2, 6)


def test_val_transform_returns_imagenet_tensor() -> None:
    image = Image.new("RGB", (24, 37), color=(8, 8, 8))
    tensor = val_transform(96)(image)
    assert tensor.shape == (3, 96, 96)
