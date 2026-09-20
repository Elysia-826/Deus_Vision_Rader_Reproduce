"""先 pad 成正方形再缩到 64。验证集 / 推理必须同一套 val_transform。

装甲裁剪往往是扁的。直接 resize 到 64×64 会把数字压扁，分类器在仿真和实拍之间对不齐。
PadToSquare 补黑边保持长宽比，和港科大「先 pad 再缩」一致。
"""

from __future__ import annotations

from typing import Final

from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


class PadToSquare:
    """短边对称黑边，保持图案不被非等比拉伸压扁。"""

    def __init__(self, fill: int = 0) -> None:
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        edge = max(width, height)
        pad_x = edge - width
        pad_y = edge - height
        padding = (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2)
        return F.pad(img, padding, fill=self.fill, padding_mode="constant")


def train_transform(imgsz: int) -> transforms.Compose:
    """港科大几何/颜色增强 + torchvision RandomErasing。"""
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            PadToSquare(),
            transforms.Resize((imgsz, imgsz)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )


def val_transform(imgsz: int) -> transforms.Compose:
    return transforms.Compose(
        [
            PadToSquare(),
            transforms.Resize((imgsz, imgsz)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
        ]
    )
