"""Sample 列表上的 torch Dataset。"""

from __future__ import annotations

from collections.abc import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from models.pattern.samples import Sample

Transform = Callable[[Image.Image], torch.Tensor]


class PatternDataset(Dataset[tuple[torch.Tensor, int]]):  # noqa: MUTABLE_OK — Dataset 按索引读盘
    """每张图一个图案 id。transform 在构造时钉死。"""

    def __init__(self, samples: tuple[Sample, ...], transform: Transform) -> None:
        self._samples = samples
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item = self._samples[index]
        image = Image.open(item.path).convert("RGB")
        return self._transform(image), int(item.pattern)

    @property
    def samples(self) -> tuple[Sample, ...]:
        return self._samples
