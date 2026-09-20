"""Prepare the flat armor dump as a YOLO detect dataset.

Keeps dataset/armor untouched. Writes:
  dataset/armor_yolo/     images+labels split train:val:test = 7:2:1
  dataset/armor_review/   near-duplicate / exact-duplicate clusters for review
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = {0: "dead", 1: "red", 2: "blue"}
SPLITS = ("train", "val", "test")
SPLIT_RATIO = (0.7, 0.2, 0.1)


@dataclass(slots=True)
class Sample:
    stem: str
    image: Path
    label: Path
    md5: str = ""
    dhash: int = 0
    ahash: int = 0
    sharpness: float = 0.0
    width: int = 0
    height: int = 0
    class_ids: frozenset[int] = field(default_factory=frozenset)
    box_count: int = 0


def hamming64(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def pack_bits(flags: np.ndarray) -> int:
    value = 0
    for bit in flags.ravel():
        value = (value << 1) | int(bit)
    return value


def perceptual_hashes(image_path: Path, hash_size: int = 8) -> tuple[int, int, float, int, int]:
    gray = Image.open(image_path).convert("L")
    width, height = gray.size
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 0, 0, 0.0, width, height
    sharpness = float(arr.var())

    d_img = np.asarray(
        gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    dhash = pack_bits(d_img[:, 1:] > d_img[:, :-1])

    a_img = np.asarray(
        gray.resize((hash_size, hash_size), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    ahash = pack_bits(a_img > a_img.mean())
    return dhash, ahash, sharpness, width, height


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def parse_label(path: Path) -> tuple[frozenset[int], int]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return frozenset(), 0
    ids: list[int] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        ids.append(int(float(parts[0])))
    return frozenset(ids), len(ids)


def load_samples(src: Path) -> list[Sample]:
    images = sorted(src.glob("*.jpg"))
    samples: list[Sample] = []
    total = len(images)
    for idx, image in enumerate(images, start=1):
        label = image.with_suffix(".txt")
        if not label.exists():
            raise FileNotFoundError(f"missing label for {image.name}")
        class_ids, box_count = parse_label(label)
        dhash, ahash, sharpness, width, height = perceptual_hashes(image)
        samples.append(
            Sample(
                stem=image.stem,
                image=image,
                label=label,
                md5=file_md5(image),
                dhash=dhash,
                ahash=ahash,
                sharpness=sharpness,
                width=width,
                height=height,
                class_ids=class_ids,
                box_count=box_count,
            )
        )
        if idx % 500 == 0 or idx == total:
            print(f"  hashed {idx}/{total}")
    return samples


def similar_geometry(left: Sample, right: Sample) -> bool:
    if min(left.width, left.height, right.width, right.height) <= 0:
        return False
    ar_l = left.width / left.height
    ar_r = right.width / right.height
    if max(ar_l, ar_r) / min(ar_l, ar_r) > 1.35:
        return False
    area_l = left.width * left.height
    area_r = right.width * right.height
    return max(area_l, area_r) / min(area_l, area_r) <= 4.0


def is_near_duplicate(left: Sample, right: Sample, dhash_th: int, ahash_th: int) -> bool:
    if left.md5 == right.md5:
        return True
    if not similar_geometry(left, right):
        return False
    d_dist = hamming64(left.dhash, right.dhash)
    a_dist = hamming64(left.ahash, right.ahash)
    return d_dist <= dhash_th and a_dist <= ahash_th


def cluster_samples(
    samples: list[Sample],
    dhash_th: int,
    ahash_th: int,
) -> list[list[Sample]]:
    """Star clusters around the sharpest remaining seed. Not transitive."""
    remaining = set(range(len(samples)))
    clusters: list[list[Sample]] = []
    while remaining:
        seed_idx = max(remaining, key=lambda i: (samples[i].sharpness, samples[i].box_count))
        seed = samples[seed_idx]
        members = [seed]
        leftovers: set[int] = set()
        for idx in remaining:
            if idx == seed_idx:
                continue
            if is_near_duplicate(seed, samples[idx], dhash_th, ahash_th):
                members.append(samples[idx])
            else:
                leftovers.add(idx)
        remaining = leftovers
        members.sort(key=lambda sample: sample.stem)
        clusters.append(members)
        if len(clusters) % 200 == 0:
            print(f"  clustered {len(samples) - len(remaining)}/{len(samples)}")
    clusters.sort(key=lambda group: (-len(group), group[0].stem))
    print(f"  clustered {len(samples)}/{len(samples)}")
    return clusters


def pick_keeper(cluster: list[Sample]) -> Sample:
    return max(cluster, key=lambda s: (s.sharpness, s.box_count, -int(s.stem)))


def stratified_split(keepers: list[Sample], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    buckets: dict[str, list[Sample]] = defaultdict(list)
    for sample in keepers:
        key = ",".join(str(cid) for cid in sorted(sample.class_ids)) or "empty"
        buckets[key].append(sample)

    assignment: dict[str, str] = {}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * SPLIT_RATIO[0])
        n_val = int(n * SPLIT_RATIO[1])
        n_test = int(n * SPLIT_RATIO[2])
        leftover = n - n_train - n_val - n_test
        n_train += leftover
        if n >= 3 and n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)
        if n >= 2 and n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        cuts = [n_train, n_train + n_val]
        parts = [bucket[: cuts[0]], bucket[cuts[0] : cuts[1]], bucket[cuts[1] :]]
        for split, members in zip(SPLITS, parts, strict=True):
            for sample in members:
                assignment[sample.stem] = split
    return assignment


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def copy_pair(sample: Sample, image_dir: Path, label_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image, image_dir / sample.image.name)
    shutil.copy2(sample.label, label_dir / sample.label.name)


def write_yaml(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Auto-generated from dataset/prepare_armor_dataset.py",
                "# classes match HKUST RM armor stage: dead / red / blue",
                "path: .",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "nc: 3",
                "names:",
                "  0: dead",
                "  1: red",
                "  2: blue",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_review_report(
    review_root: Path,
    clusters: list[list[Sample]],
    assignment: dict[str, str],
    dhash_th: int,
    ahash_th: int,
) -> None:
    index_path = review_root / "review_index.csv"
    similar = [cluster for cluster in clusters if len(cluster) > 1]
    with index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cluster_id",
                "cluster_size",
                "role",
                "stem",
                "kept_in_split",
                "md5",
                "exact_dup_of_keeper",
                "dhash_dist_to_keeper",
                "ahash_dist_to_keeper",
                "box_count",
                "classes",
            ]
        )
        for idx, cluster in enumerate(similar, start=1):
            cluster_id = f"cluster_{idx:04d}"
            cluster_dir = review_root / cluster_id
            cluster_dir.mkdir(parents=True, exist_ok=True)
            keeper = pick_keeper(cluster)
            for sample in cluster:
                role = "KEEP" if sample.stem == keeper.stem else "REVIEW"
                shutil.copy2(sample.image, cluster_dir / f"{role}__{sample.image.name}")
                shutil.copy2(sample.label, cluster_dir / f"{role}__{sample.label.name}")
                writer.writerow(
                    [
                        cluster_id,
                        len(cluster),
                        role,
                        sample.stem,
                        assignment.get(keeper.stem, ""),
                        sample.md5,
                        sample.md5 == keeper.md5,
                        hamming64(sample.dhash, keeper.dhash),
                        hamming64(sample.ahash, keeper.ahash),
                        sample.box_count,
                        "|".join(CLASS_NAMES[i] for i in sorted(sample.class_ids)) or "empty",
                    ]
                )

    lines = [
        "装甲板近重复审查说明",
        "",
        f"判定：同一 MD5 = 完全重复；否则须同时满足 dHash<={dhash_th}、",
        f"aHash<={ahash_th}，且宽高比/面积接近。组内每张都相对最清晰那张比较，不做传递合并。",
        "每个相似组只把最清晰的一张（KEEP）放进 armor_yolo 训练/验证/测试。",
        "本组全部成员都复制到本目录，文件名前缀 KEEP__ / REVIEW__。",
        "你审完后，若要把某张 REVIEW 加回去，把它和对应 txt 拷回 armor_yolo 对应 split。",
        "对照页：用浏览器打开 index.html。",
        "",
        f"相似/重复组数：{len(similar)}",
        f"组内图片总数：{sum(len(c) for c in similar)}",
        f"从训练集剔除的多余图：{sum(len(c) - 1 for c in similar)}",
        "",
    ]
    (review_root / "README.txt").write_text("\n".join(lines), encoding="utf-8")
    write_review_html(review_root)


def write_review_html(review_root: Path) -> None:
    rows = list(csv.DictReader((review_root / "review_index.csv").open(encoding="utf-8-sig")))
    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster_id"]].append(row)
    chunks = [
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'><title>armor 近重复审查</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}",
        ".cluster{border:1px solid #333;margin:18px 0;padding:10px;background:#1a1a1a}",
        ".meta{color:#aaa;font-size:13px} .row{display:flex;flex-wrap:wrap;gap:8px}",
        "figure{margin:0;width:140px} img{width:140px;height:140px;object-fit:contain;background:#000;border:1px solid #444}",
        ".keep img{border-color:#4caf50} figcaption{font-size:11px;text-align:center}</style></head><body>",
        f"<h1>armor 近重复 / 高度相似审查（{len(by_cluster)} 组）</h1>",
        "<p class='meta'>绿框 = KEEP（已进 YOLO）；其余为 REVIEW。</p>",
    ]
    for cid, members in by_cluster.items():
        keep = next(item for item in members if item["role"] == "KEEP")
        members = sorted(members, key=lambda item: (item["role"] != "KEEP", item["stem"]))
        chunks.append(f"<section class='cluster'><h2>{cid} · {len(members)} 张</h2>")
        chunks.append(
            f"<p class='meta'>KEEP={keep['stem']} → {keep['kept_in_split']}</p><div class='row'>"
        )
        for item in members:
            css = "keep" if item["role"] == "KEEP" else "review"
            name = f"{item['role']}__{item['stem']}.jpg"
            chunks.append(
                f"<figure class='{css}'><img src='{cid}/{name}' loading='lazy'>"
                f"<figcaption>{item['role']} {item['stem']} {item['classes']}</figcaption></figure>"
            )
        chunks.append("</div></section>")
    chunks.append("</body></html>")
    (review_root / "index.html").write_text("".join(chunks), encoding="utf-8")


def count_boxes(samples: list[Sample]) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for sample in samples:
        text = sample.label.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            cid = int(float(line.split()[0]))
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def write_summary(
    yolo_root: Path,
    review_root: Path,
    all_samples: list[Sample],
    keepers: list[Sample],
    assignment: dict[str, str],
    clusters: list[list[Sample]],
    dhash_th: int,
    ahash_th: int,
    seed: int,
) -> None:
    similar = [c for c in clusters if len(c) > 1]
    by_split: dict[str, list[Sample]] = {name: [] for name in SPLITS}
    for sample in keepers:
        by_split[assignment[sample.stem]].append(sample)

    lines = [
        "armor 数据集整理报告",
        "",
        f"原始目录：dataset/armor  （未改动，{len(all_samples)} 对 jpg+txt）",
        f"YOLO 目录：{yolo_root}",
        f"审查目录：{review_root}",
        f"随机种子：{seed}",
        f"相似阈值：dHash<={dhash_th} 且 aHash<={ahash_th}，且几何接近（非传递）",
        "",
        f"完全重复 MD5 组：{sum(1 for c in similar if len({s.md5 for s in c}) == 1)}",
        f"相似/重复组：{len(similar)}",
        f"剔除多余图：{sum(len(c) - 1 for c in similar)}",
        f"进入划分的独立样本：{len(keepers)}",
        "",
        "划分（按图）：",
    ]
    for name in SPLITS:
        members = by_split[name]
        boxes = count_boxes(members)
        lines.append(
            f"  {name}: {len(members)} 图  "
            f"dead={boxes[0]} red={boxes[1]} blue={boxes[2]}"
        )
    lines.extend(
        [
            "",
            "YOLO 目录结构：",
            "  images/{train,val,test}/*.jpg",
            "  labels/{train,val,test}/*.txt",
            "  data.yaml",
            "",
        ]
    )
    (yolo_root / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def prepare(src: Path, yolo_root: Path, review_root: Path, dhash_th: int, ahash_th: int, seed: int) -> None:
    print(f"scanning {src} ...")
    samples = load_samples(src)
    print(f"loaded {len(samples)} pairs, clustering ...")
    clusters = cluster_samples(samples, dhash_th=dhash_th, ahash_th=ahash_th)
    keepers_list = [pick_keeper(cluster) for cluster in clusters]
    assignment = stratified_split(keepers_list, seed=seed)

    reset_dir(yolo_root)
    reset_dir(review_root)
    for name in SPLITS:
        (yolo_root / "images" / name).mkdir(parents=True)
        (yolo_root / "labels" / name).mkdir(parents=True)

    for sample in keepers_list:
        split = assignment[sample.stem]
        copy_pair(sample, yolo_root / "images" / split, yolo_root / "labels" / split)

    write_yaml(yolo_root / "data.yaml")
    write_review_report(review_root, clusters, assignment, dhash_th, ahash_th)
    write_summary(
        yolo_root,
        review_root,
        samples,
        keepers_list,
        assignment,
        clusters,
        dhash_th,
        ahash_th,
        seed,
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Organize armor dump into YOLO splits + review folder")
    parser.add_argument("--src", type=Path, default=root / "armor")
    parser.add_argument("--yolo-out", type=Path, default=root / "armor_yolo")
    parser.add_argument("--review-out", type=Path, default=root / "armor_review")
    parser.add_argument("--dhash-th", type=int, default=6)
    parser.add_argument("--ahash-th", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(args.src, args.yolo_out, args.review_out, args.dhash_th, args.ahash_th, args.seed)
