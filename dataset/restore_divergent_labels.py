"""Put REVIEW samples whose labels differ from KEEP back into armor_yolo.

A REVIEW image is restored when it is not an exact MD5 copy of KEEP and any of:
  - class set differs
  - box count differs
  - a matched box has a different class
  - any box IoU against KEEP is < 0.5
Restored files go into the same split as that cluster's KEEP.
"""

from __future__ import annotations

import csv
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "armor"
YOLO = ROOT / "armor_yolo"
REVIEW = ROOT / "armor_review"
IOU_TH = 0.5
CLASS_NAMES = {0: "dead", 1: "red", 2: "blue"}


def parse_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cid, cx, cy, w, h = (float(x) for x in parts)
        boxes.append((int(cid), cx, cy, w, h))
    return boxes


def iou(a: tuple[int, float, float, float, float], b: tuple[int, float, float, float, float]) -> float:
    ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax2, ay2 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx2, by2 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def divergence(keep: list[tuple[int, float, float, float, float]], other: list[tuple[int, float, float, float, float]]) -> str:
    keep_cls = tuple(sorted(box[0] for box in keep))
    other_cls = tuple(sorted(box[0] for box in other))
    used: set[int] = set()
    ious: list[float] = []
    class_mismatch = 0
    for box in other:
        best_i, best = -1, -1.0
        for idx, keep_box in enumerate(keep):
            if idx in used:
                continue
            score = iou(box, keep_box)
            if score > best:
                best_i, best = idx, score
        if best_i >= 0:
            used.add(best_i)
            ious.append(best)
            if keep[best_i][0] != box[0]:
                class_mismatch += 1
        else:
            ious.append(0.0)
    reasons: list[str] = []
    if keep_cls != other_cls:
        reasons.append("class_set")
    if len(keep) != len(other):
        reasons.append("box_count")
    if class_mismatch:
        reasons.append("class_swap")
    if len(keep) != len(used) or any(score < IOU_TH for score in ious):
        reasons.append("low_iou")
    return "+".join(reasons)


def count_split_boxes(split: str) -> dict[int, int]:
    counts = {0: 0, 1: 0, 2: 0}
    for label in (YOLO / "labels" / split).glob("*.txt"):
        for box in parse_boxes(label):
            counts[box[0]] = counts.get(box[0], 0) + 1
    return counts


def main() -> None:
    rows = list(csv.DictReader((REVIEW / "review_index.csv").open(encoding="utf-8-sig")))
    by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster_id"]].append(row)

    restored: list[dict[str, str]] = []
    reason_counter: Counter[str] = Counter()
    split_counter: Counter[str] = Counter()

    for cid, members in by_cluster.items():
        keep = next(item for item in members if item["role"] == "KEEP")
        split = keep["kept_in_split"]
        keep_boxes = parse_boxes(REVIEW / cid / f"KEEP__{keep['stem']}.txt")
        for member in members:
            member["action"] = "keep_in_yolo" if member["role"] == "KEEP" else "left_in_review"
            member["restore_reason"] = ""
            if member["role"] != "REVIEW":
                continue
            if member["exact_dup_of_keeper"] == "True":
                continue
            other_boxes = parse_boxes(REVIEW / cid / f"REVIEW__{member['stem']}.txt")
            reason = divergence(keep_boxes, other_boxes)
            if not reason:
                continue
            stem = member["stem"]
            src_img = SRC / f"{stem}.jpg"
            src_lbl = SRC / f"{stem}.txt"
            dst_img = YOLO / "images" / split / f"{stem}.jpg"
            dst_lbl = YOLO / "labels" / split / f"{stem}.txt"
            if dst_img.exists() or dst_lbl.exists():
                raise FileExistsError(f"{stem} already in armor_yolo/{split}")
            shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            member["action"] = "restored_to_yolo"
            member["restore_reason"] = reason
            split_counter[split] += 1
            for part in reason.split("+"):
                reason_counter[part] += 1
            restored.append(
                {
                    "cluster_id": cid,
                    "stem": stem,
                    "split": split,
                    "reason": reason,
                    "keep_stem": keep["stem"],
                }
            )

    fieldnames = list(rows[0].keys())
    if "action" not in fieldnames:
        fieldnames.extend(["action", "restore_reason"])
    with (REVIEW / "review_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (REVIEW / "restored_to_yolo.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cluster_id", "stem", "split", "reason", "keep_stem"])
        writer.writeheader()
        writer.writerows(restored)

    lines = [
        "标签差异回灌报告",
        "",
        "规则：REVIEW 相对 KEEP，类别集合/框数不同，或框 IoU<0.5，则加回 armor_yolo。",
        "完全重复 MD5 不加回。加回的图进入该组 KEEP 所在 split，避免视觉泄漏。",
        "",
        f"加回：{len(restored)}",
        f"按 split：{dict(split_counter)}",
        f"原因：{dict(reason_counter)}",
        "",
        "加回后 armor_yolo 规模：",
    ]
    for split in ("train", "val", "test"):
        n_img = len(list((YOLO / "images" / split).glob("*.jpg")))
        n_lbl = len(list((YOLO / "labels" / split).glob("*.txt")))
        boxes = count_split_boxes(split)
        lines.append(
            f"  {split}: {n_img} 图 / {n_lbl} 标  "
            f"dead={boxes[0]} red={boxes[1]} blue={boxes[2]}"
        )
    (REVIEW / "RESTORE_SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
