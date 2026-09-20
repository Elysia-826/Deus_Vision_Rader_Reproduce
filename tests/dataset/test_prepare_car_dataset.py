from pathlib import Path

from prepare_car_dataset import Sample, split_counts, stratified_split


def test_split_counts_uses_7_2_1_and_remainder_goes_to_train() -> None:
    n_train, n_val, n_test = split_counts(3022)
    assert n_train + n_val + n_test == 3022
    assert n_train == 2116
    assert n_val == 604
    assert n_test == 302


def test_stratified_split_keeps_every_stem_once() -> None:
    samples = [
        Sample(stem=f"{i:04d}", image=Path(f"{i:04d}.jpg"), label=Path(f"{i:04d}.txt"), box_count=n)
        for i, n in enumerate([1, 1, 1, 5, 5, 5, 9, 9, 9, 12])
    ]
    assignment = stratified_split(samples, seed=42)
    assert set(assignment) == {s.stem for s in samples}
    assert set(assignment.values()) <= {"train", "val", "test"}
