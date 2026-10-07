from pathlib import Path

from models.pattern.schema import PatternTrainJob
from models.pattern.trainer import inverse_frequency_weights


def test_job_defaults_match_efficientnet_b0_recipe(tmp_path: Path) -> None:
    job = PatternTrainJob(data=tmp_path)
    assert job.imgsz == 96
    assert job.epochs == 100
    assert job.batch == 32
    assert job.learning_rate == 0.0003
    assert job.backbone == "efficientnet_b0"
    assert job.init_weights is None


def test_inverse_frequency_upweights_rare_class() -> None:
    weights = inverse_frequency_weights((100, 100, 100, 100, 10, 10))
    assert weights[4] > weights[0]
    assert weights[5] > weights[0]
