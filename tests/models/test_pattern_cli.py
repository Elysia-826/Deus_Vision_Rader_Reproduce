from typer.testing import CliRunner

from models.pattern.train import app


def test_bare_invoke_does_not_train() -> None:
    result = CliRunner().invoke(app, [])
    assert result.exit_code != 0
    assert "best weights" not in result.stdout


def test_help_mentions_v3_small_and_64() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "64" in result.stdout
    assert "mobilenetv3-small" in result.stdout
