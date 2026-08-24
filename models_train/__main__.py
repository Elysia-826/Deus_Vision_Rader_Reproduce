"""支持 `python -m models`，和 `python -m models.train` 走同一个 Typer app。"""

from models.train import app

app()
