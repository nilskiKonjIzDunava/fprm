"""Format converters for plotting input."""
import json
from pathlib import Path


def write_curves(curves: dict, path: str | Path) -> None:
    """Write curves dict to JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(curves, f, indent=2)
