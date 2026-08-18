import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for path in [
    REPO_ROOT / "deployment" / "api",
    REPO_ROOT / "deployment" / "reddit",
    REPO_ROOT / "deployment" / "bert_training",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
