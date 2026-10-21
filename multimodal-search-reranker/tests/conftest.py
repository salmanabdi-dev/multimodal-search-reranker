import sys
from pathlib import Path

# Allow running the tests from a checkout without `pip install -e .`
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
