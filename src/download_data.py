"""Download the chest X-ray dataset from Kaggle into data/raw (credentials from .env)."""
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET = "paultimothymooney/chest-xray-pneumonia"


def load_env(path: Path) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


def main() -> None:
    load_env(ROOT / ".env")
    if not (os.environ.get("KAGGLE_API_TOKEN")
            or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        raise SystemExit("No Kaggle credentials found. Fill in .env first.")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    RAW.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} -> {RAW}")
    api.dataset_download_files(DATASET, path=str(RAW), quiet=False)

    zip_path = RAW / "chest-xray-pneumonia.zip"
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    zip_path.unlink()
    n = sum(1 for _ in RAW.rglob("*.jpeg"))
    print(f"Done. {n} images under {RAW}")


if __name__ == "__main__":
    main()
