from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from dotenv import load_dotenv

from mushroom_model.config import DATA_DIR


COMPETITION = "playground-series-s4e8"

EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "sample_submission.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the Kaggle competition dataset."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload the dataset even if it already exists.",
    )

    return parser.parse_args()


def dataset_exists() -> bool:
    return all(
        (DATA_DIR / filename).exists()
        for filename in EXPECTED_FILES
    )


def verify_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "The Kaggle CLI was not found.\n"
            "Install it with:\n\n"
            "    uv add kaggle"
        )


def find_downloaded_zip() -> Path:
    expected_zip = DATA_DIR / f"{COMPETITION}.zip"

    if expected_zip.exists():
        return expected_zip

    zip_files = list(DATA_DIR.glob("*.zip"))

    if len(zip_files) == 1:
        return zip_files[0]

    if not zip_files:
        raise RuntimeError(
            "Kaggle download finished, but no ZIP file was found "
            f"in {DATA_DIR}."
        )

    raise RuntimeError(
        "Multiple ZIP files were found in the data directory. "
        "Unable to determine which one is the competition dataset."
    )


def extract_zip(zip_path: Path) -> None:
    print()
    print(f"Extracting: {zip_path.name}")

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(DATA_DIR)

    print("Extraction complete.")


def verify_dataset() -> None:
    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (DATA_DIR / filename).exists()
    ]

    if missing_files:
        missing = "\n".join(
            f"  - {filename}"
            for filename in missing_files
        )

        raise RuntimeError(
            "Dataset extraction completed, but expected files "
            f"are missing:\n{missing}"
        )


def print_dataset_summary() -> None:
    print()
    print("Dataset ready:")

    for filename in EXPECTED_FILES:
        path = DATA_DIR / filename
        size_mb = path.stat().st_size / (1024 * 1024)

        print(
            f"  {path.name:<24}"
            f"{size_mb:>10.2f} MB"
        )


def download_dataset(force: bool = False) -> None:
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if dataset_exists() and not force:
        print("Dataset already exists:")

        for filename in EXPECTED_FILES:
            print(f"  {DATA_DIR / filename}")

        print()
        print("Nothing to download.")
        print("Use --force to download it again.")
        return

    verify_kaggle_cli()

    command = [
        "kaggle",
        "competitions",
        "download",
        COMPETITION,
        "--path",
        str(DATA_DIR),
    ]

    if force:
        command.append("--force")

    print(
        f"Downloading Kaggle competition: {COMPETITION}"
    )
    print(f"Destination: {DATA_DIR}")
    print()

    try:
        subprocess.run(
            command,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Kaggle download failed.\n\n"
            "Make sure you:\n"
            "  1. Are authenticated with Kaggle.\n"
            "  2. Have accepted the competition rules.\n"
            "  3. Have internet access."
        ) from exc

    zip_path = find_downloaded_zip()

    extract_zip(zip_path)

    verify_dataset()

    print()
    print(f"Removing archive: {zip_path.name}")
    zip_path.unlink()

    print_dataset_summary()


def main() -> None:
    load_dotenv()

    args = parse_args()

    download_dataset(
        force=args.force,
    )


if __name__ == "__main__":
    main()
