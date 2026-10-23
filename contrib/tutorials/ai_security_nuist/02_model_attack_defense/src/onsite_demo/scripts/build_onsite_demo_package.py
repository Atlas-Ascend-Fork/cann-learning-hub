from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path


ONSITE_ROOT = Path(__file__).resolve().parents[1]
CHAPTER_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[4]
if str(ONSITE_ROOT) not in sys.path:
    sys.path.insert(0, str(ONSITE_ROOT))

from demo_lib.paths import ensure_dir, resolve_project_root
from demo_lib.package_contract import CHAPTER_NAME, NOTEBOOK_NAMES, require_course_root, zip_member


EXCLUDE_DIR_NAMES = {
    ".tmp_pycache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".venv",
    ".venv_torch",
    "archive",
    "jupyter_runtime",
    ".Trash-1000",
    "kernel_meta",
    "offload",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".zip"}
EXCLUDE_RELATIVE_PREFIXES = {
    "02_model_attack_defense/src/onsite_demo/outputs",
    "pytorch_onsite_demo_package",
}
REQUIRED_RELATIVE_PATHS = [
    "README.md",
    *(f"{CHAPTER_NAME}/{name}" for name in NOTEBOOK_NAMES),
    "02_model_attack_defense/src/final_summary.json",
    "02_model_attack_defense/src/data/train",
    "02_model_attack_defense/src/data/test",
    "02_model_attack_defense/src/train.py",
    "02_model_attack_defense/src/onsite_demo/demo_lib/package_contract.py",
    "02_model_attack_defense/src/onsite_demo/demo_lib/detection.py",
    "02_model_attack_defense/src/onsite_demo/demo_lib/detection_visualization.py",
    "02_model_attack_defense/src/onsite_demo/demo_lib/training.py",
    "02_model_attack_defense/src/onsite_demo/scripts/run_detection_demo_smoke.py",
    "02_model_attack_defense/src/onsite_demo/scripts/run_full_onsite_demo.py",
    "02_model_attack_defense/src/onsite_demo/scripts/run_cloud_live_10epoch_acceptance.py",
    "02_model_attack_defense/src/onsite_demo/scripts/final_preupload_audit.py",
    "02_model_attack_defense/src/onsite_demo/scripts/validate_onsite_demo.py",
    "02_model_attack_defense/src/README.md",
    "02_model_attack_defense/src/onsite_demo/configs/demo_config.json",
    "02_model_attack_defense/src/assets/evidence/server_artifacts",
    "02_model_attack_defense/src/assets/results/metrics/final_summary.json",
    "02_model_attack_defense/src/assets/results/detection",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the onsite demo package zip.")
    parser.add_argument(
        "--output-zip",
        default=str(
            CHAPTER_ROOT
            / "src"
            / "onsite_demo"
            / "outputs"
            / "cloud_live_acceptance"
            / "ai_model_backdoor_attack_and_detection_cloud_live.zip"
        ),
    )
    return parser.parse_args()


def _relative_text(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT)).replace("\\", "/")


def _zip_arcname(path: Path) -> str:
    return zip_member(_relative_text(path))


def _should_skip(path: Path, zip_path: Path) -> bool:
    relative_text = _relative_text(path)
    if path == zip_path:
        return True
    if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    return any((relative_text == prefix or relative_text.startswith(prefix + "/")) for prefix in EXCLUDE_RELATIVE_PREFIXES)


def _assert_required_paths() -> None:
    require_course_root(PACKAGE_ROOT)
    missing = [relative for relative in REQUIRED_RELATIVE_PATHS if not (PACKAGE_ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build package ZIP because required paths are missing: {missing}")


def main() -> Path:
    args = parse_args()
    resolve_project_root()
    zip_path = Path(args.output_zip)
    _assert_required_paths()
    ensure_dir(zip_path.parent)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKAGE_ROOT.rglob("*")):
            if _should_skip(path, zip_path):
                continue
            if path.is_dir():
                continue
            archive.write(path, arcname=_zip_arcname(path))

    print(f"package zip = {zip_path}")
    print(f"package size bytes = {zip_path.stat().st_size}")
    return zip_path


if __name__ == "__main__":
    main()
