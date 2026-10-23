"""Path contract for packaging and auditing; independent of notebook runtime."""

from pathlib import Path, PurePosixPath


COURSE_NAME = "ai_security_nuist"
CHAPTER_NAME = "02_model_attack_defense"
NOTEBOOK_NAMES = (
    "02.01_badnets_attack.ipynb",
    "02.02_neural_cleanse_detection.ipynb",
    "02.03_bnp_defend.ipynb",
)
# The existing live acceptance flow covers attack and detection, not BNP defence.
LIVE_NOTEBOOK_NAMES = NOTEBOOK_NAMES[:2]
CHAPTER_README = "src/README.md"
COURSE_REQUIRED_ENTRIES = {"README.md", CHAPTER_NAME}
CHAPTER_REQUIRED_ENTRIES = {*NOTEBOOK_NAMES, "src"}


def require_course_root(course_root: Path) -> None:
    """Reject a standalone chapter's arbitrary parent before traversing it."""
    if course_root.name != COURSE_NAME or not all(
        (course_root / entry).exists() for entry in COURSE_REQUIRED_ENTRIES
    ):
        raise FileNotFoundError(
            "Full course layout required: contrib/tutorials/ai_security_nuist/"
            "README.md and 02_model_attack_defense/. "
            f"Refusing to use {course_root} as the package root."
        )


def zip_member(relative_path: str) -> str:
    """Map a course-relative POSIX path to the builder's archive layout."""
    return str(PurePosixPath(COURSE_NAME) / relative_path)


def missing_zip_entries(names: list[str]) -> list[str]:
    """Files must match exactly; directories must contain at least one file."""
    files = [
        "README.md",
        *(f"{CHAPTER_NAME}/{name}" for name in NOTEBOOK_NAMES),
        f"{CHAPTER_NAME}/{CHAPTER_README}",
        f"{CHAPTER_NAME}/src/assets/results/metrics/final_summary.json",
    ]
    directories = [
        f"{CHAPTER_NAME}/src/data/train",
        f"{CHAPTER_NAME}/src/data/test",
        f"{CHAPTER_NAME}/src/assets/evidence/server_artifacts",
        f"{CHAPTER_NAME}/src/assets/results/detection",
    ]
    missing = [zip_member(path) for path in files if zip_member(path) not in names]
    for path in directories:
        prefix = zip_member(path) + "/"
        if not any(name.startswith(prefix) and not name.endswith("/") for name in names):
            missing.append(prefix)
    return missing
