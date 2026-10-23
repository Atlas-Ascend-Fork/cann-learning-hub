"""Resource-free regression tests: python -B -m unittest discover -s src/onsite_demo/tests."""

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ONSITE = Path(__file__).resolve().parents[1]
CHAPTER = ONSITE.parents[1]
sys.path.insert(0, str(ONSITE))
from demo_lib import package_contract as contract


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ONSITE / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_static_functions(name, functions):
    """Load only readers/checks, without importing training or device libraries."""
    tree = ast.parse((ONSITE / "scripts" / f"{name}.py").read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
    namespace = dict(vars(contract), json=json)
    # Postpone annotations so these isolated checks do not need typing imports.
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(source, name, "exec"), namespace)
    return namespace


class PackageContractTests(unittest.TestCase):
    def setUp(self):
        self.builder = load_script("build_onsite_demo_package")
        self.audit = load_script("final_preupload_audit")

    def make_course(self, base):
        course = base / contract.COURSE_NAME
        directories = {"train", "test", "server_artifacts", "detection"}
        for relative in self.builder.REQUIRED_RELATIVE_PATHS:
            path = course / relative
            if path.name in directories:
                path.mkdir(parents=True, exist_ok=True)
                (path / "fixture.txt").write_text("test fixture", encoding="utf-8")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("test fixture", encoding="utf-8")
        return course

    def test_real_builder_zip_passes_preupload_gate_and_excludes_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            course = self.make_course(Path(temporary))
            chapter = course / contract.CHAPTER_NAME
            output = chapter / "src/onsite_demo/outputs"
            excluded = [output / "old_run/result.json", chapter / "kernel_meta/cache.json", chapter / "offload/cache.bin"]
            for path in excluded:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("excluded", encoding="utf-8")
            zip_path = output / "cloud_live_acceptance/ai_model_backdoor_attack_and_detection_cloud_live.zip"
            with patch.object(self.builder, "PACKAGE_ROOT", course), patch.object(sys, "argv", ["builder", "--output-zip", str(zip_path)]), contextlib.redirect_stdout(io.StringIO()):
                self.builder.main()
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
            self.assertEqual([], contract.missing_zip_entries(names))
            self.assertTrue(self.audit._check_candidate_zip({"OUTPUT_ROOT": output})[0])
            for path in excluded:
                self.assertNotIn(contract.zip_member(path.relative_to(course).as_posix()), names)

    def test_wrong_zip_roots_and_notebook_suffix_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            course = self.make_course(Path(temporary))
            names = [contract.zip_member(p.relative_to(course).as_posix()) for p in course.rglob("*") if p.is_file()]
            self.assertEqual([], contract.missing_zip_entries(names))
            self.assertTrue(contract.missing_zip_entries(["extra/" + name for name in names]))
            self.assertTrue(contract.missing_zip_entries([name.split("/", 1)[1] for name in names]))
            for notebook in contract.NOTEBOOK_NAMES:
                target = contract.zip_member(f"{contract.CHAPTER_NAME}/{notebook}")
                changed = [name + ".bak" if name == target else name for name in names]
                self.assertIn(target, contract.missing_zip_entries(changed))

    def test_directory_entries_alone_do_not_satisfy_asset_checks(self):
        names = [contract.zip_member(f"{contract.CHAPTER_NAME}/src/data/train") + "/"]
        self.assertIn(names[0], contract.missing_zip_entries(names))

    def test_standalone_chapter_parent_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(self.builder, "PACKAGE_ROOT", Path(temporary)):
                with self.assertRaisesRegex(FileNotFoundError, "Full course layout required"):
                    self.builder._assert_required_paths()

    def test_resources_still_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            course = self.make_course(Path(temporary))
            summary = course / contract.CHAPTER_NAME / "src/assets/results/metrics/final_summary.json"
            summary.unlink()
            with patch.object(self.builder, "PACKAGE_ROOT", course):
                with self.assertRaisesRegex(FileNotFoundError, "final_summary.json"):
                    self.builder._assert_required_paths()

    def test_extra_chapters_and_runtime_directories_allowed(self):
        self.assertTrue(contract.COURSE_REQUIRED_ENTRIES <= {"README.md", contract.CHAPTER_NAME, "another_chapter"})
        self.assertTrue(contract.CHAPTER_REQUIRED_ENTRIES <= {*contract.NOTEBOOK_NAMES, "src", "offload", "kernel_meta"})

    def test_current_notebooks_and_readme_load_without_old_paths(self):
        functions = load_static_functions("validate_onsite_demo", {"_load_notebook", "_readme_text"})
        paths = {"PROJECT_ROOT": CHAPTER}
        notebook = functions["_load_notebook"](paths)
        self.assertGreater(len(notebook["cells"]), 0)
        readme = functions["_readme_text"](paths)
        for token in ("cloud_live", "--mode cloud_live --epochs 10 --batch-size 8", "run_cloud_live_10epoch_acceptance.py", "RUN_PROFILE", "ipywidgets", "jupyterlab_widgets"):
            self.assertIn(token, readme)
        for name in contract.NOTEBOOK_NAMES:
            self.assertIn("cells", json.loads((CHAPTER / name).read_text(encoding="utf-8")))

    def test_cloud_acceptance_reader_uses_current_paths(self):
        functions = load_static_functions("run_cloud_live_10epoch_acceptance", {"_no_bad_claims"})
        functions["_line_has_bad_claim"] = lambda line: False
        self.assertTrue(functions["_no_bad_claims"]({"PROJECT_ROOT": CHAPTER}))


if __name__ == "__main__":
    unittest.main()
