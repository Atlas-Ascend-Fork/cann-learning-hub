"""Helpers for the LongCat-Next single-NPU notebook workflow.

The tutorial no longer vendors a recipes snapshot: chapter 2 bootstraps a full
clone of the upstream cann-recipes-infer repository pinned at
``INFER_BASE_COMMIT`` into the gitignored run workspace and applies the
single-NPU patch shipped under ``patches/``; later chapters reuse that clone.

The LongCat-Next recipe writes one structured ``result.json`` per request under
``models/longcat_next/res/<date>/<case>/results/<request_id>/``，so unlike the
Qwen tutorial no log-text parsing is required: metrics are read straight from
the JSON records.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

MODEL_REPO_ID = "meituan-longcat/LongCat-Next"
REQUIRED_CHECKPOINT_ENTRIES = (
    "model.safetensors.index.json",
    "assets",
    "image_decoder",
    "cosy24k_vocoder",
)
MIN_FREE_DISK_GB = 160  # 151GB BF16 权重 + 余量

# 上游 cann-recipes-infer 仓（含 LongCat-Next 多卡推理实现），运行时全量克隆；
# 固定回退到 INFER_BASE_COMMIT（单卡 patch 的基线提交）后 git am 单卡 patch。
INFER_REPO_URL = "https://gitcode.com/cann/cann-recipes-infer.git"
INFER_BASE_COMMIT = "969160e"
INFER_REPO_DIRNAME = "cann-recipes-infer"
SINGLE_NPU_PATCH_NAME = "longcat-next-single-npu.patch"
INFER_REPO_STATE_FILENAME = "infer_repo_state.json"


def _free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / (1024 ** 3)


def _validate_checkpoint(path: Path) -> None:
    """Verify that a checkpoint directory contains everything the recipe needs."""
    missing = [entry for entry in REQUIRED_CHECKPOINT_ENTRIES if not (path / entry).exists()]
    if missing:
        raise FileNotFoundError(
            f"权重目录不完整，缺少 {missing}（路径: {path}）。"
            "请检查下载是否中断后重试（snapshot 下载支持断点续传）。"
        )


def _ensure_free_space(download_root: Path) -> float:
    download_root.mkdir(parents=True, exist_ok=True)
    free_gb = _free_disk_gb(download_root)
    if free_gb < MIN_FREE_DISK_GB:
        raise RuntimeError(
            f"下载目录 {download_root} 所在磁盘仅剩 {free_gb:.0f}GB，"
            f"低于 {MIN_FREE_DISK_GB}GB 要求（权重约 151GB）。"
            "请设置 LONGCAT_NEXT_DOWNLOAD_DIR 指向大容量磁盘，"
            "或设置 LONGCAT_NEXT_MODEL_PATH 指向已就绪的本地权重目录。"
        )
    return free_gb


def _hub_root() -> Path:
    """Locate the cann-learning-hub repository root from this module's path."""
    for root in Path(__file__).resolve().parents:
        if (root / "tutorials" / "llm_inference").is_dir():
            return root
    raise RuntimeError("无法定位 cann-learning-hub 仓库根（未找到 tutorials/llm_inference）。")


def _tutorial_workspace_dir() -> Path:
    """The gitignored run workspace ``Sources/model_inference_optimization/longcat_next``."""
    return _hub_root() / "Sources" / "model_inference_optimization" / "longcat_next"


def _tutorial_dir() -> Path:
    """The tutorial directory ``tutorials/llm_inference/longcat_next``."""
    return _hub_root() / "tutorials" / "llm_inference" / "longcat_next"


def _model_path_cache_file() -> Path:
    """Locate the persistent resolved-model-path cache in the tutorial workspace.

    The cache lives under the notebook runtime workspace
    ``Sources/model_inference_optimization/longcat_next/`` (gitignored), so a
    successful resolution in chapter 2 is reused by chapters 3-6 and by later
    kernel sessions; deleting the workspace resets it.
    """
    return _tutorial_workspace_dir() / "resolved_model_path.txt"


def _write_model_path_cache(path: Path) -> None:
    """Persist a successfully resolved model path for later chapters/sessions."""
    cache_file = _model_path_cache_file()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(str(path), encoding="utf-8")


def _download_from_modelscope(download_root: Path) -> Path:
    try:
        from modelscope import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "ModelScope 下载通道不可用。请安装 modelscope（pip install modelscope==1.38.1），"
            "或设置 LONGCAT_NEXT_MODEL_PATH 指向本地模型目录，"
            "或设置 LONGCAT_NEXT_DOWNLOAD_BACKEND=hf 使用 Hugging Face 镜像。"
        ) from exc
    # 快路径：缓存文件缺失但 ModelScope 快照缓存可能已完整（例如用户删除了
    # resolved_model_path.txt），纯本地校验、不联网、无需磁盘空间。
    try:
        resolved = snapshot_download(MODEL_REPO_ID, cache_dir=str(download_root), local_files_only=True)
        print("[模型下载] 命中 ModelScope 本地缓存，跳过网络校验", flush=True)
        return Path(resolved)
    except Exception:
        pass
    free_gb = _ensure_free_space(download_root)
    print("[模型下载] 本地缓存未命中，开始远程下载"
          "（约 151GB，视带宽约 25 分钟~4 小时，支持断点续传）", flush=True)
    print(f"[模型下载] 下载目录 = {download_root}（磁盘剩余 {free_gb:.0f}GB）", flush=True)
    resolved = snapshot_download(MODEL_REPO_ID, cache_dir=str(download_root), max_workers=4)
    return Path(resolved)


def _download_from_huggingface(download_root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub 不可用。请安装 huggingface-hub，"
            "或设置 LONGCAT_NEXT_MODEL_PATH 指向本地模型目录。"
        ) from exc
    # 国内网络默认走 hf-mirror.com，可通过 HF_ENDPOINT 覆盖（如已配代理可设为官方端点）。
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # 快路径：与 ModelScope 分支同理，先纯本地校验。
    try:
        resolved = snapshot_download(MODEL_REPO_ID, cache_dir=str(download_root), local_files_only=True)
        print("[模型下载] 命中 Hugging Face 本地缓存，跳过网络校验", flush=True)
        return Path(resolved)
    except Exception:
        pass
    free_gb = _ensure_free_space(download_root)
    print("[模型下载] 本地缓存未命中，开始远程下载"
          "（约 151GB，视带宽约 25 分钟~4 小时，支持断点续传）", flush=True)
    print(f"[模型下载] 端点 = {os.environ['HF_ENDPOINT']}，下载目录 = {download_root}"
          f"（磁盘剩余 {free_gb:.0f}GB）", flush=True)
    resolved = snapshot_download(MODEL_REPO_ID, cache_dir=str(download_root), max_workers=4)
    return Path(resolved)


def resolve_model_path(quiet: bool = False) -> str:
    """Resolve the LongCat-Next checkpoint directory.

    Priority:
    1. ``LONGCAT_NEXT_MODEL_PATH`` pointing at an existing local checkpoint
       (validated, used directly, and written to the cache).
    2. The persistent cache file ``resolved_model_path.txt`` in the tutorial
       workspace -- written by any earlier successful resolution (e.g. the
       chapter-2 download), reused across chapters and kernel sessions.
    3. Remote download of ``meituan-longcat/LongCat-Next`` (about 151GB,
       resumable): a ``local_files_only`` probe of the snapshot cache runs
       before the network; backend ``modelscope`` (default) or ``hf``;
       requires >=160GB free disk. Successful resolutions are cached back.
    """
    # ① 显式环境变量优先（并刷新缓存）
    env_path = os.environ.get("LONGCAT_NEXT_MODEL_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if (path / "model.safetensors.index.json").is_file():
            _write_model_path_cache(path)
            if not quiet:
                print(f"[模型路径] 使用环境变量指定的本地权重: {path}")
            return str(path)
        raise FileNotFoundError(
            f"LONGCAT_NEXT_MODEL_PATH 指向的目录缺少 model.safetensors.index.json: {path}"
        )

    # ② 复用此前章节/会话已解析的路径（秒回，不触发下载）
    cache_file = _model_path_cache_file()
    if cache_file.is_file():
        cached = Path(cache_file.read_text(encoding="utf-8").strip()).expanduser()
        if (cached / "model.safetensors.index.json").is_file():
            # 不受 quiet 抑制：解释"本次为何没有下载"。
            print(f"[模型路径] 复用已解析的模型路径: {cached}")
            print(f"[模型路径] （如需重新解析，删除缓存文件 {cache_file}）")
            return str(cached)
        cache_file.unlink(missing_ok=True)
        print(f"[模型路径] 缓存的路径已失效（{cached}），重新解析")

    # ③ 远程分支：中性提示先行，确认未命中后才给出体积警告
    backend = os.environ.get("LONGCAT_NEXT_DOWNLOAD_BACKEND", "modelscope").strip().lower()
    print("[模型下载] 未检测到本地权重与已解析缓存，正在检查缓存/下载权重 ...", flush=True)
    if backend in {"modelscope", "ms"}:
        default_root = Path.home() / ".cache/modelscope"
        download_root = Path(os.environ.get("LONGCAT_NEXT_DOWNLOAD_DIR", default_root)).expanduser()
        resolved = _download_from_modelscope(download_root)
    elif backend in {"huggingface", "hf"}:
        default_root = Path.home() / ".cache/huggingface/hub"
        download_root = Path(os.environ.get("LONGCAT_NEXT_DOWNLOAD_DIR", default_root)).expanduser()
        resolved = _download_from_huggingface(download_root)
    else:
        raise ValueError("LONGCAT_NEXT_DOWNLOAD_BACKEND 只能是 'modelscope' 或 'huggingface'。")

    _validate_checkpoint(resolved)
    _write_model_path_cache(resolved)
    print(f"[模型下载] 完成，路径 = {resolved}", flush=True)
    return str(resolved)


# ---------------------------------------------------------------------------
# Upstream cann-recipes-infer bootstrap (clone + pinned base + single-NPU patch)
# ---------------------------------------------------------------------------


def _git(repo_dir: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["git", "-C", str(repo_dir), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} 失败（{repo_dir}）：{detail}")
    return proc


def _read_infer_repo_state(state_file: Path) -> dict[str, Any]:
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state if isinstance(state, dict) else {}


def _write_infer_repo_state(state_file: Path, **state: str) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_single_npu_patch(repo_dir: Path, patch_file: Path) -> None:
    """``git am`` the single-NPU patch with an inline committer identity."""
    if not patch_file.is_file():
        raise FileNotFoundError(f"未找到单卡推理 patch: {patch_file}")
    am = subprocess.run(
        [
            "git", "-C", str(repo_dir),
            "-c", "user.name=longcat-tutorial",
            "-c", "user.email=longcat-tutorial@local",
            "am", str(patch_file),
        ],
        capture_output=True,
        text=True,
    )
    if am.returncode != 0:
        subprocess.run(
            ["git", "-C", str(repo_dir), "am", "--abort"],
            capture_output=True,
            text=True,
        )
        detail = (am.stderr or am.stdout or "").strip()
        raise RuntimeError(
            f"git am 应用单卡 patch 失败（基线 {INFER_BASE_COMMIT}）：\n{detail}\n"
            f"可手工恢复：cd {repo_dir} && git checkout -f --detach {INFER_BASE_COMMIT} && "
            f"git -c user.name=tutorial -c user.email=tutorial@local am {patch_file}"
        )


def _fix_infer_set_env_cann_path(repo_dir: Path) -> None:
    """Replace the upstream set_env.sh cann_path placeholder with the local CANN.

    上游 ``executor/scripts/set_env.sh`` 的 ``cann_path`` 默认为
    ``your_cann_pkgs_path`` 占位符，不替换会在推理启动时 source 失败；这里用当前
    环境的 ``ASCEND_TOOLKIT_HOME``（或 ``ASCEND_HOME_PATH``）替换，等价于
    patches/README.md 3.2 节的手工 sed 步骤。幂等：已替换过则直接跳过。
    """
    set_env_sh = repo_dir / "executor" / "scripts" / "set_env.sh"
    if not set_env_sh.is_file():
        return
    text = set_env_sh.read_text(encoding="utf-8")
    if "your_cann_pkgs_path" not in text:
        return
    cann_home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    if not cann_home:
        raise EnvironmentError(
            "上游 set_env.sh 的 cann_path 为占位符，替换需要 ASCEND_TOOLKIT_HOME 或 "
            "ASCEND_HOME_PATH 环境变量；请先 source CANN 环境再运行本 Notebook。"
        )
    if not (Path(cann_home) / "bin" / "setenv.bash").is_file():
        print(f"[infer 仓] 提示：{cann_home}/bin/setenv.bash 不存在，set_env.sh 可能仍会告警")
    set_env_sh.write_text(text.replace("your_cann_pkgs_path", cann_home), encoding="utf-8")
    print(f"[infer 仓] 已将 set_env.sh 的 cann_path 替换为 {cann_home}")


def _find_sklearn_bundled_libgomp() -> Path | None:
    """Locate the libgomp bundled inside the importable sklearn, if any.

    只定位、不导入：``importlib.util.find_spec`` 不会执行 ``sklearn/__init__.py``，
    在已加载 torch 的 kernel 进程里调用也安全（不会触发 libgomp 加载）。返回的是
    实际将被 ``import sklearn`` 使用的那份安装（混装环境下不会找错副本）。conda /
    源码编译的 sklearn 链接系统 libgomp、不带 ``scikit_learn.libs``，返回 None。
    """
    try:
        spec = importlib.util.find_spec("sklearn")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    libs_dir = Path(spec.origin).resolve().parent.parent / "scikit_learn.libs"
    if not libs_dir.is_dir():
        return None
    matches = sorted(libs_dir.glob("libgomp*.so*"))
    return matches[0] if matches else None


def _apply_gomp_tls_workaround(repo_dir: Path) -> None:
    """Append the libgomp static-TLS workaround into the cloned set_env.sh.

    背景：PyPI 版 torch（如 2.8.0+cpu）的 ``libtorch_cpu.so`` 会同时加载系统
    libgomp 与自带的 ``torch.libs/libgomp-*.so``；推理进程里 transformers 又会
    级联导入 sklearn（librosa 的传递依赖，PyPI wheel 自带第三份 libgomp，SONAME
    带 hash）。老 glibc（如 Ubuntu 20.04 的 2.31）static TLS 预留装不下第三份，
    报 ``cannot allocate memory in static TLS block`` 导致推理进程直接退出。

    修复：把 sklearn 自带的 libgomp 写入 ``LD_PRELOAD``，使其在进程启动阶段完成
    TLS 分配（启动期计入，不挤 dlopen 预留区）。该库 SONAME 特殊，不参与
    ``libgomp.so.1`` 的符号解析，对 torch / CANN 的 OpenMP 行为无干扰。

    幂等与自愈：

    - sklearn 未安装或无自带 libgomp（conda / 源码版，本就不会触发）→ 跳过；
    - 已有同路径 workaround 块 → 跳过；路径随 sklearn 升级变化 → 整块替换；
    - 已有 ``LD_PRELOAD``（如用户或 CANN 设置）→ 追加在末位，不动其优先级；
    - 克隆被 ``checkout -f`` 还原后，本函数随 ``ensure_infer_repo`` 重新执行。
    """
    set_env_sh = repo_dir / "executor" / "scripts" / "set_env.sh"
    if not set_env_sh.is_file():
        return
    libgomp = _find_sklearn_bundled_libgomp()
    if libgomp is None:
        return
    begin = "# >>> longcat-tutorial libgomp TLS workaround BEGIN >>>"
    end = "# <<< longcat-tutorial libgomp TLS workaround END <<<"
    block = "\n".join(
        [
            begin,
            "# torch（PyPI wheel，两份 libgomp）+ sklearn wheel（第三份）超出老 glibc",
            "# 的 static TLS 预留而崩溃；预载使其在进程启动阶段完成 TLS 分配。",
            f'if [ -f "{libgomp}" ]; then',
            '    case ":${LD_PRELOAD:-}:" in',
            f'        *":{libgomp}:"*) ;;',
            f'        *) export LD_PRELOAD="${{LD_PRELOAD:+${{LD_PRELOAD}}:}}{libgomp}" ;;',
            "    esac",
            "fi",
            end,
        ]
    )
    text = set_env_sh.read_text(encoding="utf-8")
    begin_idx = text.find(begin)
    end_idx = text.find(end)
    if begin_idx != -1 and end_idx != -1 and end_idx > begin_idx:
        if text[begin_idx : end_idx + len(end)] == block:
            return
        text = text[:begin_idx] + block + text[end_idx + len(end) :]
        action = "已更新"
    else:
        # 无块（首次）或 marker 不成对（手工编辑残留）：统一在文件末尾写入完整块
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block + "\n"
        action = "已追加"
    set_env_sh.write_text(text, encoding="utf-8")
    print(f"[infer 仓] {action} libgomp static-TLS workaround（LD_PRELOAD <- {libgomp}）")


_PYTHON_GATE_ORIGINAL = (
    "def require_versions():\n"
    "    if sys.version_info[:2] != (3, 12):\n"
    "        raise RuntimeError(\n"
    '            f"Python={sys.version.split()[0]}; required 3.12 for the validated CANN 9.1.0 environment."\n'
    "        )\n"
)
_PYTHON_GATE_RELAXED = (
    "def require_versions():\n"
    "    if sys.version_info[:2] not in ((3, 11), (3, 12)):\n"
    "        raise RuntimeError(\n"
    '            f"Python={sys.version.split()[0]}; required 3.11 or 3.12 for the validated CANN 9.1.0 environment."\n'
    "        )\n"
)


def _relax_python_version_gate(repo_dir: Path) -> None:
    """放宽课程 patch 的 Python 版本门禁：3.12 -> 3.11/3.12。

    单卡 patch 在 ``require_versions()`` 把解释器硬性钉在 3.12（配套 CANN 9.1.0
    验证环境）；部分运行机只有 3.11（import 链与依赖版本均满足），故在克隆仓
    层面放宽为 3.11/3.12 皆可，3.13+ 与 3.10- 仍被拒绝。课程 patch 文件本身
    不动：patch 保留上游验证矩阵，本函数属于教程引导层的机器适配（与 cann_path
    替换、libgomp TLS workaround 同类）。幂等：已放宽则跳过；上游片段变化
    （patch 更新导致原文不匹配）时同样跳过，不盲改。
    """
    target = repo_dir / "models" / "longcat_next" / "models" / "model_infer.py"
    if not target.is_file():
        return
    text = target.read_text(encoding="utf-8")
    if _PYTHON_GATE_RELAXED in text or _PYTHON_GATE_ORIGINAL not in text:
        return
    target.write_text(text.replace(_PYTHON_GATE_ORIGINAL, _PYTHON_GATE_RELAXED), encoding="utf-8")
    print("[infer 仓] 已将 require_versions 的 Python 门禁放宽为 3.11/3.12（原仅 3.12）")


def ensure_infer_repo(
    source_dir: Path | None = None,
    patch_file: Path | None = None,
    *,
    quiet: bool = False,
) -> Path:
    """Bootstrap the upstream cann-recipes-infer clone with the single-NPU patch.

    目标状态（位于 gitignored 运行目录 ``Sources/model_inference_optimization/
    longcat_next/cann-recipes-infer``）：

    1. 全量克隆 ``INFER_REPO_URL``（不限深度，master 后续演进不影响回退）；
    2. ``git checkout`` 到固定基线 ``INFER_BASE_COMMIT``（单卡 patch 即基于该
       提交制作，``git am`` 必定干净应用）；
    3. ``git am`` 应用 ``patches/longcat-next-single-npu.patch``。

    幂等与自愈：

    - 未克隆 → 执行上述 1-3；
    - 已克隆且 HEAD 与状态文件记录的 patch 提交一致 → 直接复用（离线可用）；
    - 已克隆但状态不符（am 中断 / 误切分支等）→ ``git am --abort`` 清理残留后
      ``git checkout -f`` 回基线重新应用；checkout -f 不清理 untracked 文件，
      运行 YAML（config/*.yaml）、requests/assets 软链与 res/ 结果均保留。

    返回克隆根目录。
    """
    if source_dir is None:
        source_dir = _tutorial_workspace_dir()
    if patch_file is None:
        patch_file = _tutorial_dir() / "patches" / SINGLE_NPU_PATCH_NAME
    repo_dir = source_dir / INFER_REPO_DIRNAME
    state_file = source_dir / INFER_REPO_STATE_FILENAME
    state = _read_infer_repo_state(state_file)

    def _record_state(head: str) -> None:
        _write_infer_repo_state(
            state_file,
            repo_url=INFER_REPO_URL,
            base_commit=INFER_BASE_COMMIT,
            patch_commit=head,
        )

    if (repo_dir / ".git").is_dir():
        head = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()
        if state.get("base_commit") == INFER_BASE_COMMIT and state.get("patch_commit") == head:
            if not quiet:
                print(f"[infer 仓] 复用已就绪的克隆: {repo_dir}")
                print(f"[infer 仓] 基线 {INFER_BASE_COMMIT} + 单卡 patch（HEAD={head[:12]}）")
        else:
            if not quiet:
                print(
                    f"[infer 仓] 克隆已存在但状态与记录不符（HEAD={head[:12]}），"
                    "回退基线并重新应用 patch ..."
                )
            _git(repo_dir, "am", "--abort", check=False)
            _git(repo_dir, "checkout", "-f", "--detach", INFER_BASE_COMMIT)
            _apply_single_npu_patch(repo_dir, patch_file)
            head = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()
            _record_state(head)
            if not quiet:
                print(
                    f"[infer 仓] 已在基线 {INFER_BASE_COMMIT} 上重新应用单卡 patch"
                    f"（HEAD={head[:12]}）"
                )
    else:
        if repo_dir.exists():
            if not quiet:
                print(f"[infer 仓] {repo_dir} 不是有效 git 克隆，删除后重新克隆 ...")
            shutil.rmtree(repo_dir)
        source_dir.mkdir(parents=True, exist_ok=True)
        if not quiet:
            print(f"[infer 仓] 开始克隆 {INFER_REPO_URL}（全量克隆，约 300MB，仅首次运行需要）")
        subprocess.run(["git", "clone", INFER_REPO_URL, str(repo_dir)], check=True)
        _git(repo_dir, "checkout", "-f", "--detach", INFER_BASE_COMMIT)
        _apply_single_npu_patch(repo_dir, patch_file)
        head = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()
        _record_state(head)
        if not quiet:
            print(f"[infer 仓] 单卡 patch 已应用（基线 {INFER_BASE_COMMIT}，HEAD={head[:12]}）")

    _fix_infer_set_env_cann_path(repo_dir)
    _apply_gomp_tls_workaround(repo_dir)
    _relax_python_version_gate(repo_dir)
    return repo_dir


def ensure_request_assets(model_path: str, requests_dir: Path) -> Path:
    """Make sure ``requests/assets`` points at the checkpoint assets folder.

    Media paths inside a request manifest are resolved relative to the manifest
    directory, so the official demo assets (book.png / math1.wav / ...) are
    exposed through a symlink instead of copying files. The cloned infer repo
    is reused across chapters and sessions, so a stale symlink pointing at
    another (possibly removed) checkpoint is re-pointed at the current one.
    """
    assets_link = requests_dir / "assets"
    checkpoint_assets = Path(model_path) / "assets"
    if not checkpoint_assets.is_dir():
        raise FileNotFoundError(f"权重目录缺少 assets 素材: {checkpoint_assets}")
    if assets_link.is_symlink():
        try:
            if assets_link.resolve() == checkpoint_assets.resolve():
                return assets_link
        except OSError:
            pass
        assets_link.unlink()
        print(f"[素材准备] 软链指向其他目录，已重指 -> {checkpoint_assets}")
    elif assets_link.exists():
        return assets_link
    assets_link.symlink_to(checkpoint_assets, target_is_directory=True)
    print(f"[素材准备] 已创建软链 {assets_link} -> {checkpoint_assets}")
    return assets_link


def prepare_runtime_yaml(
    template_path: Path,
    output_path: Path,
    *,
    model_path: str | None = None,
    updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy a recipe YAML template, fill the model path and apply overrides.

    ``updates`` maps dotted config keys to values, e.g.
    ``{"model_config.enable_profiler": True}``.
    """
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    if model_path is not None:
        config.setdefault("model_config", {})["model_path"] = str(model_path)
    for dotted_key, value in (updates or {}).items():
        cursor: dict[str, Any] = config
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config


def print_yaml_focus(yaml_path: Path) -> None:
    """Print the LongCat-Next-relevant fields of a recipe YAML."""
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_config = config.get("model_config", {})
    custom = model_config.get("custom_params", {})
    multimodal = custom.get("multimodal", {})
    scheduler = config.get("scheduler_config", {})
    data_config = config.get("data_config", {})
    print("YAML 文件 =", yaml_path)
    print("model_path =", model_config.get("model_path"))
    print("exe_mode =", model_config.get("exe_mode"))
    print("enable_profiler =", model_config.get("enable_profiler"))
    print("enable_moe_cpu_offload =", custom.get("enable_moe_cpu_offload"))
    print("enable_ngram_cpu_offload =", custom.get("enable_ngram_cpu_offload"))
    print("moe_chunk_max_len =", custom.get("moe_chunk_max_len"))
    print("request_file =", multimodal.get("request_file"))
    print("head_exe_mode =", multimodal.get("head_exe_mode"))
    print("enable_thinking =", multimodal.get("enable_thinking"))
    print("dataset =", data_config.get("dataset"))
    print("input_truncated_len =", data_config.get("input_truncated_len"))
    print("max_prefill_tokens =", scheduler.get("max_prefill_tokens"))
    print("max_new_tokens =", scheduler.get("max_new_tokens"))
    print("batch_size =", scheduler.get("batch_size"))


def find_recipe_case_dir(recipe_root: Path, yaml_path: Path) -> Path | None:
    """Locate the ``res/<date>/<model>_<yaml>`` directory for one run."""
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    model_name = config.get("model_config", {}).get("model_name", "longcat_next")
    date = time.strftime("%Y%m%d")
    expected = recipe_root / "models" / "longcat_next" / "res" / date / f"{model_name}_{yaml_path.stem}"
    if expected.is_dir():
        return expected

    res_root = recipe_root / "models" / "longcat_next" / "res"
    candidates = sorted(
        res_root.glob(f"**/{model_name}_{yaml_path.stem}"),
        key=lambda path: path.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def ensure_recipe_log_ok(log_path: Path | None) -> None:
    """Fail loudly when log_0.log is missing or contains error markers."""
    if log_path is None or not log_path.is_file():
        raise RuntimeError("Recipe inference log_0.log was not found.")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    failure_markers = ("Traceback (most recent call last)", "[ERROR]")
    if any(marker in text for marker in failure_markers):
        tail = "\n".join(text.splitlines()[-30:])
        print("[Recipe 推理] log_0.log 检测到错误，最近日志如下：")
        print(tail)
        raise RuntimeError(f"Recipe inference failed; see {log_path}")


def read_request_manifest(yaml_path: Path, recipe_root: Path | None = None) -> list[dict[str, Any]]:
    """Read the request manifest referenced by a recipe YAML.

    ``request_file`` inside the YAML is resolved relative to
    ``<recipe_root>/models/longcat_next``. When ``recipe_root`` is omitted it
    is derived by walking up from the YAML to the directory containing
    ``executor/scripts/infer.sh`` (the cloned cann-recipes-infer root); this
    works both for template YAMLs shipped in the clone and for notebook
    runtime YAMLs written into the clone's ``config/`` directory.
    """
    if recipe_root is None:
        for candidate in (yaml_path, *yaml_path.parents):
            if (candidate / "executor" / "scripts" / "infer.sh").is_file():
                recipe_root = candidate
                break
        else:
            raise RuntimeError(
                f"无法从 {yaml_path} 定位 infer 仓根目录（需包含 executor/scripts/infer.sh），"
                "请显式传入 recipe_root"
            )
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    manifest_rel = config["model_config"]["custom_params"]["multimodal"]["request_file"]
    manifest_path = (Path(recipe_root) / "models" / "longcat_next" / manifest_rel).resolve()
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    return document["requests"]


def collect_request_metrics(
    recipe_root: Path,
    work_dir: Path,
    metrics_tag: str,
    yaml_path: Path,
) -> dict[str, Any]:
    """Collect per-request metrics from the result.json files of one run."""
    case_dir = find_recipe_case_dir(recipe_root, yaml_path)
    if case_dir is None:
        raise RuntimeError(f"未找到 recipes 结果目录：{yaml_path.stem}")
    log_path = case_dir / "log_0.log"

    work_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = work_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ensure_recipe_log_ok(log_path)

    request_rows = []
    for result_path in sorted((case_dir / "results").glob("*/result.json")):
        record = json.loads(result_path.read_text(encoding="utf-8"))
        request_rows.append(
            {
                "id": record["id"],
                "task": record["task"],
                "generated_sequence_tokens": record.get("generated_sequence_tokens"),
                "autoregressive_seconds": round(record.get("autoregressive_seconds", 0.0), 3),
                "text": record.get("text", ""),
                "image_files": [Path(p).name for p in record.get("image_files", [])],
                "audio_files": [Path(p).name for p in record.get("audio_files", [])],
                "head_exe_mode": record.get("head_exe_mode"),
                "reached_token_budget": record.get("reached_token_budget"),
            }
        )
    if not request_rows:
        raise RuntimeError(f"结果目录中没有 result.json：{case_dir / 'results'}")

    generated_tokens = int(sum(row["generated_sequence_tokens"] for row in request_rows))
    total_time_s = float(sum(row["autoregressive_seconds"] for row in request_rows))
    tokens_per_second = generated_tokens / total_time_s if total_time_s > 0 else 0.0

    summary = {
        "metrics_tag": metrics_tag,
        "recipe_yaml_path": str(yaml_path),
        "recipe_case_dir": str(case_dir),
        "recipe_log_path": str(log_path),
        "num_requests": len(request_rows),
        "generated_tokens": generated_tokens,
        "total_time_s": round(total_time_s, 3),
        "tokens_per_second": round(tokens_per_second, 4),
        "requests": request_rows,
    }
    metrics_path = metrics_dir / f"{metrics_tag}_summary.json"
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("指标文件 =", metrics_path)
    print("结果目录 =", case_dir)
    return summary
