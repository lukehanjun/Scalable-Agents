from __future__ import annotations

import json
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HarnessEvaluation:
    run_id: str
    resolved_ids: list[str]
    resolved_count: int
    accuracy: float
    results_path: str | None
    stdout_tail: str
    stderr_tail: str


def load_swebench_tasks(
    *,
    dataset_name: str,
    split: str,
    limit: int,
    cache_dir: str | Path,
    seed: int = 7,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: `datasets`. Install dependencies with `pip install -e .` first."
        ) from exc

    if limit <= 0:
        return []

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        dataset = load_dataset(dataset_name, split=split, cache_dir=str(cache_dir.resolve()))
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            "Failed to initialize SWE-bench dataset cache. "
            f"Cache path `{cache_dir}` may be too long on Windows. "
            "Set a shorter path via SWEBENCH_DATASET_CACHE (example: C:\\hf_cache)."
        ) from exc
    # Keep deterministic ordering so repeated runs are comparable.
    ordered = dataset.shuffle(seed=seed) if limit < len(dataset) else dataset
    rows = ordered.select(range(min(limit, len(ordered))))
    tasks: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        tasks.append(
            {
                "instance_id": row.get("instance_id", f"instance-{index}"),
                "problem_statement": row.get("problem_statement", ""),
                "repo": row.get("repo", ""),
                "repo_path": None,
                "base_commit": row.get("base_commit"),
                "reference_patch": row.get("patch"),
            }
        )
    return tasks


def ensure_swebench_harness(
    *,
    workspace_root: str | Path,
    install_if_missing: bool = True,
) -> Path:
    workspace_root = Path(workspace_root)
    external_dir = workspace_root / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    swebench_repo = external_dir / "SWE-bench"
    if not swebench_repo.exists():
        _run(
            ["git", "clone", "https://github.com/SWE-bench/SWE-bench.git", str(swebench_repo)],
            cwd=external_dir,
        )

    has_swebench = importlib.util.find_spec("swebench") is not None
    if install_if_missing and not has_swebench:
        _run([sys.executable, "-m", "pip", "install", "-e", str(swebench_repo)], cwd=workspace_root)

    return swebench_repo


def prepare_task_repositories(
    *,
    tasks: list[Any],
    repos_root: str | Path,
) -> None:
    repos_root = Path(repos_root).resolve()
    repos_root.mkdir(parents=True, exist_ok=True)
    current_commit_by_repo: dict[str, str | None] = {}

    for task in tasks:
        repo_name = getattr(task, "repo", None)
        if repo_name is None and isinstance(task, dict):
            repo_name = task.get("repo")
        if not repo_name:
            continue
        repo_key = repo_name.replace("/", "__")
        local_repo = repos_root / repo_key
        clone_url = f"https://github.com/{repo_name}.git"
        with _repo_lock(repos_root, repo_key):
            if not _is_git_repo(local_repo):
                _clone_repo(clone_url=clone_url, local_repo=local_repo)

            target_commit = getattr(task, "base_commit", None)
            if target_commit is None and isinstance(task, dict):
                target_commit = task.get("base_commit")
            if target_commit and current_commit_by_repo.get(repo_key) != target_commit:
                if not _has_commit(local_repo, target_commit):
                    try:
                        _run(["git", "fetch", "origin", target_commit], cwd=local_repo)
                    except RuntimeError:
                        _run(["git", "fetch", "--all", "--tags"], cwd=local_repo)
                _checkout_commit_with_recovery(
                    local_repo=local_repo,
                    target_commit=target_commit,
                    clone_url=clone_url,
                )
                current_commit_by_repo[repo_key] = target_commit
        repo_path = str(local_repo.resolve())
        if isinstance(task, dict):
            task["repo_path"] = repo_path
        else:
            setattr(task, "repo_path", repo_path)


def run_harness_evaluation(
    *,
    dataset_name: str,
    predictions_path: str | Path,
    run_id: str,
    max_workers: int,
    instance_ids: list[str],
    swebench_repo: str | Path,
    output_root: str | Path,
) -> HarnessEvaluation:
    swebench_repo = Path(swebench_repo)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])

    completed = _run(command, cwd=swebench_repo)
    results_path = _find_results_json(run_id, output_root, swebench_repo)
    resolved_ids = _extract_resolved_ids(results_path)
    resolved_count = len(resolved_ids)
    accuracy = round(resolved_count / max(1, len(instance_ids)), 4)
    return HarnessEvaluation(
        run_id=run_id,
        resolved_ids=resolved_ids,
        resolved_count=resolved_count,
        accuracy=accuracy,
        results_path=str(results_path) if results_path else None,
        stdout_tail=completed.stdout[-4000:],
        stderr_tail=completed.stderr[-4000:],
    )


def _find_results_json(run_id: str, output_root: Path, swebench_repo: Path) -> Path | None:
    candidate_paths = [
        output_root / "evaluation_results" / run_id / "results.json",
        swebench_repo / "evaluation_results" / run_id / "results.json",
    ]
    for path in candidate_paths:
        if path.exists():
            return path

    for base in (output_root, swebench_repo):
        for path in base.rglob("results.json"):
            if run_id in str(path.parent):
                return path
    return None


def _extract_resolved_ids(results_path: Path | None) -> list[str]:
    if not results_path or not results_path.exists():
        return []
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    for key in ("resolved_ids", "resolved_instances", "resolved"):
        value = payload.get(key)
        if isinstance(value, list):
            return [str(item) for item in value]

    report = payload.get("report")
    if isinstance(report, dict):
        value = report.get("resolved_ids")
        if isinstance(value, list):
            return [str(item) for item in value]
    return []


def _run(command: list[str], cwd: Path | str | None = None) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-2000:]
        stdout_tail = (completed.stdout or "")[-2000:]
        combined = f"{stderr_tail}\n{stdout_tail}"
        if "No module named 'resource'" in combined:
            raise RuntimeError(
                "SWE-bench harness requires POSIX/Unix runtime and is not supported on native Windows Python "
                "because the `resource` module is unavailable. Run harness in WSL2/Linux or Docker."
            )
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stderr:\n{stderr_tail}\nstdout:\n{stdout_tail}"
        )
    return completed


def _clone_repo(*, clone_url: str, local_repo: Path, filtered: bool = True) -> None:
    if local_repo.exists() and not local_repo.is_dir():
        raise RuntimeError(f"Repo path exists and is not a directory: {local_repo}")
    if local_repo.exists():
        _force_remove_dir(local_repo)
    local_repo.parent.mkdir(parents=True, exist_ok=True)

    staging_repo = local_repo.parent / f"{local_repo.name}.staging-{uuid.uuid4().hex[:8]}"
    if staging_repo.exists():
        _force_remove_dir(staging_repo)

    clone_command = ["git", "clone", "--no-checkout"]
    if filtered:
        clone_command.extend(["--filter=blob:none"])
    clone_command.extend([clone_url, str(staging_repo)])
    last_error: RuntimeError | None = None
    for attempt in range(1, 3):
        try:
            if local_repo.exists():
                _force_remove_dir(local_repo)
            if staging_repo.exists():
                _force_remove_dir(staging_repo)
            _run(clone_command)
            os.replace(str(staging_repo), str(local_repo))
            return
        except RuntimeError as exc:
            last_error = exc
            # Clean up partially cloned directories before retry.
            if local_repo.exists():
                _force_remove_dir(local_repo)
            if staging_repo.exists():
                _force_remove_dir(staging_repo)
            if attempt == 2:
                break
        except OSError as exc:
            last_error = RuntimeError(str(exc))
            if staging_repo.exists():
                _force_remove_dir(staging_repo)
            if attempt == 2:
                break
    raise RuntimeError(
        f"Failed to clone repo after retries: {clone_url}\n{last_error}"
    )


def _checkout_commit_with_recovery(*, local_repo: Path, target_commit: str, clone_url: str) -> None:
    current_head = _rev_parse(local_repo, "HEAD")
    if current_head == target_commit:
        return

    try:
        _run(["git", "checkout", "--force", target_commit], cwd=local_repo)
        return
    except RuntimeError as first_error:
        first_error_text = str(first_error).lower()
        # Try broad ref fetches before falling back to re-clone.
        try:
            _run(["git", "fetch", "origin", "--tags", "--prune"], cwd=local_repo)
            _run(["git", "fetch", "origin", "+refs/heads/*:refs/remotes/origin/*"], cwd=local_repo)
            _run(["git", "checkout", "--force", target_commit], cwd=local_repo)
            return
        except RuntimeError as second_error:
            second_error_text = str(second_error).lower()
            if (
                "reference is not a tree" not in first_error_text
                and "reference is not a tree" not in second_error_text
                and "unknown revision" not in second_error_text
                and "bad object" not in second_error_text
            ):
                raise second_error

            # Fallback: re-clone without blob filtering, then retry checkout.
            _force_remove_dir(local_repo)
            _clone_repo(clone_url=clone_url, local_repo=local_repo, filtered=False)
            _run(["git", "fetch", "origin", "--tags", "--prune"], cwd=local_repo)
            _run(["git", "fetch", "origin", "+refs/heads/*:refs/remotes/origin/*"], cwd=local_repo)
            _run(["git", "checkout", "--force", target_commit], cwd=local_repo)


def _has_commit(repo_dir: Path, commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _is_git_repo(path: Path) -> bool:
    return path.is_dir() and (path / ".git").exists()


def _rev_parse(repo_dir: Path, ref: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", ref],
        cwd=str(repo_dir),
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return (completed.stdout or "").strip() or None


def _repo_lock(repos_root: Path, repo_key: str):
    from filelock import FileLock

    lock_dir = repos_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = lock_dir / f"{repo_key}.lock"
    return FileLock(str(lock_file), timeout=1800)


def _force_remove_dir(path: Path) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        path.unlink(missing_ok=True)
        return

    def _onerror(func: Any, p: str, exc_info: Any) -> None:
        try:
            os.chmod(p, stat.S_IWRITE)
        except OSError:
            pass
        func(p)

    for _ in range(3):
        shutil.rmtree(path, onerror=_onerror, ignore_errors=True)
        if not path.exists():
            return
        time.sleep(0.25)
    raise RuntimeError(
        f"Unable to remove existing directory before clone: {path}. "
        "Close processes that may hold file handles (editor terminals, indexers) and retry."
    )
