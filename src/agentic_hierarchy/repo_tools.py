from __future__ import annotations

import os
import re
from pathlib import Path

from .models import RepoSnippet


class LocalRepoContextBuilder:
    EXCLUDED_DIRS = {".git", ".venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache"}

    def __init__(self, repo_path: str | Path | None) -> None:
        self.repo_path = Path(repo_path).resolve() if repo_path else None

    def exists(self) -> bool:
        return self.repo_path is not None and self.repo_path.exists()

    def build_context(
        self,
        problem_statement: str,
        *,
        max_files: int = 8,
        max_chars_per_file: int = 3_500,
    ) -> list[RepoSnippet]:
        if not self.exists():
            return []

        keywords = self._keywords(problem_statement)
        candidates: list[tuple[float, Path]] = []
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if any(part in self.EXCLUDED_DIRS for part in path.parts):
                continue
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".lock", ".pyc"}:
                continue
            relative = path.relative_to(self.repo_path).as_posix().lower()
            score = self._path_score(relative, keywords)
            if score > 0:
                candidates.append((score, path))

        candidates.sort(key=lambda item: item[0], reverse=True)
        snippets: list[RepoSnippet] = []
        for score, path in candidates[:max_files]:
            content = self._read_file(path, max_chars_per_file)
            if not content:
                continue
            snippets.append(
                RepoSnippet(
                    path=path.relative_to(self.repo_path).as_posix(),
                    content=content,
                    score=round(score, 4),
                )
            )
        return snippets

    @staticmethod
    def summarize(snippets: list[RepoSnippet]) -> str:
        if not snippets:
            return "No local repository context was available."
        parts = []
        for snippet in snippets:
            parts.append(f"FILE: {snippet.path}\n{snippet.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _keywords(problem_statement: str) -> set[str]:
        words = {
            word
            for word in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]+", problem_statement.lower())
            if len(word) >= 3
        }
        common_noise = {
            "that", "with", "from", "under", "style", "issues", "issue", "should",
            "would", "could", "there", "their", "about", "while", "remain",
        }
        return words - common_noise

    @staticmethod
    def _path_score(relative_path: str, keywords: set[str]) -> float:
        if not keywords:
            return 0.05
        score = 0.0
        filename = relative_path.rsplit("/", 1)[-1]
        for keyword in keywords:
            if keyword in filename:
                score += 2.5
            elif keyword in relative_path:
                score += 1.0
        if "/test" in relative_path or "test_" in filename:
            score += 0.4
        if filename.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".java", ".go", ".rs")):
            score += 0.25
        return score

    @staticmethod
    def _read_file(path: Path, max_chars: int) -> str:
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
        if len(raw) > max_chars:
            return raw[:max_chars] + "\n...[truncated]"
        return raw


def resolve_repo_path(task_repo_path: str | None, repo_root: str | None, repo_name: str) -> str | None:
    if task_repo_path:
        path = Path(task_repo_path)
        return str(path.resolve()) if path.exists() else str(path)
    if not repo_root:
        return None
    root = Path(repo_root)
    candidates = [
        root / repo_name.replace("/", "__"),
        root / repo_name.replace("/", "-"),
        root / repo_name.split("/")[-1],
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str(candidates[0])
