"""Relative links in the README, docs and saved report must resolve inside the checkout."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "benchmarks" / "act-transfer" / "v1"

pytestmark = pytest.mark.skipif(
    not (ROOT / "pyproject.toml").is_file() or not V1.is_dir(),
    reason="needs a source checkout with benchmarks/act-transfer/v1",
)

MARKDOWN_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")
HTML_LINK = re.compile(r"""(?:href|src)="([^"]+)\"""")


def _markdown_files() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", ROOT / "profiles" / "README.md"]
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted(V1.glob("*.md"))
    files.append(V1 / "report" / "report.md")
    return [f for f in files if f.is_file()]


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "data:"))


def _slug(heading: str) -> str:
    text = re.sub(r"[^\w\s-]", "", heading.strip().lower())
    return re.sub(r"\s", "-", text)


def _anchors(path: Path) -> set[str]:
    headings = re.findall(r"^#+\s+(.*)$", path.read_text(encoding="utf-8"), flags=re.M)
    return {_slug(h) for h in headings}


def _check(source: Path, targets: list[str]) -> list[str]:
    problems = []
    for target in targets:
        if _is_external(target):
            continue
        path_part, _, anchor = target.partition("#")
        resolved = source if not path_part else (source.parent / path_part).resolve()
        if not resolved.exists():
            problems.append(f"{source.relative_to(ROOT)}: missing {target}")
        elif anchor and resolved.suffix == ".md" and anchor not in _anchors(resolved):
            problems.append(f"{source.relative_to(ROOT)}: no heading for {target}")
    return problems


@pytest.mark.parametrize("path", _markdown_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_markdown_links_resolve(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    targets = [a or b for a, b in MARKDOWN_LINK.findall(text)]
    assert not _check(path, targets)


def test_report_html_links_resolve() -> None:
    html = V1 / "report" / "index.html"
    targets = HTML_LINK.findall(html.read_text(encoding="utf-8"))
    assert targets
    assert not _check(html, targets)


def test_readme_media_and_social_preview_exist() -> None:
    media = V1 / "media"
    for name in ("comparison.gif", "comparison.mp4", "social-preview.png"):
        assert (media / name).is_file(), name
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "benchmarks/act-transfer/v1/media/comparison.gif" in readme
