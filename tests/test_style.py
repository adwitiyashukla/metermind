from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_CHARACTERS = "".join(
    chr(code) for code in (
        0x2014, 0x2013, 0x2012, 0x2015, 0x2010, 0x2011, 0x2212,
        0x2018, 0x2019, 0x201C, 0x201D, 0x2026, 0x00B7, 0x2022,
        0x2192, 0x2190, 0x21D2,
    )
)

TEXT_SUFFIXES = {".py", ".md", ".yml", ".yaml", ".sql", ".txt", ".toml", ".cfg", ".ini", ".sh"}


def _tracked_text_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return []
    files = []
    for line in result.stdout.splitlines():
        path = REPO_ROOT / line.strip()
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return sorted(files)


@pytest.mark.parametrize("path", _tracked_text_files(), ids=lambda p: p.name)
def test_no_typographic_characters(path: Path):
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip(f"{path.name} is not UTF-8 text")

    offenders = [
        f"  line {n}: {line.strip()[:90]}"
        for n, line in enumerate(content.splitlines(), 1)
        if any(character in line for character in FORBIDDEN_CHARACTERS)
    ]
    assert not offenders, (
        f"{path.relative_to(REPO_ROOT)} contains non-ASCII typographic characters:\n"
        + "\n".join(offenders)
    )


@pytest.mark.parametrize("path", _tracked_text_files(), ids=lambda p: p.name)
def test_no_html_entities_that_render_as_typography(path: Path):
    import re

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        pytest.skip(f"{path.name} is not UTF-8 text")

    names = ("mdash", "ndash", "hellip", "middot", "bull", "rarr", "larr", "le", "ge")
    banned = {"&" + name + ";" for name in names}
    found = {m for m in re.findall(r"&[a-z]+;", content) if m in banned}
    assert not found, f"{path.relative_to(REPO_ROOT)} contains {sorted(found)}"


def test_no_absolute_local_paths():
    import re

    pattern = re.compile(r"[A-Z]:\\\\?Users\\\\?", re.IGNORECASE)
    offenders = []
    for path in _tracked_text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if pattern.search(content):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"absolute local paths found in {offenders}"
