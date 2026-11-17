"""Build a reviewable source candidate using an explicit allowlist; never publish it."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from base.version import VERSION

ROOT_FILES = (
    ".gitignore",
    ".gitattributes",
    "README.md",
    "RUNNING.md",
    "THIRD_PARTY_NOTICES.md",
    "app.py",
    "new_main.py",
    "demo_backend.py",
    "bootstrap_database.py",
    "config.example.ini",
    "docker-compose.yml",
    "run_demo.ps1",
    "run_full.ps1",
    "pyproject.toml",
    "requirements-demo.txt",
    "requirements-dev.txt",
    "requirements-eval.txt",
    "requirements.txt",
)
PUBLIC_DOCS = ("项目架构.md", "发布说明.md", "验证记录.md")


def candidates():
    selected = [ROOT / name for name in ROOT_FILES]
    for folder in ("base", "mysql_qa", "rag_qa", "evals", "scripts", "tests"):
        selected.extend(
            p
            for p in (ROOT / folder).rglob("*.py")
            if not {"models", "data", "classify_data", "__pycache__"}.intersection(
                p.relative_to(ROOT).parts
            )
        )
    for folder in ("examples", "static", ".github"):
        selected.extend(p for p in (ROOT / folder).rglob("*") if p.is_file())
    selected.extend(ROOT / "docs" / name for name in PUBLIC_DOCS)
    selected.extend((ROOT / "docs/assets").glob("*.png"))
    selected.append(ROOT / "reports/release_validation.json")
    return sorted(set(p for p in selected if p.is_file()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = (args.output or ROOT / f".runtime/releases/edurag-{VERSION}-{stamp}").resolve()
    if target == ROOT or target in ROOT.parents or target.exists():
        raise ValueError("发布候选目录必须是尚不存在的新目录，不能覆盖现有项目")
    pattern = re.compile(
        r"sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"
        r"|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    entries = []
    for source in candidates():
        content = source.read_bytes()
        if source.suffix.lower() != ".png":
            content = content.replace(b"\r\n", b"\n")
        if pattern.search(content.decode("utf-8", errors="ignore")):
            raise ValueError(f"候选文件命中凭证模式，需人工检查：{source.relative_to(ROOT)}")
        entries.append((source, content))
    target.mkdir(parents=True)
    manifest = []
    for source, content in entries:
        relative = source.relative_to(ROOT)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest.append(
            {"path": relative.as_posix(), "sha256": hashlib.sha256(content).hexdigest()}
        )
    summary = {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "publication_status": "local_candidate_only_upstream_license_pending",
        "original_course_assets_included": False,
        "files": manifest,
    }
    (target / "RELEASE_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"directory": str(target), "files": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
