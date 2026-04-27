# -*- coding: utf-8 -*-
"""Streamlit parser for local agent/skill markdown files.

Scopes:
1) .agents/skills/*/SKILL.md
2) Any local */agents directory (e.g., everything-claude-code/agents)

No network calls, no destructive operations.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import streamlit as st


def normalize_name(name: str) -> str:
    value = (name or "").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unnamed-skill"


def yaml_quote(value: str) -> str:
    return '"' + (value or "").replace("\\", "\\\\").replace('"', '\\"') + '"'


def parse_front_matter(text: str) -> Tuple[Dict[str, str], str, bool]:
    if not text.startswith("---"):
        return {}, text, False

    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, False

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, text, False

    meta: Dict[str, str] = {}
    for line in lines[1:end_idx]:
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        value = v.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        meta[key] = value

    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return meta, body, True


def infer_heading(markdown_body: str) -> str:
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = re.sub(r"^#+\s*", "", stripped).strip()
            if heading:
                return heading
    return ""


def infer_first_paragraph(markdown_body: str) -> str:
    paragraph: List[str] = []
    in_code_fence = False
    for line in markdown_body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith("#"):
            continue
        paragraph.append(stripped)
    return " ".join(paragraph).strip()


def read_text_safely(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_skill_file(path: Path) -> Dict[str, str]:
    text = read_text_safely(path)
    meta, body, has_yaml = parse_front_matter(text)

    skill_name = (
        meta.get("name")
        or infer_heading(body)
        or path.parent.name
        or path.stem
    )
    description = (meta.get("description") or infer_first_paragraph(body)).strip()

    return {
        "source_path": str(path),
        "folder_name": path.parent.name,
        "skill_name": skill_name.strip(),
        "description": description,
        "has_yaml": "yes" if has_yaml else "no",
    }


def detect_candidate_dirs(repo_root: Path) -> List[Path]:
    found: set[Path] = set()

    skills_root = repo_root / ".agents" / "skills"
    if skills_root.is_dir():
        found.add(skills_root.resolve())

    for d in repo_root.rglob("*"):
        if not d.is_dir():
            continue
        parts = set(d.parts)
        if ".git" in parts or "__pycache__" in parts:
            continue
        if d.name.lower() == "agents":
            found.add(d.resolve())

    return sorted(found)


def parse_manual_paths(repo_root: Path, manual_input: str) -> List[Path]:
    paths: List[Path] = []
    if not manual_input.strip():
        return paths
    tokens = re.split(r"[\n,;]+", manual_input)
    for token in tokens:
        p = token.strip().strip('"').strip("'")
        if not p:
            continue
        path = Path(p)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        if path.exists():
            paths.append(path)
    return paths


def collect_markdown_files(source_dirs: Iterable[Path], repo_root: Path) -> List[Path]:
    files: set[Path] = set()
    skills_root = (repo_root / ".agents" / "skills").resolve()

    for src in source_dirs:
        if src.is_file() and src.suffix.lower() == ".md":
            files.add(src.resolve())
            continue
        if not src.is_dir():
            continue

        resolved = src.resolve()
        if resolved == skills_root:
            for p in resolved.glob("*/SKILL.md"):
                if p.is_file():
                    files.add(p.resolve())
            continue

        for p in resolved.rglob("*.md"):
            if p.is_file():
                files.add(p.resolve())

    return sorted(files)


def filter_records(records: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    q = (query or "").strip().lower()
    if not q:
        return records

    out: List[Dict[str, str]] = []
    for row in records:
        haystack = " ".join(
            [
                row.get("source_path", ""),
                row.get("folder_name", ""),
                row.get("skill_name", ""),
                row.get("description", ""),
            ]
        ).lower()
        if q in haystack:
            out.append(row)
    return out


def make_csv_bytes(rows: List[Dict[str, str]]) -> bytes:
    fields = ["source_path", "folder_name", "skill_name", "description", "has_yaml"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in fields})
    return buf.getvalue().encode("utf-8")


def make_json_bytes(rows: List[Dict[str, str]]) -> bytes:
    fields = ["source_path", "folder_name", "skill_name", "description", "has_yaml"]
    safe_rows = [{k: row.get(k, "") for k in fields} for row in rows]
    return json.dumps(safe_rows, ensure_ascii=False, indent=2).encode("utf-8")


def import_or_convert(rows: List[Dict[str, str]], repo_root: Path, overwrite: bool) -> Tuple[int, int, int]:
    target_root = repo_root / ".agents" / "skills"
    target_root.mkdir(parents=True, exist_ok=True)

    imported = 0
    skipped = 0
    failed = 0

    for row in rows:
        src = Path(row["source_path"])
        if not src.exists():
            failed += 1
            continue

        normalized = normalize_name(row.get("skill_name", "") or row.get("folder_name", "") or src.stem)
        dst_dir = target_root / normalized
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "SKILL.md"

        if dst.exists() and not overwrite:
            skipped += 1
            continue

        try:
            source_text = read_text_safely(src)
            _, _, has_yaml = parse_front_matter(source_text)
            if has_yaml:
                out_text = source_text
            else:
                desc = (row.get("description") or "No description provided").strip()
                out_text = (
                    "---\n"
                    f"name: {yaml_quote(normalized)}\n"
                    f"description: {yaml_quote(desc)}\n"
                    "---\n\n"
                    f"{source_text.strip()}\n"
                )
            dst.write_text(out_text, encoding="utf-8")
            imported += 1
        except Exception:
            failed += 1

    return imported, skipped, failed


def main() -> None:
    st.set_page_config(page_title="Skill Parser Tool", page_icon="🧩", layout="wide")
    st.title("🧩 Skill / Agent Markdown Parser")
    st.caption("Local-only parser: scans skill/agent markdown, supports import/convert, and CSV/JSON export.")

    repo_root = Path.cwd()
    auto_dirs = detect_candidate_dirs(repo_root)

    st.subheader("Source Selection")
    use_auto = st.checkbox("Use auto-detected source directories", value=True)
    with st.expander("Auto-detected directories", expanded=False):
        if auto_dirs:
            for d in auto_dirs:
                st.code(str(d))
        else:
            st.write("No auto-detected directories found.")

    manual_input = st.text_area(
        "Manual path input (one per line, or comma-separated; files or directories)",
        value="",
        height=100,
        placeholder=".agents/skills\neverything-claude-code/agents",
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        do_scan = st.button("Scan Markdown Files", type="primary", use_container_width=True)
    with col_b:
        overwrite = st.checkbox("Overwrite existing imported SKILL.md", value=False)

    if not do_scan and "parsed_records" not in st.session_state:
        st.info("Click 'Scan Markdown Files' to parse skills.")
        return

    if do_scan:
        selected_dirs: List[Path] = []
        if use_auto:
            selected_dirs.extend(auto_dirs)
        selected_dirs.extend(parse_manual_paths(repo_root, manual_input))

        # Deduplicate while preserving simple deterministic order
        uniq_dirs = sorted(set(d.resolve() for d in selected_dirs))
        md_files = collect_markdown_files(uniq_dirs, repo_root)
        records = [parse_skill_file(p) for p in md_files]
        st.session_state["parsed_records"] = records
        st.session_state["selected_dirs"] = [str(p) for p in uniq_dirs]

    records = st.session_state.get("parsed_records", [])
    if not records:
        st.warning("No markdown files found in selected directories.")
        return

    st.subheader("Quick Stats")
    total = len(records)
    missing_yaml = sum(1 for r in records if r.get("has_yaml") != "yes")
    missing_desc = sum(1 for r in records if not r.get("description", "").strip())
    m1, m2, m3 = st.columns(3)
    m1.metric("# Skills/Agents", total)
    m2.metric("Missing YAML", missing_yaml)
    m3.metric("Missing Description", missing_desc)

    query = st.text_input("Search", value="", placeholder="Filter by path, folder, skill name, or description")
    filtered = filter_records(records, query)

    st.subheader("Parsed Table")
    st.write(f"Showing {len(filtered)} / {len(records)} rows")
    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "source_path": st.column_config.TextColumn("source_path", width="large"),
            "folder_name": st.column_config.TextColumn("folder_name", width="small"),
            "skill_name": st.column_config.TextColumn("skill_name", width="medium"),
            "description": st.column_config.TextColumn("description", width="large"),
            "has_yaml": st.column_config.TextColumn("has_yaml", width="small"),
        },
    )

    st.subheader("Actions")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("Import/Convert", use_container_width=True):
            imported, skipped, failed = import_or_convert(filtered, repo_root, overwrite=overwrite)
            st.success(f"Imported/converted: {imported}, skipped: {skipped}, failed: {failed}")
    with c2:
        st.download_button(
            "Export CSV",
            data=make_csv_bytes(filtered),
            file_name="parsed_skills.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "Export JSON",
            data=make_json_bytes(filtered),
            file_name="parsed_skills.json",
            mime="application/json",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

