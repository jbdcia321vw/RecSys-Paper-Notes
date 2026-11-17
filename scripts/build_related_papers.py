#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_related_papers.py

为每篇论文笔记计算标题 + 摘要的语义嵌入，并按余弦相似度在笔记底部生成
5 篇相关论文推荐。摘要优先从 .cache_pdfs/<arxiv_id>.pdf 提取；PDF 缺失或
提取失败时，回退到笔记 description 与 TL;DR。

用法：
  python scripts/build_related_papers.py
  python scripts/build_related_papers.py --top-k 5 --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import hashlib
import json
import posixpath
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_nav_and_index import collect_all_notes, render_paper_dl

START_MARKER = "<!-- related-papers:start -->"
END_MARKER = "<!-- related-papers:end -->"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_manifest(repo_root: Path) -> dict:
    path = repo_root / "data" / "manifest.jsonl"
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        note_path = item.get("note_path", "")
        if note_path:
            rows[note_path] = item
    return rows


def index_pdfs(cache_dir: Path) -> dict:
    pdfs = defaultdict(list)
    if not cache_dir.exists():
        return pdfs
    for path in cache_dir.glob("*.pdf"):
        base = re.sub(r"v\d+$", "", path.stem)
        pdfs[base].append(path)
    for paths in pdfs.values():
        paths.sort(key=lambda p: p.name)
    return pdfs


def arxiv_id_for(note: dict, manifest_row: dict) -> str:
    arxiv_id = manifest_row.get("arxiv_id") or note.get("arxiv_id") or ""
    if not arxiv_id:
        match = re.search(r"arxiv\.org/abs/([^/?#]+)", note.get("paper_url", ""))
        if match:
            arxiv_id = match.group(1)
    return re.sub(r"v\d+$", "", arxiv_id)


def resolve_pdf(arxiv_id: str, pdfs: dict):
    if not arxiv_id:
        return None
    candidates = pdfs.get(arxiv_id, [])
    return candidates[0] if candidates else None


def extract_abstract(pdf_path: Path) -> tuple:
    try:
        import pymupdf
    except ImportError:
        return "", "pymupdf-missing"

    try:
        doc = pymupdf.open(pdf_path)
    except Exception:
        return "", "open-failed"

    try:
        page_count = min(2, len(doc))
        text = "\n".join(doc[i].get_text() for i in range(page_count))
    finally:
        doc.close()

    text = text.replace("\u00ad", "").replace("-\n", "")
    text = re.sub(r"[ \t]+", " ", text)
    match = re.search(r"(?i)\babstract\b", text)
    if not match:
        return "", "marker-missing"

    rest = text[match.end():]
    stop = re.search(
        r"(?i)\n\s*(?:ccs concepts|keywords?|(?:1\.?\s+)?introduction|i\.\s+introduction)\b",
        rest,
    )
    abstract = rest[: stop.start() if stop else 4000]
    abstract = re.sub(r"\s+", " ", abstract).strip()
    if len(abstract) < 80:
        return "", "too-short"
    return abstract[:4000], "pdf"


def extract_tldr(note_path: Path) -> str:
    text = note_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^## TL;DR[^\n]*\n(.*?)(?=\n## |\Z)", text, re.DOTALL | re.MULTILINE)
    if not match:
        return ""
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value[:1800]


def build_embedding_text(note: dict, abstract: str, note_path: Path) -> str:
    title = note.get("title", "").strip()
    if abstract:
        return f"{title}\n\n{abstract}"
    description = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", note.get("description", "")).strip()
    tldr = extract_tldr(note_path)
    parts = [title, description, tldr]
    return "\n\n".join(part for part in parts if part)


def load_embedding_cache(cache_path: Path, model_name: str, paths: list, hashes: list):
    if not cache_path.exists():
        return None
    try:
        data = np.load(cache_path, allow_pickle=False)
        cached_model = str(data["model"].item())
        cached_paths = [str(x) for x in data["paths"].tolist()]
        cached_hashes = [str(x) for x in data["hashes"].tolist()]
        vectors = data["vectors"]
    except Exception:
        return None
    if cached_model != model_name or cached_paths != paths or cached_hashes != hashes:
        return None
    return vectors


def encode_texts(model_name: str, texts: list):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def relative_note_url(source_rel: str, target_rel: str) -> str:
    source_dir = posixpath.dirname(source_rel)
    source_page_dir = posixpath.join(
        source_dir, posixpath.splitext(posixpath.basename(source_rel))[0]
    )
    target_dir = posixpath.splitext(target_rel)[0]
    rel = posixpath.relpath(target_dir, source_page_dir)
    return rel.rstrip("/") + "/"


def render_related_block(note: dict, related: list) -> str:
    if not related:
        return ""

    def href_for(item: dict) -> str:
        return relative_note_url(note["rel_from_docs"], item["rel_from_docs"])

    lines = [
        START_MARKER,
        "## 相关论文推荐",
        "",
        '<div class="conf-index related-papers">',
        render_paper_dl(related, href_for),
        "</div>",
        END_MARKER,
    ]
    return "\n".join(lines)


def upsert_related_block(note_path: Path, block: str):
    text = note_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\n*" + re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER) + r"\s*",
        re.DOTALL,
    )
    text = pattern.sub("\n", text).rstrip()
    if block:
        text = f"{text}\n\n{block}\n"
    note_path.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成论文语义相关推荐")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"嵌入模型（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--top-k", type=int, default=5, help="每篇笔记推荐数量（默认 5）")
    parser.add_argument("--cache-dir", default=".cache_pdfs", help="PDF 缓存目录")
    parser.add_argument("--output-dir", default="data/related", help="嵌入与相关推荐输出目录")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 篇，用于调试")
    parser.add_argument("--no-write-notes", action="store_true", help="只生成数据，不修改笔记")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    docs_dir = REPO_ROOT / "docs"
    cache_dir = REPO_ROOT / args.cache_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    notes = collect_all_notes(docs_dir)
    if args.limit:
        notes = notes[: args.limit]
    if not notes:
        print("[Related] 未找到论文笔记。")
        return 1

    manifest = load_manifest(REPO_ROOT)
    pdfs = index_pdfs(cache_dir)
    texts = []
    sources = defaultdict(int)

    for note in notes:
        rel = note["rel_from_docs"]
        manifest_row = manifest.get(f"docs/{rel}", {})
        arxiv_id = arxiv_id_for(note, manifest_row)
        pdf_path = resolve_pdf(arxiv_id, pdfs)
        abstract = ""
        source = "fallback"
        if pdf_path:
            abstract, source = extract_abstract(pdf_path)
        sources[source] += 1
        if source not in {"pdf", "fallback"}:
            sources["fallback"] += 1
        texts.append(build_embedding_text(note, abstract, note["path"]))

    paths = [note["rel_from_docs"] for note in notes]
    hashes = [text_hash(text) for text in texts]
    embedding_path = output_dir / "embeddings.npz"
    vectors = load_embedding_cache(embedding_path, args.model, paths, hashes)
    if vectors is None:
        print(f"[Related] 计算 {len(texts)} 篇论文嵌入：{args.model}", flush=True)
        vectors = encode_texts(args.model, texts)
        np.savez_compressed(
            embedding_path,
            model=np.array(args.model),
            paths=np.array(paths),
            hashes=np.array(hashes),
            vectors=vectors,
        )
    else:
        print(f"[Related] 复用已有嵌入：{embedding_path}", flush=True)

    similarity = vectors @ vectors.T
    np.fill_diagonal(similarity, -1.0)
    mapping = {}

    for index, note in enumerate(notes):
        seen_titles = {normalize_title(note.get("title", ""))}
        related = []
        for other_index in np.argsort(-similarity[index]):
            if len(related) >= args.top_k:
                break
            other = notes[int(other_index)]
            title_key = normalize_title(other.get("title", ""))
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            related.append(
                {
                    "rel_from_docs": other["rel_from_docs"],
                    "title": other.get("title", ""),
                    "description": other.get("description", ""),
                    "score": float(similarity[index][int(other_index)]),
                    "venue": other.get("venue", ""),
                    "year": other.get("year", ""),
                    "categories": other.get("categories", []),
                }
            )
        mapping[note["rel_from_docs"]] = related
        if not args.no_write_notes:
            block = render_related_block(note, related)
            upsert_related_block(note["path"], block)

    (output_dir / "related_papers.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"[Related] 完成 {len(notes)} 篇：PDF 摘要 {sources.get('pdf', 0)}，"
        f"回退 {sources.get('fallback', 0)}；输出 {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
