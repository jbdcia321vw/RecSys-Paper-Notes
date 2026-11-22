#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_oa_fallback.py — 探测 DBLP 目录中未上 arXiv 的论文能否通过 OpenAlex / Unpaywall / Semantic Scholar / 顶会直链下载免费 PDF。
用法示例：
  python scripts/test_oa_fallback.py --venue SIGIR --year 2024 --limit 15
  python scripts/test_oa_fallback.py --venue RecSys --year 2025 --limit 15
"""

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _http_get_json(url: str, timeout: int = 15) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "RecSys-Paper-Notes/3.9 (mailto:recsys-notes@papernotes.org)",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _check_pdf_downloadable(url: str, timeout: int = 15) -> tuple:
    """真正请求前几个字节，确认是 HTTP 200 且文件头为 %PDF-（排除 Cloudflare 人机验证页）。"""
    if not url:
        return False, "empty"
    for proxy_url in ("", "http://127.0.0.1:7897"):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RecSys-Paper-Notes/3.9",
                    "Accept": "application/pdf,*/*",
                },
            )
            if proxy_url:
                opener = urllib.request.build_opener(
                    urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
                )
                ctx = opener.open(req, timeout=timeout)
            else:
                ctx = urllib.request.urlopen(req, timeout=timeout)
            with ctx as resp:
                head = resp.read(8)
                if head.startswith(b"%PDF-"):
                    return True, f"HTTP {resp.status} (%PDF)"
                return False, f"非PDF内容 ({head!r})"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code}"
        except Exception as e:
            last_err = str(e)[:60]
    return False, last_err


def _extract_doi(ee_url: str) -> str:
    m = re.search(r"doi\.org/(10\.\d{4,9}/[^\s\"'<>]+)", ee_url or "", re.IGNORECASE)
    return m.group(1).rstrip("/") if m else ""


def _direct_venue_pdf_candidate(ee_url: str) -> str:
    if not ee_url:
        return ""
    u = ee_url.strip()
    if u.lower().endswith(".pdf"):
        return u
    if "openreview.net/forum?id=" in u:
        return u.replace("openreview.net/forum?id=", "openreview.net/pdf?id=")
    if "aclanthology.org/" in u:
        return u.rstrip("/") + ".pdf"
    if "proceedings.mlr.press/" in u and u.endswith(".html"):
        stem = u.rsplit("/", 1)[-1][:-5]
        return u.rsplit("/", 1)[0] + f"/{stem}/{stem}.pdf"
    return ""


def probe_paper(title: str, ee_url: str) -> dict:
    doi = _extract_doi(ee_url)
    candidates = []

    # 1. 顶会官网直链 (OpenReview / ACL Anthology / PMLR / 直接 .pdf)
    direct_pdf = _direct_venue_pdf_candidate(ee_url)
    if direct_pdf:
        candidates.append(("Venue-Direct", direct_pdf))

    # 2. OpenAlex API (按 DOI 或标题)
    try:
        if doi:
            oa_url = f"https://api.openalex.org/works/https://doi.org/{urllib.parse.quote(doi)}?mailto=recsys-notes@papernotes.org"
            oa = _http_get_json(oa_url)
        else:
            oa_url = f"https://api.openalex.org/works?search={urllib.parse.quote(title[:120])}&per-page=1&mailto=recsys-notes@papernotes.org"
            res = _http_get_json(oa_url).get("results") or []
            oa = res[0] if res else {}
        best_oa = oa.get("best_oa_location") or {}
        if best_oa.get("pdf_url"):
            candidates.append(("OpenAlex", best_oa["pdf_url"]))
        for loc in (oa.get("locations") or []):
            pu = (loc or {}).get("pdf_url")
            if pu and all(pu != c[1] for c in candidates):
                candidates.append(("OpenAlex-Loc", pu))
    except Exception:
        pass

    # 3. Unpaywall API (仅需 DOI)
    if doi:
        try:
            up_url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email=recsys-notes@papernotes.org"
            up = _http_get_json(up_url)
            best_up = up.get("best_oa_location") or {}
            if best_up.get("url_for_pdf"):
                candidates.append(("Unpaywall", best_up["url_for_pdf"]))
        except Exception:
            pass

    # 4. Semantic Scholar API (按 DOI 或标题)
    try:
        if doi:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi)}?fields=title,openAccessPdf"
            s2 = _http_get_json(s2_url)
        else:
            s2_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={urllib.parse.quote(title[:120])}&limit=1&fields=title,openAccessPdf"
            rows = _http_get_json(s2_url).get("data") or []
            s2 = rows[0] if rows else {}
        oa_pdf = (s2.get("openAccessPdf") or {}).get("url")
        if oa_pdf:
            candidates.append(("SemanticScholar", oa_pdf))
    except Exception:
        pass

    tried = []
    for src, pdf_url in candidates:
        ok, status = _check_pdf_downloadable(pdf_url)
        tried.append((src, pdf_url, ok, status))
        if ok:
            return {"ok": True, "source": src, "pdf_url": pdf_url, "status": status, "tried": tried}
    return {"ok": False, "source": "-", "pdf_url": "", "status": tried[-1][3] if tried else "no_oa_url", "tried": tried}


def main():
    parser = argparse.ArgumentParser(description="测试非 arXiv DBLP 论文通过开放获取 API 下载 PDF 的可行性")
    parser.add_argument("--venue", default="SIGIR")
    parser.add_argument("--year", default="2024")
    parser.add_argument("--limit", type=int, default=15)
    args = parser.parse_args()

    cache_file = REPO_ROOT / "data" / "dblp_cache" / f"{args.venue}_{args.year}.json"
    if not cache_file.exists():
        print(f"[Error] 未找到 DBLP 缓存文件: {cache_file}，请先运行 build_curated_list.py --use-dblp")
        return

    dblp_rows = json.loads(cache_file.read_text(encoding="utf-8"))
    csv_file = REPO_ROOT / "data" / "papers" / f"{args.venue}{args.year}.csv"
    existing_norms = set()
    if csv_file.exists():
        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if r.get("title"):
                    existing_norms.add(re.sub(r"[^a-z0-9]", "", r["title"].lower()))

    missed = [
        r for r in dblp_rows
        if re.sub(r"[^a-z0-9]", "", (r.get("title") or "").lower()) not in existing_norms
    ]
    print(f"[Probe] {args.venue} {args.year} DBLP 缓存共 {len(dblp_rows)} 篇，未在 CSV (arXiv) 中的共 {len(missed)} 篇，抽样实测前 {min(args.limit, len(missed))} 篇...\n")

    ok_count = 0
    for idx, row in enumerate(missed[: args.limit], 1):
        title = row.get("title", "")
        ee = row.get("ee", "")
        res = probe_paper(title, ee)
        if res["ok"]:
            ok_count += 1
            print(f"[{idx:02d}] [OK - {res['source']}] {title[:65]}")
            print(f"     PDF: {res['pdf_url']} ({res['status']})")
        else:
            print(f"[{idx:02d}] [MISS] {title[:65]} -> {res['status']}")
        time.sleep(0.5)

    print(f"\n[Summary] 实测 {min(args.limit, len(missed))} 篇未上 arXiv 的 DBLP 论文，成功直接下载真实 PDF: {ok_count} 篇！")


if __name__ == "__main__":
    main()