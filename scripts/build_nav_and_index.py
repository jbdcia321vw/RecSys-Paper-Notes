#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_nav_and_index.py
扫描 docs/venues/<VENUE>/<YEAR>/<category>/ 下的推荐系统论文笔记，自动生成：

1. 按推荐系统 16 大细分类别聚合的跨会议索引页 (docs/topics/<category>.md)
2. 按会议年份聚合的索引页 (docs/years/<year>.md)
3. 按顶会/顶刊 + 年份 + 分类划分的索引页
   (docs/venues/<VENUE>/index.md, docs/venues/<VENUE>/<YEAR>/index.md, .../<category>/index.md)
4. 全站总览首页 (docs/index.md)
5. 双维度 + 年份导航配置文件 (mkdocs.yml)
"""

import html
import json
import re
from pathlib import Path
from collections import defaultdict

from init_clean_recsys_repo import VENUES, CATEGORIES, SUPPORTED_YEARS

REPO_SLUG = "RecSys-Paper-Notes"

CAT_ICONS = {
    "collaborative_filtering": "🤝",
    "graph_rec": "🕸️",
    "self_supervised_denoising": "🔄",
    "sequential_rec": "⏱️",
    "generative_rec": "✨",
    "llm_rec": "🧠",
    "agent_conversational_rec": "🗣️",
    "multimodal_rec": "🧩",
    "social_group_rec": "👥",
    "cold_start_rec": "❄️",
    "cross_domain_multi_scenario": "🌐",
    "poi_spatiotemporal_rec": "📍",
    "ctr_feature_interaction": "🎯",
    "multi_task_reranking": "⚖️",
    "rl_bandit_rec": "🎮",
    "debias_fairness_industrial": "🛡️",
}

CATEGORY_ABBR = {
    "collaborative_filtering": "CF",
    "graph_rec": "Graph",
    "self_supervised_denoising": "SSL",
    "sequential_rec": "Seq",
    "generative_rec": "GenRec",
    "llm_rec": "LLM4Rec",
    "agent_conversational_rec": "CRS",
    "multimodal_rec": "MM",
    "social_group_rec": "Social",
    "cold_start_rec": "Cold",
    "cross_domain_multi_scenario": "Cross",
    "poi_spatiotemporal_rec": "POI",
    "ctr_feature_interaction": "CTR",
    "multi_task_reranking": "MTL",
    "rl_bandit_rec": "RL",
    "debias_fairness_industrial": "Debias",
}

TOPIC_TAG_SYNONYMS = {
    "collaborative_filtering": {"协同过滤", "双塔召回", "双塔模型", "矩阵分解"},
    "graph_rec": {"图神经网络", "知识图谱", "知识图谱推荐", "图推荐"},
    "self_supervised_denoising": {"自监督学习", "自监督对比学习", "对比学习", "去噪", "交互去噪"},
    "sequential_rec": {"序列推荐", "序列建模", "会话推荐", "长序列建模"},
    "generative_rec": {
        "生成式端到端",
        "端到端生成式",
        "生成式推荐",
        "语义ID",
        "语义 ID",
        "Semantic ID",
        "语义ID分词",
        "生成式检索",
        "扩散推荐",
        "扩散模型",
        "Flow Matching",
        "RQ-VAE",
        "残差量化",
        "向量量化",
        "向量量化码本",
        "生成式重排",
    },
    "llm_rec": {"大语言模型推荐", "大模型推荐", "LLM推荐", "LLM4Rec", "LLM Agent", "LLM-as-a-Judge"},
    "agent_conversational_rec": {"对话推荐", "智能体", "LLM Agent", "用户模拟", "智能体仿真"},
    "multimodal_rec": {"多模态推荐", "多模态", "MM"},
    "social_group_rec": {"社交推荐", "群组推荐", "社会选择聚合"},
    "cold_start_rec": {"冷启动", "长尾推荐", "长尾", "元学习", "少样本"},
    "cross_domain_multi_scenario": {"跨域推荐", "多场景推荐", "跨域序列推荐", "负迁移抑制"},
    "poi_spatiotemporal_rec": {"时空轨迹与POI推荐", "本地生活推荐", "POI推荐", "时空推荐"},
    "ctr_feature_interaction": {"CTR预估与特征交叉", "CTR预估", "CVR预估", "特征交叉", "自动出价"},
    "multi_task_reranking": {"多任务学习", "列表级重排", "重排", "生成式重排"},
    "rl_bandit_rec": {"强化学习", "Bandit", "Contextual Bandit", "离线强化学习", "自动出价"},
    "debias_fairness_industrial": {
        "去偏公平与工业架构",
        "去偏公平与可信推荐",
        "去偏公平",
        "可信推荐",
        "工业架构",
        "流行度去偏",
    },
}

PORTAL_STYLE = """<style>
.recsys-hero {
  text-align: center;
  padding: 2.6rem 1.5rem 2.2rem;
  margin-bottom: 2rem;
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(63,81,181,0.10) 0%, rgba(33,150,243,0.06) 50%, rgba(156,39,176,0.08) 100%);
  border: 1px solid rgba(63,81,181,0.16);
}
.recsys-hero h1 {
  font-size: 2.2rem;
  font-weight: 800;
  margin: 0 0 0.6rem 0;
  letter-spacing: -0.02em;
}
.recsys-hero p {
  font-size: 1.05rem;
  opacity: 0.82;
  max-width: 740px;
  margin: 0 auto 1.4rem;
}
.recsys-stats {
  display: flex;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.recsys-stat-pill {
  padding: 0.45rem 1.1rem;
  border-radius: 999px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.2);
  font-size: 0.9rem;
  font-weight: 600;
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.recsys-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
  margin: 1.2rem 0 2.2rem;
}
.recsys-card {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 1.15rem 1.25rem;
  border-radius: 12px;
  background: var(--md-default-bg-color);
  border: 1px solid rgba(128,128,128,0.18);
  text-decoration: none !important;
  color: var(--md-default-fg-color) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.recsys-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 8px 24px rgba(0,0,0,0.08);
  border-color: var(--md-primary-fg-color);
}
.recsys-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.45rem;
}
.recsys-card-title {
  font-size: 1.02rem;
  font-weight: 700;
}
.recsys-badge {
  font-size: 0.75rem;
  padding: 0.15rem 0.55rem;
  border-radius: 999px;
  background: rgba(63,81,181,0.12);
  color: var(--md-primary-fg-color);
  font-weight: 600;
  white-space: nowrap;
}
.recsys-card-desc {
  font-size: 0.84rem;
  opacity: 0.75;
  line-height: 1.45;
}
</style>"""


def _field(frontmatter: str, name: str) -> str:
    m = re.search(rf'^{name}:\s*["\']?(.*?)["\']?\s*$', frontmatter, re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_frontmatter(md_path: Path) -> dict:
    """解析单篇论文 Markdown 笔记顶部的 YAML Frontmatter 元数据。"""
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    meta = {
        "title": md_path.stem.replace("_", " ").title(),
        "description": "",
        "date": "",
        "venue": "",
        "year": "",
        "paper_id": "",
        "arxiv_id": "",
        "paper_url": "",
        "code_url": "",
        "categories": [],
        "tags": [],
        "path": md_path,
    }
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        for line in text.splitlines():
            if line.startswith("# "):
                meta["title"] = line[2:].strip()
                break
        return meta

    fm = m.group(1)
    meta["title"] = _field(fm, "title").replace("[Paper Note] ", "").strip()
    meta["description"] = _field(fm, "description")
    meta["date"] = _field(fm, "date")
    meta["venue"] = _field(fm, "venue")
    meta["year"] = _field(fm, "year")
    meta["paper_id"] = _field(fm, "paper_id")
    meta["arxiv_id"] = _field(fm, "arxiv_id")
    meta["paper_url"] = _field(fm, "paper_url")
    meta["code_url"] = _field(fm, "code_url")

    cats = re.findall(r"^\s*-\s*([a-z0-9_]+)\s*$", fm, re.MULTILINE)
    inline = re.search(r"^categories:\s*\[(.*?)\]", fm, re.MULTILINE)
    if inline:
        cats += [c.strip().strip("'\"") for c in inline.group(1).split(",")]
    valid_cats = []
    for c in cats:
        if c in CATEGORIES and c not in valid_cats:
            valid_cats.append(c)
    meta["categories"] = valid_cats

    tags = []
    in_tags = False
    for line in fm.splitlines():
        if re.match(r"^tags:\s*", line):
            in_tags = True
            inline_tags = re.match(r"^tags:\s*\[(.*?)\]", line)
            if inline_tags:
                tags.extend(t.strip().strip("'\"") for t in inline_tags.group(1).split(","))
                in_tags = False
            continue
        if not in_tags:
            continue
        tag_match = re.match(r"^\s*-\s*(.+?)\s*$", line)
        if tag_match:
            tags.append(tag_match.group(1).strip("'\""))
        elif line.strip():
            in_tags = False
    meta["tags"] = [t for t in tags if t]
    return meta


def md_cell(value: str) -> str:
    """转义 Markdown 表格单元格中的竖线与换行。"""
    return (value or "").replace("|", "\\|").replace("\n", " ").strip()


def format_resource_links(item: dict) -> str:
    """生成论文原文与开源代码的快捷点击链接标签。"""
    links = []
    if item.get("paper_url"):
        links.append(f"[📄 原文]({item['paper_url']})")
    if item.get("code_url"):
        links.append(f"[💻 代码]({item['code_url']})")
    return " · ".join(links) if links else "-"


def _esc(value: str) -> str:
    return html.escape(value or "", quote=True)


def note_url(rel_from_docs: str) -> str:
    """把 venues/.../note.md 转成 use_directory_urls 下的 note/ 目录链接。"""
    return rel_from_docs[:-3] + "/" if rel_from_docs.endswith(".md") else rel_from_docs.rstrip("/") + "/"


def render_paper_tags(item: dict) -> str:
    """渲染论文标题旁的出处、类别与年份标签。"""
    tags = []
    venue_code = item.get("venue", "")
    venue = VENUES.get(venue_code, {})
    venue_name = venue.get("name", venue_code)
    if venue_name:
        tags.append(
            f'<span class="paper-tag paper-tag--venue" '
            f'title="发表出处：{_esc(venue.get("full_name", venue_name))}">{_esc(venue_name)}</span>'
        )

    for category in item.get("categories", []):
        cinfo = CATEGORIES.get(category)
        if not cinfo:
            continue
        abbr = CATEGORY_ABBR.get(category, category)
        tags.append(
            f'<span class="paper-tag paper-tag--category" '
            f'title="{_esc(cinfo["title"])}">{_esc(abbr)}</span>'
        )

    year = item.get("year", "")
    if year:
        tags.append(
            f'<span class="paper-tag paper-tag--year" title="发表年份">{_esc(year)}</span>'
        )

    return f'<span class="paper-tags">{"".join(tags)}</span>' if tags else ""


PIPELINE_TAG_TERMS = {"召回", "粗排", "精排", "重排", "排序"}


def is_pipeline_tag(tag: str) -> bool:
    """判断 tag 是否只由召回/精排/重排等推荐链路阶段词组成。"""
    parts = [part for part in re.split(r"[\s/+、,，&和与及\-]+", tag) if part]
    return bool(parts) and all(part in PIPELINE_TAG_TERMS for part in parts)


def render_hot_tags(items: list, excluded: set, limit: int = 8) -> str:
    """根据论文 frontmatter tags 生成高频主题摘要。"""
    excluded_norm = {re.sub(r"\s+", "", value).lower() for value in excluded if value}
    counts = defaultdict(int)
    for item in items:
        for tag in item.get("tags", []):
            tag = (tag or "").strip()
            if (
                tag
                and not is_pipeline_tag(tag)
                and re.sub(r"\s+", "", tag).lower() not in excluded_norm
            ):
                counts[tag] += 1
    hot = sorted(
        ((tag, count) for tag, count in counts.items() if count >= 2),
        key=lambda pair: (-pair[1], pair[0]),
    )[:limit]
    if not hot:
        return ""
    return "🔥 **高频主题：** " + " · ".join(f"{tag} ×{count}" for tag, count in hot)


def _is_facet_tag(tag: str, excluded_norm: set) -> bool:
    """Return whether a frontmatter tag is useful as a list-page facet."""
    tag = (tag or "").strip()
    if not tag:
        return False
    if re.search(r"(?:19|20)\d{2}", tag):
        return False
    return _normalize_term(tag) not in excluded_norm


def render_tag_filter(items: list, excluded: set) -> str:
    """Render the client-side AND/OR tag filter for a generated paper list."""
    excluded_norm = {_normalize_term(value) for value in excluded if value}
    counts = defaultdict(int)
    for item in items:
        for tag in item.get("tags", []):
            tag = (tag or "").strip()
            if _is_facet_tag(tag, excluded_norm):
                counts[tag] += 1
    if not counts:
        return ""

    chips = []
    for tag, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0].lower())):
        chips.append(
            f'<button type="button" class="paper-tag-filter__chip" '
            f'data-tag="{_esc(tag)}" aria-pressed="false">'
            f'{_esc(tag)}<span class="paper-tag-filter__count">{count}</span></button>'
        )

    return "\n".join(
        [
            '<div class="paper-tag-filter" data-paper-tag-filter>',
            '  <div class="paper-tag-filter__head">',
            '    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>',
            '    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>',
            '    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">',
            '      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>',
            '      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>',
            "    </span>",
            '    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>',
            "  </div>",
            '  <div class="paper-tag-filter__chips">',
            "\n".join(chips),
            "  </div>",
            '  <div class="paper-tag-filter__status" data-tag-status></div>',
            "</div>",
        ]
    )


def render_filter_layout(filter_html: str, content_html: str) -> str:
    return "\n".join(
        [
            '<div class="pn-filter-layout">',
            '  <aside class="pn-filter-rail">',
            filter_html,
            "  </aside>",
            '  <div class="pn-filter-list">',
            content_html,
            "  </div>",
            "</div>",
        ]
    )


def render_paper_dl(items: list, href_for) -> str:
    """按原版 Paper-Notes 的 <dl><dt><dd> 结构渲染论文列表。"""
    lines = ['<dl data-paper-list>']
    for item in items:
        title = _esc(item.get("title", ""))
        desc = re.sub(r"^(?:\[[^\]]+\]\s*)+", "", item.get("description") or "").strip()
        desc = _esc(desc or "点击查看核心思想与实验协议精读")
        href = _esc(href_for(item))
        tags_json = _esc(
            json.dumps(item.get("tags", []), ensure_ascii=False, separators=(",", ":"))
        )
        lines.append(
            f'<dt data-paper-tags="{tags_json}"><strong><a href="{href}">{title}</a></strong>{render_paper_tags(item)}</dt>'
        )
        lines.append("<dd>")
        lines.append(f"<p>{desc}</p>")
        lines.append("</dd>")
    lines.append("</dl>")
    return "\n".join(lines)


def parse_note_path(md_file: Path, docs_dir: Path):
    """解析 venues/<VENUE>/<YEAR>/<category>/<slug>.md 路径。"""
    try:
        rel = md_file.relative_to(docs_dir)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 5 or parts[0] != "venues":
        return None
    venue, year, category = parts[1], parts[2], parts[3]
    if not re.fullmatch(r"\d{4}", year):
        return None
    if venue not in VENUES or category not in CATEGORIES:
        return None
    return venue, year, category


def collect_all_notes(docs_dir: Path):
    """遍历 docs/venues/<VENUE>/<YEAR>/<category>/ 收集所有论文解读笔记。"""
    venues_dir = docs_dir / "venues"
    notes = []
    if not venues_dir.exists():
        return notes

    for md_file in sorted(venues_dir.rglob("*.md")):
        if md_file.name == "index.md":
            continue
        parsed = parse_note_path(md_file, docs_dir)
        if not parsed:
            print(f"[Warn] 跳过不符合 venues/<VENUE>/<YEAR>/<category>/ 结构的文件: {md_file}")
            continue
        venue, year, category = parsed
        meta = parse_frontmatter(md_file)
        meta["venue"] = venue
        meta["year"] = year
        meta["category"] = category
        if category not in meta["categories"]:
            meta["categories"].insert(0, category)
        meta["rel_from_docs"] = md_file.relative_to(docs_dir).as_posix()
        notes.append(meta)
    return notes


def discover_years(docs_dir: Path, notes: list) -> list:
    """从配置、笔记与目录中发现所有年份，按倒序返回。"""
    years = set(SUPPORTED_YEARS)
    for n in notes:
        if n.get("year"):
            years.add(n["year"])
    venues_dir = docs_dir / "venues"
    if venues_dir.exists():
        for venue_dir in venues_dir.iterdir():
            if not venue_dir.is_dir():
                continue
            for child in venue_dir.iterdir():
                if child.is_dir() and re.fullmatch(r"\d{4}", child.name):
                    years.add(child.name)
    return sorted(years, reverse=True)


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", "", value or "").lower()


def _venue_tag_terms(notes: list) -> set:
    years = {str(n.get("year", "")) for n in notes if n.get("year")}
    terms = set(years)
    for code, vinfo in VENUES.items():
        name = vinfo.get("name", "")
        terms.update(value for value in (code, name, vinfo.get("full_name", "")) if value)
        parts = [
            part.strip()
            for part in re.split(r"[/、,，&]+", re.sub(r"\([^)]*\)", "", name))
            if part.strip()
        ]
        terms.update(parts)
        if len(parts) > 1:
            terms.update(sep.join(parts) for sep in ("/", " & ", " ", "、", ", "))
    terms.update(f"{term} {year}" for term in list(terms) for year in years)
    suffixes = ("Findings", "Demo", "Short", "Industry", "Track")
    terms.update(f"{term} {suffix}" for term in list(terms) for suffix in suffixes)
    terms.update(
        f"{term} {year} {suffix}"
        for term in list(terms)
        for year in years
        for suffix in suffixes
    )
    return terms


def _topic_hot_tag_terms(items: list, cat_slug: str, cinfo: dict, notes: list) -> set:
    title_norm = _normalize_term(cinfo["title"])
    abbr = CATEGORY_ABBR.get(cat_slug, cat_slug)
    abbr_norm = _normalize_term(abbr)
    excluded = {
        cinfo["title"],
        cinfo["en_title"],
        abbr,
        *TOPIC_TAG_SYNONYMS.get(cat_slug, set()),
        *(info["title"] for info in CATEGORIES.values()),
        *(CATEGORY_ABBR.get(slug, slug) for slug in CATEGORIES),
    }
    for item in items:
        for tag in item.get("tags", []):
            tag = (tag or "").strip()
            tag_norm = _normalize_term(tag)
            if tag_norm and (tag_norm in title_norm or tag_norm == abbr_norm):
                excluded.add(tag)
    excluded.update(_venue_tag_terms(notes))
    return excluded


def build_topic_pages(docs_dir: Path, notes: list):
    """生成 docs/topics/ 下的 16 大推荐系统细分方向跨会议聚合页。"""
    topics_dir = docs_dir / "topics"
    topics_dir.mkdir(parents=True, exist_ok=True)

    cat_map = defaultdict(list)
    for n in notes:
        for c in n["categories"]:
            if c in CATEGORIES:
                cat_map[c].append(n)

    cards_html = ['<div class="recsys-grid">']
    for cat_slug, cinfo in CATEGORIES.items():
        items = cat_map.get(cat_slug, [])
        icon = CAT_ICONS.get(cat_slug, "📌")
        cards_html.append(
            f'  <a class="recsys-card" href="{cat_slug}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {cinfo["title"]}</span>'
            f'      <span class="recsys-badge">{len(items)} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{cinfo["desc"]}</div>'
            f'  </a>'
        )
    cards_html.append("</div>")

    topic_index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        f"# 🔬 按推荐系统细分类别浏览 ({len(CATEGORIES)} 大正交方向)",
        "",
        "按**推荐系统研究子方向**跨会议、跨年份聚合所有论文解读笔记。每篇均附带精读笔记、**论文原文直达链接**与**开源代码仓库链接（如有）**。",
        "",
        "\n".join(cards_html),
    ]

    for cat_slug, cinfo in CATEGORIES.items():
        items = cat_map.get(cat_slug, [])
        icon = CAT_ICONS.get(cat_slug, "📌")
        page_lines = [
            "---",
            "hide:",
            "  - toc",
            "---",
            "",
            f"# {icon} {cinfo['title']} ({cinfo['en_title']})",
            "",
            f"> **研究范畴**：{cinfo['desc']}",
            "",
            f"当前方向共收录 **{len(items)}** 篇顶会/顶刊论文解读：",
            "",
        ]
        tag_filter = ""
        if items:
            hot = render_hot_tags(items, _topic_hot_tag_terms(items, cat_slug, cinfo, notes))
            if hot:
                page_lines.extend([hot.replace("高频主题", "高频交叉主题"), ""])
            tag_filter = render_tag_filter(items, _venue_tag_terms(notes))
        if not items:
            page_lines.append("*暂无论文笔记。*")
        else:
            sorted_items = sorted(items, key=lambda x: (x["venue"], x["year"], x["title"].lower()))
            paper_list = render_paper_dl(
                sorted_items, lambda item: "../../" + note_url(item["rel_from_docs"])
            )
            if tag_filter:
                page_lines.append(render_filter_layout(tag_filter, paper_list))
            else:
                page_lines.append(paper_list)

        (topics_dir / f"{cat_slug}.md").write_text("\n".join(page_lines) + "\n", encoding="utf-8")

    (topics_dir / "index.md").write_text("\n".join(topic_index_lines) + "\n", encoding="utf-8")


def build_year_pages(docs_dir: Path, notes: list, years: list):
    """生成 docs/years/<year>.md 跨会刊年份聚合页。"""
    years_dir = docs_dir / "years"
    years_dir.mkdir(parents=True, exist_ok=True)

    year_map = defaultdict(lambda: defaultdict(int))
    for n in notes:
        year_map[n["year"]][n["venue"]] += 1

    cards_html = ['<div class="recsys-grid">']
    for year in years:
        total = sum(year_map[year].values())
        cards_html.append(
            f'  <a class="recsys-card" href="{year}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
            f'      <span class="recsys-badge">{total} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">按会议/期刊浏览 {year} 年收录的推荐系统论文解读</div>'
            f'  </a>'
        )
    cards_html.append("</div>")

    index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        "# 🗓️ 按会议年份浏览",
        "",
        "按论文发表年份跨会刊聚合。每个年份页面可继续进入对应会议/期刊的年度目录。",
        "",
        "\n".join(cards_html),
    ]
    (years_dir / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    for year in years:
        total = sum(year_map[year].values())
        venue_cards = ['<div class="recsys-grid">']
        for venue_code, vinfo in VENUES.items():
            count = year_map[year].get(venue_code, 0)
            icon = "🏛️" if vinfo["type"] == "conference" else "📖"
            venue_cards.append(
                f'  <a class="recsys-card" href="../../venues/{venue_code}/{year}/">'
                f'    <div class="recsys-card-head">'
                f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
                f'      <span class="recsys-badge">{count} 篇</span>'
                f'    </div>'
                f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
                f'  </a>'
            )
        venue_cards.append("</div>")
        lines = [
            "---",
            "hide:",
            "  - toc",
            "---",
            "",
            PORTAL_STYLE,
            "",
            f"# 🗓️ {year} 年推荐系统论文解读",
            "",
            f"当前年份共收录 **{total}** 篇论文解读。",
            "",
            "\n".join(venue_cards),
        ]
        (years_dir / f"{year}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_venue_pages(docs_dir: Path, notes: list, years: list):
    """生成 venues/index.md、venues/<VENUE>/index.md、venues/<VENUE>/<YEAR>/index.md 与分类页。"""
    venues_dir = docs_dir / "venues"
    venues_dir.mkdir(parents=True, exist_ok=True)

    venue_map = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for n in notes:
        venue_map[n["venue"]][n["year"]][n["category"]].append(n)

    venue_cards = ['<div class="recsys-grid">']
    for venue_code, vinfo in VENUES.items():
        total_v = sum(
            len(items)
            for year_items in venue_map[venue_code].values()
            for items in year_items.values()
        )
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        venue_cards.append(
            f'  <a class="recsys-card" href="{venue_code}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
            f'      <span class="recsys-badge">{total_v} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
            f'  </a>'
        )
    venue_cards.append("</div>")

    venue_index_lines = [
        "---",
        "hide:",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        "# 🏛️ 按顶会与顶刊浏览 (22 个推荐系统核心会议与期刊)",
        "",
        "\n".join(venue_cards),
    ]
    (venues_dir / "index.md").write_text("\n".join(venue_index_lines) + "\n", encoding="utf-8")

    for venue_code, vinfo in VENUES.items():
        venue_dir = venues_dir / venue_code
        venue_dir.mkdir(parents=True, exist_ok=True)
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"

        year_cards = ['<div class="recsys-grid">']
        for year in years:
            total_y = sum(len(items) for items in venue_map[venue_code][year].values())
            year_cards.append(
                f'  <a class="recsys-card" href="{year}/">'
                f'    <div class="recsys-card-head">'
                f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
                f'      <span class="recsys-badge">{total_y} 篇</span>'
                f'    </div>'
                f'    <div class="recsys-card-desc">{vinfo["name"]} {year} 年收录的推荐系统论文解读</div>'
                f'  </a>'
            )
        year_cards.append("</div>")

        venue_lines = [
            "---",
            "hide:",
            "  - toc",
            "---",
            "",
            PORTAL_STYLE,
            "",
            f"# {icon} {vinfo['name']} 推荐系统论文解读汇总",
            "",
            f"- **英文全称**: {vinfo['full_name']}",
            f"- **级别定位**: {vinfo['tier']}",
            "",
            "## 按年份浏览",
            "",
            "\n".join(year_cards),
        ]
        (venue_dir / "index.md").write_text("\n".join(venue_lines) + "\n", encoding="utf-8")

        for year in years:
            year_dir = venue_dir / year
            year_dir.mkdir(parents=True, exist_ok=True)
            total_y = sum(len(items) for items in venue_map[venue_code][year].values())
            active_cats = [
                (cat_slug, cinfo, venue_map[venue_code][year].get(cat_slug, []))
                for cat_slug, cinfo in CATEGORIES.items()
                if venue_map[venue_code][year].get(cat_slug, [])
            ]

            year_items = [
                item
                for items in venue_map[venue_code][year].values()
                for item in items
            ]
            hot = render_hot_tags(
                year_items,
                {
                    vinfo["name"],
                    f"{vinfo['name']} {year}",
                    year,
                    *(cinfo["title"] for cinfo in CATEGORIES.values()),
                },
            )
            year_lines = [
                "---",
                "hide:",
                "  - toc",
                "---",
                "",
                f"# {icon} {vinfo['name']} {year} 推荐系统论文解读",
                "",
                f"{total_y} 篇 {vinfo['name']} {year} 论文解读，覆盖 {len(active_cats)} 个推荐系统细分方向。"
                "每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。",
                "",
            ]
            if hot:
                year_lines.extend([hot, ""])
            tag_filter = render_tag_filter(year_items, _venue_tag_terms(notes))
            section_lines = ['<div class="conf-index">']
            for cat_slug, cinfo, c_items in active_cats:
                c_icon = CAT_ICONS.get(cat_slug, "📌")
                sorted_items = sorted(c_items, key=lambda x: x["title"].lower())
                section_lines.append('<section class="paper-section" data-paper-section>')
                section_lines.append("<hr />")
                section_lines.append(f'<h2 id="{cat_slug}">{c_icon} {cinfo["title"]} ({len(c_items)})</h2>')
                section_lines.append(
                    render_paper_dl(
                        sorted_items[:10],
                        lambda item, cs=cat_slug: f"{cs}/{item['path'].stem}/",
                    )
                )
                if len(c_items) > 10:
                    section_lines.append(
                        f'<p><a href="{cat_slug}/">查看全部{len(c_items)}篇「{cinfo["title"]}」论文 →</a></p>'
                    )
                section_lines.append("</section>")
            section_lines.append("</div>")
            sections_html = "\n".join(section_lines)
            if tag_filter:
                year_lines.append(render_filter_layout(tag_filter, sections_html))
            else:
                year_lines.append(sections_html)
            (year_dir / "index.md").write_text("\n".join(year_lines) + "\n", encoding="utf-8")

            for cat_slug, cinfo in CATEGORIES.items():
                c_dir = year_dir / cat_slug
                c_items = venue_map[venue_code][year].get(cat_slug, [])
                c_icon = CAT_ICONS.get(cat_slug, "📌")
                c_lines = [
                    "---",
                    "hide:",
                    "  - toc",
                    "---",
                    "",
                    f"# {c_icon} {cinfo['title']}",
                    "",
                    f"**{vinfo['name']} {year}** · **{len(c_items)}** 篇论文解读",
                    "",
                    f"> **方向说明**：{cinfo['desc']}",
                    "",
                ]
                if not c_items:
                    c_lines.append("*当前会议在该年份、该细分方向下暂无论文笔记。*")
                else:
                    cross_links = []
                    for other_year, other_cats in venue_map[venue_code].items():
                        if other_year == year:
                            continue
                        count = len(other_cats.get(cat_slug, []))
                        if count:
                            cross_links.append(
                                f"[{vinfo['name']} {other_year} ({count})](../../{other_year}/{cat_slug}/)"
                            )
                    for other_venue, other_years in venue_map.items():
                        if other_venue == venue_code:
                            continue
                        for other_year, other_cats in other_years.items():
                            count = len(other_cats.get(cat_slug, []))
                            if count:
                                cross_links.append(
                                    f"[{VENUES[other_venue]['name']} {other_year} ({count})]"
                                    f"(../../../{other_venue}/{other_year}/{cat_slug}/)"
                                )
                    if cross_links:
                        c_lines.append("📌 **同领域跨会议浏览：** " + " · ".join(cross_links))
                        c_lines.append("")

                    hot = render_hot_tags(
                        c_items,
                        {vinfo["name"], f"{vinfo['name']} {year}", year, cinfo["title"]},
                    )
                    if hot:
                        c_lines.append(hot)
                        c_lines.append("")

                    tag_filter = render_tag_filter(c_items, _venue_tag_terms(notes))
                    sorted_items = sorted(c_items, key=lambda x: x["title"].lower())
                    paper_list = render_paper_dl(sorted_items, lambda item: item["path"].stem + "/")
                    if tag_filter:
                        c_lines.append(render_filter_layout(tag_filter, paper_list))
                    else:
                        c_lines.append(paper_list)
                (c_dir / "index.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")


SEARCH_PAGE = """---
hide:
  - toc
---

# 🔎 搜索论文笔记

按标题、关键词与标签检索全站论文。标签筛选默认使用 **AND**（同时包含），可切换为 **OR**（任一匹配）。

<div id="pn-search-root">
  <section class="pn-search-main">
    <div id="pn-search-box">
      <input id="pn-search-input" type="search" autocomplete="off" spellcheck="false" />
    </div>
    <div id="pn-search-status"></div>
    <ol id="pn-search-results"></ol>
  </section>
  <aside class="pn-search-filters">
    <fieldset id="pn-search-conference-filter" disabled>
      <legend>会议筛选</legend>
      <div id="pn-search-conferences"></div>
    </fieldset>
    <fieldset id="pn-search-tag-filter" disabled>
      <legend>标签筛选（默认 AND）</legend>
      <div id="pn-search-tags"></div>
      <div class="pn-tag-actions">
        <span class="pn-tag-mode" id="pn-search-tag-mode" role="group" aria-label="标签匹配模式">
          <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
          <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
        </span>
        <button type="button" id="pn-search-tag-clear" disabled>清除标签</button>
      </div>
    </fieldset>
  </aside>
</div>
"""


def build_search_page(docs_dir: Path):
    """Generate the full-page search UI consumed by javascripts/search-page.js."""
    (docs_dir / "search.md").write_text(SEARCH_PAGE, encoding="utf-8")


def build_home_and_mkdocs_yml(repo_root: Path, notes: list, years: list, github_user: str = "jbdcia321vw"):
    """生成全宽门户卡片式 docs/index.md 与按会刊/年份分组的 mkdocs.yml。"""
    docs_dir = repo_root / "docs"
    cat_counts = defaultdict(int)
    venue_counts = defaultdict(int)
    year_counts = defaultdict(int)
    for n in notes:
        for c in n["categories"]:
            cat_counts[c] += 1
        venue_counts[n["venue"]] += 1
        year_counts[n["year"]] += 1

    topic_cards = ['<div class="recsys-grid">']
    for cat_slug, cinfo in CATEGORIES.items():
        icon = CAT_ICONS.get(cat_slug, "📌")
        topic_cards.append(
            f'  <a class="recsys-card" href="topics/{cat_slug}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {cinfo["title"]}</span>'
            f'      <span class="recsys-badge">{cat_counts[cat_slug]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{cinfo["desc"]}</div>'
            f'  </a>'
        )
    topic_cards.append("</div>")

    year_cards = ['<div class="recsys-grid">']
    for year in years:
        year_cards.append(
            f'  <a class="recsys-card" href="years/{year}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">🗓️ {year} 年</span>'
            f'      <span class="recsys-badge">{year_counts[year]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">按会议/期刊浏览 {year} 年收录的论文解读</div>'
            f'  </a>'
        )
    year_cards.append("</div>")

    venue_cards = ['<div class="recsys-grid">']
    for venue_code, vinfo in VENUES.items():
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        venue_cards.append(
            f'  <a class="recsys-card" href="venues/{venue_code}/">'
            f'    <div class="recsys-card-head">'
            f'      <span class="recsys-card-title">{icon} {vinfo["name"]}</span>'
            f'      <span class="recsys-badge">{venue_counts[venue_code]} 篇</span>'
            f'    </div>'
            f'    <div class="recsys-card-desc">{vinfo["tier"]} · {vinfo["full_name"]}</div>'
            f'  </a>'
        )
    venue_cards.append("</div>")

    home_lines = [
        "---",
        "hide:",
        "  - navigation",
        "  - toc",
        "---",
        "",
        PORTAL_STYLE,
        "",
        '<div class="recsys-hero">',
        '  <h1>📚 RecSys Paper Notes</h1>',
        f'  <p>5 分钟读懂一篇推荐系统与信息检索顶会/顶刊论文 · 支持「{len(CATEGORIES)} 大正交细分方向 × {len(VENUES)} 个核心会刊 × 年份」三维检索</p>',
        '  <div class="recsys-stats">',
        f'    <span class="recsys-stat-pill">📖 已收录 {len(notes)} 篇精读笔记</span>',
        f'    <span class="recsys-stat-pill">🔬 {len(CATEGORIES)} 大推荐系统细分方向</span>',
        f'    <span class="recsys-stat-pill">🏛️ {len(VENUES)} 个顶会与顶级期刊</span>',
        f'    <span class="recsys-stat-pill">🗓️ {len(years)} 个年份</span>',
        '  </div>',
        '</div>',
        "",
        f"## 🔬 按推荐系统细分类别浏览 ({len(CATEGORIES)} 大正交方向)",
        "",
        "\n".join(topic_cards),
        "",
        "## 🗓️ 按会议年份浏览",
        "",
        "\n".join(year_cards),
        "",
        f"## 🏛️ 按顶会与顶刊浏览 ({len(VENUES)} 个会刊)",
        "",
        "\n".join(venue_cards),
    ]
    (docs_dir / "index.md").write_text("\n".join(home_lines) + "\n", encoding="utf-8")

    has_overrides = (repo_root / "overrides").exists()
    custom_dir_line = "  custom_dir: overrides\n" if has_overrides else ""
    logo_line = "  logo: assets/papernotes-logo-horizontal.svg\n" if (docs_dir / "assets" / "papernotes-logo-horizontal.svg").exists() else ""
    favicon_line = "  favicon: assets/favicon.svg\n" if (docs_dir / "assets" / "favicon.svg").exists() else ""

    extra_css_block = ""
    if (docs_dir / "stylesheets" / "extra.css").exists():
        extra_css_block = "extra_css:\n  - stylesheets/extra.css\n"

    js_list = []
    for js_rel in [
        "javascripts/mathjax.js",
        "javascripts/search-page.js",
        "javascripts/tag-filter.js",
        "javascripts/title-link.js",
    ]:
        if (docs_dir / js_rel).exists():
            js_list.append(f"  - {js_rel}")
    extra_js_block = "extra_javascript:\n" + "\n".join(js_list) + "\n"

    hooks_list = []
    for h_rel in [
        "hooks/set_build_date.py",
        "hooks/tags_order.py",
        "hooks/slim_search_index.py",
    ]:
        if (repo_root / h_rel).exists():
            hooks_list.append(f"  - {h_rel}")
    hooks_block = ("hooks:\n" + "\n".join(hooks_list) + "\n") if hooks_list else ""

    topic_nav_items = ["      - 全部推荐细分方向总览: topics/index.md"]
    for cat_slug, cinfo in CATEGORIES.items():
        icon = CAT_ICONS.get(cat_slug, "📌")
        topic_nav_items.append(f'      - "{icon} {cinfo["title"]}": topics/{cat_slug}.md')

    year_nav_items = ["      - 全部年份总览: years/index.md"]
    for year in years:
        year_nav_items.append(f'      - "{year} 年": years/{year}.md')

    conf_nav_items = ["      - 全部顶会与顶刊总览: venues/index.md"]
    for venue_code, vinfo in VENUES.items():
        icon = "🏛️" if vinfo["type"] == "conference" else "📖"
        conf_nav_items.append(f'      - "{icon} {vinfo["name"]}":')
        conf_nav_items.append(f"          - 会刊总览: venues/{venue_code}/index.md")
        for year in years:
            conf_nav_items.append(f'          - "{year}": venues/{venue_code}/{year}/index.md')

    mkdocs_content = f"""site_name: RecSys Paper Notes · 推荐系统论文精读库
site_description: RecSys Paper Notes 推荐系统与信息检索顶会顶刊论文精读库，覆盖 {len(VENUES)} 个会刊、{len(CATEGORIES)} 大方向，按会刊/年份/主题分类，每篇 5 分钟读懂核心思想
site_url: https://{github_user}.github.io/{REPO_SLUG}/
repo_url: https://github.com/{github_user}/{REPO_SLUG}
repo_name: {github_user}/{REPO_SLUG}

theme:
  name: material
{custom_dir_line}{logo_line}{favicon_line}  language: zh
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - navigation.indexes
    - search.suggest
    - search.highlight
    - content.code.copy
  palette:
    - scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: 切换至浅色模式
    - scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: 切换至深色模式

plugins:
  - search:
      lang:
        - zh
        - en
  - tags

{hooks_block}{extra_css_block}{extra_js_block}
markdown_extensions:
  - admonition
  - tables
  - attr_list
  - md_in_html
  - pymdownx.details
  - pymdownx.arithmatex:
      generic: true
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format

nav:
  - 🏠 首页总览: index.md
  - 🔎 全站搜索: search.md
  - 🔬 按推荐细分类别 ({len(CATEGORIES)}大类):
{chr(10).join(topic_nav_items)}
  - 🗓️ 按会议年份浏览:
{chr(10).join(year_nav_items)}
  - 🏛️ 按顶会与顶刊 ({len(VENUES)}个会刊):
{chr(10).join(conf_nav_items)}
  - ℹ️ 关于本站: about.md
"""
    (repo_root / "mkdocs.yml").write_text(mkdocs_content, encoding="utf-8")
    print(f"[Build] 已生成卡片式首页、分类/年份/会刊索引与 mkdocs.yml（共索引 {len(notes)} 篇笔记，{len(years)} 个年份）。")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    docs_dir = root / "docs"
    all_notes = collect_all_notes(docs_dir)
    all_years = discover_years(docs_dir, all_notes)
    build_topic_pages(docs_dir, all_notes)
    build_year_pages(docs_dir, all_notes, all_years)
    build_venue_pages(docs_dir, all_notes, all_years)
    build_search_page(docs_dir)
    build_home_and_mkdocs_yml(root, all_notes, all_years)
