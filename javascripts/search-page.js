/* Full-page search results for paper-notes.
 *
 * Loaded on every page via `extra_javascript`, but only activates on the
 * dedicated /search/ page (detected by the #pn-search-root container). It
 * reuses the slimmed `search/search_index.json` we already ship (title +
 * keywords + tags after hooks/slim_search_index.py), so no lunr.js and no
 * Material search worker are involved — the scoring is a small, predictable,
 * CJK-aware token overlap that we fully control.
 *
 * Why a custom scorer instead of lunr: the corpus fields are short (title +
 * ~5 keywords), so weighted token coverage + a title-substring bonus ranks
 * better and more predictably than BM25's length-normalisation (which the
 * dropdown UI already has to fight with a client-side re-rank).
 */
(function () {
  "use strict";

  var LANG =
    (document.documentElement.lang || "zh").toLowerCase().indexOf("en") === 0
      ? "en"
      : "zh";

  var T = {
    zh: {
      placeholder: "搜索论文笔记…",
      loading: "正在加载搜索索引…",
      empty: "输入关键词或选择标签开始搜索（标题 + 关键词 + 标签）",
      none: "没有找到匹配的笔记。",
      count: function (n, from, to, page, pages) {
        return pages > 1
          ? "找到 " + n + " 条结果（第 " + from + "-" + to + " 条 / 共 " + pages + " 页）"
          : "找到 " + n + " 条结果";
      },
      deep: function (q) {
        return "在 Bing 上全文深搜「" + q + "」";
      },
      failed: "搜索索引加载失败，请稍后重试。",
      prev: "上一页",
      next: "下一页",
      allConferences: "全部会议"
    },
    en: {
      placeholder: "Search paper notes…",
      loading: "Loading search index…",
      empty: "Type a keyword or pick tags to search (title + keywords + tags)",
      none: "No matching notes found.",
      count: function (n, from, to, page, pages) {
        return pages > 1
          ? n + " results (showing " + from + "-" + to + " of page " + page + "/" + pages + ")"
          : n + " results";
      },
      deep: function (q) {
        return "Full-text search \u201C" + q + "\u201D on Bing";
      },
      failed: "Failed to load the search index. Please try again later.",
      prev: "Previous",
      next: "Next",
      allConferences: "All conferences"
    }
  }[LANG];

  var PAGE_SIZE = 50;
  var TAG_LIMIT = 10;
  var MAX_PAGE_LINKS = 7;
  var docsCache = null; // pre-tokenised docs, built once per page load
  var conferenceNames = [];
  var building = false;
  var waiters = [];

  function base() {
    var i = location.pathname.indexOf("/search/");
    return i >= 0 ? location.pathname.slice(0, i + 1) : "/";
  }

  function expand(q) {
    // Reuse the bilingual (中↔英) expansion dictionary from overrides/main.html
    // when available, so the full page matches the dropdown's recall.
    try {
      return typeof window.__pnExpand === "function" ? window.__pnExpand(q) : q;
    } catch (e) {
      return q;
    }
  }

  // CJK-aware tokenizer: Latin tokens (len >= 2) + CJK unigrams & bigrams.
  // Applied identically at index time and query time so the two are consistent.
  function tokenize(s) {
    if (!s) return [];
    s = s.toLowerCase().replace(/\$[^$]*\$/g, " ").replace(/\u200b/g, " ");
    var parts = s.split(/[^a-z0-9\u4e00-\u9fff]+/);
    var out = [];
    for (var i = 0; i < parts.length; i++) {
      var p = parts[i];
      if (!p) continue;
      if (/[\u4e00-\u9fff]/.test(p)) {
        var buf = "";
        for (var j = 0; j < p.length; j++) {
          var ch = p.charAt(j);
          if (ch >= "\u4e00" && ch <= "\u9fff") {
            if (buf) {
              if (buf.length >= 2) out.push(buf);
              buf = "";
            }
            out.push(ch); // unigram
            var nx = p.charAt(j + 1);
            if (nx >= "\u4e00" && nx <= "\u9fff") out.push(ch + nx); // bigram
          } else {
            buf += ch;
          }
        }
        if (buf && buf.length >= 2) out.push(buf);
      } else if (p.length >= 2) {
        out.push(p);
      }
    }
    return out;
  }

  function addTokens(map, text, weight) {
    var toks = tokenize(text);
    for (var i = 0; i < toks.length; i++) {
      var t = toks[i];
      if (!map[t] || map[t] < weight) map[t] = weight;
    }
  }

  function prettyArea(a) {
    return a.replace(/[_-]+/g, " ").replace(/\b\w/g, function (c) {
      return c.toUpperCase();
    });
  }

  function prettyConference(conference) {
    return conference.replace(/(\D)(\d{4})$/, "$1 $2");
  }

  function conferenceParts(conference) {
    var match = /^(.*?)(\d{4})$/.exec(conference);
    return {
      name: match ? match[1] : conference,
      year: match ? parseInt(match[2], 10) : 0
    };
  }

  function siteConferenceOrder() {
    // Homepage cards and primary navigation share the same generated order.
    var links = document.querySelectorAll(".md-sidebar--primary a[href]");
    var siteBase = base();
    var seen = Object.create(null);
    var order = [];
    for (var i = 0; i < links.length; i++) {
      try {
        var pathname = new URL(links[i].href, location.href).pathname;
        if (pathname.indexOf(siteBase) !== 0) continue;
        var conference = pathname.slice(siteBase.length).replace(/\/+$/, "");
        if (!/^[a-z]+\d{4}$/i.test(conference) || seen[conference]) continue;
        seen[conference] = true;
        order.push(conference);
      } catch (e) {}
    }
    return order;
  }

  function conferenceInputs(filter) {
    return filter.querySelectorAll("input[data-conference]");
  }

  function selectedConferences(filter) {
    var inputs = conferenceInputs(filter);
    var selected = [];
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].checked) selected.push(inputs[i].getAttribute("data-conference"));
    }
    return selected;
  }

  function syncAllConferences(filter) {
    var all = filter.querySelector("#pn-search-all-conferences");
    var inputs = conferenceInputs(filter);
    var checked = 0;
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].checked) checked++;
    }
    all.checked = checked === inputs.length;
    all.indeterminate = checked > 0 && checked < inputs.length;
  }

  function applyConferenceSelection(filter, selected) {
    var wanted = null;
    if (selected !== null) {
      wanted = Object.create(null);
      for (var i = 0; i < selected.length; i++) wanted[selected[i]] = true;
    }
    var inputs = conferenceInputs(filter);
    for (var j = 0; j < inputs.length; j++) {
      inputs[j].checked = wanted === null || !!wanted[inputs[j].getAttribute("data-conference")];
    }
    syncAllConferences(filter);
    return selectedConferences(filter);
  }

  function tagInputs(filter) {
    return filter.querySelectorAll("input[data-tag]");
  }

  function selectedTags(filter) {
    var inputs = tagInputs(filter);
    var selected = [];
    for (var i = 0; i < inputs.length; i++) {
      if (inputs[i].checked) selected.push(inputs[i].getAttribute("data-tag"));
    }
    return selected;
  }

  function syncTagClear(filter) {
    var clear = filter.querySelector("#pn-search-tag-clear");
    if (clear) clear.disabled = selectedTags(filter).length === 0;
  }

  function applyTagSelection(filter, selected) {
    var wanted = null;
    if (selected !== null) {
      wanted = Object.create(null);
      for (var i = 0; i < selected.length; i++) wanted[selected[i]] = true;
    }
    var inputs = tagInputs(filter);
    for (var j = 0; j < inputs.length; j++) {
      inputs[j].checked = wanted === null || !!wanted[inputs[j].getAttribute("data-tag")];
    }
    syncTagClear(filter);
    return selectedTags(filter);
  }

  function setTagMode(filter, mode) {
    mode = mode === "or" ? "or" : "and";
    var buttons = filter.querySelectorAll("[data-tag-mode]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute(
        "aria-pressed",
        buttons[i].getAttribute("data-tag-mode") === mode ? "true" : "false"
      );
    }
    return mode;
  }

  function selectedTagMode(filter) {
    var button = filter.querySelector('[data-tag-mode][aria-pressed="true"]');
    return button && button.getAttribute("data-tag-mode") === "or" ? "or" : "and";
  }

  function isUsefulTag(tag) {
    tag = (tag || "").trim();
    return !!tag && !/(?:19|20)\d{2}/.test(tag);
  }

  function appendTagOption(container, tag, count) {
    var label = document.createElement("label");
    label.className = "pn-conference-option pn-tag-option";
    var input = document.createElement("input");
    input.type = "checkbox";
    input.setAttribute("data-tag", tag);
    var text = document.createElement("span");
    text.textContent = tag;
    var num = document.createElement("em");
    num.className = "pn-tag-count";
    num.textContent = count;
    text.appendChild(num);
    label.appendChild(input);
    label.appendChild(text);
    container.appendChild(label);
  }

  function populateTags(filter, selected, mode) {
    var counts = Object.create(null);
    for (var i = 0; i < docsCache.length; i++) {
      var tags = docsCache[i].tagList || [];
      for (var j = 0; j < tags.length; j++) {
        var tag = tags[j];
        if (isUsefulTag(tag)) counts[tag] = (counts[tag] || 0) + 1;
      }
    }

    var chosen = selected || [];
    for (var s = 0; s < chosen.length; s++) {
      if (!(chosen[s] in counts)) counts[chosen[s]] = 0;
    }

    var names = Object.keys(counts).sort(function (a, b) {
      return counts[b] - counts[a] || a.localeCompare(b);
    });
    names = names.slice(0, TAG_LIMIT);
    var container = filter.querySelector("#pn-search-tags");
    container.innerHTML = "";
    for (var k = 0; k < names.length; k++) {
      appendTagOption(container, names[k], counts[names[k]]);
    }
    filter.disabled = false;
    setTagMode(filter, mode);
    return applyTagSelection(filter, selected);
  }

  function appendConferenceOption(container, conference, labelText, isAll) {
    var label = document.createElement("label");
    label.className = "pn-conference-option" + (isAll ? " pn-conference-all" : "");
    var input = document.createElement("input");
    input.type = "checkbox";
    if (isAll) input.id = "pn-search-all-conferences";
    else input.setAttribute("data-conference", conference);
    var text = document.createElement("span");
    text.textContent = labelText;
    label.appendChild(input);
    label.appendChild(text);
    container.appendChild(label);
  }

  function populateConferences(filter, selected) {
    var seen = Object.create(null);
    var conferences = [];
    for (var i = 0; i < docsCache.length; i++) {
      var conference = docsCache[i].venue;
      if (!seen[conference]) {
        seen[conference] = true;
        conferences.push(conference);
      }
    }
    var siteOrder = siteConferenceOrder();
    conferences.sort(function (a, b) {
      var ai = siteOrder.indexOf(a);
      var bi = siteOrder.indexOf(b);
      if (ai !== bi) {
        return (ai < 0 ? siteOrder.length : ai) - (bi < 0 ? siteOrder.length : bi);
      }
      var ap = conferenceParts(a);
      var bp = conferenceParts(b);
      return ap.name.localeCompare(bp.name) || bp.year - ap.year || a.localeCompare(b);
    });

    conferenceNames = conferences;
    var container = filter.querySelector("#pn-search-conferences");
    container.innerHTML = "";
    appendConferenceOption(container, "", T.allConferences, true);
    for (var j = 0; j < conferences.length; j++) {
      appendConferenceOption(container, conferences[j], prettyConference(conferences[j]), false);
    }
    filter.disabled = false;
    return applyConferenceSelection(filter, selected);
  }

  function buildDocs(raw) {
    var docs = [];
    for (var i = 0; i < raw.length; i++) {
      var d = raw[i];
      var loc = d.location || "";
      if (!loc || loc.indexOf("#") >= 0) continue; // section anchors
      var segs = loc.replace(/\/+$/, "").split("/");
      if (segs.length < 3) continue; // skip conf-root / area-index pages
      var title = (d.title || "").trim();
      if (!title) continue;
      var kw = d.text || "";
      var kwList = kw ? kw.split(/[,\u3001]\s*/).filter(Boolean) : [];
      var tagList = d.tags || [];
      var tags = tagList.join(" ");
      var map = Object.create(null);
      addTokens(map, title, 3);
      addTokens(map, kw, 2);
      addTokens(map, tags, 2);
      docs.push({
        loc: loc,
        title: title,
        venue: segs[0],
        area: prettyArea(segs[1]),
        kw: kw,
        kwList: kwList,
        tagList: tagList,
        map: map,
        titleNorm: title.toLowerCase(),
        kwNorm: kw.toLowerCase(),
        tlen: title.length
      });
    }
    return docs;
  }

  function ensureIndex(cb) {
    if (docsCache) {
      cb(null);
      return;
    }
    waiters.push(cb);
    if (building) return;
    building = true;
    fetch(base() + "search/search_index.json")
      .then(function (r) {
        return r.json();
      })
      .then(function (data) {
        docsCache = buildDocs(data.docs || []);
        flush(null);
      })
      .catch(function (err) {
        flush(err);
      });
    function flush(err) {
      building = false;
      var ws = waiters;
      waiters = [];
      for (var i = 0; i < ws.length; i++) ws[i](err);
    }
  }

  function tagMatch(docTags, tags, tagMode) {
    if (!tags || !tags.length) return true;
    if (tagMode === "or") {
      for (var i = 0; i < tags.length; i++) {
        if (docTags.indexOf(tags[i]) >= 0) return true;
      }
      return false;
    }
    for (var j = 0; j < tags.length; j++) {
      if (docTags.indexOf(tags[j]) < 0) return false;
    }
    return true;
  }

  function runSearch(q, conferences, tags, tagMode) {
    var qn = q.toLowerCase().replace(/\s+/g, " ").trim();
    var qToks = [];
    var seen = Object.create(null);
    var raw = tokenize(expand(q));
    for (var i = 0; i < raw.length; i++) {
      if (!seen[raw[i]]) {
        seen[raw[i]] = 1;
        qToks.push(raw[i]);
      }
    }
    if (!qToks.length && (!tags || !tags.length)) return [];
    var conferenceSet = null;
    if (conferences !== null) {
      conferenceSet = Object.create(null);
      for (var c = 0; c < conferences.length; c++) conferenceSet[conferences[c]] = true;
    }
    var hits = [];
    for (var d = 0; d < docsCache.length; d++) {
      var doc = docsCache[d];
      if (conferenceSet && !conferenceSet[doc.venue]) continue;
      if (!tagMatch(doc.tagList, tags, tagMode)) continue;
      var s = 0,
        matched = 0;
      for (var k = 0; k < qToks.length; k++) {
        var w = doc.map[qToks[k]];
        if (w) {
          s += w;
          matched++;
        }
      }
      if (qToks.length) {
        if (s <= 0) continue;
        s *= 0.4 + 0.6 * (matched / qToks.length); // favour matching all terms
        if (qn.length >= 2) {
          if (doc.titleNorm.indexOf(qn) >= 0) s += 12; // near-exact title match
          else if (doc.kwNorm.indexOf(qn) >= 0) s += 4;
        }
      }
      hits.push({ doc: doc, s: s });
    }
    hits.sort(function (a, b) {
      return b.s - a.s || a.doc.tlen - b.doc.tlen;
    });
    return hits;
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // Build the set of strings to highlight from a query: whole whitespace-
  // separated phrases (exact user input + bilingual expansions) plus the
  // search tokens (CJK bigrams + Latin words; single CJK chars are dropped to
  // avoid highlighting one stray character everywhere). Longest first so
  // "\u56fe\u50cf\u62fc\u8d34" wins over its "\u56fe\u50cf" sub-bigram during range marking.
  function highlightTerms(q) {
    var ex = expand(q);
    var set = Object.create(null);
    var phrases = ex.toLowerCase().split(/\s+/);
    for (var i = 0; i < phrases.length; i++) {
      var p = phrases[i].replace(/[^a-z0-9\u4e00-\u9fff]+/g, "");
      if (p.length >= 2) set[p] = 1;
    }
    var toks = tokenize(ex);
    for (var j = 0; j < toks.length; j++) {
      if (toks[j].length >= 2) set[toks[j]] = 1;
    }
    var out = [];
    for (var k in set) out.push(k);
    out.sort(function (a, b) {
      return b.length - a.length;
    });
    return out;
  }

  // HTML-escape `text` while wrapping any span that matches a highlight term
  // in <mark>. Matching is done on the raw (case-folded) string and the output
  // is escaped per-character, so HTML entities can never be corrupted.
  function hl(text, terms) {
    var n = text.length;
    if (!n || !terms.length) return esc(text);
    var low = text.toLowerCase();
    var mark = new Array(n);
    for (var i = 0; i < n; i++) mark[i] = false;
    for (var t = 0; t < terms.length; t++) {
      var term = terms[t];
      var tl = term.length;
      if (!tl) continue;
      var from = 0,
        idx;
      while ((idx = low.indexOf(term, from)) !== -1) {
        for (var m = idx; m < idx + tl; m++) mark[m] = true;
        from = idx + tl;
      }
    }
    var out = "",
      open = false;
    for (var c = 0; c < n; c++) {
      if (mark[c] && !open) {
        out += '<mark class="pn-mark">';
        open = true;
      } else if (!mark[c] && open) {
        out += "</mark>";
        open = false;
      }
      out += esc(text.charAt(c));
    }
    if (open) out += "</mark>";
    return out;
  }

  function deepHost() {
    if (/papernotes\.org$/.test(location.hostname)) return location.hostname;
    return LANG === "en" ? "en.papernotes.org" : "papernotes.org";
  }

  function deepLink(q) {
    var href =
      "https://www.bing.com/search?q=site:" +
      deepHost() +
      "+" +
      encodeURIComponent(q);
    return (
      '<a class="pn-deep" href="' +
      href +
      '" target="_blank" rel="noopener">' +
      esc(T.deep(q)) +
      "</a>"
    );
  }

  function getQ() {
    var m = /[?&]q=([^&]*)/.exec(location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : "";
  }

  function getPage() {
    var m = /[?&]page=(\d+)/.exec(location.search);
    var p = m ? parseInt(m[1], 10) : 1;
    return p > 0 ? p : 1;
  }

  function getConferences() {
    var m = /[?&]conference=([^&]*)/.exec(location.search);
    if (!m) return null;
    try {
      var value = decodeURIComponent(m[1].replace(/\+/g, " "));
      return value === "none" ? [] : value.split(",").filter(Boolean);
    } catch (e) {
      return null;
    }
  }

  function getTags() {
    var m = /[?&]tags=([^&]*)/.exec(location.search);
    if (!m) return [];
    try {
      return decodeURIComponent(m[1].replace(/\+/g, " ")).split(",").filter(Boolean);
    } catch (e) {
      return [];
    }
  }

  function getTagMode() {
    var m = /[?&]tag_mode=(and|or)/.exec(location.search);
    return m && m[1] === "or" ? "or" : "and";
  }

  function searchUrl(q, page, conferences, tags, tagMode) {
    var nq = (q || "").trim();
    var params = [];
    if (nq) params.push("q=" + encodeURIComponent(nq));
    if (conferences !== null && conferences.length !== conferenceNames.length) {
      params.push(
        "conference=" + encodeURIComponent(conferences.length ? conferences.join(",") : "none")
      );
    }
    if (tags && tags.length) {
      params.push("tags=" + encodeURIComponent(tags.join(",")));
      params.push("tag_mode=" + (tagMode === "or" ? "or" : "and"));
    }
    if ((nq || (tags && tags.length)) && page && page > 1) params.push("page=" + page);
    return base() + "search/" + (params.length ? "?" + params.join("&") : "");
  }

  function replaceSearchUrl(q, page, conferences, tags, tagMode) {
    try {
      history.replaceState(null, "", searchUrl(q, page, conferences, tags, tagMode));
    } catch (e) {}
  }

  function ensurePager(list) {
    var pager = document.getElementById("pn-search-pagination");
    if (!pager) {
      pager = document.createElement("nav");
      pager.id = "pn-search-pagination";
      pager.className = "pn-pager";
      pager.setAttribute("aria-label", LANG === "zh" ? "搜索结果分页" : "Search result pages");
      list.insertAdjacentElement("afterend", pager);
    }
    return pager;
  }

  function pageAnchor(q, conferences, tags, tagMode, page, label, className) {
    return (
      '<button type="button" class="pn-page ' +
      className +
      '" data-page="' +
      page +
      '" data-query="' +
      encodeURIComponent(q) +
      '" data-conferences="' +
      encodeURIComponent((conferences || []).join(",")) +
      '" data-tags="' +
      encodeURIComponent((tags || []).join(",")) +
      '" data-tag-mode="' +
      (tagMode === "or" ? "or" : "and") +
      '">' +
      esc(label) +
      "</button>"
    );
  }

  function renderPager(pager, q, conferences, tags, tagMode, page, totalPages) {
    if (!pager) return;
    if ((!q && (!tags || !tags.length)) || totalPages <= 1) {
      pager.innerHTML = "";
      return;
    }

    var half = Math.floor(MAX_PAGE_LINKS / 2);
    var start = Math.max(1, page - half);
    var end = Math.min(totalPages, start + MAX_PAGE_LINKS - 1);
    start = Math.max(1, end - MAX_PAGE_LINKS + 1);

    var html = "";
    html +=
      page > 1
        ? pageAnchor(q, conferences, tags, tagMode, page - 1, T.prev, "pn-prev")
        : '<span class="pn-page pn-prev pn-disabled">' + esc(T.prev) + "</span>";

    if (start > 1) {
      html += pageAnchor(q, conferences, tags, tagMode, 1, "1", "pn-number");
      if (start > 2) html += '<span class="pn-page pn-ellipsis">…</span>';
    }

    for (var p = start; p <= end; p++) {
      html +=
        p === page
          ? '<span class="pn-page pn-number pn-current" aria-current="page">' + p + "</span>"
          : pageAnchor(q, conferences, tags, tagMode, p, String(p), "pn-number");
    }

    if (end < totalPages) {
      if (end < totalPages - 1) html += '<span class="pn-page pn-ellipsis">…</span>';
      html += pageAnchor(q, conferences, tags, tagMode, totalPages, String(totalPages), "pn-number");
    }

    html +=
      page < totalPages
        ? pageAnchor(q, conferences, tags, tagMode, page + 1, T.next, "pn-next")
        : '<span class="pn-page pn-next pn-disabled">' + esc(T.next) + "</span>";

    pager.innerHTML = html;
  }

  function render(status, list, pager, q, page, conferences, tags, tagMode) {
    q = (q || "").trim();
    tags = tags || [];
    if (!q && !tags.length) {
      status.textContent = T.empty;
      list.innerHTML = "";
      renderPager(pager, q, conferences, tags, tagMode, 1, 0);
      return;
    }
    status.textContent = T.loading;
    list.innerHTML = "";
    renderPager(pager, q, conferences, tags, tagMode, 1, 0);
    ensureIndex(function (err) {
      if (err) {
        status.textContent = T.failed;
        return;
      }
      var hits = runSearch(q, conferences, tags, tagMode);
      if (!hits.length) {
        replaceSearchUrl(q, 1, conferences, tags, tagMode);
        status.innerHTML = esc(T.none) + (q ? " · " + deepLink(q) : "");
        return;
      }
      var totalPages = Math.ceil(hits.length / PAGE_SIZE);
      page = Math.min(Math.max(page || 1, 1), totalPages);
      replaceSearchUrl(q, page, conferences, tags, tagMode);
      var start = (page - 1) * PAGE_SIZE;
      var end = Math.min(start + PAGE_SIZE, hits.length);
      status.innerHTML =
        T.count(hits.length, start + 1, end, page, totalPages) +
        (q ? " · " + deepLink(q) : "");
      var terms = highlightTerms(q);
      var b = base();
      var html = "";
      for (var i = start; i < end; i++) {
        var doc = hits[i].doc;
        html +=
          '<li class="pn-hit">' +
          '<a class="pn-hit-link" href="' +
          b +
          doc.loc +
          '">' +
          hl(doc.title, terms) +
          "</a>" +
          '<div class="pn-hit-meta">' +
          esc(doc.venue) +
          " · " +
          esc(doc.area) +
          "</div>" +
          (doc.kwList && doc.kwList.length
            ? '<div class="pn-hit-kw">' +
              doc.kwList
                .map(function (k) {
                  return '<span class="pn-kw">' + hl(k, terms) + "</span>";
                })
                .join("") +
              "</div>"
            : "") +
          "</li>";
      }
      list.innerHTML = html;
      renderPager(pager, q, conferences, tags, tagMode, page, totalPages);
    });
  }

  var styleInjected = false;
  function injectStyle() {
    if (styleInjected) return;
    styleInjected = true;
    var css =
      "#pn-search-root{max-width:46rem;margin:0 auto}" +
      "#pn-search-box{margin:.4rem 0 1rem}" +
      "#pn-search-input{box-sizing:border-box;width:100%;min-width:0;font-size:1rem;padding:.6rem .8rem;border:1px solid var(--md-default-fg-color--lighter);border-radius:.4rem;background:var(--md-default-bg-color);color:var(--md-default-fg-color)}" +
      "#pn-search-input:focus{outline:none;border-color:var(--md-accent-fg-color)}" +
      "#pn-search-conference-filter{min-width:0;border:0;padding:0;margin:.65rem 0 0}" +
      "#pn-search-conference-filter legend{padding:0;color:var(--md-default-fg-color--light);font-size:.78rem;line-height:1.5}" +
      "#pn-search-conferences{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.3rem}" +
      ".pn-conference-option{display:inline-flex;align-items:center;gap:.28rem;padding:.25rem .48rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem;background:var(--md-default-bg-color);font-size:.76rem;line-height:1.45;white-space:nowrap;cursor:pointer}" +
      ".pn-conference-option:hover{border-color:var(--md-accent-fg-color)}" +
      ".pn-conference-option input{width:.85rem;height:.85rem;margin:0;accent-color:var(--md-accent-fg-color)}" +
      ".pn-conference-option input:checked+span{color:var(--md-typeset-a-color);font-weight:600}" +
      "#pn-search-conference-filter:disabled{opacity:.6}" +
      "#pn-search-tag-filter{min-width:0;border:0;padding:0;margin:.65rem 0 0}" +
      "#pn-search-tag-filter legend{padding:0;color:var(--md-default-fg-color--light);font-size:.78rem;line-height:1.5}" +
      "#pn-search-tags{display:flex;flex-wrap:wrap;gap:.35rem;margin-top:.3rem;max-height:9rem;overflow:auto;padding:.1rem 0}" +
      "#pn-search-tag-filter:disabled{opacity:.6}" +
      ".pn-tag-count{font-style:normal;opacity:.7;font-size:.65rem;margin-left:.2rem}" +
      ".pn-tag-actions{display:flex;align-items:center;gap:.6rem;margin-top:.4rem;flex-wrap:wrap}" +
      ".pn-tag-mode{display:inline-flex;overflow:hidden;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem}" +
      ".pn-tag-mode button{padding:.12rem .42rem;border:0;background:transparent;color:var(--md-default-fg-color--light);font:inherit;font-size:.66rem;font-weight:700;line-height:1.5;cursor:pointer}" +
      ".pn-tag-mode button[aria-pressed=true]{background:var(--md-accent-fg-color);color:var(--md-primary-bg-color)}" +
      "#pn-search-tag-clear{padding:.16rem .48rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem;background:transparent;color:var(--md-typeset-a-color);font:inherit;font-size:.68rem;line-height:1.5;cursor:pointer}" +
      "#pn-search-tag-clear:disabled{color:var(--md-default-fg-color--lighter);cursor:default}" +
      "#pn-search-status{color:var(--md-default-fg-color--light);font-size:.8rem;margin-bottom:.4rem}" +
      "#pn-search-results{list-style:none;margin:0;padding:0}" +
      ".pn-hit{padding:.7rem 0;border-bottom:1px solid var(--md-default-fg-color--lightest)}" +
      ".pn-hit-link{font-weight:600;font-size:1rem;line-height:1.4}" +
      ".pn-hit-meta{font-size:.72rem;color:var(--md-default-fg-color--light);margin-top:.15rem}" +
      ".pn-hit-kw{margin-top:.35rem;display:flex;flex-wrap:wrap;gap:.3rem}" +
      ".pn-kw{font-size:.72rem;color:var(--md-default-fg-color--light);background:var(--md-default-fg-color--lightest);border-radius:.6rem;padding:.08rem .5rem;line-height:1.7;white-space:nowrap}" +
      ".pn-pager{display:flex;align-items:center;justify-content:center;gap:.35rem;flex-wrap:wrap;margin:1rem 0 0}" +
      ".pn-page{min-width:2rem;padding:.28rem .55rem;border:1px solid var(--md-default-fg-color--lightest);border-radius:.35rem;text-align:center;font-size:.78rem;line-height:1.5;background:var(--md-default-bg-color)}" +
      "button.pn-page{font:inherit;color:var(--md-typeset-a-color);cursor:pointer}" +
      ".pn-page:hover{border-color:var(--md-accent-fg-color);text-decoration:none}" +
      ".pn-current{color:var(--md-primary-bg-color);background:var(--md-primary-fg-color);border-color:var(--md-primary-fg-color);font-weight:700}" +
      ".pn-disabled,.pn-ellipsis{color:var(--md-default-fg-color--light);background:transparent}" +
      ".pn-disabled{pointer-events:none}" +
      // Matched-term highlight, mainstream-search style: a warm bold colour that
      // pops on both the purple title links and the grey keyword chips, plus a
      // faint tinted pill. All rules are scoped under `.md-typeset` so they beat
      // Material's own `.md-typeset mark` default (blue text / yellow box) which
      // otherwise wins on specificity. `.pn-kw .pn-mark` drops the pill inside
      // chips (the chip already has its own background) and just recolours text.
      ".md-typeset .pn-mark{color:#e8590c;background:rgba(232,89,12,.14);font-weight:700;border-radius:.2rem;padding:0 .12em}" +
      ".md-typeset .pn-kw .pn-mark{background:none;padding:0}" +
      '[data-md-color-scheme="slate"] .md-typeset .pn-mark{color:#ffa94d;background:rgba(255,169,77,.16)}' +
      '[data-md-color-scheme="slate"] .md-typeset .pn-kw .pn-mark{background:none}' +
      ".pn-deep{font-size:.8rem;white-space:nowrap}" +
      "body.pn-on-search .md-search{display:none!important}" +
      'body.pn-on-search label[for="__search"]{display:none!important}' +
      // Hiding the header search leaves Material thinking search is "open"
      // (#__search stays checked), which fires its
      // `[data-md-toggle=search]:checked ~ .md-header .md-header__option{max-width:0;opacity:0}`
      // rule and collapses the dark/light palette toggle. Force it back (our
      // !important beats Material's non-important collapse rule).
      "body.pn-on-search .md-header__option{max-width:none!important;opacity:1!important}";
    var st = document.createElement("style");
    st.textContent = css;
    document.head.appendChild(st);
  }

  function init() {
    var root = document.getElementById("pn-search-root");
    document.body.classList.toggle("pn-on-search", !!root);
    if (!root) return; // not the search page
    injectStyle();
    var input = document.getElementById("pn-search-input");
    var conferenceFilter = document.getElementById("pn-search-conference-filter");
    var tagFilter = document.getElementById("pn-search-tag-filter");
    var status = document.getElementById("pn-search-status");
    var list = document.getElementById("pn-search-results");
    if (!input || !conferenceFilter || !status || !list) return;
    var pager = ensurePager(list);

    input.placeholder = T.placeholder;

    var initialQ = getQ();
    var initialPage = getPage();
    var initialConferences = getConferences();
    var initialTags = tagFilter ? getTags() : [];
    var initialTagMode = tagFilter ? getTagMode() : "and";
    if (initialQ) input.value = initialQ;

    var timer = null;
    function update(immediate, page, queryOverride, conferencesOverride, tagsOverride, tagModeOverride) {
      var q = typeof queryOverride === "string" ? queryOverride : input.value;
      if (typeof queryOverride === "string" && input.value !== q) input.value = q;
      var conferences = Array.isArray(conferencesOverride)
        ? applyConferenceSelection(conferenceFilter, conferencesOverride)
        : selectedConferences(conferenceFilter);
      var tags = tagFilter
        ? Array.isArray(tagsOverride)
          ? applyTagSelection(tagFilter, tagsOverride)
          : selectedTags(tagFilter)
        : [];
      var tagMode = tagFilter
        ? tagModeOverride
          ? setTagMode(tagFilter, tagModeOverride)
          : selectedTagMode(tagFilter)
        : "and";
      if (tagFilter) syncTagClear(tagFilter);
      var nq = q.trim();
      page = nq || tags.length ? page || 1 : 1;
      replaceSearchUrl(nq, page, conferences, tags, tagMode);
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (immediate) {
        render(status, list, pager, q, page, conferences, tags, tagMode);
      } else {
        timer = setTimeout(function () {
          render(status, list, pager, q, page, conferences, tags, tagMode);
        }, 180);
      }
    }

    input.addEventListener("input", function () {
      update(false, 1);
    });
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter") {
        e.preventDefault();
        update(true, 1);
      }
    });
    conferenceFilter.addEventListener("change", function (e) {
      if (e.target && e.target.id === "pn-search-all-conferences") {
        var inputs = conferenceInputs(conferenceFilter);
        for (var i = 0; i < inputs.length; i++) inputs[i].checked = e.target.checked;
      }
      syncAllConferences(conferenceFilter);
      update(true, 1);
    });
    if (tagFilter) {
      tagFilter.addEventListener("change", function (e) {
        if (e.target && e.target.hasAttribute("data-tag")) update(true, 1);
      });
      var tagClear = tagFilter.querySelector("#pn-search-tag-clear");
      if (tagClear) {
        tagClear.addEventListener("click", function () {
          applyTagSelection(tagFilter, []);
          update(true, 1);
        });
      }
      var tagModeGroup = tagFilter.querySelector("#pn-search-tag-mode");
      if (tagModeGroup) {
        tagModeGroup.addEventListener("click", function (e) {
          var button = e.target.closest("[data-tag-mode]");
          if (!button) return;
          setTagMode(tagFilter, button.getAttribute("data-tag-mode"));
          update(true, 1);
        });
      }
    }
    pager.addEventListener("click", function (e) {
      var target = e.target.closest("[data-page]");
      if (!target) return;
      e.preventDefault();
      var query = "";
      try {
        query = decodeURIComponent(target.getAttribute("data-query") || "");
      } catch (err) {}
      try {
        var href = target.getAttribute("href");
        var hrefQuery = href ? new URL(href, location.href).searchParams.get("q") : "";
        if (hrefQuery) query = hrefQuery;
      } catch (err) {}
      var conferences = [];
      try {
        var encoded = decodeURIComponent(target.getAttribute("data-conferences") || "");
        conferences = encoded ? encoded.split(",") : [];
      } catch (err) {}
      var tags = [];
      try {
        var encodedTags = decodeURIComponent(target.getAttribute("data-tags") || "");
        tags = encodedTags ? encodedTags.split(",") : [];
      } catch (err) {}
      var tagMode = target.getAttribute("data-tag-mode") === "or" ? "or" : "and";
      update(
        true,
        parseInt(target.getAttribute("data-page"), 10) || 1,
        query || input.value || getQ(),
        conferences,
        tags,
        tagMode
      );
      try {
        status.scrollIntoView({ block: "start" });
      } catch (err) {}
    });

    status.textContent = T.loading;
    ensureIndex(function (err) {
      if (err) {
        status.textContent = T.failed;
        return;
      }
      initialConferences = populateConferences(conferenceFilter, initialConferences);
      if (tagFilter) {
        initialTags = populateTags(tagFilter, initialTags, initialTagMode);
        initialTagMode = selectedTagMode(tagFilter);
      }
      render(
        status,
        list,
        pager,
        initialQ,
        initialPage,
        initialConferences,
        initialTags,
        initialTagMode
      );
    });
    try {
      input.focus();
    } catch (e) {}
  }

  // Material instant navigation replaces <main> without a full reload, so hook
  // its document$ observable (emits on first load and after each instant nav).
  // Fall back to DOMContentLoaded when Material's bundle is unavailable.
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
