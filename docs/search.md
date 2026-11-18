---
hide:
  - toc
---

# 🔎 搜索论文笔记

按标题、关键词与标签检索全站论文。标签筛选默认使用 **AND**（同时包含），可切换为 **OR**（任一匹配）。

<div id="pn-search-root">
  <div id="pn-search-box">
    <input id="pn-search-input" type="search" autocomplete="off" spellcheck="false" />
  </div>
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
  <div id="pn-search-status"></div>
  <ol id="pn-search-results"></ol>
</div>
