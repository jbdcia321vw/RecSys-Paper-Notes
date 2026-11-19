(function () {
  "use strict";

  var LANG =
    (document.documentElement.lang || "zh").toLowerCase().indexOf("en") === 0
      ? "en"
      : "zh";

  var T = {
    zh: {
      all: function (n) {
        return "共 " + n + " 篇论文";
      },
      shown: function (visible, total) {
        return "显示 " + visible + " / " + total + " 篇论文";
      }
    },
    en: {
      all: function (n) {
        return n + " papers";
      },
      shown: function (visible, total) {
        return "Showing " + visible + " / " + total + " papers";
      }
    }
  }[LANG];

  function parseTags(dt) {
    var raw = dt.getAttribute("data-paper-tags") || "[]";
    try {
      var value = JSON.parse(raw);
      return Array.isArray(value) ? value : [];
    } catch (e) {
      return [];
    }
  }

  function mountToRightRail(root) {
    var aside = document.querySelector(".md-sidebar--secondary");
    if (!aside) return;
    var inner = aside.querySelector(".md-sidebar__inner");
    if (!inner) return;
    var nav = aside.querySelector(".md-nav--secondary");
    if (!nav) {
      nav = document.createElement("nav");
      nav.className = "md-nav md-nav--secondary";
      inner.appendChild(nav);
    }

    var existing = nav.querySelector(":scope > .pn-tag-rail");
    if (existing) existing.parentNode.removeChild(existing);

    var wrap = document.createElement("div");
    wrap.className = "pn-tag-rail";
    var body = document.createElement("div");
    body.className = "md-typeset pn-tag-rail__body";
    body.appendChild(root);
    wrap.appendChild(body);
    nav.appendChild(wrap);
  }

  function setup(root) {
    if (root.getAttribute("data-tag-filter-ready") === "1") return;
    root.setAttribute("data-tag-filter-ready", "1");

    var chips = root.querySelectorAll("button[data-tag]");
    var clear = root.querySelector("[data-tag-clear]");
    var status = root.querySelector("[data-tag-status]");
    var modeButtons = root.querySelectorAll("[data-tag-mode]");
    var lists = document.querySelectorAll("dl[data-paper-list]");
    var selected = Object.create(null);
    var mode = "and";

    var params = new URLSearchParams(location.search);
    var initialTags = (params.get("tags") || "").split(",").filter(Boolean);
    if (params.get("tag_mode") === "or") mode = "or";
    for (var i = 0; i < initialTags.length; i++) selected[initialTags[i]] = true;

    function selectedKeys() {
      return Object.keys(selected);
    }

    function matches(tags) {
      var keys = selectedKeys();
      if (!keys.length) return true;
      if (mode === "or") {
        for (var i = 0; i < keys.length; i++) {
          if (tags.indexOf(keys[i]) >= 0) return true;
        }
        return false;
      }
      for (var j = 0; j < keys.length; j++) {
        if (tags.indexOf(keys[j]) < 0) return false;
      }
      return true;
    }

    function syncControls() {
      for (var i = 0; i < chips.length; i++) {
        chips[i].setAttribute(
          "aria-pressed",
          selected[chips[i].getAttribute("data-tag")] ? "true" : "false"
        );
      }
      if (clear) clear.disabled = selectedKeys().length === 0;
      for (var j = 0; j < modeButtons.length; j++) {
        modeButtons[j].setAttribute(
          "aria-pressed",
          modeButtons[j].getAttribute("data-tag-mode") === mode ? "true" : "false"
        );
      }
    }

    function syncUrl() {
      var keys = selectedKeys();
      var url = new URL(location.href);
      if (keys.length) {
        url.searchParams.set("tags", keys.join(","));
        url.searchParams.set("tag_mode", mode);
      } else {
        url.searchParams.delete("tags");
        url.searchParams.delete("tag_mode");
      }
      try {
        history.replaceState(null, "", url);
      } catch (e) {}
    }

    function apply() {
      var total = 0;
      var visible = 0;
      for (var i = 0; i < lists.length; i++) {
        var dts = lists[i].querySelectorAll("dt[data-paper-tags]");
        for (var j = 0; j < dts.length; j++) {
          var dt = dts[j];
          total++;
          var show = matches(parseTags(dt));
          dt.classList.toggle("pn-tag-hidden", !show);
          var dd = dt.nextElementSibling;
          if (dd && dd.tagName === "DD") dd.classList.toggle("pn-tag-hidden", !show);
          if (show) visible++;
        }
      }

      var sections = document.querySelectorAll("section[data-paper-section]");
      for (var k = 0; k < sections.length; k++) {
        var hasVisible = sections[k].querySelector(
          "dt[data-paper-tags]:not(.pn-tag-hidden)"
        );
        sections[k].classList.toggle("pn-tag-hidden", !hasVisible);
      }

      if (status) {
        status.textContent = selectedKeys().length
          ? T.shown(visible, total)
          : T.all(total);
      }
      syncControls();
      syncUrl();
    }

    for (var c = 0; c < chips.length; c++) {
      chips[c].addEventListener("click", function () {
        var tag = this.getAttribute("data-tag");
        if (selected[tag]) delete selected[tag];
        else selected[tag] = true;
        apply();
      });
    }

    if (clear) {
      clear.addEventListener("click", function () {
        selected = Object.create(null);
        apply();
      });
    }

    for (var m = 0; m < modeButtons.length; m++) {
      modeButtons[m].addEventListener("click", function () {
        mode = this.getAttribute("data-tag-mode") === "or" ? "or" : "and";
        apply();
      });
    }

    apply();
  }

  function init() {
    var roots = document.querySelectorAll("[data-paper-tag-filter]");
    document.body.classList.toggle("pn-has-tag-filter", roots.length > 0);
    for (var i = 0; i < roots.length; i++) {
      mountToRightRail(roots[i]);
      setup(roots[i]);
    }
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(init);
  } else {
    document.addEventListener("DOMContentLoaded", init);
  }
})();
