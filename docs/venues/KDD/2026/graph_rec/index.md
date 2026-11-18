# 🕸️ 图神经网络与知识图谱推荐

**KDD 2026** · **1** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [KDD 2025 (3)](../../2025/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="空间翻转" aria-pressed="false">空间翻转<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱滤波" aria-pressed="false">谱滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频率信号缩放" aria-pressed="false">频率信号缩放<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;频率信号缩放&quot;,&quot;空间翻转&quot;]"><strong><a href="how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
</dl>
