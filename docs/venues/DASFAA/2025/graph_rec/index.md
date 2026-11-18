# 🕸️ 图神经网络与知识图谱推荐

**DASFAA 2025** · **1** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动推荐" aria-pressed="false">冷启动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图预训练" aria-pressed="false">超图预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="边信息融合" aria-pressed="false">边信息融合<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图预训练&quot;,&quot;边信息融合&quot;,&quot;冷启动推荐&quot;]"><strong><a href="genet_unleashing_the_power_of_side_information_for_recommendation_via_hyper/">GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将异构边信息统一构建为超图，通过超边预测与双层自监督对比预训练，再以 LightGCN/GRU 微调，Top-N 与序列推荐最高提升 38%，冷启动提升 175%+。</p>
</dd>
</dl>
