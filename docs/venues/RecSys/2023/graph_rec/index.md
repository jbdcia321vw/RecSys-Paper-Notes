---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**RecSys 2023** · **3** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [RecSys 2021 (1)](../../2021/graph_rec/) · [RecSys 2022 (1)](../../2022/graph_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/graph_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/graph_rec/) · [CIKM 2022 (5)](../../../CIKM/2022/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2021 (1)](../../../DASFAA/2021/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2021 (2)](../../../ICDE/2021/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/graph_rec/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/graph_rec/) · [KDD 2022 (1)](../../../KDD/2022/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/graph_rec/) · [SIGIR 2022 (7)](../../../SIGIR/2022/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/graph_rec/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/graph_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (3)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/graph_rec/) · [WSDM 2022 (7)](../../../WSDM/2022/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2021 (4)](../../../WWW/2021/graph_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/graph_rec/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示 ≥2 篇论文的标签；默认 AND</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;全局新闻图&quot;,&quot;GGNN&quot;,&quot;实体图增强&quot;]"><strong><a href="going_beyond_local_global_graph_enhanced_personalized_news_recommendations/">Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用跨用户全局新闻图(GGNN)与全局实体图增强新闻/候选表示，MIND-large AUC 69.54、nDCG@10 44.19，兼顾多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低比特量化&quot;,&quot;梯度直通估计器&quot;,&quot;Hessian 迹估计&quot;]"><strong><a href="hessian_aware_quantized_node_embeddings_for_recommendation/">Hessian-aware Quantized Node Embeddings for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用二阶 Hessian 迹修正 STE 的梯度失配，实现任意比特 GNN 节点量化嵌入，1-bit 下 Recall@50 较 HashGNN 平均提升约 16%，推理加速 3.6×。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;混合曲率流形&quot;,&quot;几何感知 Margin&quot;]"><strong><a href="knowledge_based_multiple_adaptive_spaces_fusion_for_recommendation/">Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习曲率的统一空间兼容双曲/欧氏/球面并注意力融合多子空间，配合几何感知 margin，在三个 KG 数据集采样排序下 HR@20 最高提升 1.9%。</p>
</dd>
</dl>
