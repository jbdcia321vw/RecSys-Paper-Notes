---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**CIKM 2022** · **5** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [CIKM 2021 (7)](../../2021/graph_rec/) · [CIKM 2023 (2)](../../2023/graph_rec/) · [CIKM 2024 (1)](../../2024/graph_rec/) · [CIKM 2025 (3)](../../2025/graph_rec/) · [CIKM 2026 (3)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/graph_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [DASFAA 2021 (1)](../../../DASFAA/2021/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2021 (2)](../../../ICDE/2021/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/graph_rec/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/graph_rec/) · [KDD 2022 (1)](../../../KDD/2022/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/graph_rec/) · [RecSys 2022 (1)](../../../RecSys/2022/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/graph_rec/) · [SIGIR 2022 (7)](../../../SIGIR/2022/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/graph_rec/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/graph_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (3)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/graph_rec/) · [WSDM 2022 (7)](../../../WSDM/2022/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2021 (4)](../../../WWW/2021/graph_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/graph_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表征学习&quot;,&quot;强化学习元路径选择&quot;,&quot;异质信息网络&quot;,&quot;元路径推荐&quot;]"><strong><a href="automatic_meta_path_discovery_for_effective_graph_based_recommendation/">Automatic Meta-Path Discovery for Effective Graph-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出RMS强化学习元路径自动选择与HRec注意力聚合推荐器；在Yelp/Douban/TCL的Leave-One-Out、1正+499负采样排序下，HR@3较最佳基线最高提升12.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;上下文漂移解耦&quot;,&quot;随机游走图嵌入&quot;,&quot;多头交互注意力&quot;]"><strong><a href="hysage_a_hybrid_static_and_adaptive_graph_embedding_network_for_context_dri/">HySAGE: A Hybrid Static and Adaptive Graph Embedding Network for Context-Drifting Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>解耦静态交互图嵌入与漂移上下文自适应嵌入，注意力交互融合，全库未交互排序下 HR@10 最高提升 28.4%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多层交互对比学习&quot;,&quot;知识图谱推荐&quot;,&quot;非局部图&quot;]"><strong><a href="improving_knowledge_aware_recommendation_with_multi_level_interactive_contr/">Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 KGIC：以 GNN 层为视图做图内 CF-KG 与图间局部-非局部两层交互对比，三数据集 AUC 相对提升 0.67%–3.35%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;子图级图神经网络&quot;,&quot;捆绑包推荐&quot;,&quot;迁移推荐&quot;]"><strong><a href="suger_a_subgraph_based_graph_convolutional_network_method_for_bundle_recomm/">SUGER: A Subgraph-based Graph Convolutional Network Method for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以用户-捆绑包为中心抽取异质 k 跳子图，用两级 RGCN 图级编码器打分，并删除中心边消除标签泄漏；Recall@20 在 Youshu/Netease 上相对 BGCN 提升 39%/90%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低秩分解&quot;,&quot;图卷积简化&quot;]"><strong><a href="svd_gcn_a_simplified_graph_convolution_paradigm_for_recommendation/">SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用截断 SVD 替代 GCN 邻域聚合，仅取 K 个奇异向量并重归一化，全库排序精度提升且百倍加速。</p>
</dd>
</dl>
