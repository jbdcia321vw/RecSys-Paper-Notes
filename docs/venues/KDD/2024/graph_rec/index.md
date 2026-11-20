---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**KDD 2024** · **5** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [KDD 2023 (3)](../../2023/graph_rec/) · [KDD 2025 (3)](../../2025/graph_rec/) · [KDD 2026 (1)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;三部图元路径&quot;,&quot;对比学习&quot;]"><strong><a href="consistency_and_discrepancy_based_contrastive_tripartite_graph_learning_for/">Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可达/不可达元路径导出一致性与差异性系数，经 GCN 极限理论免训练预计算，配合 CD 对比损失解决三部图推荐极端冷启动。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱图滤波&quot;,&quot;广义图归一化&quot;,&quot;免训练协同过滤&quot;]"><strong><a href="how_powerful_is_graph_filtering_for_recommendation/">How Powerful is Graph Filtering for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以谱锐度视角提出广义图归一化 G²N 与个性化图滤波 IGF，免训练闭式 SGFCF 在四个密度差异数据集上 nDCG@10 最高提升 14.3%，训练耗时降低三个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;偏序推荐图&quot;,&quot;多行为协同过滤&quot;,&quot;偏序 BPR 采样&quot;]"><strong><a href="multi_behavior_collaborative_filtering_with_partial_order_graph_convolution/">Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以分级偏序秩函数融合多行为图为单张加权 POG，POGCN 图卷积 + 偏序 BPR 采样，全库排序下三基准 Mean NDCG 提升 19.67%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;知识图谱推荐&quot;,&quot;健康行为干预&quot;]"><strong><a href="nudgerank_digital_algorithmic_nudging_for_personalized_health/">NudgeRank: Digital Algorithmic Nudging for Personalized Health</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可扩展异构知识图谱+注意力图卷积实现百万级人群个性化健康提醒，离线Precision@3提升45%，线上日步数+6.17%、MVPA+7.61%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Box Embedding&quot;,&quot;标签感知推荐&quot;]"><strong><a href="when_box_meets_graph_neural_network_in_tag_aware_recommendation/">When Box Meets Graph Neural Network in Tag-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户/物品/标签统一建模为超矩形盒，用交并逻辑算子替代 GNN 均值聚合，并以 Gumbel 平滑盒体积作为匹配分，全库排序下全面超越 LFGCF。</p>
</dd>
</dl>
