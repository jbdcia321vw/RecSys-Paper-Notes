---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**KDD 2023** · **3** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [KDD 2022 (1)](../../2022/graph_rec/) · [KDD 2024 (5)](../../2024/graph_rec/) · [KDD 2025 (3)](../../2025/graph_rec/) · [KDD 2026 (1)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/graph_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/graph_rec/) · [CIKM 2022 (5)](../../../CIKM/2022/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2021 (1)](../../../DASFAA/2021/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2021 (2)](../../../ICDE/2021/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/graph_rec/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/graph_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/graph_rec/) · [RecSys 2022 (1)](../../../RecSys/2022/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/graph_rec/) · [SIGIR 2022 (7)](../../../SIGIR/2022/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/graph_rec/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/graph_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (3)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/graph_rec/) · [WSDM 2022 (7)](../../../WSDM/2022/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2021 (4)](../../../WWW/2021/graph_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/graph_rec/)

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
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;LightGCN&quot;,&quot;多准则推荐&quot;,&quot;PairNorm&quot;]"><strong><a href="criteria_tell_you_more_than_ratings_criteria_preference_aware_light_graph_c/">Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多准则评分展开为准则-物品扩展二部图，用轻量图卷积 + 用户准则偏好/物品准则双嵌入建模，全库排序下 P@5 最高提升 141.2%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;有效电导&quot;,&quot;难负样本采样&quot;,&quot;数据子采样&quot;]"><strong><a href="graph_based_model_agnostic_data_subsampling_for_recommendation_systems/">Graph-Based Model-Agnostic Data Subsampling for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用用户-物品二部图的有效电导度量负样本难度并做边传播平滑，实现免 pilot 模型的数据子采样，KuaiRec/MIND 上 AUC 稳定超越均匀采样且可与模型法互补融合。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱合理化&quot;,&quot;掩码自编码器&quot;,&quot;跨视图对比学习&quot;]"><strong><a href="knowledge_graph_self_supervised_rationalization_for_recommendation/">Knowledge Graph Self-Supervised Rationalization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KGRec 用注意力合理化分数驱动掩码重建生成式任务与去噪跨视图对比学习，Full-Ranking 下 Recall@20 较 SOTA 提升 2.3%–4.2%。</p>
</dd>
</dl>
