---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐

**WSDM 2021** · **2** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [WSDM 2022 (1)](../../2022/cold_start_rec/) · [WSDM 2023 (1)](../../2023/cold_start_rec/) · [WSDM 2024 (1)](../../2024/cold_start_rec/) · [WSDM 2025 (1)](../../2025/cold_start_rec/) · [WSDM 2026 (1)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/cold_start_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/cold_start_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2021 (1)](../../../ACM_MM/2021/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/cold_start_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/cold_start_rec/) · [CIKM 2023 (3)](../../../CIKM/2023/cold_start_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2022 (1)](../../../ICDE/2022/cold_start_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICDM 2021 (1)](../../../ICDM/2021/cold_start_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/cold_start_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/cold_start_rec/) · [KDD 2021 (2)](../../../KDD/2021/cold_start_rec/) · [KDD 2023 (2)](../../../KDD/2023/cold_start_rec/) · [KDD 2024 (3)](../../../KDD/2024/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/cold_start_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/cold_start_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/cold_start_rec/) · [SIGIR 2023 (3)](../../../SIGIR/2023/cold_start_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/cold_start_rec/) · [IEEE TKDE 2023 (2)](../../../TKDE/2023/cold_start_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/cold_start_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/cold_start_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/cold_start_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/cold_start_rec/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cold_start_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;伪标签 Pseudo-Labelling&quot;,&quot;知识图谱 GNN&quot;,&quot;协同训练&quot;]"><strong><a href="alleviating_cold_start_problems_in_recommendation_through_pseudo_labelling_/">Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在知识图谱上把未观测样本伪标签化为弱正例，配合 KG 路径采样、频率感知负采样与双模型协同训练，显著提升新用户/新物品的召回。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/表征学习&quot;,&quot;GNN预训练&quot;,&quot;元学习&quot;,&quot;自适应邻居采样&quot;]"><strong><a href="pre_training_graph_neural_networks_for_cold_start_users_and_items_represent/">Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出元学习式GNN预训练、自注意力元聚合器与自适应邻居采样器，直接优化冷启动用户/物品表征，在三个公开数据集的下游推荐上优于GNN基线。</p>
</dd>
</dl>
