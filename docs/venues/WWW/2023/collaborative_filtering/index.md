---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**WWW (TheWebConf) 2023** · **6** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2021 (6)](../../2021/collaborative_filtering/) · [WWW (TheWebConf) 2022 (6)](../../2022/collaborative_filtering/) · [WWW (TheWebConf) 2024 (2)](../../2024/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (6)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2021 (3)](../../../AAAI/2021/collaborative_filtering/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/collaborative_filtering/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2021 (4)](../../../CIKM/2021/collaborative_filtering/) · [CIKM 2022 (5)](../../../CIKM/2022/collaborative_filtering/) · [CIKM 2023 (3)](../../../CIKM/2023/collaborative_filtering/) · [CIKM 2024 (4)](../../../CIKM/2024/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2021 (1)](../../../ECIR/2021/collaborative_filtering/) · [ECIR 2022 (1)](../../../ECIR/2022/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICDE 2021 (2)](../../../ICDE/2021/collaborative_filtering/) · [ICDE 2023 (1)](../../../ICDE/2023/collaborative_filtering/) · [ICDE 2024 (2)](../../../ICDE/2024/collaborative_filtering/) · [ICDM 2021 (2)](../../../ICDM/2021/collaborative_filtering/) · [ICDM 2023 (1)](../../../ICDM/2023/collaborative_filtering/) · [ICDM 2024 (1)](../../../ICDM/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/collaborative_filtering/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/collaborative_filtering/) · [KDD 2021 (6)](../../../KDD/2021/collaborative_filtering/) · [KDD 2022 (3)](../../../KDD/2022/collaborative_filtering/) · [KDD 2023 (1)](../../../KDD/2023/collaborative_filtering/) · [KDD 2024 (1)](../../../KDD/2024/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2021 (4)](../../../RecSys/2021/collaborative_filtering/) · [RecSys 2022 (2)](../../../RecSys/2022/collaborative_filtering/) · [RecSys 2023 (7)](../../../RecSys/2023/collaborative_filtering/) · [RecSys 2024 (4)](../../../RecSys/2024/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2021 (7)](../../../SIGIR/2021/collaborative_filtering/) · [SIGIR 2022 (4)](../../../SIGIR/2022/collaborative_filtering/) · [SIGIR 2023 (6)](../../../SIGIR/2023/collaborative_filtering/) · [SIGIR 2024 (4)](../../../SIGIR/2024/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/collaborative_filtering/) · [IEEE TKDE 2021 (2)](../../../TKDE/2021/collaborative_filtering/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/collaborative_filtering/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/collaborative_filtering/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/collaborative_filtering/) · [ACM TOIS 2022 (2)](../../../TOIS/2022/collaborative_filtering/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/collaborative_filtering/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/collaborative_filtering/) · [ACM TORS 2023 (1)](../../../TORS/2023/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [ACM TWEB 2022 (1)](../../../TWEB/2022/collaborative_filtering/) · [WSDM 2021 (1)](../../../WSDM/2021/collaborative_filtering/) · [WSDM 2022 (1)](../../../WSDM/2022/collaborative_filtering/) · [WSDM 2023 (1)](../../../WSDM/2023/collaborative_filtering/) · [WSDM 2024 (3)](../../../WSDM/2024/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/)

🔥 **高频主题：** WWW 2023 ×6 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;易到难课程学习&quot;,&quot;异构模型集成&quot;]"><strong><a href="distillation_from_heterogeneous_models_for_top_k_recommendation/">Distillation from Heterogeneous Models for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用教师训练轨迹构造由易到难的排序知识课程，将异构推荐模型集成蒸馏进轻量学生，R@10 最高提升 24.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;聚类分治难负例&quot;,&quot;Prompt 式多任务学习&quot;,&quot;双塔 ANN 检索&quot;]"><strong><a href="divide_and_conquer_towards_better_embedding_based_retrieval_for_recommender/">Divide and Conquer: Towards Better Embedding-based Retrieval for Recommender Systems From a Multi-task Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将候选库语义聚类后分簇并行 EBR，仅用同簇难负例训练并配合 Prompt 式多任务适配，KuaiRand Recall 最高提升 42.7%，线上 A/B 全指标正向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图划分&quot;,&quot;物品相似度模型&quot;,&quot;谱聚类&quot;]"><strong><a href="fine_tuning_partition_aware_item_similarities_for_efficient_and_scalable_re/">Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用谱划分把物品相似度建模限制在子划分内，并以全局谱信息微调；4 数据集上全库排序指标超越 GCN，提速 10×、省 95% 参数。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;对比学习&quot;,&quot;图神经网络&quot;]"><strong><a href="graph_less_collaborative_filtering/">Graph-less Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双层知识蒸馏把 GCN 教师压缩为 MLP 学生，配合梯度自适应对比正则，在 Full-Ranking 协议下兼顾精度与推理效率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负采样&quot;,&quot;OPAUC&quot;,&quot;DRO&quot;]"><strong><a href="on_the_theories_behind_hard_negative_sampling_for_recommendation/">On the Theories Behind Hard Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>证明BPR难负采样等价优化OPAUC，并给出可控硬度与按K调采样策略。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（Top-N 排序）&quot;,&quot;可解释推荐&quot;,&quot;口味簇学习&quot;,&quot;稀疏隶属矩阵&quot;,&quot;直通梯度估计&quot;]"><strong><a href="towards_explainable_collaborative_filtering_with_taste_clusters_learning/">Towards Explainable Collaborative Filtering with Taste Clusters Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用稀疏「口味簇」隶属矩阵实现自解释 CF，Forest 集成补偿稀疏化损失，三数据集 Recall@20 相对最优基线提升 8.0%–14.6%。</p>
</dd>
</dl>
