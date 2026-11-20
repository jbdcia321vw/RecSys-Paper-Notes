---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**CIKM 2023** · **5** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [CIKM 2022 (2)](../../2022/multi_task_reranking/) · [CIKM 2024 (1)](../../2024/multi_task_reranking/) · [CIKM 2025 (4)](../../2025/multi_task_reranking/) · [CIKM 2026 (5)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/multi_task_reranking/) · [KDD 2022 (1)](../../../KDD/2022/multi_task_reranking/) · [KDD 2023 (3)](../../../KDD/2023/multi_task_reranking/) · [KDD 2024 (3)](../../../KDD/2024/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2023 (1)](../../../RecSys/2023/multi_task_reranking/) · [RecSys 2024 (5)](../../../RecSys/2024/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2022 (1)](../../../SIGIR/2022/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/multi_task_reranking/) · [WWW (TheWebConf) 2023 (3)](../../../WWW/2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;粗排&quot;,&quot;排序一致性对齐&quot;,&quot;分块采样&quot;,&quot;ECPM 排序学习&quot;]"><strong><a href="copr_consistency_oriented_pre_ranking_for_online_advertising/">COPR: Consistency-Oriented Pre-Ranking for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将粗排↔精排的分数对齐松弛为 ECPM 排序对齐，用分块采样与 ΔNDCG 加权 pairwise 损失优化一致性，淘宝线上 +12.3% CTR、+5.6% RPM。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;任务特定底层表征&quot;,&quot;超网络条件Transformer&quot;,&quot;SENet式特征重标定&quot;]"><strong><a href="deep_task_specific_bottom_representation_network_for_multi_task_recommendat/">Deep Task-specific Bottom Representation Network for Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CIKM 2023 多任务推荐论文，提出 DTRN 以超网络条件 Transformer 与 SENet 式任务特定底层表征缓解负迁移；在电商与 Tenrec 样本级 AUC/LogLoss 上全面优于 MMoE/PLE/AITM，在线进入率 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;稀疏MoE路由&quot;,&quot;多模态特征融合&quot;]"><strong><a href="efficient_multi_task_learning_via_generalist_recommender/">Efficient Multi-Task Learning via Generalist Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GRec 用多模态 Wide&amp;Deep + 并行 Transformer + 任务-句子级稀疏 MoE 路由，在电信场景离线 AP 与线上 A/B 全面超越 MMoE 与单任务模型。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;投影解耦&quot;,&quot;多专家网络&quot;]"><strong><a href="parallel_knowledge_enhancement_based_framework_for_multi_behavior_recommend/">Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以并行流知识投影增强级联多行为 GNN，并用投影解耦多专家消除 MTL 梯度冲突，三数据集 HR@10 最高提升 95.16%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排/模型选择&quot;,&quot;帕累托前沿&quot;,&quot;多目标优化&quot;,&quot;校准推荐&quot;]"><strong><a href="post_hoc_selection_of_pareto_optimal_solutions_in_search_and_recommendation/">Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 PDU 从帕累托前沿事后挑出唯一可部署解：用样本级 utopia 距离替代聚合点距离，并支持用户级校准。</p>
</dd>
</dl>
