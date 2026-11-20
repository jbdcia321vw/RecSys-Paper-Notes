---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**RecSys 2026** · **7** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [RecSys 2023 (1)](../../2023/multi_task_reranking/) · [RecSys 2024 (5)](../../2024/multi_task_reranking/) · [RecSys 2025 (5)](../../2025/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [CIKM 2023 (5)](../../../CIKM/2023/multi_task_reranking/) · [CIKM 2024 (1)](../../../CIKM/2024/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2023 (3)](../../../KDD/2023/multi_task_reranking/) · [KDD 2024 (3)](../../../KDD/2024/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2023 (3)](../../../WWW/2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 多任务学习 ×7

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;子空间划分&quot;,&quot;多目标检索&quot;]"><strong><a href="embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;异构内容排序&quot;,&quot;MoE&quot;]"><strong><a href="heterogeneous_ranking_in_industrial_scale_recommender_systems_a_case_study/">Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 Google Discover 异构信息流的 HA-MoE 多门控专家架构，配合 LENS 可观测性与 DL-AUC 双层评估，离线与线上均获稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;辅助头&quot;,&quot;联合分布逼近&quot;,&quot;协方差建模&quot;]"><strong><a href="learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;轻量排序头&quot;,&quot;在线持续学习&quot;,&quot;任务冲突隔离&quot;]"><strong><a href="lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;分段加权&quot;,&quot;MMoE&quot;]"><strong><a href="multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;位置偏置&quot;,&quot;置换一致性&quot;,&quot;LLM 列表重排&quot;]"><strong><a href="position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;成对比较特征&quot;,&quot;上下文感知列表重排&quot;]"><strong><a href="towards_full_candidate_interaction_a_comprehensive_comparison_network_for_b/">Towards Full Candidate Interaction: A Comprehensive Comparison Network for Better Route Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出比较优先范式，用非重叠路段构造显式比较张量并在成对空间直接推理，线上CR +1.2%、偏航率 -0.8%。</p>
</dd>
</dl>
