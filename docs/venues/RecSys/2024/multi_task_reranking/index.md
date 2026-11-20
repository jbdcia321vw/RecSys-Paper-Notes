---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**RecSys 2024** · **5** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [RecSys 2023 (1)](../../2023/multi_task_reranking/) · [RecSys 2025 (5)](../../2025/multi_task_reranking/) · [RecSys 2026 (7)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [CIKM 2022 (2)](../../../CIKM/2022/multi_task_reranking/) · [CIKM 2023 (5)](../../../CIKM/2023/multi_task_reranking/) · [CIKM 2024 (1)](../../../CIKM/2024/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/multi_task_reranking/) · [KDD 2022 (1)](../../../KDD/2022/multi_task_reranking/) · [KDD 2023 (3)](../../../KDD/2023/multi_task_reranking/) · [KDD 2024 (3)](../../../KDD/2024/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [SIGIR 2022 (1)](../../../SIGIR/2022/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/multi_task_reranking/) · [WWW (TheWebConf) 2023 (3)](../../../WWW/2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;服务时在线学习&quot;,&quot;代理评估器&quot;,&quot;电商推荐&quot;]"><strong><a href="do_not_wait_learning_re_ranking_model_without_user_feedback_at_serving_time/">Do Not Wait: Learning Re-Ranking Model Without User Feedback At Serving Time in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用代理评估器在服务时刻为每个请求在线生成瞬态参数修正，无需等待延迟反馈即提升重排效果，离线 NDCG@10 +0.65%、线上购买数 +2.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;统一上下文建模&quot;,&quot;特征交叉&quot;,&quot;个性化搜索&quot;]"><strong><a href="joint_modeling_of_search_and_recommendations_via_an_unified_contextual_reco/">Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Netflix 以 {user, query, country, source entity, task} 为共享上下文训练单一深度排序器，插补缺失上下文并做特征交叉，统一服务搜索与推荐，个性化后离线指标 +7%/+10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回&quot;,&quot;Pareto 前沿近似&quot;,&quot;会话推荐&quot;]"><strong><a href="pareto_front_approximation_for_multi_objective_session_based_recommender_sy/">Pareto Front Approximation for Multi-Objective Session-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将偏好向量条件化注入 TRON 会话 Transformer，配合 KL 非均匀性正则，单模型覆盖 CTR–CVR 全 Pareto 前沿并经线上 A/B 验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;跨内容类型排序&quot;,&quot;曝光预算与多样性&quot;,&quot;线上 A/B 测试&quot;]"><strong><a href="ranking_across_different_content_types_the_robust_beauty_of_multinomial_ble/">Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式混合(MB)在打分后按内容类型采样组列表，实现可解释、可重训稳定的跨内容类型曝光预算，Amazon Music 线上 A/B 播客时长 +18.82% 且整体互动 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Hybrid Targets&quot;,&quot;梯度调节&quot;,&quot;标签嵌入&quot;]"><strong><a href="touch_the_core_exploring_task_dependence_among_hybrid_targets_for_recommend/">Touch the Core: Exploring Task Dependence Among Hybrid Targets for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次提出混合目标（离散转化+连续核心值）多任务学习，用标签嵌入与信息融合显式传递依赖，并设计梯度方向/量级调节策略，离线三数据集与线上 A/B 均显著提升。</p>
</dd>
</dl>
