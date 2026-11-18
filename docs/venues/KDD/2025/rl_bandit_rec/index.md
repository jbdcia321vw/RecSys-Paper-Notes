# 🎮 强化学习与动态探索推荐

**KDD 2025** · **5** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [KDD 2026 (4)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Ad Load Optimization" aria-pressed="false">Ad Load Optimization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Clustering of Neural Bandits" aria-pressed="false">Clustering of Neural Bandits<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="In-Context RL" aria-pressed="false">In-Context RL<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Loss of Plasticity" aria-pressed="false">Loss of Plasticity<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Offline RL" aria-pressed="false">Offline RL<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Robust MDP" aria-pressed="false">Robust MDP<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Selective Reinitialization" aria-pressed="false">Selective Reinitialization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/策略决策" aria-pressed="false">召回/策略决策<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果状态编码" aria-pressed="false">因果状态编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子模优化" aria-pressed="false">子模优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联名营销推荐" aria-pressed="false">联名营销推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预算分配" aria-pressed="false">预算分配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预算规划" aria-pressed="false">预算规划<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/策略决策&quot;,&quot;Contextual Bandit&quot;,&quot;子模优化&quot;,&quot;联名营销推荐&quot;]"><strong><a href="a_unified_online_offline_framework_for_co_branding_campaign_recommendations/">A Unified Online-Offline Framework for Co-Branding Campaign Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个联名营销推荐系统性研究：二分图建模+方差感知CUCB在线学习+部分枚举离线预算分配，理论保证近优regret与1-1/e近似，真实数据收益提升12%-73%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;粗排&quot;,&quot;自动出价&quot;,&quot;预算规划&quot;,&quot;In-Context RL&quot;]"><strong><a href="an_adaptable_budget_planner_for_enhancing_budget_constrained_auto_bidding_i/">An Adaptable Budget Planner for Enhancing Budget-Constrained Auto-Bidding in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将竞价周期切分为多阶段，用 in-context RL 逐 episode 调整预算分配，few-shot 适配新广告主，仿真与线上 A/B 均提升累计价值。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;离线强化学习&quot;,&quot;因果状态编码&quot;,&quot;预算分配&quot;]"><strong><a href="multi_task_offline_reinforcement_learning_for_online_advertising_in_recomme/">Multi-task Offline Reinforcement Learning for Online Advertising in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用因果状态编码器+因果注意力+多任务解码器把广告渠道推荐与预算分配统一进离线RL，KuaiRand/Criteo 上 F1 提升约 7.5%/16.2%，线上 CTR +0.08。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Clustering of Neural Bandits&quot;,&quot;Loss of Plasticity&quot;,&quot;Selective Reinitialization&quot;]"><strong><a href="revisiting_clustering_of_neural_bandits_selective_reinitialization_for_miti/">Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>针对神经聚类 Bandit 的可塑性丧失问题，提出选择性重初始化框架 SeRe，配合自适应变点检测，在六个推荐数据集上最多降低 12.82% 累积遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Offline RL&quot;,&quot;Robust MDP&quot;,&quot;Ad Load Optimization&quot;]"><strong><a href="session_level_dynamic_ad_load_optimization_using_offline_robust_reinforceme/">Session-Level Dynamic Ad Load Optimization using Offline Robust Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将会话级广告加载建模为鲁棒MDP，用IPM不确定集导出免额外开销的鲁棒对决DQN，离线AUCC提升80%+，线上A/B双位数增益。</p>
</dd>
</dl>
