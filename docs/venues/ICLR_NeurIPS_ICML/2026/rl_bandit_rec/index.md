# 🎮 强化学习与动态探索推荐

**ICLR / NeurIPS / ICML (RecSys) 2026** · **4** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** ICML ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Combinatorial Semi-Bandit" aria-pressed="false">Contextual Combinatorial Semi-Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Estimation-to-Decisions" aria-pressed="false">Estimation-to-Decisions<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文 Bandit" aria-pressed="false">上下文 Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主动推荐" aria-pressed="false">主动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态 LTR" aria-pressed="false">动态 LTR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双边匹配" aria-pressed="false">双边匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对数障碍探索" aria-pressed="false">对数障碍探索<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;留存优化&quot;,&quot;双边匹配&quot;,&quot;动态 LTR&quot;]"><strong><a href="beyond_match_maximization_and_fairness_retention_optimized_two_sided_matchi/">Beyond Match Maximization and Fairness: Retention-Optimized Two-Sided Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双边匹配推荐重构为留存最大化问题，用凹性 Jensen 下界把 NP-hard 排序松弛为 O(NlogN) 的 argsort 打分，在合成与真实约会平台数据上以约 70% 匹配量取得最高留存率。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;激励相容探索&quot;,&quot;上下文 Bandit&quot;,&quot;机制设计&quot;]"><strong><a href="incentivized_exploration_with_stochastic_covariates_a_two_stage_mechanism_d/">Incentivized Exploration with Stochastic Covariates: A Two-Stage Mechanism Design for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向在线随机用户协变量的激励相容探索，提出两阶段 RCB 算法（冷启动补贴 + 逆比例间隙采样），在满足 ε-DBIC 约束下达到 $\tilde{O}(\sqrt{KdT})$ 遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Semi-Bandit&quot;,&quot;Estimation-to-Decisions&quot;,&quot;对数障碍探索&quot;]"><strong><a href="optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;主动推荐&quot;,&quot;策略梯度修正&quot;]"><strong><a href="prorl_effective_reinforcement_learning_for_proactive_recommendation_via_rec/">ProRL: Effective Reinforcement Learning for Proactive Recommendation via Rectified Policy Gradient Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示主动推荐中策略梯度的长度捷径与高方差缺陷，提出逐步奖励中心化与位置特定优势估计，三数据集全面超越 SOTA。</p>
</dd>
</dl>
