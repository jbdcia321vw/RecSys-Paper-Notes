---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**RecSys 2024** · **5** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [RecSys 2025 (2)](../../2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文组合赌博机 (CCB)&quot;,&quot;离线策略评估 (OPE)&quot;,&quot;奖励分解&quot;]"><strong><a href="effective_off_policy_evaluation_and_learning_in_contextual_combinatorial_ba/">Effective Off-Policy Evaluation and Learning in Contextual Combinatorial Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>因子化动作空间 + 主/残差效应分解的 OPCB 估计器，仅对主动作重要性采样、残差回归，KuaiRec 上 MSE 较 IPS 降 92%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/融合权重决策&quot;,&quot;Off-Policy Learning&quot;,&quot;Counterfactual Risk Minimisation&quot;,&quot;Effective Sample Size&quot;,&quot;Multi-Objective Scalarisation&quot;]"><strong><a href="multi_objective_recommendation_via_multivariate_policy_learning/">Multi-Objective Recommendation via Multivariate Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多目标融合权重视为连续动作，用 CRM/SNIPS 学习策略，提出 ESS 校正提升小样本置信区间覆盖率，线上 A/B 显著提升留存与时长。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;排序策略学习（离线评估）&quot;,&quot;上下文赌博机&quot;,&quot;逆倾向得分&quot;,&quot;基线校正&quot;]"><strong><a href="optimal_baseline_corrections_for_off_policy_contextual_bandits/">Optimal Baseline Corrections for Off-Policy Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 IPS/SNIPS/BanditNet/DR 统一为 β-IPS 基线校正，闭式求方差最优 β，在 OBP 与 ZOZOTOWN 上降低 OPE 误差并加速 OPL。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线列表探索&quot;,&quot;Engagement-Utility 错位&quot;,&quot;流行度偏差&quot;]"><strong><a href="the_fault_in_our_recommendations_on_the_perils_of_optimizing_the_measurable/">The Fault in Our Recommendations: On the Perils of Optimizing the Measurable</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用重复选择模型揭示 engagement 与 utility 错位，提出 PEAR 后验探索；δ=0.99 时 utility +53.4% 仅损 engagement 0.12%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;离线评估/策略学习&quot;,&quot;逆倾向得分 IPS&quot;,&quot;反事实方差缩减&quot;]"><strong><a href="text_rm_ope_off_policy_estimation_with_pairs_of_policies/">Δ-OPE: Off-Policy Estimation with Pairs of Policies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 off-policy 评估从单策略估值改为成对策略差值估计，用加性控制变量 Δβ-IPS 显著降方差，仿真与线上 A/B 双验证。</p>
</dd>
</dl>
