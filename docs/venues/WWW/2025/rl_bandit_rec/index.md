# 🎮 强化学习与动态探索推荐

**WWW (TheWebConf) 2025** · **4** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (3)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/)

🔥 **高频主题：** WWW 2025 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Combinatorial Semi-Bandit" aria-pressed="false">Contextual Combinatorial Semi-Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TD 分解" aria-pressed="false">TD 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层 UCB" aria-pressed="false">分层 UCB<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动作探索鲁棒性" aria-pressed="false">动作探索鲁棒性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序策略优化" aria-pressed="false">召回/排序策略优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可学习偏置项" aria-pressed="false">可学习偏置项<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="奖励塑形" aria-pressed="false">奖励塑形<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="安全探索" aria-pressed="false">安全探索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散世界模型" aria-pressed="false">扩散世界模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态抽象" aria-pressed="false">状态抽象<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期价值估计" aria-pressed="false">长期价值估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非平稳环境" aria-pressed="false">非平稳环境<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;非平稳环境&quot;,&quot;状态抽象&quot;,&quot;安全探索&quot;]"><strong><a href="auro_reinforcement_learning_for_adaptive_user_retention_optimization_in_rec/">AURO: Reinforcement Learning for Adaptive User Retention Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用值对齐状态抽象+拒绝采样守护探索，解决留存优化RL在非平稳环境下的适应与隐式冷启动难题，KuaiSim与线上A/B全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Semi-Bandit&quot;,&quot;可学习偏置项&quot;,&quot;分层 UCB&quot;]"><strong><a href="interactive_visualization_recommendation_with_hier_sucb/">Interactive Visualization Recommendation with Hier-SUCB</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将可视化推荐建模为带可学习偏置项的分层上下文组合半臂老虎机，把遗憾界从 $O(\sqrt{T\ln^3(nm^2T\ln T)})$降至$O(\sqrt{T\ln^3(m^2T\ln T)})$，在 Plot.ly 上 80/160 轮超越离线 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;离线强化学习&quot;,&quot;扩散世界模型&quot;,&quot;奖励塑形&quot;]"><strong><a href="reward_balancing_revisited_enhancing_offline_reinforcement_learning_for_rec/">Reward Balancing Revisited: Enhancing Offline Reinforcement Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散世界模型显式建模奖励不确定性，并以衰减加权的交互/熵双惩罚平衡多样性与长期收益，在三个真实数据集上刷新离线RL推荐SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/排序策略优化&quot;,&quot;TD 分解&quot;,&quot;长期价值估计&quot;,&quot;动作探索鲁棒性&quot;]"><strong><a href="value_function_decomposition_in_markov_recommendation_process/">Value Function Decomposition in Markov Recommendation Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将标准 TD 损失分解为策略随机与用户环境随机两个子目标，实现更快收敛、更强探索鲁棒性的长期价值估计。</p>
</dd>
</dl>
