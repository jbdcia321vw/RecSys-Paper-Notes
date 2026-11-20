---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**WWW (TheWebConf) 2023** · **6** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2024 (4)](../../2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2022 (3)](../../../AAAI/2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/rl_bandit_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../../ICLR_NeurIPS_ICML/2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (3)](../../../ICLR_NeurIPS_ICML/2023/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2022 (4)](../../../SIGIR/2022/rl_bandit_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/)

🔥 **高频主题：** WWW 2023 ×6 · Actor-Critic ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Actor-Critic" aria-pressed="false">Actor-Critic<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/列表推荐&quot;,&quot;Latent Action Space&quot;,&quot;Actor-Critic&quot;,&quot;动作空间对齐&quot;]"><strong><a href="exploration_and_regularization_of_the_latent_action_space_in_recommendation/">Exploration and Regularization of the Latent Action Space in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>HAC将列表推荐分解为潜在超动作与效应动作，共享critic并以逆池化对齐和BCE监督稳定RL，在三个模拟在线环境取得约1%–6%的长期收益提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/曝光分配策略&quot;,&quot;Contextual Bandit&quot;,&quot;Exposure Constraints&quot;]"><strong><a href="learning_with_exposure_constraints_in_recommendation_systems/">Learning with Exposure Constraints in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将提供者最低曝光约束建模为分阶段上下文老虎机，提出EES/DP*/LCB*，达到˜O(T^{2/3}) regret并给出Ω(T^{2/3})下界。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/混排&quot;,&quot;Thompson Sampling&quot;,&quot;Personalized PageRank&quot;,&quot;探索-利用平衡&quot;]"><strong><a href="pie_personalized_interest_exploration_for_large_scale_recommender_systems/">PIE: Personalized Interest Exploration for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 PPR 构建用户-创作者探索空间 + Thompson 采样在线探索 + 概率化混排，在 Facebook Watch 十亿级用户四周 A/B 中 SCC +3.50% 而互动指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;长期留存优化&quot;,&quot;无限视界 MDP&quot;,&quot;Actor-Critic&quot;]"><strong><a href="reinforcing_user_retention_in_a_billion_scale_short_video_recommender_syste/">Reinforcing User Retention in a Billion Scale Short Video Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频留存优化建模为无限视界 request-based MDP，以最小化累积回归间隔为目标，用奖励归一化、活跃度分组策略与软行为正则改造 Actor-Critic，快手全量 A/B 中 DAU +0.2%、次日留存 +0.053%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Constrained MDP&quot;,&quot;Actor-Critic&quot;,&quot;多目标软约束&quot;]"><strong><a href="two_stage_constrained_actor_critic_for_short_video_recommendation/">Two-Stage Constrained Actor-Critic for Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频推荐建模为 CMDP，用多 Critic 分离评估稠密/稀疏奖励，两阶段软 KL 约束 Actor，离线 WatchTime +2.23%、线上 +0.379%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;Decision Transformer&quot;,&quot;User Retention&quot;,&quot;Contrastive Learning&quot;]"><strong><a href="user_retention_oriented_recommendation_with_decision_transformer/">User Retention-oriented Recommendation with Decision Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将留存优化转为决策Transformer自回归生成，以奖励提示和加权对比学习缓解离线RL不稳定，在IQiYi/ML-1m提升留存与准确率。</p>
</dd>
</dl>
