---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**ICLR / NeurIPS / ICML (RecSys) 2025** · **3** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual MDP" aria-pressed="false">Contextual MDP<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Knowledge Tracing" aria-pressed="false">Knowledge Tracing<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Model-Based Value Estimation" aria-pressed="false">Model-Based Value Estimation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两阶段最大似然估计" aria-pressed="false">两阶段最大似然估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="出价策略" aria-pressed="false">出价策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分位数回归" aria-pressed="false">分位数回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="延迟奖励" aria-pressed="false">延迟奖励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="心理奖励模型" aria-pressed="false">心理奖励模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端策略优化" aria-pressed="false">端到端策略优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义表征学习" aria-pressed="false">语义表征学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆强化学习" aria-pressed="false">逆强化学习<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;arXiv 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;出价策略&quot;,&quot;Contextual MDP&quot;,&quot;延迟奖励&quot;,&quot;两阶段最大似然估计&quot;]"><strong><a href="learning_personalized_ad_impact_via_contextual_reinforcement_learning_under/">Learning Personalized Ad Impact via Contextual Reinforcement Learning under Delayed Rewards</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将自动出价建模为带延迟泊松奖励的CMDP，提出两阶段MLE与数据分割，实现个性化出价近最优遗憾界。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;逆强化学习&quot;,&quot;心理奖励模型&quot;,&quot;分位数回归&quot;]"><strong><a href="mtrec_learning_to_align_with_user_preferences_via_mental_reward_models/">MTRec: Learning to Align with User Preferences via Mental Reward Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分布化逆强化学习(QR-IQL)从行为轨迹反推用户心理奖励分布，作为辅助对齐信号注入CTR/RL推荐模型，离线AUC/NCIS全面提升，工业短视频A/B观看时长+7%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR/NeurIPS/ICML 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端策略优化&quot;,&quot;Knowledge Tracing&quot;,&quot;Model-Based Value Estimation&quot;,&quot;语义表征学习&quot;]"><strong><a href="personalized_exercise_recommendation_with_semantically_grounded_knowledge_t/">Personalized Exercise Recommendation with Semantically-Grounded Knowledge Tracing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>ExRec 用 LLM 自动标注知识点+对比学习语义表征+KT 校准，并以 KT 模型做 Model-Based Value Estimation 稳定 Q-learning，在四个教育推荐任务上显著提升知识状态增益。</p>
</dd>
</dl>
  </div>
</div>
