---
hide:
  - toc
---

# 🏛️ DASFAA 2026 推荐系统论文解读

3 篇 DASFAA 2026 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义规划" aria-pressed="false">LLM 语义规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好优化" aria-pressed="false">偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层强化学习" aria-pressed="false">分层强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态负采样" aria-pressed="false">动态负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解释一致性" aria-pressed="false">解释一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程强化学习" aria-pressed="false">课程强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过滤气泡缓解" aria-pressed="false">过滤气泡缓解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;偏好优化&quot;,&quot;动态负采样&quot;]"><strong><a href="llm_rec/dynamicpo_dynamic_preference_optimization_for_recommendation/">DynamicPO: Dynamic Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示多负样本 DPO 的偏好优化坍缩现象，提出动态边界负样本选择与双边际动态 β 调整，即插即用提升 LLM 推荐 HitRatio@1。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;课程强化学习&quot;,&quot;解释一致性&quot;]"><strong><a href="agent_conversational_rec/curr_rlcer_curriculum_reinforcement_learning_for_coherence_explainable_reco/">Curr-RLCER: Curriculum Reinforcement Learning For Coherence Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用三阶段课程强化学习（CTR→评分→解释生成）训练 LLM，并设计一致性奖励与评估方案，使解释与评分双向对齐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;分层强化学习&quot;,&quot;LLM 语义规划&quot;,&quot;过滤气泡缓解&quot;]"><strong><a href="rl_bandit_rec/llm_enhanced_reinforcement_learning_for_long_term_user_satisfaction_in_inte/">LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>LLM 做类别级语义规划、PPO 做物品级策略，KuaiSim 全会话仿真下 Tint/Rcum 显著超越 DNaIR 等 SOTA。</p>
</dd>
</dl>
</section>
</div>
