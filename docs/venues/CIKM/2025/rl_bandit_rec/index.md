# 🎮 强化学习与动态探索推荐

**CIKM 2025** · **5** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [CIKM 2026 (2)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decision Transformer" aria-pressed="false">Decision Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Expectile 回归" aria-pressed="false">Expectile 回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 先验" aria-pressed="false">LLM 先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="专家轨迹引导" aria-pressed="false">专家轨迹引导<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="关键词剪枝" aria-pressed="false">关键词剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/投放策略" aria-pressed="false">召回/投放策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散策略" aria-pressed="false">扩散策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散规划器" aria-pressed="false">扩散规划器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模仿学习" aria-pressed="false">模仿学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户留存" aria-pressed="false">用户留存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="能量引导采样" aria-pressed="false">能量引导采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为克隆" aria-pressed="false">行为克隆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;扩散策略&quot;,&quot;离线强化学习&quot;,&quot;能量引导采样&quot;]"><strong><a href="energy_guided_diffusion_sampling_for_long_term_user_behavior_prediction_in_/">Energy-Guided Diffusion Sampling for Long-Term User Behavior Prediction in Reinforcement Learning-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散策略替代行为克隆以提升离线RL4RS策略表达能力，并以Q值能量引导采样降低随机性，六数据集与仿真环境全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;扩散规划器&quot;,&quot;自动出价&quot;,&quot;专家轨迹引导&quot;]"><strong><a href="expert_guided_diffusion_planner_for_auto_bidding/">Expert-Guided Diffusion Planner for Auto-bidding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用专家轨迹引导条件扩散规划器生成出价序列，配合混合强制训练与跳步采样，线上转化+11.29%、收入+12.36%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/投放策略&quot;,&quot;LLM Agent&quot;,&quot;Contextual Bandit&quot;,&quot;关键词剪枝&quot;]"><strong><a href="kp_agent_keyword_pruning_in_sponsored_search_advertising_via_llm_powered_co/">KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将关键词剪枝建模为上下文赌博机，用 LLM Agent 生成代码片段执行剪枝，美团真实数据上累计利润最高提升 49.28%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;Decision Transformer&quot;,&quot;Expectile 回归&quot;,&quot;LLM 先验&quot;]"><strong><a href="maximum_in_support_return_modeling_for_dynamic_recommendation_with_language/">Maximum In-Support Return Modeling for Dynamic Recommendation with Language Model Prior</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将轨迹拼接从训练期移至动作推理期，用 expectile 回归估计最大在支撑回报并自适应截断历史，配合 LLM 先验初始化 DT，在 5 个离线数据集与 VirtualTB 仿真上全面超越 EDT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;模仿学习&quot;,&quot;用户留存&quot;,&quot;行为克隆&quot;]"><strong><a href="stratified_expert_cloning_for_retention_aware_recommendation_at_scale/">Stratified Expert Cloning for Retention-Aware Recommendation at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层专家行为克隆替代 RL 探索，配合动作熵正则与自适应专家选择，在快手双平台 A/B 中提升活跃天数 0.098%/0.122%。</p>
</dd>
</dl>
