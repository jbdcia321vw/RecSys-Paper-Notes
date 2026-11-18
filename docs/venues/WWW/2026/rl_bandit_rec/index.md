# 🎮 强化学习与动态探索推荐

**WWW (TheWebConf) 2026** · **3** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2025 (4)](../../2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (2)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/)

🔥 **高频主题：** WWW 2026 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 微调" aria-pressed="false">GRPO 微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 上下文学习" aria-pressed="false">LLM 上下文学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文赌博机" aria-pressed="false">上下文赌博机<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="出价与预算分配" aria-pressed="false">出价与预算分配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双阶段智能体" aria-pressed="false">双阶段智能体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="库存约束" aria-pressed="false">库存约束<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散生成规划" aria-pressed="false">扩散生成规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线策略学习" aria-pressed="false">离线策略学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨渠道预算分配" aria-pressed="false">跨渠道预算分配<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;粗排&quot;,&quot;扩散生成规划&quot;,&quot;自动出价&quot;,&quot;跨渠道预算分配&quot;]"><strong><a href="ahbid_an_adaptable_hierarchical_bidding_framework_for_cross_channel_adverti/">AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用扩散模型生成跨渠道预算/约束目标轨迹、结合约束执行与实时控制出价，离线与线上 A/B 收益提升 13.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;出价与预算分配&quot;,&quot;LLM 上下文学习&quot;,&quot;GRPO 微调&quot;,&quot;双阶段智能体&quot;]"><strong><a href="dara_few_shot_budget_allocation_in_online_advertising_via_in_context_decisi/">DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双阶段 LLM 智能体分离少样本规划与精细优化，配合动态参考策略 GRPO-Adaptive 微调，在真实与合成广告环境中显著降低边际 ROI 方差。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;库存约束&quot;,&quot;离线策略学习&quot;,&quot;上下文赌博机&quot;]"><strong><a href="off_policy_learning_with_limited_supply/">Off-Policy Learning with Limited Supply</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次形式化库存受限下的离线策略学习，提出按“相对奖励差”分配稀缺物品的 OPLS，在合成与 KuaiRec 上稳定超越贪心 OPL。</p>
</dd>
</dl>
