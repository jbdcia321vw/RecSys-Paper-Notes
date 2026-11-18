# 🎮 强化学习与动态探索推荐

**CIKM 2026** · **2** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [CIKM 2025 (5)](../../2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (4)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/rl_bandit_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Off-Policy Learning" aria-pressed="false">Off-Policy Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Top-k Policy Decomposition" aria-pressed="false">Top-k Policy Decomposition<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免模拟器部署推理" aria-pressed="false">免模拟器部署推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="进化搜索专家合成" aria-pressed="false">进化搜索专家合成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称 Actor-Critic" aria-pressed="false">非对称 Actor-Critic<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Off-Policy Learning&quot;,&quot;Top-k Policy Decomposition&quot;]"><strong><a href="efficient_offline_learning_of_ranking_policies_via_top_k_policy_decompositi/">Efficient Offline Learning of Ranking Policies via Top-$k$ Policy Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将排序策略分解为 top-$k$ 策略梯度与 bottom 回归两段，在 CPC 条件下无偏且大幅降方差，小样本与大规模排序空间下显著优于 IPS/DR/RIPS/IIPS。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;进化搜索专家合成&quot;,&quot;非对称 Actor-Critic&quot;,&quot;免模拟器部署推理&quot;]"><strong><a href="evol_simulator_guided_evolutionary_expert_synthesis_for_deployment_free_lea/">EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用知识追踪模拟器内进化搜索合成专家路径，蒸馏为仅凭初始掌握度盲规划的非对称 Actor-Critic，实现免模拟器部署的学习路径推荐。</p>
</dd>
</dl>
