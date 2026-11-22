---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**AAAI (RecSys Track) 2021** · **6** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2022 (3)](../../2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2021 (1)](../../../CIKM/2021/rl_bandit_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/rl_bandit_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../../ICLR_NeurIPS_ICML/2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (3)](../../../ICLR_NeurIPS_ICML/2023/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2021 (2)](../../../KDD/2021/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (5)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2022 (4)](../../../SIGIR/2022/rl_bandit_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2021 (5)](../../../WWW/2021/rl_bandit_rec/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** AAAI ×4 · AAAI 2021 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/端到端策略学习&quot;,&quot;Offline RL&quot;,&quot;外推误差约束&quot;,&quot;离散随机策略&quot;]"><strong><a href="a_general_offline_reinforcement_learning_framework_for_interactive_recommen/">A General Offline Reinforcement Learning Framework for Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将交互推荐写成概率生成模型的变分推断，用离散随机 Actor-Critic 加五种行为策略约束抑制外推误差，全库排序下 NDCG@10 提升 13.8%~25.2%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Bandit&quot;,&quot;Dispersion Function&quot;,&quot;多样性推荐&quot;]"><strong><a href="a_hybrid_bandit_framework_for_diversified_recommendation/">A Hybrid Bandit Framework for Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用模块函数刻画相关性、散度函数刻画多样性，在线联合学习用户对二者的个性化偏好，贪心+UCB 求解并给出无间隙 regret 上界。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;广告投放实验/创意分配&quot;,&quot;Contextual Thompson Sampling&quot;,&quot;实验即服务 (EaaS)&quot;,&quot;互斥子人群拆分 (DA)&quot;]"><strong><a href="comparison_lift_bandit_based_experimentation_system_for_online_advertising/">Comparison Lift: Bandit-based Experimentation System for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>京东广告 EaaS 系统：将重叠目标人群拆为互斥子人群，在子人群上跑 Contextual Thompson Sampling，再聚合回人群层做创意-人群最优组合推断；线上 1,547 次实验平均提升测试期点击 +27%、广告主 CTR +46%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;DQN 多子动作联合决策&quot;,&quot;Dueling 价值-优势分解&quot;,&quot;广告-推荐列表混排&quot;]"><strong><a href="dear_deep_reinforcement_learning_for_online_advertising_impression_in_recom/">DEAR: Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用单一 Dueling-DQN 同时决策「是否插广告/插哪条/插哪个槽位」，把前向复杂度从 O(|A|·L) 降到 O(|A|)，在抖音 100 万会话上累积奖励较 HDQN 提升 6.71%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;Thompson Sampling&quot;,&quot;动态规划&quot;]"><strong><a href="efficient_optimal_selection_for_composited_advertising_creatives_with_tree_/">Efficient Optimal Selection for Composited Advertising Creatives with Tree Structure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 ingredient 树 + 元素二部图参数化创意 CTR，动态规划精确搜索最优创意，并将 Thompson sampling 下沉到参数后验实现高效探索。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/策略层端到端&quot;,&quot;世界模型 (World Model)&quot;,&quot;通用价值函数 (UVF)&quot;,&quot;解耦价值估计&quot;]"><strong><a href="reinforcement_learning_with_a_disentangled_universal_value_function_for_ite/">Reinforcement Learning with a Disentangled Universal Value Function for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以目标导向世界模型解耦环境动态与高方差奖励，构建通用价值函数 $Q=g^{\top}M(s,a,g)$，在三个仿真环境与网易线上场景提升累积点击 3.7%–10.7%。</p>
</dd>
</dl>
