---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**SIGIR 2023** · **4** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [SIGIR 2022 (4)](../../2022/rl_bandit_rec/) · [SIGIR 2024 (4)](../../2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2021 (6)](../../../AAAI/2021/rl_bandit_rec/) · [AAAI (RecSys Track) 2022 (3)](../../../AAAI/2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2021 (1)](../../../CIKM/2021/rl_bandit_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/rl_bandit_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../../ICLR_NeurIPS_ICML/2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (3)](../../../ICLR_NeurIPS_ICML/2023/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2021 (2)](../../../KDD/2021/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (5)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2021 (5)](../../../WWW/2021/rl_bandit_rec/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** 离线强化学习 ×2 · 端到端生成式 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;离线强化学习&quot;,&quot;马太效应&quot;,&quot;熵惩罚&quot;]"><strong><a href="alleviating_matthew_effect_of_offline_reinforcement_learning_in_interactive/">Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>指出离线 RL 的保守性惩罚会放大马太效应，提出熵惩罚项 DORL，在 KuaiRec/KuaiRand 交互环境中累计收益提升 79.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;离线强化学习&quot;,&quot;Decision Transformer&quot;,&quot;因果奖励估计&quot;]"><strong><a href="causal_decision_transformer_for_recommender_systems_via_offline_reinforceme/">Causal Decision Transformer for Recommender Systems via Offline Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以因果决策 Transformer 将离线推荐改写为轨迹序列建模，绕开奖励函数手工设计，在 6 个真实数据集与 VirtualTB 模拟器上取得召回/CTR 领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;状态增广&quot;,&quot;对比学习&quot;]"><strong><a href="contrastive_state_augmentations_for_reinforcement_learning_based_recommende/">Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用四种局部状态增广平滑离线 RL 的 Q 值面，并对增广视图施加对比损失，在三个数据集与模拟在线环境中稳定提升序列推荐收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;策略学习/端到端排序&quot;,&quot;真实工业数据集&quot;,&quot;离线策略评估&quot;,&quot;Slate 推荐&quot;]"><strong><a href="rl4rs_a_real_world_dataset_for_reinforcement_learning_based_recommender_sys/">RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>发布首个开源真实工业级 RL 推荐数据集（Slate/SeqSlate），配套四层评估框架与数据理解工具；PPO 线上 GMV 较 DIEN 提升 23.9%。</p>
</dd>
</dl>
