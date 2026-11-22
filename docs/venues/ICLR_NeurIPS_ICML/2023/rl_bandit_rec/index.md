---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**ICLR / NeurIPS / ICML (RecSys) 2023** · **3** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../2021/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2021 (6)](../../../AAAI/2021/rl_bandit_rec/) · [AAAI (RecSys Track) 2022 (3)](../../../AAAI/2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2021 (1)](../../../CIKM/2021/rl_bandit_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/rl_bandit_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2021 (2)](../../../KDD/2021/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (5)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2022 (4)](../../../SIGIR/2022/rl_bandit_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2021 (5)](../../../WWW/2021/rl_bandit_rec/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/rl_bandit_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/会话级RL&quot;,&quot;用户模拟器&quot;,&quot;长期留存&quot;]"><strong><a href="kuaisim_a_comprehensive_simulator_for_recommender_systems/">KuaiSim: A Comprehensive Simulator for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出多行为、跨会话推荐模拟器 KuaiSim，统一请求级列表、整会话与留存优化，并在 KuaiRand/ML-1m 给出可复现基准。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线策略&quot;,&quot;Interaction-Grounded Learning&quot;,&quot;潜在奖励学习&quot;,&quot;个性化去偏&quot;]"><strong><a href="personalized_reward_learning_with_interaction_grounded_learning_igl/">Personalized Reward Learning with Interaction-Grounded Learning (IGL)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用逆运动学与3状态潜在奖励学习个性化隐式反馈，在仿真、Facebook新闻与生产图像推荐中提升公平性并抑制负反馈。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;残差策略学习&quot;,&quot;离线强化学习&quot;,&quot;条件变分自编码器&quot;]"><strong><a href="resact_reinforcing_long_term_engagement_in_sequential_recommendation_with_r/">ResAct: Reinforcing Long-term Engagement in Sequential Recommendation with Residual Actor</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 CVAE 重构在线策略并叠加残差 Actor 做近端策略改进，配双重信息论正则，NCIS 离线评估下 Session Length 提升 35%。</p>
</dd>
</dl>
