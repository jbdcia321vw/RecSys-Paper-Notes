---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**WWW (TheWebConf) 2021** · **5** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (6)](../../2023/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (4)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2021 (6)](../../../AAAI/2021/rl_bandit_rec/) · [AAAI (RecSys Track) 2022 (3)](../../../AAAI/2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [CIKM 2021 (1)](../../../CIKM/2021/rl_bandit_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/rl_bandit_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../../CIKM/2025/rl_bandit_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/rl_bandit_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../../ICLR_NeurIPS_ICML/2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (3)](../../../ICLR_NeurIPS_ICML/2023/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2021 (2)](../../../KDD/2021/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (5)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2022 (4)](../../../SIGIR/2022/rl_bandit_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/)

🔥 **高频主题：** WWW 2021 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/在线创意选择&quot;,&quot;Visual-aware Ranking&quot;,&quot;Hybrid Bandit&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="a_hybrid_bandit_model_with_visual_priors_for_creative_ranking_in_display_ad/">A Hybrid Bandit Model with Visual Priors for Creative Ranking in Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>视觉list-wise先验热启动商品/创意双线性混合Bandit，在CreativeRanking回放协议下sCTR达3.320%、Regret降至78.11%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;深度上下文老虎机&quot;,&quot;引导式 Bootstrap&quot;,&quot;不确定性探索&quot;]"><strong><a href="guideboot_guided_bootstrap_for_deep_contextual_bandits/">GuideBoot: Guided Bootstrap for Deep Contextual Bandits in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>WWW 2021｜强化学习与探索推荐｜以预测不确定性反比生成伪样本引导深度 bootstrap 探索，五大模型袋随机决策，广告仿真平均奖励最高提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/粗排&quot;,&quot;Policy Gradient&quot;,&quot;下游自适应剪枝&quot;,&quot;奖励塑形&quot;]"><strong><a href="optimizing_ad_pruning_of_sponsored_search_with_reinforcement_learning/">Optimizing AD Pruning of Sponsored Search with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将下游精排/拍卖视为黑盒环境，用策略梯度的序列式选品智能体替代 eCPM 规则剪枝，百度在线 A/B 取得 CPM +1.95%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Pareto 优化&quot;,&quot;多目标确定性策略梯度&quot;]"><strong><a href="reinforcement_recommendation_with_user_multi_aspect_preference/">Reinforcement Recommendation with User Multi-aspect Preference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将单值 reward 扩展为多维度方面级 reward 向量，用 Pareto-DPG 多 critic 单 actor 学得帕累托有效策略，并理论分析梯度偏差上界，三个数据集 F1/NDCG 相对最优基线提升 4.4%–9.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线竞价与预算分配&quot;,&quot;Bandits with Knapsacks&quot;,&quot;跨平台预算优化&quot;]"><strong><a href="stochastic_bandits_for_multi_platform_budget_optimization_in_online_adverti/">Stochastic Bandits for Multi-platform Budget Optimization in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将跨平台预算分配建模为带背包的随机赌博机，设计乘性权重原始-对偶算法实现 $O(OPT\sqrt{mn/B})$ 级 regret 并验证预算均匀耗尽。</p>
</dd>
</dl>
