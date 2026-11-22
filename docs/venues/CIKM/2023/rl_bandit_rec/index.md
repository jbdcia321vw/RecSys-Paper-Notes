---
hide:
  - toc
---

# 🎮 强化学习与动态探索推荐

**CIKM 2023** · **2** 篇论文解读

> **方向说明**：Contextual Bandit 探索与利用 (E&E)、Offline RL 长期留存与生命周期价值 (LTV) 策略优化

📌 **同领域跨会议浏览：** [CIKM 2021 (1)](../../2021/rl_bandit_rec/) · [CIKM 2024 (2)](../../2024/rl_bandit_rec/) · [CIKM 2025 (5)](../../2025/rl_bandit_rec/) · [CIKM 2026 (3)](../../2026/rl_bandit_rec/) · [AAAI (RecSys Track) 2021 (6)](../../../AAAI/2021/rl_bandit_rec/) · [AAAI (RecSys Track) 2022 (3)](../../../AAAI/2022/rl_bandit_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/rl_bandit_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/rl_bandit_rec/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/rl_bandit_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/rl_bandit_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/rl_bandit_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/rl_bandit_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/rl_bandit_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/rl_bandit_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/rl_bandit_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/rl_bandit_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (5)](../../../ICLR_NeurIPS_ICML/2022/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (3)](../../../ICLR_NeurIPS_ICML/2023/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (3)](../../../ICLR_NeurIPS_ICML/2025/rl_bandit_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/rl_bandit_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/rl_bandit_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/rl_bandit_rec/) · [KDD 2021 (2)](../../../KDD/2021/rl_bandit_rec/) · [KDD 2022 (3)](../../../KDD/2022/rl_bandit_rec/) · [KDD 2023 (1)](../../../KDD/2023/rl_bandit_rec/) · [KDD 2024 (3)](../../../KDD/2024/rl_bandit_rec/) · [KDD 2025 (5)](../../../KDD/2025/rl_bandit_rec/) · [KDD 2026 (5)](../../../KDD/2026/rl_bandit_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/rl_bandit_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/rl_bandit_rec/) · [RecSys 2023 (4)](../../../RecSys/2023/rl_bandit_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/rl_bandit_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/rl_bandit_rec/) · [RecSys 2026 (1)](../../../RecSys/2026/rl_bandit_rec/) · [SIGIR 2022 (4)](../../../SIGIR/2022/rl_bandit_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/rl_bandit_rec/) · [SIGIR 2024 (4)](../../../SIGIR/2024/rl_bandit_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/rl_bandit_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/rl_bandit_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/rl_bandit_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/rl_bandit_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/rl_bandit_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/rl_bandit_rec/) · [WSDM 2022 (2)](../../../WSDM/2022/rl_bandit_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/rl_bandit_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2021 (5)](../../../WWW/2021/rl_bandit_rec/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/rl_bandit_rec/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/rl_bandit_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/rl_bandit_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/rl_bandit_rec/)

🔥 **高频主题：** Contextual Bandit ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;嵌入维度搜索&quot;,&quot;流式推荐&quot;]"><strong><a href="dynamic_embedding_size_search_with_minimum_regret_for_streaming_recommender/">Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将流式推荐中的动态嵌入维度搜索建模为非平稳 LinUCB 赌博机，用频率+兴趣多样性指标做上下文，取得亚线性遗憾与更低内存/时延。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/在线探索&quot;,&quot;Epistemic Neural Network&quot;,&quot;Thompson Sampling&quot;,&quot;Contextual Bandit&quot;]"><strong><a href="scalable_neural_contextual_bandit_for_recommender_systems/">Scalable Neural Contextual Bandit for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 ENR：用 EpiNet 式认知不确定性头与 Thompson 采样替代神经老虎机的高代价矩阵求逆，在 MIND/KuaiRec 在线评测中 CTR +9%、评分 +6%，交互节省 29–56%。</p>
</dd>
</dl>
