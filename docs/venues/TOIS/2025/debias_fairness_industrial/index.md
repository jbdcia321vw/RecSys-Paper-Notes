# 🛡️ 去偏公平、可信隐私与工业架构

**ACM TOIS 2025** · **4** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ACM TOIS 2026 (1)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** TOIS ×4 · 去偏公平与可信推荐 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平表征学习" aria-pressed="false">公平表征学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="函数空间正则" aria-pressed="false">函数空间正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布距离MMD" aria-pressed="false">分布距离MMD<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="后处理/模型修正" aria-pressed="false">后处理/模型修正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体强化学习" aria-pressed="false">多智能体强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性遗忘" aria-pressed="false">属性遗忘<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;嵌入表压缩&quot;,&quot;轻量化推荐基准&quot;,&quot;端侧部署&quot;]"><strong><a href="a_thorough_performance_benchmarking_on_lightweight_embedding_based_recommen/">A Thorough Performance Benchmarking on Lightweight Embedding-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一协议横评 7 类轻量嵌入压缩方法，提出 MagPrune 强基线，并首次量化跨任务迁移性与树莓派端侧效率瓶颈。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果推断&quot;,&quot;多模态推荐&quot;,&quot;公平表征学习&quot;]"><strong><a href="causality_inspired_fair_representation_learning_for_multimodal_recommendati/">Causality-Inspired Fair Representation Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>因果启发的模态解耦与公平/不公平关系挖掘，实现多模态推荐的反事实公平，公平性显著优于 SOTA 且精度可比。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;多智能体强化学习&quot;,&quot;黑盒对抗攻击&quot;,&quot;社交推荐&quot;]"><strong><a href="multi_agent_attacks_for_black_box_social_recommendations/">Multi-agent Attacks for Black-box Social Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多智能体强化学习协同生成冷启动物品画像与跨社区社交画像，对黑盒社交推荐发起无目标投毒/规避攻击，全库排序下 NDCG@10 相对下降约 19%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;后处理/模型修正&quot;,&quot;属性遗忘&quot;,&quot;分布距离MMD&quot;,&quot;函数空间正则&quot;]"><strong><a href="post_training_attribute_unlearning_in_recommender_systems/">Post-Training Attribute Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出锚点分布MMD与函数空间排序正则的PoT-AU双损失，多类属性遗忘复杂度降至O(T)且推荐性能近无损。</p>
</dd>
</dl>
