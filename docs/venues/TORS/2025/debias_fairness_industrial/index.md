---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**ACM TORS 2025** · **7** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ACM TORS 2026 (3)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** TORS ×7 · 去偏公平与可信推荐 ×7

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全链路（召回/排序/解释）" aria-pressed="false">全链路（召回/排序/解释）<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态重排" aria-pressed="false">动态重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单品公平" aria-pressed="false">单品公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排+重排" aria-pressed="false">召回+精排+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/重排/端到端" aria-pressed="false">召回/重排/端到端<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可信推荐 TRS" aria-pressed="false">可信推荐 TRS<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积溢出影响" aria-pressed="false">图卷积溢出影响<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体公平" aria-pressed="false">多智能体公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="度量归一化" aria-pressed="false">度量归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响函数" aria-pressed="false">影响函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐遗忘" aria-pressed="false">推荐遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据投毒攻击" aria-pressed="false">数据投毒攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光公平" aria-pressed="false">曝光公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光感知推荐" aria-pressed="false">曝光感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型更新/遗忘" aria-pressed="false">模型更新/遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户簇定向攻击" aria-pressed="false">用户簇定向攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵分解" aria-pressed="false">矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社会选择" aria-pressed="false">社会选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经网络压缩" aria-pressed="false">神经网络压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="系统性文献综述" aria-pressed="false">系统性文献综述<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测与度量" aria-pressed="false">评测与度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视角权衡" aria-pressed="false">跨视角权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护" aria-pressed="false">隐私保护<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;联邦学习&quot;,&quot;神经网络压缩&quot;,&quot;隐私保护&quot;]"><strong><a href="a_privacy_preserving_system_for_movie_recommendations_using_federated_learn/">A Privacy Preserving System for Movie Recommendations Using Federated Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端到端联邦电影推荐系统，提出 FedQ 队列式联邦训练缓解非独立同分布与小样本问题，结合 NNC 压缩降低通信开销，在 MovieLens 25M 上验证 16 万客户端规模。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;全链路（召回/排序/解释）&quot;,&quot;可信推荐 TRS&quot;,&quot;跨视角权衡&quot;]"><strong><a href="a_survey_on_trustworthy_recommender_systems/">A Survey on Trustworthy Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理可解释、公平、隐私、鲁棒、可控五大可信推荐维度及其交叉关系，提出跨视角关系矩阵与评估局限反思。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;社会选择&quot;,&quot;多智能体公平&quot;,&quot;动态重排&quot;]"><strong><a href="dynamic_fairness_aware_recommendation_through_multi_agent_social_choice/">Dynamic Fairness-aware Recommendation through Multi-agent Social Choice</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多利益相关方公平推荐建模为「分配+聚合」两阶段社会选择问题，SCRUF-D 框架动态调度多个公平性智能体，在真实与合成数据上实现多维权衡。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;数据投毒攻击&quot;,&quot;矩阵分解&quot;,&quot;用户簇定向攻击&quot;]"><strong><a href="evaluating_impact_of_user_cluster_targeted_attacks_in_matrix_factorisation_/">Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统分析 MF 推荐中面向用户簇的投毒攻击，揭示 V 矩阵是攻击传播主通道，簇内真实评分量决定脆弱性。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测与度量&quot;,&quot;单品公平&quot;,&quot;曝光公平&quot;,&quot;度量归一化&quot;]"><strong><a href="evaluation_measures_of_individual_item_fairness_for_recommender_systems_a_c/">Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统批判 8 种单品曝光公平度量，修正 5 类理论缺陷并给出归一化方案与选用指南。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/重排/端到端&quot;,&quot;曝光感知推荐&quot;,&quot;系统性文献综述&quot;,&quot;曝光偏差&quot;]"><strong><a href="impression_aware_recommender_systems/">Impression-Aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首篇曝光感知推荐系统(IARS)系统性综述，提出统一理论框架与三维分类体系，厘清模型、数据集与评估协议。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型更新/遗忘&quot;,&quot;影响函数&quot;,&quot;推荐遗忘&quot;,&quot;图卷积溢出影响&quot;]"><strong><a href="recommendation_unlearning_via_influence_function/">Recommendation Unlearning via Influence Function</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用影响函数一步更新模型实现推荐遗忘，联合直接损失影响与图卷积溢出影响，剪枝加速后较重训提速 250 倍以上。</p>
</dd>
</dl>
  </div>
</div>
