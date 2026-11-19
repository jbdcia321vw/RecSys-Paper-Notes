---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**RecSys 2024** · **12** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [RecSys 2025 (9)](../../2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2024 (2)](../../../CIKM/2024/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2024 (4)](../../../ECIR/2024/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDE 2024 (3)](../../../ICDE/2024/debias_fairness_industrial/) · [ICDM 2024 (1)](../../../ICDM/2024/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (7)](../../../ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2024 (2)](../../../IJCAI/2024/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2024 (10)](../../../KDD/2024/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [SIGIR 2024 (5)](../../../SIGIR/2024/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2024 (1)](../../../TKDD/2024/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2024 (7)](../../../TOIS/2024/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2024 (4)](../../../TORS/2024/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2024 (20)](../../../WWW/2024/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×2 · 可解释推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;多任务学习&quot;,&quot;教师偏差泄漏&quot;]"><strong><a href="bridging_the_gap_unpacking_the_hidden_challenges_in_knowledge_distillation_/">Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>针对推荐蒸馏中的数据漂移、教师选型与多学生标签共享三大工业挑战，提出持续更新教师 + 分离 logits 的辅助蒸馏框架，线上互动率最高提升 1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;post-filter&quot;,&quot;multi-sided fairness&quot;,&quot;POI recommendation&quot;]"><strong><a href="capri_fair_integration_of_multi_sided_fairness_in_contextual_poi_recommenda/">CAPRI-FAIR: Integration of Multi-sided Fairness in Contextual POI Recommendation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>免训练后置重排同时注入POI提供者与消费者公平，在Yelp/Gowalla上以Precision与GCE评测权衡长尾曝光和不活跃用户精度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/候选集排序&quot;,&quot;新闻推荐数据集&quot;,&quot;Beyond-Accuracy 评估&quot;,&quot;性别导向偏置&quot;]"><strong><a href="eb_nerd_a_large_scale_dataset_for_news_recommendation/">EB-NeRD: A Large-Scale Dataset for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>发布百万用户丹麦新闻推荐数据集与 RecSys&#x27;24 Challenge 基准，通过打乱位置的曝光日志揭示流行度偏置与性别导向差异。</p>
</dd>
<dt data-paper-tags="[&quot;AI Magazine&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/前端解释层&quot;,&quot;可解释推荐&quot;,&quot;音乐推荐系统&quot;,&quot;知识图谱多跳解释&quot;]"><strong><a href="explainability_in_music_recommender_systems/">Explainability in Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>立场综述：把 RS 解释目标与 XAI 目标对齐，梳理特征/样例/知识图谱三类解释及用户研究与离线指标双轨评估，并剖析工业 MRS 的‘简洁 vs 复杂’矛盾。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平与可信隐私&quot;,&quot;解释层/决策支持&quot;,&quot;多利益相关方推荐&quot;,&quot;职业知识图谱&quot;,&quot;可解释推荐&quot;]"><strong><a href="explainable_multi_stakeholder_job_recommender_systems/">Explainable Multi-Stakeholder Job Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>招聘推荐三方利益相关方解释偏好研究 + 职业知识图谱注意力 GNN（OKRA），发现用户无法区分真实/随机解释，主张转向决策支持式解释。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;后处理(图数据增广)&quot;,&quot;消费者公平性 (Consumer Fairness)&quot;,&quot;图协同过滤增广&quot;,&quot;采样策略&quot;]"><strong><a href="fair_augmentation_for_graph_collaborative_filtering/">Fair Augmentation for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>复现并形式化“公平图增广”后处理范式，推导 3 类新采样策略，在 5 数据集×16 模型上揭示大图高效用模型有效、公平知识不可迁移。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边匹配市场&quot;,&quot;嫉妒自由&quot;,&quot;Nash Social Welfare&quot;]"><strong><a href="fair_reciprocal_recommendation_in_matching_markets/">Fair Reciprocal Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用NSW与Frank-Wolfe交替优化双边互惠推荐，在保持匹配数同时近零嫉妒，合成/真实约会数据验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;上下文 Bandit&quot;,&quot;贝叶斯决策论&quot;,&quot;变分推断&quot;]"><strong><a href="position_paper_why_the_shooting_in_the_dark_method_dominates_recommender_sy/">Position Paper: Why the Shooting in the Dark Method Dominates Recommender Systems Practice; A Call to Abandon Anti-Utopian Thinking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>论证工业界「离线代理指标盲选 + A/B 验证」缺乏理论依据，主张用定制似然与变分贝叶斯直接瞄准奖励优化。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;线上评测与实验平台&quot;,&quot;A/B 测试&quot;,&quot;代理指标与统计功效&quot;,&quot;多重假设检验校正&quot;]"><strong><a href="powerful_a_b_testing_metrics_and_where_to_find_them/">Powerful A/B-Testing Metrics and Where to Find Them</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用数百个历史 A/B 实验构建指标校验管线，量化 I/II/III 类错误与统计功效；联合 DAU/Engagers/TimeSpent 在 95% 置信下相对降低 II 类错误 35%，或把所需样本量缩小 3.5 倍。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;计算社会选择&quot;,&quot;异构公平性&quot;,&quot;公平感知重排&quot;]"><strong><a href="social_choice_for_heterogeneous_fairness_in_recommendation/">Social Choice for Heterogeneous Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用计算社会选择的多智能体架构 SCRUF-D 将三类异构提供方公平性定义分别交由独立 Agent，在分配—聚合两阶段中实现多目标公平重排，且几乎不损 nDCG。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据采集与评测基准&quot;,&quot;用户信念 (Beliefs)&quot;,&quot;选择行为建模&quot;,&quot;反事实评估&quot;]"><strong><a href="the_movielens_beliefs_dataset_collecting_pre_choice_data_for_online_recomme/">The MovieLens Beliefs Dataset: Collecting Pre-Choice Data for Online Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出在 MovieLens 首页以非侵入方式采集用户对未观看影片的“选择前信念”，开源 3,199 用户/28,457 条信念+172 万评分数据集，并量化响应选择偏差与推荐-信念覆盖率。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平与可信推荐&quot;,&quot;公平性度量选择&quot;,&quot;医疗推荐&quot;,&quot;用户研究&quot;,&quot;情境敏感设计&quot;]"><strong><a href="understanding_fairness_in_recommender_systems_a_healthcare_perspective/">Understanding Fairness in Recommender Systems: A Healthcare Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>131人问卷研究：在高危/低危两种医疗推荐场景下让公众从四种公平度量中选择，发现公众理解度低、偏好随情境漂移且无共识。</p>
</dd>
</dl>
