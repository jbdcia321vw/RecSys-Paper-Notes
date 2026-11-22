---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**RecSys 2025** · **17** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [RecSys 2021 (4)](../../2021/collaborative_filtering/) · [RecSys 2022 (2)](../../2022/collaborative_filtering/) · [RecSys 2023 (7)](../../2023/collaborative_filtering/) · [RecSys 2024 (4)](../../2024/collaborative_filtering/) · [RecSys 2026 (8)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2021 (3)](../../../AAAI/2021/collaborative_filtering/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/collaborative_filtering/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2021 (4)](../../../CIKM/2021/collaborative_filtering/) · [CIKM 2022 (5)](../../../CIKM/2022/collaborative_filtering/) · [CIKM 2023 (3)](../../../CIKM/2023/collaborative_filtering/) · [CIKM 2024 (4)](../../../CIKM/2024/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2021 (1)](../../../ECIR/2021/collaborative_filtering/) · [ECIR 2022 (1)](../../../ECIR/2022/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICDE 2021 (2)](../../../ICDE/2021/collaborative_filtering/) · [ICDE 2023 (1)](../../../ICDE/2023/collaborative_filtering/) · [ICDE 2024 (2)](../../../ICDE/2024/collaborative_filtering/) · [ICDM 2021 (2)](../../../ICDM/2021/collaborative_filtering/) · [ICDM 2023 (1)](../../../ICDM/2023/collaborative_filtering/) · [ICDM 2024 (1)](../../../ICDM/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/collaborative_filtering/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/collaborative_filtering/) · [KDD 2021 (6)](../../../KDD/2021/collaborative_filtering/) · [KDD 2022 (3)](../../../KDD/2022/collaborative_filtering/) · [KDD 2023 (1)](../../../KDD/2023/collaborative_filtering/) · [KDD 2024 (1)](../../../KDD/2024/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [SIGIR 2021 (7)](../../../SIGIR/2021/collaborative_filtering/) · [SIGIR 2022 (4)](../../../SIGIR/2022/collaborative_filtering/) · [SIGIR 2023 (6)](../../../SIGIR/2023/collaborative_filtering/) · [SIGIR 2024 (4)](../../../SIGIR/2024/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/collaborative_filtering/) · [IEEE TKDE 2021 (2)](../../../TKDE/2021/collaborative_filtering/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/collaborative_filtering/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/collaborative_filtering/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/collaborative_filtering/) · [ACM TOIS 2022 (2)](../../../TOIS/2022/collaborative_filtering/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/collaborative_filtering/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/collaborative_filtering/) · [ACM TORS 2023 (1)](../../../TORS/2023/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [ACM TWEB 2022 (1)](../../../TWEB/2022/collaborative_filtering/) · [WSDM 2021 (1)](../../../WSDM/2021/collaborative_filtering/) · [WSDM 2022 (1)](../../../WSDM/2022/collaborative_filtering/) · [WSDM 2023 (1)](../../../WSDM/2023/collaborative_filtering/) · [WSDM 2024 (3)](../../../WSDM/2024/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/collaborative_filtering/) · [WWW (TheWebConf) 2022 (6)](../../../WWW/2022/collaborative_filtering/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/collaborative_filtering/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** Two-Tower ×2 · 曝光偏差 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Two-Tower" aria-pressed="false">Two-Tower<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/偏好推断&quot;,&quot;离散选择模型&quot;,&quot;核密度估计&quot;,&quot;曝光偏差&quot;]"><strong><a href="a_non_parametric_choice_model_that_learns_how_users_choose_between_recommen/">A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 sigmoid 核密度估计学习误差分布，无需假设 MNL/ENL 即可同时推断用户偏好与选择机制，抗竞争型曝光偏差。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排&quot;,&quot;因果推断&quot;,&quot;上下文老虎机&quot;,&quot;跨渠道营销&quot;]"><strong><a href="agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;子 ID 分解&quot;,&quot;个性化流行度&quot;,&quot;准确率-新颖度权衡&quot;]"><strong><a href="balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;采样 Softmax&quot;,&quot;logQ 修正&quot;,&quot;批内负采样&quot;]"><strong><a href="correcting_the_logq_correction_revisiting_sampled_softmax_for_large_scale_r/">Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>指出标准 logQ 修正误把正样本当作采样项，提出带置信度权重的改进修正，在公开与工业数据上稳定超越标准 logQ。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;极值理论&quot;,&quot;量子特征选择&quot;,&quot;QUBO/QAOA&quot;]"><strong><a href="estimating_quantum_execution_requirements_for_feature_selection_in_recommen/">Estimating Quantum Execution Requirements for Feature Selection in Recommender Systems Using Extreme Value Theory</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用极值理论对 QAOA 求解 QUBO 特征选择的输出建模，估计达到经典 SA 水平所需的量子执行次数，并在模拟器与真实 QPU 上验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;信息瓶颈&quot;,&quot;多行为推荐&quot;,&quot;图去噪&quot;]"><strong><a href="hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;Learning-to-Rank&quot;,&quot;Two-Tower&quot;,&quot;GBDT vs DNN&quot;]"><strong><a href="industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;实体引导兴趣建模&quot;,&quot;对比预训练&quot;,&quot;双塔用户编码器&quot;]"><strong><a href="ip2_entity_guided_interest_probing_for_personalized_news_recommendation/">IP2: Entity-Guided Interest Probing for Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以实体-标题对比预训练探测新闻内实体焦点，并用跨注意力双塔用户编码器建模新闻间实体引导兴趣，MIND-large AUC 达 72.06%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;特征工程/在线推理&quot;,&quot;Spark-Keras 等价翻译&quot;,&quot;训练-服务一致性&quot;,&quot;特征预处理管线&quot;]"><strong><a href="kamae_bridging_spark_and_keras_for_seamless_ml_preprocessing/">Kamae: Bridging Spark and Keras for Seamless ML Preprocessing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 PySpark 预处理管线一对一翻译为等价 Keras 层，消除离线/在线特征工程重复实现，Expedia 线上延迟降 61%、成本降 58%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多分辨率地理嵌入&quot;,&quot;H3 六边形网格&quot;,&quot;双塔对比学习&quot;]"><strong><a href="location_matters_leveraging_multi_resolution_geo_embeddings_for_housing_sea/">Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 H3 多分辨率六边形网格与城市嵌入注入双塔对比学习，缓解房源地理稀疏，离线重排模拟租金转化提升 85%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;离线评估&quot;,&quot;离策略估计&quot;,&quot;元分析&quot;,&quot;BLUE&quot;]"><strong><a href="meta_off_policy_estimation/">Meta Off-Policy Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多个 OPE 估计器视为相关研究做固定效应元分析，用 BLUE 闭式解融合并给出保守置信区间，OBD 上方差降超 50%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;离线评测&quot;,&quot;负采样策略&quot;,&quot;曝光偏差&quot;,&quot;元评测&quot;]"><strong><a href="on_the_reliability_of_sampling_strategies_in_offline_recommender_evaluation/">On the Reliability of Sampling Strategies in Offline Recommender Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以全观测 KuaiRec 为真值，四维度（分辨率/保真度/鲁棒性/预测力）系统解耦曝光偏差与采样偏差，给出采样策略选择指南。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正交Procrustes&quot;,&quot;低秩SVD&quot;,&quot;嵌入空间稳定化&quot;]"><strong><a href="orthogonal_low_rank_embedding_stabilization/">Orthogonal Low Rank Embedding Stabilization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用低秩SVD+正交Procrustes把重训后的用户/物品嵌入对齐到固定参考空间，保内积无损，跨期余弦相似度从0.01升至0.82</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双效用行为建模&quot;,&quot;局部贪心最优策略&quot;,&quot;站外选项估计&quot;]"><strong><a href="recommendation_and_temptation/">Recommendation and Temptation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户选择解耦为 enrichment 与 temptation 双效用并引入站外选项，证明局部贪心最优并给出可估计算子，MovieLens 与仿真全面超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Two-Tower&quot;,&quot;内容特征召回&quot;,&quot;ANN 分层索引&quot;]"><strong><a href="suggest_complement_inspire_story_of_two_tower_recommendations_at_allegro_co/">Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以内容特征双塔为统一底座，仅改查询塔与索引逻辑即复用至相似/互补/灵感三类召回，两年线上 A/B 验证 CTR 与 GMV 双增。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;稀疏自编码器&quot;,&quot;嵌入压缩&quot;,&quot;向量检索&quot;]"><strong><a href="the_future_is_sparse_embedding_compression_for_scalable_retrieval_in_recomm/">The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用余弦重建的稀疏自编码器把稠密向量压成高维 k-稀疏向量，免重训骨干，12× 压缩仅损 1.35% CTR，线上显著优于同尺寸 Matryoshka。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;评测与可复现性审计&quot;,&quot;超参数优化&quot;,&quot;框架默认值&quot;,&quot;早停策略&quot;]"><strong><a href="the_hidden_cost_of_defaults_in_recommender_system_evaluation/">The Hidden Cost of Defaults in Recommender System Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>审计 RecBole 未文档化的 early_stop=10 默认策略如何静默截断随机搜索与贝叶斯优化，量化搜索路径不稳定性与可复现性风险。</p>
</dd>
</dl>
