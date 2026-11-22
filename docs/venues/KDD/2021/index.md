---
hide:
  - toc
---

# 🏛️ KDD 2021 推荐系统论文解读

24 篇 KDD 2021 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×2 · 去偏公平与工业架构 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;损失函数自动搜索&quot;,&quot;Gumbel-softmax 可微硬选择&quot;,&quot;DARTS 双层优化&quot;]"><strong><a href="collaborative_filtering/autoloss_automated_loss_function_search_in_recommendations/">AutoLoss: Automated Loss Function Search in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以控制器网络逐样本生成候选损失权重并经 Gumbel-softmax 硬选择，DARTS 双层优化自动搜索损失函数，Criteo/ML-20m 上 AUC 全面超越 SLF</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/排序模型设计&quot;,&quot;AutoML&quot;,&quot;性能预测器&quot;]"><strong><a href="collaborative_filtering/efficient_data_specific_model_search_for_collaborative_filtering/">Efficient Data-specific Model Search for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AutoCF 将CF拆为四阶段统一空间，用随机搜索+性能预测器为数据集定制模型，全库排序Recall@20较SOTA提升20.1%–22.7%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;拉普拉斯特征映射初始化&quot;,&quot;流形学习与残差双损失&quot;]"><strong><a href="collaborative_filtering/initialization_matters_regularizing_manifold_informed_initialization_for_ne/">Initialization Matters: Regularizing Manifold-informed Initialization for Neural Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用带流行度正则的拉普拉斯特征映射（LEPORID）为神经推荐系统提供流形感知初始化，并配残差双损失网络 DLR2，在三个数据集全库排序上显著超越 KNN 与 SOTA 神经模型。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;深度哈希编码&quot;,&quot;无嵌入表参数化&quot;,&quot;OOV 冷启动&quot;]"><strong><a href="collaborative_filtering/learning_to_embed_categorical_features_without_embedding_tables_for_recomme/">Learning to Embed Categorical Features without Embedding Tables for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用稠密哈希编码加深度嵌入网络彻底取代 one-hot 嵌入表，在 1/4 参数下逼近全量嵌入 AUC 并原生支持 OOV。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;关系拓扑保持&quot;,&quot;模型压缩&quot;]"><strong><a href="collaborative_filtering/topology_distillation_for_recommender_system/">Topology Distillation for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将教师表示空间的关系拓扑分层（组级原型+组内实体级）蒸馏给轻量学生，Full-Ranking 下 Recall@50 较 SOTA DE 提升 3.0–7.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;低秩回归&quot;,&quot;矩阵分解&quot;,&quot;闭式解&quot;]"><strong><a href="collaborative_filtering/towards_a_better_understanding_of_linear_models_for_recommendation/">Towards a Better Understanding of Linear Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用统一闭式解揭示低秩回归与矩阵分解的奇异值缩放差异，并提出BPR超参搜索与近邻模型增强，在ML-20M/Netflix全库排序上逼近甚至超越EASE。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回（Top-N图协同过滤）&quot;,&quot;三训练&quot;,&quot;社交增强视图&quot;,&quot;邻域判别对比学习&quot;]"><strong><a href="self_supervised_denoising/socially_aware_self_supervised_tri_training_for_recommendation/">Socially-Aware Self-Supervised Tri-Training for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>SEPT用好友/分享三角视图与偏好视图三编码器互伪标，邻域判别对比学习；5折全库排序，Yelp NDCG@10较LightGCN提升10.33%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;多样性驱动强化学习&quot;,&quot;约束Trie解码&quot;,&quot;查询改写&quot;]"><strong><a href="generative_rec/diversity_driven_query_rewriting_in_search_advertising/">Diversity driven Query Rewriting in Search Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>CLOVER 以人类评审模型为奖励、用多样性驱动 RL 微调 ProphetNet，并在 trie 约束下解码查询改写；Bing 线上点击 +12.83%、收入 +21.29%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;半个性化分群&quot;,&quot;冷用户嵌入预测&quot;]"><strong><a href="cold_start_rec/a_semi_personalized_system_for_user_cold_start_recommendation_on_music_stre/">A Semi-Personalized System for User Cold Start Recommendation on Music Streaming Apps</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出DNN预测冷用户嵌入并映射到暖用户分群的半个性化框架，在Deezer全库Top-50与在线A/B显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习&quot;,&quot;混合专家&quot;,&quot;Look-alike 人群扩展&quot;]"><strong><a href="cold_start_rec/learning_to_expand_audience_via_meta_hybrid_experts_and_critics_for_recomme/">Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用两阶段元学习模拟「理解种子—扩展人群」，并以任务驱动门控的混合专家/评论家建模任务关系，微信十亿级线上 CVR 提升约 7.9%–15.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;对偶学习&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/dual_attentive_sequential_learning_for_cross_domain_click_through_rate_pred/">Dual Attentive Sequential Learning for Cross-Domain Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用正交对偶映射对齐双域用户表征、用跨域拼接 Query 的双注意力做匹配，三数据集 AUC/HR@10 全面领先，线上 VV +7.07%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;数值特征嵌入&quot;,&quot;软离散化&quot;,&quot;元嵌入&quot;]"><strong><a href="ctr_feature_interaction/an_embedding_learning_framework_for_numerical_features_in_ctr_prediction/">An Embedding Learning Framework for Numerical Features in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AutoDis 用字段元嵌入 + 可微软离散化 + 加权聚合，为数值特征端到端学习“连续但可区分”的嵌入，三数据集 AUC 提升 0.17–0.23%，线上 CTR +2.1%、eCPM +2.7%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;图神经网络Embedding增强&quot;,&quot;特征稀疏&quot;,&quot;课程学习&quot;]"><strong><a href="ctr_feature_interaction/dual_graph_enhanced_embedding_neural_network_for_ctr_prediction/">Dual Graph enhanced Embedding Neural Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用属性图+协同图双图卷积精炼Embedding，以分治式字段卷积与课程学习式组织学习缓解特征/行为稀疏，Alimama AUC 提升 8.83%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多任务学习&quot;,&quot;序列依赖&quot;,&quot;信息迁移&quot;]"><strong><a href="ctr_feature_interaction/modeling_the_sequential_dependence_among_audience_multi_step_conversions_wi/">Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AITM 用自适应信息迁移模块在输出层向量空间建模多步转化的顺序依赖，配合行为期望校准损失，在美团信用卡广告与 Ali-CCP 上均取得稳定 AUC 提升，线上端到端转化率显著改善。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排+拍卖机制设计&quot;,&quot;可微排序 NeuralSort&quot;,&quot;端到端自动机制设计&quot;]"><strong><a href="ctr_feature_interaction/neural_auction_end_to_end_learning_of_auction_mechanisms_for_e_commerce_adv/">Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出可微深度神经拍卖 DNA，以 DeepSet 集合编码、部分单调 MIN-MAX 打分与 NeuralSort 松弛排序，在淘宝离线与在线 A/B 中显著提升多目标拍卖指标。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;广告主侧策略推荐（出价/定向决策）&quot;,&quot;Contextual Bandit&quot;,&quot;Dropout as Thompson Sampling&quot;,&quot;偏好向量学习&quot;]"><strong><a href="ctr_feature_interaction/we_know_what_you_want_an_advertising_strategy_recommender_system_for_online/">We Know What You Want: An Advertising Strategy Recommender System for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 MLP 将广告主特征映射为 KPI 偏好向量 w，以 σ(w^T·v) 预测策略采纳率，Dropout 近似 Thompson Sampling 平衡探索，仿真累计后悔较随机策略降 61.2%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/广告排序&quot;,&quot;高斯过程&quot;,&quot;上下文 Bandit&quot;,&quot;不确定性估计&quot;]"><strong><a href="rl_bandit_rec/exploration_in_online_advertising_systems_with_deep_uncertainty_aware_learn/">Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将深度核高斯过程嵌入 CTR 模型以估计不确定性，结合 UCB/TS 排序做探索，线上社福 +8.2%、收入 +8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/流量分配（实验层）&quot;,&quot;Multi-Armed Bandit&quot;,&quot;D-Optimal Design&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="rl_bandit_rec/towards_the_d_optimal_online_experiment_design_for_recommender_selection/">Towards the D-Optimal Online Experiment Design for Recommender Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用统计学 D-最优设计替代均匀随机探索来分配多推荐器流量，在线性/贝叶斯/非线性奖励下最大化 Fisher 信息增益，MovieLens 在线模拟与 Walmart.com 杂货相似商品部署验证累积召回显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;逆倾向加权&quot;,&quot;曝光偏差&quot;]"><strong><a href="debias_fairness_industrial/contrastive_learning_for_debiased_candidate_generation_in_large_scale_recom/">Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>证明对比损失等价IPW去偏，提出队列CLRec/Multi-CLRec，淘宝四月A/B提升CTR并降马太效应。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果推断与后门调整&quot;,&quot;偏差放大 (Bias Amplification)&quot;]"><strong><a href="debias_fairness_industrial/deconfounded_recommendation_for_alleviating_bias_amplification/">Deconfounded Recommendation for Alleviating Bias Amplification</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以因果图揭示物品组分布混杂子导致偏差放大，用后门调整近似算子改造 FM/NFM，在 Full-Ranking 下同时提升精度并降低 C_KL。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端（云端训练 + 端侧推理）&quot;,&quot;端云协同学习&quot;,&quot;模型套模型蒸馏&quot;,&quot;元学习参数基&quot;]"><strong><a href="debias_fairness_industrial/device_cloud_collaborative_learning_for_recommendation/">Device-Cloud Collaborative Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>端云协同 DCCL：MetaPatch 低秩元补丁做端侧个性化，MoMoDistill 模型套模型蒸馏反哺云端，LOO+1:100 采样排序下 Taobao HR@1 相对提升 6.79%，长尾用户增益最大。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;弹性嵌入 (Elastic Embeddings)&quot;,&quot;端侧轻量化与免重训定制&quot;,&quot;神经架构搜索 (NAS)&quot;]"><strong><a href="debias_fairness_industrial/learning_elastic_embeddings_for_customizing_on_device_recommenders/">Learning Elastic Embeddings for Customizing On-Device Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将物品嵌入切成可选的块，配合多样性正则与性能估计器驱动的进化搜索，一次训练即可在任意端侧内存预算下免重训定制推荐器，5MB 极紧预算下 Recall@50 提升最高 51%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/推理阶段&quot;,&quot;因果推断&quot;,&quot;反事实推理&quot;,&quot;流行度去偏&quot;]"><strong><a href="debias_fairness_industrial/model_agnostic_counterfactual_reasoning_for_eliminating_popularity_bias_in_/">Model-Agnostic Counterfactual Reasoning for Eliminating Popularity Bias in Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出模型无关因果图、多任务训练与反事实推理去除物品流行度直接效应，在五个真实数据集全库排序下显著优于DICE等基线。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与工业架构&quot;,&quot;训练基础设施 (端到端)&quot;,&quot;混合并行 Hybrid Parallelism&quot;,&quot;通信压缩 Dynamic Thresholding&quot;,&quot;Top-K 稀疏化&quot;]"><strong><a href="debias_fairness_industrial/training_recommender_systems_at_scale_communication_efficient_model_and_dat/">Training Recommender Systems at Scale: Communication-Efficient Model and Data Parallelism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 DCT 双压缩框架，用延迟更新硬阈值对 DP 梯度与 MP 激活/梯度稀疏化，实现 100×/20× 通信压缩，产线端到端训练加速 37% 且无损精度。</p>
</dd>
</dl>
</section>
</div>
