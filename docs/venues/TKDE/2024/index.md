---
hide:
  - toc
---

# 📖 IEEE TKDE 2024 推荐系统论文解读

20 篇 IEEE TKDE 2024 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDE ×16 · TKDE 2024 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;端侧排序&quot;,&quot;弹性嵌入&quot;,&quot;端侧推荐&quot;]"><strong><a href="collaborative_filtering/personalized_elastic_embedding_learning_for_on_device_recommendation/">Personalized Elastic Embedding Learning for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>一次训练生成按端侧预算裁剪的用户组个性化弹性物品嵌入，组内精调与控制器赋权，在 Amazon/Yelp 全库排序下优于 RULE。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;马尔可夫图扩散&quot;,&quot;InfoBPR 排序损失&quot;,&quot;图稀疏化&quot;]"><strong><a href="graph_rec/mgdcf_distance_learning_via_markov_graph_diffusion_for_neural_collaborative/">MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>证明 LightGCN/APPNP 等价于 1 层上下文编码 NRL，用马尔可夫图扩散权衡两类距离，并配 InfoBPR 多负样本损失，Amazon-Book NDCG@20 较 LightGCN 提升 47%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排与可解释生成&quot;,&quot;反事实解释&quot;,&quot;知识图谱路径推理&quot;,&quot;强化学习策略梯度&quot;]"><strong><a href="graph_rec/reinforced_path_reasoning_for_counterfactual_explainable_recommendation/">Reinforced Path Reasoning for Counterfactual Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>TKDE 2024｜图神经网络与知识图谱推荐｜在协同知识图谱上用两步注意力路径采样器压缩动作空间，以强化学习优化反事实解释策略，生成物品属性级反事实解释并反哺推荐训练（Yelp2018 Recall@80 +78.03%）。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;监督对比损失&quot;,&quot;协同邻居&quot;,&quot;LightGCN&quot;]"><strong><a href="self_supervised_denoising/neighborhood_enhanced_supervised_contrastive_learning_for_collaborative_fil/">Neighborhood-Enhanced Supervised Contrastive Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 ItemKNN 最近邻与交互邻居纳入对比损失正样本，缓解假阴性；三数据集全库排序 NDCG@20 较 SGL 提升 10.09%/7.09%/35.36%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自监督预训练&quot;,&quot;数据增广&quot;,&quot;InfoNCE&quot;,&quot;SELFRec&quot;]"><strong><a href="self_supervised_denoising/self_supervised_learning_for_recommender_systems_a_survey/">Self-Supervised Learning for Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次系统界定自监督推荐 (SSR)，提出对比/生成/预测/混合四分类与 JL/PF/IL 三训练范式，并开源 SELFRec 在 Full-Ranking 下给出受控实证结论。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/全库排序&quot;,&quot;图对比学习&quot;,&quot;噪声嵌入增广&quot;,&quot;跨层对比&quot;]"><strong><a href="self_supervised_denoising/xsimgcl_towards_extremely_simple_graph_contrastive_learning_for_recommendat/">XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用噪声嵌入增广与单编码器跨层对比替代图增广，四数据集全库排序显著提升且训练近 LightGCN 轻量。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;增量学习&quot;,&quot;多意图解耦&quot;,&quot;知识蒸馏&quot;]"><strong><a href="sequential_rec/a_framework_for_elastic_adaptation_of_user_multiple_intents_in_sequential_r/">A Framework for Elastic Adaptation of User Multiple Intents in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 IMA/EMA 增量多意图自适应框架：蒸馏损失保旧意图、困惑度检测新建意图、正交投影裁剪冗余意图、活跃度/聚类弹性压缩，4 数据集 NDCG 较最优增量基线提升 3.8–4.8%，省 27.3% 内存。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态会话推荐&quot;,&quot;概率价格建模&quot;]"><strong><a href="sequential_rec/beyond_co_occurrence_multi_modal_session_based_recommendation/">Beyond Co-occurrence: Multi-modal Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用伪模态对比、分层枢轴Transformer与Wasserstein自注意力统一建模图像/文本/价格，Amazon三域全库排序Prec@20提升1.45%–3.14%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;连续时间记忆网络&quot;,&quot;时序集合预测&quot;]"><strong><a href="sequential_rec/continuous_time_user_preference_modelling_for_temporal_sets_prediction/">Continuous-Time User Preference Modelling for Temporal Sets Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>连续时间记忆库显式建模用户偏好演化，全局序列+set-batch，全库Top-K评估，JingDong Recall@10提升11.48%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;神经常微分方程&quot;,&quot;动态图推荐&quot;]"><strong><a href="sequential_rec/learning_graph_ode_for_continuous_time_sequential_recommendation/">Learning Graph ODE for Continuous-Time Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GDERec 用自回归图 ODE 建模用户-物品交互图的连续演化，配时间注意力聚合不规则采样，5 数据集全库排序 R@10 提升约 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Prompt-Tuning&quot;,&quot;冷启动&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/personalized_prompt_for_sequential_recommendation/">Personalized Prompt for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户画像生成个性化软前缀 Prompt 替代全参微调，配合 Prompt 导向对比学习，在三大数据集的 Few-shot/Zero-shot 冷启动上稳定超越 Fine-tuning。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;Pre-train/Fine-tune/Prompt 范式&quot;,&quot;Semantic ID 与 ICL/CoT&quot;]"><strong><a href="llm_rec/recommender_systems_in_the_era_of_large_language_models_llms/">Recommender Systems in the Era of Large Language Models (LLMs)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以「表征（离散ID vs 文本侧信息）× 适配范式（预训练/微调/提示）」二维轴系统梳理 LLM4Rec，附 60+ 方法仓库索引与可信/幻觉等未来路线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;精排&quot;,&quot;可解释推荐&quot;,&quot;信息瓶颈&quot;,&quot;变分自编码器&quot;]"><strong><a href="agent_conversational_rec/explainable_recommender_with_geometric_information_bottleneck/">Explainable Recommender with Geometric Information Bottleneck</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用交互图 K-Means 软簇分布作几何先验，经可迁移信息瓶颈约束评论 VAE 的隐因子，兼顾评分预测与跨评论解释抽取；3 数据集 RMSE 均值 84.18，人工评测解释质量领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;贝叶斯图增广&quot;,&quot;多视图注意力&quot;,&quot;鲁棒推荐&quot;]"><strong><a href="poi_spatiotemporal_rec/bayes_enhanced_multi_view_attention_networks_for_robust_poi_recommendation/">Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用贝叶斯后验引导的节点复制图增广加三视图注意力建模个人/语义/距离依赖，在签到缺失与噪声下 R@5 最高提升 16.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;图结构学习&quot;,&quot;层次原型&quot;,&quot;对比多视图融合&quot;]"><strong><a href="poi_spatiotemporal_rec/bi_level_graph_structure_learning_for_next_poi_recommendation/">Bi-Level Graph Structure Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 EM 聚类原型构造 POI–原型层次图，并配自适应成对结构学习与多视图对比融合；全库排序下 Acc@20 最高相对提升 13.56%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;可微架构搜索&quot;,&quot;嵌入维度搜索&quot;,&quot;特征选择&quot;]"><strong><a href="ctr_feature_interaction/i_razor_a_differentiable_neural_input_razor_for_feature_selection_and_dimen/">i-Razor: A Differentiable Neural Input Razor for Feature Selection and Dimension Search in DNN-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用带 0 维吸收算子的可微算子选择把字段选择与嵌入维度搜索合并为一次性联合优化，CPT 剪枝导出配置后重训，AUC 提升且参数量降 35%+。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;结果多样性&quot;,&quot;MMR与DPP&quot;,&quot;曝光公平&quot;]"><strong><a href="multi_task_reranking/result_diversification_in_search_and_recommendation_a_survey/">Result Diversification in Search and Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次统一搜索与推荐的结果多样性分类体系：概念层区分个体级/系统级与外在/内在多样性，度量层分无关/含相关性两族，方法层按干预时机分预/中/后处理与在线 Bandit/RL。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果蒸馏&quot;,&quot;前门调整&quot;]"><strong><a href="debias_fairness_industrial/causal_distillation_for_alleviating_performance_heterogeneity_in_recommende/">Causal Distillation for Alleviating Performance Heterogeneity in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用前门调整+异构多教师蒸馏近似因果效应，在采样排序下提升尾部用户并降低性能异质性。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;本地差分隐私&quot;,&quot;属性推断攻击&quot;,&quot;自适应隐私预算&quot;]"><strong><a href="debias_fairness_industrial/comprehensive_privacy_analysis_on_federated_recommender_system_against_attr/">Comprehensive Privacy Analysis on Federated Recommender System against Attribute Inference Attacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>先用梯度式属性推断攻击逐组件量化 FedRec 泄漏差异，再提出按脆弱度分配隐私预算的自适应 LDP 机制 APM，在几乎不损 Hit@20 的前提下显著降低属性推断 F1。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;前门调整 (Front-door Adjustment)&quot;,&quot;隐混淆因子 (Hidden Confounder)&quot;,&quot;多任务因果推荐&quot;]"><strong><a href="debias_fairness_industrial/mitigating_hidden_confounding_effects_for_causal_recommendation/">Mitigating Hidden Confounding Effects for Causal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用前门调整将物品特征对点赞的因果效应拆成两个可识别部分效应，Tiktok/Kwai/Taobao 全库排序 N@50 最高提升 19.4%，低活跃用户增益尤大。</p>
</dd>
</dl>
</section>
</div>
