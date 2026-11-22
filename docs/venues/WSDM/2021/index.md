---
hide:
  - toc
---

# 🏛️ WSDM 2021 推荐系统论文解读

18 篇 WSDM 2021 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 可解释推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;超长方体用户表示&quot;,&quot;序列兴趣建模&quot;,&quot;键值记忆网络&quot;]"><strong><a href="collaborative_filtering/learning_user_representations_with_hypercuboids_for_recommender_systems/">Learning User Representations with Hypercuboids for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户由点嵌入升级为超长方体（中心+非负偏移），以内外复合距离度量用户-物品关系，全库排序下 Alibaba 线上数据 Recall@50 提升 8.5%–14.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱问答&quot;,&quot;约束建模&quot;]"><strong><a href="graph_rec/personalized_food_recommendation_as_constrained_question_answering_over_a_l/">Personalized Food Recommendation as Constrained Question Answering over a Large-scale Food Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将饮食推荐化为FoodKG约束KBQA，用查询扩展、KG增强与约束建模处理数值比较和否定约束，F1较BAMnet绝对提升60.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排训练&quot;,&quot;隐式反馈去噪&quot;,&quot;自适应损失&quot;]"><strong><a href="self_supervised_denoising/denoising_implicit_feedback_for_recommendation/">Denoising Implicit Feedback for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出自适应去噪训练ADT，用动态截断/重加权CE剪枝大损失假阳性交互；三数据集全库排序下NeuMF+T-CE平均提升15.62%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;稀疏兴趣激活&quot;,&quot;多兴趣表征&quot;,&quot;意图聚合&quot;]"><strong><a href="sequential_rec/sparse_interest_network_for_sequential_recommendation/">Sparse-Interest Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用大规模概念原型池做可微 top-K 稀疏兴趣激活，再以主动意图预测聚合多兴趣向量，Taobao HR@50 相对 MIND 提升 15.2%，工业 4B 数据 HR@50 提升 34.06%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;元路径注意力&quot;,&quot;可解释推荐&quot;]"><strong><a href="sequential_rec/temporal_meta_path_guided_explainable_recommendation/">Temporal Meta-path Guided Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用相邻物品间 item-item 元路径替代 RNN 建模时序依赖，配合双层注意力与 MLP，在 Amazon 三数据集上 HR@1 提升 30%+。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;推荐解释生成（端到端联合建模）&quot;,&quot;可解释推荐&quot;,&quot;情感对齐&quot;,&quot;约束解码&quot;]"><strong><a href="agent_conversational_rec/explanation_as_a_defense_of_recommendation/">Explanation as a Defense of Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用共享情感向量、双软门与Gumbel情感正则对齐解释与推荐，推理期MCTS约束解码；Yelp/Ratebeer上提升解释质量与情感一致性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;伪标签 Pseudo-Labelling&quot;,&quot;知识图谱 GNN&quot;,&quot;协同训练&quot;]"><strong><a href="cold_start_rec/alleviating_cold_start_problems_in_recommendation_through_pseudo_labelling_/">Alleviating Cold-Start Problems in Recommendation through Pseudo-Labelling over Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在知识图谱上把未观测样本伪标签化为弱正例，配合 KG 路径采样、频率感知负采样与双模型协同训练，显著提升新用户/新物品的召回。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/表征学习&quot;,&quot;GNN预训练&quot;,&quot;元学习&quot;,&quot;自适应邻居采样&quot;]"><strong><a href="cold_start_rec/pre_training_graph_neural_networks_for_cold_start_users_and_items_represent/">Pre-Training Graph Neural Networks for Cold-Start Users and Items Representation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出元学习式GNN预训练、自注意力元聚合器与自适应邻居采样器，直接优化冷启动用户/物品表征，在三个公开数据集的下游推荐上优于GNN基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;召回&quot;,&quot;Origin-Aware 下一目的地推荐&quot;,&quot;ST-LSTM 时空细胞状态&quot;,&quot;个性化偏好注意力&quot;]"><strong><a href="poi_spatiotemporal_rec/origin_aware_next_destination_recommendation_with_personalized_preference_a/">Origin-Aware Next Destination Recommendation with Personalized Preference Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 ST-LSTM 编码器与个性化偏好注意力解码器，将起点信息与局部/全局时空因子注入 OD 关系建模，七个 Grab 真实轨迹数据集 Acc@1 相对 SOTA 提升约 13%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;结构化剪枝&quot;,&quot;场感知特征交叉&quot;]"><strong><a href="ctr_feature_interaction/deeplight_deep_lightweight_feature_interactions_for_accelerating_ctr_predic/">DeepLight: Deep Lightweight Feature Interactions for Accelerating CTR Predictions in Ad Serving</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 DeepFwFM 场对权重矩阵建模二阶交叉，并对 DNN、场矩阵与嵌入做结构化剪枝，Criteo 46×/Avazu 27× 加速且 AUC 不降。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;细粒度特征交互&quot;,&quot;交互注意力&quot;,&quot;Pre-LN Transformer&quot;]"><strong><a href="ctr_feature_interaction/multi_interactive_attention_network_for_fine_grained_feature_learning_in_ct/">Multi-Interactive Attention Network for Fine-grained Feature Learning in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>MIAN 用 Pre-LN Transformer 编码行为序列，并对候选物品与每个细粒度用户/上下文属性做逐字段注意力交互，Amazon 与支付宝数据 AUC 提升 0.37%–0.87%，线上 CTR +0.41%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;机制设计&quot;,&quot;多目标拍卖&quot;,&quot;单调性约束&quot;,&quot;深度策略优化&quot;]"><strong><a href="ctr_feature_interaction/optimizing_multiple_performance_metrics_with_deep_gsp_auctions_for_e_commer/">Optimizing Multiple Performance Metrics with Deep GSP Auctions for E-commerce Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用深度网络替换 GSP 的静态 rank score，在单调性与平滑过渡约束下以 DDPG 端到端学习多指标拍卖排序，淘宝线上全指标共赢。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;三角训练框架&quot;,&quot;竞价词生成&quot;,&quot;中间隐变量期望嵌入&quot;]"><strong><a href="ctr_feature_interaction/triangular_bidword_generation_for_sponsored_search_auction/">Triangular Bidword Generation for Sponsored Search Auction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用高质量 &lt;query, ad&gt; 点击数据构建三角互监督，以竞价词为隐变量桥接 A↔Q，实现去噪竞价词生成；Full-List 生成评测下 Ad 侧 Conv-KNRM +18.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;对抗攻击&quot;,&quot;黑盒梯度估计&quot;,&quot;视觉感知推荐&quot;]"><strong><a href="debias_fairness_industrial/a_black_box_attack_model_for_visually_aware_recommender_systems/">A Black-Box Attack Model for Visually-Aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>通过无感知图像扰动与链式梯度近似黑盒攻击视觉感知推荐系统，在 Amazon 双数据集上将随机商品推入 top-20 命中率达 0.53。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;预测不确定性估计&quot;,&quot;神经元激活强度&quot;,&quot;集成预测方差&quot;]"><strong><a href="debias_fairness_industrial/beyond_point_estimate_inferring_ensemble_prediction_variation_from_neuron_a/">Beyond Point Estimate: Inferring Ensemble Prediction Variation from Neuron Activation Strength in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以单模型 ReLU 激活强度为辅助任务推断集成预测方差，Criteo R²=0.78、极端方差桶 AUC 0.92，免去百模集成在线推理</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/评分预测&quot;,&quot;主流偏置去偏&quot;,&quot;卷积自编码器对抗正则&quot;,&quot;评论文本建模&quot;]"><strong><a href="debias_fairness_industrial/leave_no_user_behind_towards_improving_the_utility_of_recommender_systems_f/">Leave No User Behind: Towards Improving the Utility of Recommender Systems for Non-mainstream Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在 DeepCoNN 上并联用户/物品卷积自编码器，用文本重建损失作为评分预测的对抗条件，显著提升非主流用户 per-user RMSE 而几乎不牺牲主流用户。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;组合公平 (Compositional Fairness)&quot;,&quot;反事实 Headroom 分析&quot;,&quot;分布匹配正则&quot;]"><strong><a href="debias_fairness_industrial/practical_compositional_fairness_understanding_fairness_in_multi_component_/">Practical Compositional Fairness: Understanding Fairness in Multi-Component Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>证明多组件乘性打分下公平性不会自动组合，给出对称性充分条件与反事实 headroom 测试框架，产线系统整体 PairAcc gap 由 0.1546 降至 0.0222。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;去偏公平与工业架构&quot;,&quot;重排&quot;,&quot;约束马尔可夫决策过程&quot;,&quot;长期曝光公平&quot;,&quot;约束策略优化 CPO&quot;]"><strong><a href="debias_fairness_industrial/towards_long_term_fairness_in_recommendation/">Towards Long-term Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将动态流行度分组的曝光公平建模为 CMDP，用双 Critic 的 CPO 在 400 步长程推荐中持续压制 Gini 与流行度率，NDCG@20 相对最优基线提升 26.6%–43.2%。</p>
</dd>
</dl>
</section>
</div>
