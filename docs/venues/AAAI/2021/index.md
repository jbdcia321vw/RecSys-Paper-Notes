---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2021 推荐系统论文解读

23 篇 AAAI (RecSys Track) 2021 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI 2021 ×13 · AAAI ×10 · 会话推荐 ×2 · 图神经网络 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络" aria-pressed="false">图神经网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排/自适应选题&quot;,&quot;p-VAE&quot;,&quot;信息增益&quot;]"><strong><a href="collaborative_filtering/educational_question_mining_at_scale_prediction_analysis_and_personalizatio/">Educational Question Mining At Scale: Prediction, Analysis and Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 p-VAE 补全稀疏学生-题目答题矩阵，并以信息论指标量化题目质量、难度与个性化选题。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/评分（矩阵分解）与事后解释&quot;,&quot;影响力估计&quot;,&quot;知识图谱补全&quot;,&quot;梯度回滚&quot;,&quot;可解释推荐&quot;]"><strong><a href="collaborative_filtering/explaining_neural_matrix_factorization_with_gradient_rollback/">Explaining Neural Matrix Factorization with Gradient Rollback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>训练中累积每个三元组引起的参数更新量，测试时回滚该更新即可免重训估计样本影响力；ROAR 重训验证 PD 提升 15–39 点，开销仅 3h|D| 内存。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;逆强化学习&quot;,&quot;合成反馈闭环&quot;,&quot;生成式用户模拟&quot;]"><strong><a href="collaborative_filtering/learning_to_recommend_from_sparse_data_via_generative_user_feedback/">Learning to Recommend from Sparse Data via Generative User Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用「虚拟用户」奖励估计器+反馈生成器与 CF 推荐器构成闭环，以 IRL 对抗正则在稀疏交互上稳定提升 VAE/MF 类召回 2%~9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;图Transformer&quot;,&quot;知识图谱&quot;]"><strong><a href="graph_rec/knowledge_enhanced_hierarchical_graph_transformer_network_for_multi_behavio/">Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>时间感知图Transformer融合多行为与知识图谱，Leave-One-Out+99负采样评测，Retail HR@10相对提升23.1%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何 Poincaré Ball&quot;,&quot;知识图谱注意力聚合&quot;,&quot;双层优化自适应正则&quot;]"><strong><a href="graph_rec/knowledge_enhanced_top_k_recommendation_in_poincar_ball/">Knowledge-Enhanced Top-K Recommendation in Poincaré Ball</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户/物品/KG 实体关系嵌入 Poincaré 球，用双曲注意力+Einstein 中点聚合邻居，并以双层优化自适应学习正则强度，NDCG@20 提升 3.46%–16.70%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态记忆网络&quot;,&quot;长序列建模&quot;]"><strong><a href="sequential_rec/dynamic_memory_based_attention_network_for_sequential_recommendation/">Dynamic Memory based Attention Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>DMAN 将超长行为序列切段并逐段训练，用循环注意力与动态胶囊记忆分别抽取短期/长期兴趣，全库排序下 HR@50 最高提升 15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;会话推荐&quot;,&quot;图神经网络&quot;,&quot;多任务学习&quot;]"><strong><a href="sequential_rec/graph_enhanced_multi_task_learning_of_multi_level_transition_dynamics_for_s/">Graph-Enhanced Multi-Task Learning of Multi-Level Transition Dynamics for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>MTD 用位置感知注意力建模会话内转移、用全局图编码器+互信息最大化建模跨会话转移，多任务联合学习，三数据集 Pre@10 最高提升 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;非侵入式注意力&quot;,&quot;侧信息融合&quot;,&quot;BERT4Rec&quot;]"><strong><a href="sequential_rec/non_invasive_self_attention_for_side_information_fusion_in_sequential_recom/">Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>NOVA 让侧信息只参与注意力 Q/K 计算、V 保持纯 ID 向量空间，全库排序下 ML-1m HR@10 相对提升 13.51%，开销几乎可忽略。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;对比自监督&quot;,&quot;会话推荐&quot;]"><strong><a href="sequential_rec/self_supervised_hypergraph_convolutional_networks_for_session_based_recomme/">Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>会话即超边构建双通道超图卷积，并以会话级互信息自监督增强，全库排序下 P@20 最高相对提升 19.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;多类型交互图&quot;,&quot;互信息图对比学习&quot;]"><strong><a href="social_group_rec/knowledge_aware_coupled_graph_neural_network_for_social_recommendation/">Knowledge-aware Coupled Graph Neural Network for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 KCGN，用多类型交互图 + 相对时间编码建模动态交互异质性，并以互信息估计对齐用户/物品孪生图的高阶结构；Leave-One-Out + 99 负采样协议下 Epinions HR@10 提升 4.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习少样本匹配&quot;,&quot;序列推荐&quot;,&quot;冷启动物品&quot;]"><strong><a href="cold_start_rec/cold_start_sequential_recommendation_via_meta_learner/">Cold-start Sequential Recommendation via Meta Learner</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>Mecos 用元学习把冷启动物品推荐重构为「查询序列–支持集」度量匹配，仅靠交互数据即可即插即用地提升 SR-GNN 等模型，HR@10 平均提升 70%–99%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排/用户偏好预测&quot;,&quot;元学习&quot;,&quot;自适应学习率&quot;,&quot;kd-tree&quot;]"><strong><a href="cold_start_rec/personalized_adaptive_meta_learning_for_cold_start_user_preference_predicti/">Personalized Adaptive Meta Learning for Cold-start User Preference Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用用户嵌入自适应元学习率缓解冷启动任务过拟合，MovieLens/BookCrossing/淘宝生产集验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;兴趣漂移迁移&quot;,&quot;神经主题模型&quot;]"><strong><a href="poi_spatiotemporal_rec/out_of_town_recommendation_with_travel_intention_modeling/">Out-of-Town Recommendation with Travel Intention Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用G-GNN编码家域偏好、NTM无监督挖掘旅行意图、MLP迁移兴趣漂移，异地POI召回相对提升约3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;重要性采样&quot;]"><strong><a href="ctr_feature_interaction/capturing_delayed_feedback_in_conversion_rate_prediction_via_elapsed_time_s/">Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 ES-DFM：以「已过时间采样分布」$p(e|x)$ 主动权衡标签准确性与模型新鲜度，配合有界重要性权重实现流式 CVR 无偏估计，Criteo/Taobao 上 AUC 分别提升 0.26%/0.44%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉筛选&quot;,&quot;L0正则&quot;,&quot;图神经网络&quot;]"><strong><a href="ctr_feature_interaction/detecting_beneficial_feature_interactions_for_recommender_systems/">Detecting Beneficial Feature Interactions for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>L0-SIGN用L0边预测筛选有益二阶交叉，再由SIGN聚合；Frappe ACC较最强基线提升2.50%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/端到端策略学习&quot;,&quot;Offline RL&quot;,&quot;外推误差约束&quot;,&quot;离散随机策略&quot;]"><strong><a href="rl_bandit_rec/a_general_offline_reinforcement_learning_framework_for_interactive_recommen/">A General Offline Reinforcement Learning Framework for Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将交互推荐写成概率生成模型的变分推断，用离散随机 Actor-Critic 加五种行为策略约束抑制外推误差，全库排序下 NDCG@10 提升 13.8%~25.2%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Bandit&quot;,&quot;Dispersion Function&quot;,&quot;多样性推荐&quot;]"><strong><a href="rl_bandit_rec/a_hybrid_bandit_framework_for_diversified_recommendation/">A Hybrid Bandit Framework for Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用模块函数刻画相关性、散度函数刻画多样性，在线联合学习用户对二者的个性化偏好，贪心+UCB 求解并给出无间隙 regret 上界。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;广告投放实验/创意分配&quot;,&quot;Contextual Thompson Sampling&quot;,&quot;实验即服务 (EaaS)&quot;,&quot;互斥子人群拆分 (DA)&quot;]"><strong><a href="rl_bandit_rec/comparison_lift_bandit_based_experimentation_system_for_online_advertising/">Comparison Lift: Bandit-based Experimentation System for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>京东广告 EaaS 系统：将重叠目标人群拆为互斥子人群，在子人群上跑 Contextual Thompson Sampling，再聚合回人群层做创意-人群最优组合推断；线上 1,547 次实验平均提升测试期点击 +27%、广告主 CTR +46%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;DQN 多子动作联合决策&quot;,&quot;Dueling 价值-优势分解&quot;,&quot;广告-推荐列表混排&quot;]"><strong><a href="rl_bandit_rec/dear_deep_reinforcement_learning_for_online_advertising_impression_in_recom/">DEAR: Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用单一 Dueling-DQN 同时决策「是否插广告/插哪条/插哪个槽位」，把前向复杂度从 O(|A|·L) 降到 O(|A|)，在抖音 100 万会话上累积奖励较 HDQN 提升 6.71%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;Thompson Sampling&quot;,&quot;动态规划&quot;]"><strong><a href="rl_bandit_rec/efficient_optimal_selection_for_composited_advertising_creatives_with_tree_/">Efficient Optimal Selection for Composited Advertising Creatives with Tree Structure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 ingredient 树 + 元素二部图参数化创意 CTR，动态规划精确搜索最优创意，并将 Thompson sampling 下沉到参数后验实现高效探索。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/策略层端到端&quot;,&quot;世界模型 (World Model)&quot;,&quot;通用价值函数 (UVF)&quot;,&quot;解耦价值估计&quot;]"><strong><a href="rl_bandit_rec/reinforcement_learning_with_a_disentangled_universal_value_function_for_ite/">Reinforcement Learning with a Disentangled Universal Value Function for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以目标导向世界模型解耦环境动态与高方差奖励，构建通用价值函数 $Q=g^{\top}M(s,a,g)$，在三个仿真环境与网易线上场景提升累积点击 3.7%–10.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;分解式对抗学习&quot;,&quot;正交正则&quot;,&quot;敏感属性去偏&quot;]"><strong><a href="debias_fairness_industrial/fairrec_fairness_aware_news_recommendation_with_decomposed_adversarial_lear/">FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>把用户表征解耦为偏见感知/无偏见双路，用属性预测＋对抗学习＋正交正则剥离性别偏置，全库排序下性别可推断率逼近随机，AUC 仅降 1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估（全库排序/采样排序）&quot;,&quot;经验秩分布&quot;,&quot;MLE/最大熵&quot;]"><strong><a href="debias_fairness_industrial/on_estimating_recommendation_evaluation_metrics_under_sampling/">On Estimating Recommendation Evaluation Metrics under Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>学习采样下的经验秩分布，统一估计 Recall/NDCG/AP；n=100 采样、100 次重复，WMLE 将 AP@10 偏差较 BV0.01 降低约 15.9%。</p>
</dd>
</dl>
</section>
</div>
