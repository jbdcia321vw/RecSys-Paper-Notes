---
hide:
  - toc
---

# 🏛️ RecSys 2021 推荐系统论文解读

37 篇 RecSys 2021 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** Off-Policy Evaluation ×2 · 离线评测 ×2 · 联邦协同过滤 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Off-Policy Evaluation" aria-pressed="false">Off-Policy Evaluation<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测" aria-pressed="false">离线评测<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦协同过滤" aria-pressed="false">联邦协同过滤<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排/列表组织&quot;,&quot;UMAP&quot;,&quot;HDBSCAN&quot;]"><strong><a href="collaborative_filtering/automatic_collection_creation_and_recommendation/">Automatic Collection Creation and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在ALS/双塔DNN嵌入上做UMAP降维+HDBSCAN密度聚类自动生成个性化歌单，线上A/B推荐驱动消费2.94→6.83首/日(2.3x)</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;度量学习&quot;,&quot;翻译向量&quot;,&quot;记忆网络注意力&quot;]"><strong><a href="collaborative_filtering/hierarchical_latent_relation_modeling_for_collaborative_metric_learning/">Hierarchical Latent Relation Modeling for Collaborative Metric Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用记忆网络注意力分层建模物品-物品与用户-物品潜在翻译向量，改造 CML 三元组损失；Full-Ranking 下 MovieLens/Echonest 排名指标较 AdaCML 提升约 7%/18%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;NeuMF&quot;,&quot;可复现性评估&quot;,&quot;长尾与偏差&quot;]"><strong><a href="collaborative_filtering/reenvisioning_collaborative_filtering_vs_matrix_factorization/">Reenvisioning Collaborative Filtering vs Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>复现 NCF vs MF 精度对比并扩展至长尾、多样性与偏差维度，证实 MF 精度与长尾双赢、NeuMF 覆盖率更高。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Reverse MIPS&quot;,&quot;精确检索&quot;,&quot;下界索引&quot;]"><strong><a href="collaborative_filtering/reverse_maximum_inner_product_search_how_to_efficiently_find_users_who_woul/">Reverse Maximum Inner Product Search: How to efficiently find users who would like to buy my item?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出精确反向MIPS算法Simpfer，用范数排序、下界数组与块级过滤实现O(1)剪枝，四数据集至少两个数量级加速。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;稀疏特征分解&quot;,&quot;信息增益赋权&quot;]"><strong><a href="graph_rec/sparse_feature_factorization_for_recommender_systems_with_knowledge_graphs/">Sparse Feature Factorization for Recommender Systems with Knowledge Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>KGFlex 以熵驱动信息增益为用户-特征对赋权，仅在用户相关 KG 特征上做稀疏因子分解，全库排序下精度可比、多样性与长尾覆盖显著更优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序离线评估&quot;,&quot;采样评估&quot;,&quot;Kendall Tau&quot;]"><strong><a href="sequential_rec/a_case_study_on_sampling_strategies_for_evaluating_neural_sequential_item_r/">A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>复训4个神经序列推荐模型于5数据集，证明Uniform与Popular采样排序均可能偏离全库排序，η=100无通用安全性，应优先Full-Ranking。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;正交投影去噪&quot;,&quot;用户记忆网络&quot;]"><strong><a href="sequential_rec/denoising_user_aware_memory_network_for_recommendation/">Denoising User-aware Memory Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以显式反馈正交投影净化隐式反馈噪声，并用四路用户级记忆网络建模长期兴趣，AUC 较最优基线提升 1.18/0.97 个点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时间集合预测&quot;,&quot;Tweedie 分布 VAE&quot;]"><strong><a href="sequential_rec/global_local_item_embedding_for_temporal_set_prediction/">Global-Local Item Embedding for Temporal Set Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GLOIE 用时间衰减集合和的 VAE（Tweedie 输出）刻画全局协同信号，与动态图学到的局部序列嵌入做注意力融合，在三个下一篮基准的全库排序上稳定领先 DNNTSP。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;端到端排序（next-click 预测）&quot;,&quot;分层 Transformer&quot;,&quot;神经指针网络&quot;]"><strong><a href="sequential_rec/large_scale_modeling_of_mobile_user_click_behaviors_using_deep_learning/">Large-Scale Modeling of Mobile User Click Behaviors Using Deep Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>分层 Transformer + 神经指针在真实屏幕候选元素上做 next-click 全候选排序，用户级划分下 Top-1 48.3%、Top-3 71.4%，较最强深度基线提升 25.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回/会话级排序&quot;,&quot;双曲表征&quot;,&quot;目标估计&quot;,&quot;弱监督&quot;]"><strong><a href="sequential_rec/learning_to_represent_human_motives_for_goal_directed_web_browsing/">Learning to Represent Human Motives for Goal-directed Web Browsing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 GoWeB，用心理学目标分类体系在双曲空间学习目标表征并估计页面访问目标，统一增强会话推荐、重访预测与目标聚类。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;小样本元学习&quot;,&quot;跨会话相似检索&quot;]"><strong><a href="sequential_rec/next_item_recommendations_in_short_sessions/">Next-item Recommendations in Short Sessions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将短会话下一项预测建模为小样本学习，用 SSRN 检索跨会话相似历史并做特征级调制，在 Delicious/Reddit 短会话上 Recall@5 最高提升 17.2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;排序（历史候选重排）&quot;,&quot;认知架构 ACT-R&quot;,&quot;重听行为建模&quot;]"><strong><a href="sequential_rec/predicting_music_relistening_behavior_using_the_act_r_framework/">Predicting Music Relistening Behavior Using the ACT-R Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 ACT-R 声明记忆五组件建模音乐重听，一周滑窗逐 LE 预测会话剩余曲目，R-prec 0.111，较最强单组件 +6.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;自适应序列长度&quot;,&quot;Actor-Critic 强化学习&quot;,&quot;联合损失正则&quot;]"><strong><a href="sequential_rec/sequence_adaptation_via_reinforcement_learning_in_recommender_systems/">Sequence Adaptation via Reinforcement Learning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 actor-critic 把“回看序列长度 l”建模为 MDP 动作，按用户个性化裁剪子序列并以 Q 值乘性正则交叉熵，Steam/ML-10M 上 NDCG@10 相对最优基线提升 6.5%/12.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排（评论驱动的交互式列表重排）&quot;,&quot;多模态变分自编码器&quot;,&quot;弱监督训练&quot;,&quot;多步 Critiquing&quot;]"><strong><a href="agent_conversational_rec/fast_multi_step_critiquing_for_vae_based_recommender_systems/">Fast Multi-Step Critiquing for VAE-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>M&amp;Ms-VAE 以多模态条件独立 + 专家混合后验建模交互与关键短语，配套冻结主干的自监督门控重排，三任务同时领先且评论推理最高 25.6× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;多步批判 (Multi-step Critiquing)&quot;,&quot;可解释推荐界面&quot;]"><strong><a href="agent_conversational_rec/multi_step_critiquing_user_interface_for_recommender_systems/">Multi-Step Critiquing User Interface for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建四套模型无关的多步批判式酒店推荐 Web 界面，把 T-RECS/CE-VAE 的关键词解释与自然语言理由落到真实交互，并定性暴露解释粒度与关键词模糊等失效模式。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;引力启发图自编码器&quot;,&quot;有向链接预测&quot;,&quot;工业音乐推荐&quot;]"><strong><a href="cold_start_rec/cold_start_similar_artists_ranking_with_gravity_inspired_graph_autoencoders/">Cold Start Similar Artists Ranking with Gravity-Inspired Graph Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>把新艺人相似艺人排序建模为有向属性图链接预测，用引力启发图自编码器的质量-距离解码器，全库排序下 NDCG@200 相对提升 21%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;全空间建模&quot;,&quot;多任务学习&quot;,&quot;样本选择偏差&quot;]"><strong><a href="ctr_feature_interaction/an_analysis_of_entire_space_multi_task_models_for_post_click_conversion_pre/">An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在 Twitter 广告 5B 曝光工业数据上系统消融 ESMM 的三大要素，发现仅需硬参数共享或仅需点击加权即可获得全部增益，且二者不可叠加</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;页面级序列决策&quot;,&quot;上下文 Bandit 数据采集&quot;,&quot;位置偏置 IPS&quot;]"><strong><a href="multi_task_reranking/page_level_optimization_of_e_commerce_item_recommendations/">Page-level Optimization of e-Commerce Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将商品详情页模块的选序建模为两阶段 RNN 序列决策，用模块族嵌入压缩动作空间、束搜索+IPS+Thompson 采样实现线上个性化重排，线上 CTR +2.48%、PTR +7.34%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Search Recommendation&quot;,&quot;Results Organization&quot;,&quot;Instant Search&quot;]"><strong><a href="multi_task_reranking/recommendations_and_results_organization_in_netflix_search/">Recommendations and Results Organization in Netflix Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>Netflix 搜索把协同过滤推荐注入检索，并把结果组织成带标签的横向分组列表做列表级重排，属无量化指标的工业系统论文。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;离线评测&quot;,&quot;Off-Policy Evaluation&quot;,&quot;Double/Debiased Machine Learning&quot;,&quot;Neyman Orthogonality&quot;]"><strong><a href="rl_bandit_rec/debiased_off_policy_evaluation_for_recommendation_systems/">Debiased Off-Policy Evaluation for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将 Doubly Robust 与 Double/Debiased ML 结合，用 Neyman 正交+cross-fitting 实现 √N 一致的去偏离线策略评估，线上广告 relative-RMSE 较 DR 降 13.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;离线策略评估&quot;,&quot;Off-Policy Evaluation&quot;,&quot;CDF 鲁棒性分析&quot;,&quot;反事实估计&quot;]"><strong><a href="rl_bandit_rec/evaluating_the_robustness_of_off_policy_evaluation/">Evaluating the Robustness of Off-Policy Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 IEOE，以性能 CDF/AU-CDF/CVaR 替代单点 MSE，量化 OPE 估计器对超参与评估策略的鲁棒性，在 Open Bandit 与真实电商数据上给出安全选型准则。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (16)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦协同过滤&quot;,&quot;多臂老虎机&quot;]"><strong><a href="debias_fairness_industrial/a_payload_optimization_method_for_federated_recommender_systems/">A Payload Optimization Method for Federated Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以贝叶斯 Thompson Sampling 在联邦服务器端挑选物品子集传输，实现 90% 载荷削减下仅 4%–8% 精度损失</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端推荐审计&quot;,&quot;Filter Bubble&quot;,&quot;Sockpuppet Audit&quot;,&quot;Misinformation&quot;]"><strong><a href="debias_fairness_industrial/an_audit_of_misinformation_filter_bubbles_on_youtube_bubble_bursting_and_re/">An Audit of Misinformation Filter Bubbles on YouTube: Bubble Bursting and Recent Behavior Changes</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>50 次 sockpuppet 代理审计发现 YouTube 搜索端无气泡生成、推荐端有，且连看 40 条辟谣视频可令推荐归一化分从 0.01 拉回 -0.27。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平&quot;,&quot;召回&quot;,&quot;流行度偏置&quot;,&quot;性别公平性&quot;]"><strong><a href="debias_fairness_industrial/analyzing_item_popularity_bias_of_music_recommender_systems_are_different_g/">Analyzing Item Popularity Bias of Music Recommender Systems: Are Different Genders Equally Affected?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用中位数/高阶矩/KL/Kendall τ 七维分布度量替代均值差，在 LFM-2b 上揭示七种 CF 算法对女性用户施加更强流行度偏置。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;模型窃取&quot;,&quot;数据投毒&quot;,&quot;序列推荐&quot;]"><strong><a href="debias_fairness_industrial/black_box_attacks_on_sequential_recommenders_via_data_free_model_extraction/">Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>无训练数据下用自回归合成查询 + 排序蒸馏窃取序列推荐模型，并迁移为画像污染与数据投毒攻击，ML-1M 上 Agr@10 达 0.747。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Pairwise Ranking&quot;,&quot;可解释性加权&quot;,&quot;曝光偏差IPS&quot;]"><strong><a href="debias_fairness_industrial/debiased_explainable_pairwise_ranking_from_implicit_feedback/">Debiased Explainable Pairwise Ranking from Implicit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在 BPR 上引入 item 邻域可解释性加权 EBPR，并量化其额外曝光偏差，用 IPS 邻域倾向构造无偏估计 UEBPR，NDCG/MEP 双赢且自带流行度去偏。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/离线评估&quot;,&quot;Slate Exposure&quot;,&quot;Candidate Sampling Bias&quot;]"><strong><a href="debias_fairness_industrial/finn_no_slates_dataset_a_new_sequential_dataset_logging_interactions_allvie/">FINN.no Slates Dataset: A new Sequential Dataset Logging Interactions, allViewed Items and Click Responses/No-Click for Recommender Systems Research</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>发布FINN.no 30天真实市场slate级序列日志，含曝光列表、点击/no-click与搜索/推荐来源，支撑曝光条件似然与离线策略评估。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与工业架构&quot;,&quot;离线分析/消费审计&quot;,&quot;曝光入口归因&quot;,&quot;内容多样性度量&quot;]"><strong><a href="debias_fairness_industrial/follow_the_guides_disentangling_human_and_algorithmic_curation_in_online_mu/">Follow the Guides: Disentangling Human and Algorithmic Curation in Online Music Consumption</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 Deezer 9k 用户全年 51M 播放为样本，将入口三分为自然/算法/编辑并聚类出四类用户，揭示编辑拉向流行、算法拉向长尾的相反效应与「过滤小生境」结论</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/在线模型更新&quot;,&quot;增量更新与持续学习&quot;,&quot;GRU 元生成器&quot;,&quot;灾难性遗忘&quot;]"><strong><a href="debias_fairness_industrial/learning_an_adaptive_meta_model_generator_for_incrementally_updating_recomm/">Learning an Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 GRU 元生成器把增量模型序列压缩成更好服务下一周期的模型，截断序列+多步并发训练，四数据集 AUC 相对提升约 1%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;差分隐私推荐&quot;,&quot;簇级代理用户&quot;,&quot;编辑在环&quot;]"><strong><a href="debias_fairness_industrial/nu_brief_a_privacy_aware_newsletter_personalization_engine_for_publishers/">NU:BRIEF -- A Privacy-aware Newsletter Personalization Engine for Publishers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>RecSys 2021 Demo 论文：以 k-means 兴趣簇代理匿名用户 + 差分隐私 taste vector 实现出版商通讯个性化，A/B 显示 CTR 与人工策展持平、策展耗时下降 80%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;在线评估/实验平台&quot;,&quot;Causal Effect&quot;,&quot;Interleaving&quot;,&quot;Inverse Propensity Score&quot;]"><strong><a href="debias_fairness_industrial/online_evaluation_methods_for_the_causal_effect_of_recommendations/">Online Evaluation Methods for the Causal Effect of Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个因果 interleaving 在线评估框架，用 EPI/CBI+IPS 无偏比较推荐因果效应，模拟中比 A/B 测试高效约 30 倍。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部16篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
