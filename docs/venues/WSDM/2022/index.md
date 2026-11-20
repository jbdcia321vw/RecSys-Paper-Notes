---
hide:
  - toc
---

# 🏛️ WSDM 2022 推荐系统论文解读

22 篇 WSDM 2022 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多行为推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/评测基准&quot;,&quot;数据采样与 Coreset&quot;,&quot;Kendall&#x27;s Tau 排名保持性&quot;,&quot;Propensity 去偏&quot;]"><strong><a href="collaborative_filtering/on_sampling_collaborative_filtering_datasets/">On Sampling Collaborative Filtering Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统刻画 16 种 CF 数据采样策略对算法排名保持性的影响，提出 SVP-CF 代理采样与 Data-genie 采样器推荐器，仅用 10% 数据即可保持排名，带来 5.8× 训练加速。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图表征学习&quot;,&quot;分布式训练系统&quot;]"><strong><a href="graph_rec/graph4rec_a_universal_toolkit_with_graph_neural_networks_for_recommender_sy/">Graph4Rec: A Universal Toolkit with Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 GNN4Rec 训练统一为五段式流水线，配分布式图引擎与参数服务器，在 ICF/UCF/U2I 召回协议下系统对比 8 类 GNN 并给出工程实践建议。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2020&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;离散哈希码学习&quot;,&quot;直通估计器STE&quot;]"><strong><a href="graph_rec/learning_to_hash_with_graph_neural_networks_for_recommender_systems/">Learning to Hash with Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HashGNN 端到端联合学习 GNN 连续表征与二值哈希码，融合链接重建与 Hamming 排序损失，并以 Guidance STE 消除梯度放大；全库排序 HR@50 较 GraphSage 提升 4–10%，检索提速 2.5–3.6 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;洛伦兹双曲几何&quot;,&quot;知识图谱推荐&quot;,&quot;注意力聚合&quot;]"><strong><a href="graph_rec/modeling_scale_free_graphs_with_hyperbolic_geometry_for_knowledge_aware_rec/">Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户-物品-KG 三分图嵌入洛伦兹双曲流形，用异构解耦传播与知识感知注意力建模无标度层级结构，Full-Ranking 下 Recall@20 提升 3.6–15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;跨层互关系注意力&quot;]"><strong><a href="graph_rec/multi_behavior_graph_neural_networks_for_recommender_system/">Multi-Behavior Graph Neural Networks for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MBRec 以多通道行为编码、跨行为自注意力与跨层互关系融合建模多类型交互，三数据 1:99 采样排序下 HR@10 超最强基线 4.7%–6.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;贝叶斯图神经网络&quot;,&quot;图生成式数据增强&quot;]"><strong><a href="graph_rec/preference_and_concurrence_aware_bayesian_graph_neural_networks_for_recomme/">Preference and Concurrence Aware Bayesian Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用用户偏好、物品共现与度约束的 PECO 图生成器增强 BGNN，四数据集 Recall@20 较 Node-Copy 最高提升 1.44%。</p>
</dd>
<dt data-paper-tags="[&quot;ISWC 2020&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/CTR 打分&quot;,&quot;规则学习 (Rule Learning)&quot;,&quot;知识图谱嵌入规则过滤 (RotatE)&quot;,&quot;冷启动推荐&quot;]"><strong><a href="graph_rec/rule_guided_graph_neural_networks_for_recommender_systems/">Rule-Guided Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用自动规则学习为 KG 上的 GraphSAGE 式聚合提供显式长程语义与采样先验，Dianping-Food AUC 相对最优基线提升 4.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;场景图&quot;,&quot;层次图注意力&quot;,&quot;BPR&quot;]"><strong><a href="graph_rec/scenerec_scene_based_graph_neural_networks_for_recommender_systems/">SceneRec: Scene-Based Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>把场景形式化为共现类目集合，构建物品-类目-场景三层图并用场景注意力聚合物品表征，JD 四数据集 NDCG@10 提升 5.9%–25.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;,&quot;元学习&quot;]"><strong><a href="self_supervised_denoising/contrastive_meta_learning_with_behavior_multiplicity_for_recommendation/">Contrastive Meta Learning with Behavior Multiplicity for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用行为感知图传播、跨行为 InfoNCE 与元权重网络解决多行为推荐中目标行为监督稀疏和个性化依赖建模问题。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;表示退化 (Anisotropy)&quot;,&quot;对比学习正则 (Dropout 增广 / 监督正样本)&quot;]"><strong><a href="sequential_rec/contrastive_learning_for_representation_degeneration_problem_in_sequential_/">Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐器的物品嵌入各向异性退化问题，用 Dropout 双视图与同目标难正样本的对比正则（DuoRec）在全库排序下 HR@5 最高提升 104%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构全局图&quot;,&quot;个性化会话编码&quot;]"><strong><a href="sequential_rec/heterogeneous_global_graph_neural_networks_for_personalized_session_based_r/">Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建用户-物品异构全局图并叠加双粒度偏好门控，在 Last.fm/Xing/Reddit 全库 HR/MRR 上相对 SOTA 提升 1.0%–6.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时序元学习&quot;,&quot;异构图注意力&quot;]"><strong><a href="sequential_rec/long_short_term_temporal_meta_learning_in_online_recommendation/">Long Short-Term Temporal Meta-learning in Online Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双图(全局长期/内部短期)+门控融合+时序 MAML 异步优化，在微信看一看 10 亿级行为上做小时级 AUC 与线上 A/B 验证，CTR 相对 +9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;Transformer&quot;,&quot;Attention2D&quot;,&quot;多属性序列&quot;]"><strong><a href="sequential_rec/sequential_modeling_with_multiple_attributes_for_watchlist_recommendation_i/">Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出Trans2D/Attention2D，在item×attribute二维序列上建模三类交互，用于eBay watchlist重排；P@1较最佳基线提升11.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;重排（展示层流量分配）&quot;,&quot;Thompson Sampling&quot;,&quot;动态先验&quot;,&quot;批式更新探索&quot;]"><strong><a href="cold_start_rec/dynamic_prior_thompson_sampling_for_cold_start_exploration_in_recommender_s/">Dynamic Prior Thompson Sampling for Cold-Start Exploration in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用闭式二次方程反解新臂 Beta 先验均值，使 P(新臂胜现任冠军)=ε，在线 QPTR +0.19%、后悔曝光相对降 21.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多域对抗分离 MDSN&quot;,&quot;门控方面偏好增强 GAPE&quot;]"><strong><a href="cross_domain_multi_scenario/multi_sparse_domain_collaborative_recommendation_via_enhanced_comprehensive/">Multi-Sparse-Domain Collaborative Recommendation via Enhanced Comprehensive Aspect Preference Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MSDCR 以对抗分离+门控增强学习域特有方面偏好、以域适应学习域不变偏好，在多稀疏域 Douban 数据上 leave-one-out+99 负采样评测 NDCG@5 最高相对提升 31.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;元学习个性化桥接&quot;,&quot;冷启动偏好迁移&quot;]"><strong><a href="cross_domain_multi_scenario/personalized_transfer_of_user_preferences_for_cross_domain_recommendation/">Personalized Transfer of User Preferences for Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用元网络按用户特征生成个性化桥接函数替代共享桥，配合任务导向优化，在 Amazon 三组跨域冷启动任务上 MAE 相对下降 3.6%–10.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Auto-bidding 自动出价&quot;,&quot;多智能体强化学习&quot;,&quot;信用分配&quot;]"><strong><a href="ctr_feature_interaction/a_cooperative_competitive_multi_agent_framework_for_auto_bidding_in_online_/">A Cooperative-Competitive Multi-Agent Framework for Auto-bidding in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以温度正则化信用分配构建合作-竞争混合多智能体出价框架 MAAB，配合 bar agent 与均值智能体，在离线仿真与淘宝在线 A/B 上同时提升社会福利与平台收入。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Triangle Motif&quot;,&quot;DPP Diversity&quot;]"><strong><a href="ctr_feature_interaction/triangle_graph_interest_network_for_click_through_rate_prediction/">Triangle Graph Interest Network for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以物品共现图三角形为兴趣单元，DPP 选多样相关三角；AUC 较 DMIN 最高提升 4.49%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排（交互呈现层）&quot;,&quot;探索-利用权衡&quot;,&quot;界面透明度&quot;,&quot;被试内用户实验&quot;]"><strong><a href="rl_bandit_rec/an_empirical_analysis_on_transparent_algorithmic_exploration_in_recommender/">An Empirical Analysis on Transparent Algorithmic Exploration in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 94 人被试内实验证明：把探索项从「伪装混入」改为「显式分区揭示」，可在不损感知准确率的前提下提升探索项反馈量与用户满意度。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;层级树检索&quot;,&quot;用户兴趣探索&quot;]"><strong><a href="rl_bandit_rec/show_me_the_whole_world_towards_entire_item_space_exploration_for_interacti/">Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用共点击表征自底向上聚出物品层级树，把全库 Contextual Bandit 探索改写为根到叶序列决策 (HCB) 与渐进感受野扩展 (pHCB)，IPS 模拟器上累积奖励最高提升 280%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排（包级列表组合枚举）&quot;,&quot;ZDD 零压缩决策图&quot;,&quot;固定参数可解性 (FPT)&quot;,&quot;群组公平 (Proportionality / Envy-freeness)&quot;]"><strong><a href="debias_fairness_industrial/enumerating_fair_packages_for_group_recommendations/">Enumerating Fair Packages for Group Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 ZDD 压缩枚举全部满足比例公平/无嫉妒约束的包，证明群组规模参数下 FPT，$O(n(K+g)2^g)$内产出万亿级候选并支持交集/top-$K$/采样过滤。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评测/鲁棒性评测&quot;,&quot;Robustness Evaluation&quot;,&quot;RecBole&quot;,&quot;Subpopulation Fairness&quot;]"><strong><a href="debias_fairness_industrial/rgrecsys_a_toolkit_for_robustness_evaluation_of_recommender_systems/">RGRecSys: A Toolkit for Robustness Evaluation of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出RGRecSys统一评估推荐模型在子群体、分布偏移、特征变换、攻击与稀疏下的鲁棒性，基于RecBole与MovieLens-100k做案例展示。</p>
</dd>
</dl>
</section>
</div>
