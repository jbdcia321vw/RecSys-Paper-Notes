---
hide:
  - toc
---

# 🏛️ ECIR 2024 推荐系统论文解读

15 篇 ECIR 2024 论文解读，覆盖 7 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向 Transformer&quot;,&quot;多正样本对比学习&quot;]"><strong><a href="sequential_rec/contrastive_learning_with_bidirectional_transformers_for_sequential_recomme/">Contrastive Learning with Bidirectional Transformers for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用双向 Transformer + 滑窗切分 + 完形/ dropout 混合增广构造多正样本对比学习，并动态重加权双损失；在 Beauty/Toys/ML-1M 全库排序下 NDCG@10 最高提升 21.04%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;序列 Next-Item 排序&quot;,&quot;离线强化学习&quot;,&quot;Critic Regularized Regression&quot;,&quot;DistilGPT2&quot;]"><strong><a href="sequential_rec/integrating_offline_reinforcement_learning_with_transformers_for_sequential/">Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DistilGPT2 初始化 CRR 离线 RL 做两阶段序列推荐；在 ML-25M/Yoochoose 全库排序 HR@10 显著超越 SASRec。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序（Next-Item）&quot;,&quot;模态遗忘&quot;,&quot;相关性蒸馏&quot;]"><strong><a href="sequential_rec/lightweight_modality_adaptation_to_sequential_recommendation_via_correlatio/">Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KDSR，以整体/分解模态相关性蒸馏和异步训练缓解两阶段序列推荐的模态遗忘，5 个数据集 HR@5 最高提升 23.9%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;GNN&quot;,&quot;可复现性评测&quot;]"><strong><a href="sequential_rec/performance_comparison_of_session_based_recommendation_algorithms_based_on_/">Performance Comparison of Session-based Recommendation Algorithms based on GNNs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一复现8个GNN会话推荐模型，发现简单kNN/规则基线在MRR@20上不劣甚至更优，并揭示种子与嵌入维度敏感。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/下一物品排序&quot;,&quot;自对比学习&quot;,&quot;表示均匀性&quot;]"><strong><a href="sequential_rec/self_contrastive_learning_for_session_based_recommendation/">Self Contrastive Learning for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将自对比损失作为物品表示均匀性正则，无需数据增广/正负样本构造，在三个会话推荐基准全库排序下显著提升 P@10/MRR@10。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端&quot;,&quot;检索增强生成&quot;,&quot;话题标签推荐&quot;,&quot;难负样本对比学习&quot;]"><strong><a href="generative_rec/right_retrieval_augmented_generation_for_mainstream_hashtag_recommendation/">RIGHT: Retrieval-augmented Generation for Mainstream Hashtag Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用检索器+选择器+生成器的检索增强框架做主流话题标签推荐，在 Twitter/Weibo 两数据集上 F1@1 最高提升约 10.8%，并可零样本注入 ChatGPT 带来 12.7%/18.3% 增益。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;LLaMA-LoRA&quot;,&quot;指令化序列模板&quot;]"><strong><a href="llm_rec/genrec_large_language_model_for_generative_recommendation/">GenRec: Large Language Model for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLaMA-LoRA 将用户交互历史改写为指令序列，直接生成下一物品名称而非逐候选打分；MovieLens-25M 上 HR@5 相对 P5 提升 50.3%，但稀疏 Amazon-Toys 上反而下降。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;LLM推荐&quot;,&quot;重排&quot;,&quot;Zero-Shot Ranking&quot;,&quot;Prompting &amp; Bootstrapping&quot;]"><strong><a href="llm_rec/large_language_models_are_zero_shot_rankers_for_recommender_systems/">Large Language Models are Zero-Shot Rankers for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将推荐形式化为条件排序，用提示与自举缓解顺序/位置/流行度偏差，在 ML-1M/Games 的 20 候选零样本排序中显著优于 UniSRec。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮交互在线推荐&quot;,&quot;联邦学习&quot;,&quot;本地差分隐私&quot;]"><strong><a href="agent_conversational_rec/federated_conversational_recommender_system/">Federated Conversational Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>FedCRS 将历史兴趣估计与交互式偏好诱导两阶段全部去中心化，用梯度裁剪+Laplace 噪声实现用户级 LDP，SR@15 逼近非隐私 SOTA</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;地理分组自适应路由&quot;,&quot;Co-Action Network&quot;,&quot;EM/REINFORCE 联合优化&quot;]"><strong><a href="poi_spatiotemporal_rec/an_adaptive_framework_of_geographical_group_specific_network_on_o2o_recomme/">An Adaptive Framework of Geographical Group-Specific Network on O2O Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GeoGrouse 用可学习地理分组隐变量驱动「共享+分组专用」双塔策略网络，在饿了么零售到店场景 7 天线上 A/B 中 CTR/ACR 双涨。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;后处理分桶&quot;,&quot;提供者公平&quot;,&quot;类型校准&quot;]"><strong><a href="multi_task_reranking/moregin_multi_objective_recommendation_at_the_global_and_individual_levels/">MOReGIn: Multi-Objective Recommendation at the Global and Individual Levels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>分桶后处理重排同时实现全局提供者地理公平与个体类型校准，在 Movies/Songs 双数据集上将 ΔTotal 降为 0、校准误差降一个量级。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（pairwise 训练期采样）&quot;,&quot;提供者组曝光公平&quot;,&quot;代价敏感元学习&quot;,&quot;重加权负采样&quot;]"><strong><a href="debias_fairness_industrial/a_cost_sensitive_meta_learning_strategy_for_fair_provider_exposure_in_recom/">A Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用类别代价重加权三元组负采样，使提供者组曝光对齐目录占比，且 NDCG 无损（ML-1M Exp@10 由 1.4% 升至 6.0%）。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/重排与评测协议&quot;,&quot;top-n 选择策略&quot;,&quot;超参优化偏差&quot;,&quot;显著性检验&quot;]"><strong><a href="debias_fairness_industrial/revealing_the_hidden_impact_of_top_n_metrics_on_optimization_in_recommender/">Revealing the Hidden Impact of Top-N Metrics on Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>穷举 252 种 top-n 选择策略 × 20 数据集 × 11 算法，证明 top-n 评估截断带来的优化偏差在统计上可忽略（增益 ≤1.5%）</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;图神经网络推荐&quot;,&quot;鲁棒性评估&quot;,&quot;边级扰动&quot;]"><strong><a href="debias_fairness_industrial/robustness_in_fairness_against_edge_level_perturbations_in_gnn_based_recomm/">Robustness in Fairness against Edge-level Perturbations in GNN-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ECIR 2024 去偏公平研究首次评估图推荐系统在边级扰动下的公平性鲁棒性，设计可微扰动目标，揭示消费者侧公平性极为脆弱。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序（Top-10 推荐）&quot;,&quot;Differential Privacy&quot;,&quot;Popularity Bias&quot;]"><strong><a href="debias_fairness_industrial/the_impact_of_differential_privacy_on_recommendation_accuracy_and_popularit/">The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统实证 1-Bit 差分隐私对三种协同过滤模型准确率与流行度偏差的影响，揭示 ε 越小准确率降幅越大且长尾偏好用户受损更重。</p>
</dd>
</dl>
</section>
</div>
