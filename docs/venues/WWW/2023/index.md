---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2023 推荐系统论文解读

65 篇 WWW (TheWebConf) 2023 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** WWW 2023 ×56 · WWW ×9 · Actor-Critic ×4 · 对比学习 ×4 · 多行为推荐 ×3 · CTR预估与特征交叉 ×2 · 双层优化 ×2 · 多模态推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">31</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Actor-Critic" aria-pressed="false">Actor-Critic<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="本地差分隐私" aria-pressed="false">本地差分隐私<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;易到难课程学习&quot;,&quot;异构模型集成&quot;]"><strong><a href="collaborative_filtering/distillation_from_heterogeneous_models_for_top_k_recommendation/">Distillation from Heterogeneous Models for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用教师训练轨迹构造由易到难的排序知识课程，将异构推荐模型集成蒸馏进轻量学生，R@10 最高提升 24.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;聚类分治难负例&quot;,&quot;Prompt 式多任务学习&quot;,&quot;双塔 ANN 检索&quot;]"><strong><a href="collaborative_filtering/divide_and_conquer_towards_better_embedding_based_retrieval_for_recommender/">Divide and Conquer: Towards Better Embedding-based Retrieval for Recommender Systems From a Multi-task Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将候选库语义聚类后分簇并行 EBR，仅用同簇难负例训练并配合 Prompt 式多任务适配，KuaiRand Recall 最高提升 42.7%，线上 A/B 全指标正向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图划分&quot;,&quot;物品相似度模型&quot;,&quot;谱聚类&quot;]"><strong><a href="collaborative_filtering/fine_tuning_partition_aware_item_similarities_for_efficient_and_scalable_re/">Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用谱划分把物品相似度建模限制在子划分内，并以全局谱信息微调；4 数据集上全库排序指标超越 GCN，提速 10×、省 95% 参数。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;对比学习&quot;,&quot;图神经网络&quot;]"><strong><a href="collaborative_filtering/graph_less_collaborative_filtering/">Graph-less Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双层知识蒸馏把 GCN 教师压缩为 MLP 学生，配合梯度自适应对比正则，在 Full-Ranking 协议下兼顾精度与推理效率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负采样&quot;,&quot;OPAUC&quot;,&quot;DRO&quot;]"><strong><a href="collaborative_filtering/on_the_theories_behind_hard_negative_sampling_for_recommendation/">On the Theories Behind Hard Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>证明BPR难负采样等价优化OPAUC，并给出可控硬度与按K调采样策略。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（Top-N 排序）&quot;,&quot;可解释推荐&quot;,&quot;口味簇学习&quot;,&quot;稀疏隶属矩阵&quot;,&quot;直通梯度估计&quot;]"><strong><a href="collaborative_filtering/towards_explainable_collaborative_filtering_with_taste_clusters_learning/">Towards Explainable Collaborative Filtering with Taste Clusters Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用稀疏「口味簇」隶属矩阵实现自解释 CF，Forest 集成补偿稀疏化损失，三数据集 Recall@20 相对最优基线提升 8.0%–14.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图多行为建模&quot;,&quot;代码语义-结构融合&quot;]"><strong><a href="graph_rec/code_recommendation_for_open_source_software_developers/">Code Recommendation for Open Source Software Developers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CODER 用异构图联合建模用户-文件微观贡献与用户-项目宏观行为，并以项目文件树聚合桥接两级信号，全库排序下 NDCG@20 最高提升 32.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回与表示学习&quot;,&quot;CIR拓扑度量&quot;,&quot;协同感知图卷积&quot;]"><strong><a href="graph_rec/collaboration_aware_graph_convolutional_network_for_recommender_systems/">Collaboration-Aware Graph Convolutional Network for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出CIR拓扑度量与CAGC协同感知图卷积，按邻居共同交互比例重加权，6数据集Full-Ranking下Recall@20较LightGCN平均约+9.1%，训练总时间最高降96.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;实例级高阶关系&quot;,&quot;多任务梯度解耦&quot;]"><strong><a href="graph_rec/compressed_interaction_graph_based_framework_for_multi_behavior_recommendat/">Compressed Interaction Graph based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用矩阵乘法显式构造实例级高阶交互图并以节点注意力压缩图空间，配合分离输入多专家网络解耦多任务梯度，Taobao HR@10 相对提升 20.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;自适应聚合跳数&quot;,&quot;双策略强化学习&quot;]"><strong><a href="graph_rec/dual_policy_learning_for_aggregation_optimization_in_graph_neural_network_b/">Dual Policy Learning for Aggregation Optimization in Graph Neural Network-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双 DQN 为每个用户/物品自适应选择聚合跳数，替代固定 GNN 层数超参，LightGCN/KGAT 上 nDCG 最高 +63.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;持续图学习&quot;,&quot;动态可扩展图卷积&quot;,&quot;流式推荐&quot;]"><strong><a href="graph_rec/dynamically_expandable_graph_convolution_for_streaming_recommendation/">Dynamically Expandable Graph Convolution for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以模型隔离视角对 GCN 卷积核执行剪枝-精炼-扩张，配合时序注意力初始化，实现免历史回放的流式推荐，Taobao2014 R@20 最高提升 146%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;级联图卷积&quot;,&quot;行为链依赖&quot;]"><strong><a href="graph_rec/multi_behavior_recommendation_with_cascading_graph_convolution_networks/">Multi-Behavior Recommendation with Cascading Graph Convolution Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用级联 LightGCN + 特征变换逐跳传递行为链特征、仅用目标行为监督，Beibei/Tmall 全库排序下 Recall@10 相对最强基线提升 23.2%/44.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;混合频率滤波器&quot;,&quot;非参数协同过滤&quot;]"><strong><a href="graph_rec/personalized_graph_signal_processing_for_collaborative_filtering/">Personalized Graph Signal Processing for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用个性化图信号+增广相似图+混合频率图滤波构建免训练的非参数 CF，随机划分全库排序下 NDCG 平均超越 SOTA 3.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排（训练数据级预处理）&quot;,&quot;深度强化学习策略网络&quot;,&quot;实例级去噪与数据子集迁移&quot;]"><strong><a href="self_supervised_denoising/autodenoise_automatic_data_instance_denoising_for_recommendations/">AutoDenoise: Automatic Data Instance Denoising for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 DRL 策略网络逐实例做 select/deselect 去噪，配合搜索-验证两阶段交替训练，产出可迁移的干净子集，DeepFM 等骨干 AUC 稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;可学习图掩码&quot;,&quot;掩码图自编码器&quot;,&quot;Infomax 语义相关性&quot;]"><strong><a href="self_supervised_denoising/automated_self_supervised_learning_for_recommendation/">Automated Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>AutoCF 用可学习掩码 + 图自注意力掩码自编码器替代手工图增广，在全库排序下相对最优 SSL 基线提升 7.8%–33.7% Recall@20。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;提示微调&quot;,&quot;图重构去噪&quot;]"><strong><a href="self_supervised_denoising/denoising_and_prompt_tuning_for_multi_behavior_recommendation/">Denoising and Prompt-Tuning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DPT 以三阶段范式先用重构难度无监督去噪辅助行为，再用重初始化与连续提示轻量微调，在 IJCAI/Tmall 上 HR@10 提升 1.58%–2.62%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;停留时间重加权&quot;,&quot;有效阅读筛选&quot;,&quot;多任务学习&quot;]"><strong><a href="self_supervised_denoising/reweighting_clicks_with_dwell_time_in_recommendation/">Reweighting Clicks with Dwell Time in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用停留时间定义有效阅读并以归一化 Sigmoid 重加权点击信号，工业级双塔多任务框架在微信看一看离线 AUC+5.62%、线上 CTR+2.58%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;排序&quot;,&quot;反事实推断&quot;,&quot;因果去偏&quot;,&quot;会话协同过滤&quot;]"><strong><a href="sequential_rec/a_counterfactual_collaborative_session_based_recommender_system/">A Counterfactual Collaborative Session-based Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将会话外因素建模为混杂因子，以自监督伪标签预训练+反事实协同推理消除伪相关，三数据集 R@20 最高提升 21.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自监督序列校正&quot;,&quot;逆序生成插入&quot;]"><strong><a href="sequential_rec/a_self_correcting_sequential_recommender/">A Self-Correcting Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>STEAM 用 item-wise 校正器（keep/delete/insert）+ 两个自监督纠错任务先修序列再推荐，三个数据集 5-core/LOO/99 负采样下 HR@5 最高 +5.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序打分&quot;,&quot;MLP-Mixer&quot;,&quot;可微架构搜索&quot;,&quot;长短期兴趣解耦&quot;]"><strong><a href="sequential_rec/automlp_automated_mlp_for_sequential_recommendations/">AutoMLP: Automated MLP for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双 SRSMLP 分支解耦长/短期兴趣，并以 DARTS 可微搜索短期窗口长度，线性复杂度下 Beauty 的 MRR@10 相对提升 8.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;从众权重解耦&quot;,&quot;去偏对比学习&quot;]"><strong><a href="sequential_rec/debiased_contrastive_learning_for_sequential_recommendation/">Debiased Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 DCRec，用三通道从众权重网络解耦用户真实兴趣与流行度从众，并以加权 InfoNCE 实现去偏对比学习，四数据集 LOO 全库排序 HR@1 最高提升 10.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因子化多头注意力&quot;,&quot;双兴趣解耦&quot;,&quot;负反馈序列建模&quot;]"><strong><a href="sequential_rec/dual_interest_factorization_heads_attention_for_sequential_recommendation/">Dual-interest Factorization-heads Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用头对分解注意力与反馈标签掩码建模跳/不跳混合序列，双兴趣解耦塔加 BPR 成对对比，Micro-video GAUC 提升 2.19%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;潜变量模型&quot;,&quot;变分自编码器 VAE&quot;,&quot;用户意图建模&quot;]"><strong><a href="sequential_rec/latent_user_intent_modeling_for_sequential_recommenders/">Latent User Intent Modeling for Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以条件 VAE 将用户意图建模为潜变量并注入 REINFORCE 序列召回器，三周线上 A/B 使整体满意度 +0.07%、消费多样性 +0.10%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;向量量化(VQ/OPQ)&quot;,&quot;可迁移跨域推荐&quot;,&quot;对比预训练&quot;]"><strong><a href="sequential_rec/learning_vector_quantized_item_representation_for_transferable_sequential_r/">Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 OPQ 将物品文本映射为离散码并查表得表征，配合半合成/混合域难负样本对比预训练与 Gumbel-Sinkhorn 置换对齐微调，跨域跨平台全库排序显著超越 UniSRec。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排打分&quot;,&quot;Hawkes 过程与时间核&quot;,&quot;负激励建模&quot;,&quot;静态属性兴趣&quot;]"><strong><a href="sequential_rec/modeling_temporal_positive_and_negative_excitation_for_sequential_recommend/">Modeling Temporal Positive and Negative Excitation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SDIL 用可学习正/负时间衰减核刻画历史交互的时变激励，并以属性级静态兴趣门控融合，采样排序下 Cellphones MRR 相对提升 10.45%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;Wasserstein 距离&quot;,&quot;自监督数据增广&quot;]"><strong><a href="sequential_rec/mutual_wasserstein_discrepancy_minimization_for_sequential_recommendation/">Mutual Wasserstein Discrepancy Minimization for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 2-Wasserstein 距离替代 KL 散度重建对比互信息估计，四数据集全指标相对提升 2.24%–20.10%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮交互在线召回&quot;,&quot;NDCG导向属性选择&quot;,&quot;用户表征精炼&quot;]"><strong><a href="agent_conversational_rec/enhancing_user_personalization_in_conversational_recommenders/">Enhancing User Personalization in Conversational Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 MGMCR 多正确答案设定与 UpsRec 框架：NDCG 增益贪心属性问询 + 注意力用户表征精炼器，四数据集 NDCG@10 较 UNICORN 提升 23.6%–58.6% 且轮次减半。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;自监督对比学习&quot;,&quot;免负采样训练&quot;,&quot;潜在表征 dropout&quot;]"><strong><a href="multimodal_rec/bootstrap_latent_representations_for_multi_modal_recommendation/">Bootstrap Latent Representations for Multi-modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>BM3 用隐层 dropout 自举对比视图、免负采样免辅助图，重构交互图并对齐模态特征，全库排序下 R@10 提升 3.7–20.4% 且训练快 2–9 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;对抗式图增广&quot;,&quot;跨模态对比学习&quot;]"><strong><a href="multimodal_rec/multi_modal_self_supervised_learning_for_recommendation/">Multi-Modal Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Gumbel+WGAN-GP 对抗生成模态感知交互图，再以跨模态对比学习对齐模态偏好，All-Rank 全库排序下 Recall@20 最高提升 16.04%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图神经网络&quot;,&quot;多视图共识建模&quot;]"><strong><a href="social_group_rec/consrec_learning_consensus_behind_interactions_for_group_recommendation/">ConsRec: Learning Consensus Behind Interactions for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用成员超图、物品二部图与群组加权图三视图建模群组共识，门控自适应融合，采样排序下 Mafengwo HR@5 相对最优基线提升 2.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;社交图去噪&quot;,&quot;偏好引导关系置信度&quot;,&quot;课程学习&quot;]"><strong><a href="social_group_rec/robust_preference_guided_denoising_for_graph_based_social_recommendation/">Robust Preference-Guided Denoising for Graph based Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用交互偏好引导社交图去噪，结合Transformer关系置信度、课程学习与自适应裁剪，在Ciao/Yelp/Douban上稳定提升Diffnet++/MHCN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;特征级数据增广&quot;,&quot;人群定向&quot;]"><strong><a href="cold_start_rec/bootstrapping_contrastive_learning_enhanced_music_cold_start_matching/">Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用动态自举的特征相关性矩阵驱动特征级对比增广，学习鲁棒歌曲表征，并结合聚类人群定向完成冷启歌曲快速加热，全库排序 Recall@50 提升 4.74%，线上有效播放 +38.47%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;嵌入压缩&quot;,&quot;分簇学习&quot;,&quot;GPCA&quot;]"><strong><a href="cold_start_rec/clustered_embedding_learning_for_recommender_systems/">Clustered Embedding Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出自上而下分簇嵌入学习 CEL，以梯度主成分分裂簇并动态重分配，在压缩嵌入表的同时显著提升冷物品与线上 AUC。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;神经架构搜索&quot;,&quot;超网络调制&quot;,&quot;FiLM&quot;]"><strong><a href="cold_start_rec/coldnas_search_to_modulate_for_user_cold_start_recommendation/">ColdNAS: Search to Modulate for User Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>把冷启动用户的 FiLM 调制函数与调制位置建模为 NAS 问题，经可证等价的空间变换将 $6^{CL}$搜索空间压缩到$2^{4L}$，一次性可微搜索后在三个基准上稳定超越 MAML 系与超网络系基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;模糊协同嵌入矫正&quot;,&quot;内容-共现双视图&quot;]"><strong><a href="cold_start_rec/contrastive_collaborative_filtering_for_cold_start_item_recommendation/">Contrastive Collaborative Filtering for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用内容 CF 与共现 CF 双模块的对比学习把共现协同信号隐式&quot;背&quot;进内容编码器，矫正冷启动物品的模糊协同嵌入，全库用户排序下 HR@5 提升 6.1%/29.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;零样本属性推理&quot;,&quot;会话图神经网络&quot;,&quot;双意图建模&quot;]"><strong><a href="cold_start_rec/dual_intent_enhanced_graph_neural_network_for_session_based_new_item_recomm/">Dual Intent Enhanced Graph Neural Network for Session-based New Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 GSNIR 新设定与 NirGNN：用注意力+β分布双意图模拟用户决策，并以零样本属性推理把新物品映射进 GNN 空间，Amazon G&amp;GF 上 P@20 相对提升 19.33%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/候选打分&quot;,&quot;跨语言迁移&quot;,&quot;少样本用户建模&quot;,&quot;域对齐&quot;]"><strong><a href="cold_start_rec/few_shot_news_recommendation_via_cross_lingual_transfer/">Few-shot News Recommendation via Cross-lingual Transfer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>无重叠用户/新闻的跨语言少样本新闻推荐：跨域新闻检索增广 + 文档级随机掩码 + 新闻对齐 MSE 消除域偏移，MIND→Adressa 4-shot AUC 相对提升 7.69%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;类别潜空间翻译&quot;,&quot;CycleGAN 半监督&quot;]"><strong><a href="cold_start_rec/semi_supervised_adversarial_learning_for_complementary_item_recommendation/">Semi-supervised Adversarial Learning for Complementary Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为冷启动商品维护类别专属潜空间并用 CycleGAN 式半监督对抗学习训练跨类别翻译器，类别感知场景 Clothing NDCG 提升 33%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;持续迁移学习&quot;,&quot;双塔适配器&quot;,&quot;GLU 门控&quot;]"><strong><a href="cross_domain_multi_scenario/continual_transfer_learning_for_cross_domain_click_through_rate_prediction_/">Continual Transfer Learning for Cross-Domain Click-Through Rate Prediction at Taobao</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 CTL 新任务与双塔 CTNet，冻结演进中的源域塔、持续训练目标塔并用 GLU 适配器做逐层注入，仅用增量数据即在淘宝 A→B/A→C 提升 GAUC 1.0%/3.6%，线上 CTR +2.5%/+12.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域异构图消息传递&quot;,&quot;用户兴趣对齐&quot;,&quot;梯度对齐&quot;]"><strong><a href="cross_domain_multi_scenario/cross_domain_recommendation_via_user_interest_alignment/">Cross-domain recommendation via user interest alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>WWW 2023｜跨域多场景推荐：构建统一跨域异构图并重定义消息传递，再用对比聚类与梯度对齐约束重叠用户兴趣一致；双域留一法 1:99 采样排序下最高相对提升 10.23%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;行为路径匹配&quot;,&quot;对比学习增广&quot;,&quot;序列行为建模&quot;]"><strong><a href="ctr_feature_interaction/a_deep_behavior_path_matching_network_for_click_through_rate_prediction/">A Deep Behavior Path Matching Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出“行为路径”并做路径级+点击级两级匹配，配合对比增广与自激活去噪，美团线上 CTR +1.6%、广告收入 +1.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;置信度排序损失&quot;,&quot;在线学习与模型演进&quot;,&quot;知识蒸馏替代&quot;]"><strong><a href="ctr_feature_interaction/confidence_ranking_for_ctr_prediction/">Confidence Ranking for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用“超越线上部署模型”的点式/关系式排序代理损失替代交叉熵与蒸馏，JD 广告精排 A/B 平均 +1.75% CTR。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;意图分解&quot;,&quot;触发诱导推荐&quot;,&quot;多任务联合学习&quot;]"><strong><a href="ctr_feature_interaction/deep_intention_aware_network_for_click_through_rate_prediction/">Deep Intention-Aware Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将小程序触发诱导推荐 CTR 分解为「进入意图 × 双条件 CTR」，用后验标注监督意图网络，淘宝聚划算 AUC 0.812、线上 CTR +4.74%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;权重共享NAS&quot;,&quot;特征交叉&quot;,&quot;算子平衡&quot;]"><strong><a href="ctr_feature_interaction/nasrec_weight_sharing_neural_architecture_search_for_recommender_systems/">NASRec: Weight Sharing Neural Architecture Search for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出权重共享 NAS 超网与单算子任意连接采样，自动搜索 CTR 特征交叉架构，在 Criteo/Avazu/KDD 取得领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征选择&quot;,&quot;可学习门控&quot;,&quot;Learning-by-Continuation&quot;]"><strong><a href="ctr_feature_interaction/optimizing_feature_set_for_click_through_rate_prediction/">Optimizing Feature Set for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>OptFS 以特征级门控 + 交互分解把特征集搜索空间从 C(m,2) 降到 m，Criteo 仅留 4.2% 特征仍提升 AUC。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;上下文特征&quot;,&quot;多样性校准&quot;,&quot;贪心序列重排&quot;]"><strong><a href="multi_task_reranking/caviar_context_aware_video_recommendations/">CAViaR: Context Aware Video Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用上下文特征让点式模型感知前序视频相似度，贪心式序列重排去冗余；NE 相对降低 0.85%–1.4%，线上互动 +1.7%、DAU +1.5%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回+精排级联&quot;,&quot;难负例采样&quot;,&quot;排序序蒸馏&quot;,&quot;渐近无偏KL估计&quot;]"><strong><a href="multi_task_reranking/cooperative_retriever_and_ranker_in_deep_recommenders/">Cooperative Retriever and Ranker in Deep Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CoRR 联合训练召回与精排：用自适应采样器从召回分布抽硬负例训精排，用渐近无偏 KL 估计做排序序一致性蒸馏回灌召回，四个大规模数据集 NDCG@10 提升 5–18%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;动态损失加权&quot;,&quot;会话级 MDP&quot;]"><strong><a href="multi_task_reranking/multi_task_recommendations_with_reinforcement_learning/">Multi-Task Recommendations with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将会话级交互建模为 MDP，用多 Critic 网络输出 Q 值动态调节各任务 BCE 权重，AUC 相对 SOTA 提升约 0.3–0.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/列表推荐&quot;,&quot;Latent Action Space&quot;,&quot;Actor-Critic&quot;,&quot;动作空间对齐&quot;]"><strong><a href="rl_bandit_rec/exploration_and_regularization_of_the_latent_action_space_in_recommendation/">Exploration and Regularization of the Latent Action Space in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>HAC将列表推荐分解为潜在超动作与效应动作，共享critic并以逆池化对齐和BCE监督稳定RL，在三个模拟在线环境取得约1%–6%的长期收益提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/曝光分配策略&quot;,&quot;Contextual Bandit&quot;,&quot;Exposure Constraints&quot;]"><strong><a href="rl_bandit_rec/learning_with_exposure_constraints_in_recommendation_systems/">Learning with Exposure Constraints in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将提供者最低曝光约束建模为分阶段上下文老虎机，提出EES/DP*/LCB*，达到˜O(T^{2/3}) regret并给出Ω(T^{2/3})下界。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/混排&quot;,&quot;Thompson Sampling&quot;,&quot;Personalized PageRank&quot;,&quot;探索-利用平衡&quot;]"><strong><a href="rl_bandit_rec/pie_personalized_interest_exploration_for_large_scale_recommender_systems/">PIE: Personalized Interest Exploration for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 PPR 构建用户-创作者探索空间 + Thompson 采样在线探索 + 概率化混排，在 Facebook Watch 十亿级用户四周 A/B 中 SCC +3.50% 而互动指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;长期留存优化&quot;,&quot;无限视界 MDP&quot;,&quot;Actor-Critic&quot;]"><strong><a href="rl_bandit_rec/reinforcing_user_retention_in_a_billion_scale_short_video_recommender_syste/">Reinforcing User Retention in a Billion Scale Short Video Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频留存优化建模为无限视界 request-based MDP，以最小化累积回归间隔为目标，用奖励归一化、活跃度分组策略与软行为正则改造 Actor-Critic，快手全量 A/B 中 DAU +0.2%、次日留存 +0.053%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Constrained MDP&quot;,&quot;Actor-Critic&quot;,&quot;多目标软约束&quot;]"><strong><a href="rl_bandit_rec/two_stage_constrained_actor_critic_for_short_video_recommendation/">Two-Stage Constrained Actor-Critic for Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频推荐建模为 CMDP，用多 Critic 分离评估稠密/稀疏奖励，两阶段软 KL 约束 Actor，离线 WatchTime +2.23%、线上 +0.379%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;Decision Transformer&quot;,&quot;User Retention&quot;,&quot;Contrastive Learning&quot;]"><strong><a href="rl_bandit_rec/user_retention_oriented_recommendation_with_decision_transformer/">User Retention-oriented Recommendation with Decision Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将留存优化转为决策Transformer自回归生成，以奖励提示和加权对比学习缓解离线RL不稳定，在IQiYi/ML-1m提升留存与准确率。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (11)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;逆概率加权 (IPS)&quot;,&quot;未观测混杂平衡权重&quot;]"><strong><a href="debias_fairness_industrial/balancing_unobserved_confounding_with_a_few_unbiased_ratings_in_debiased_re/">Balancing Unobserved Confounding with a Few Unbiased Ratings in Debiased Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用少量无偏评分学习熵最大平衡权重，重加权 IPS/DR/AutoDebias 损失以对抗未观测混杂与模型误设，模型无关且即插即用，NDCG@5 最高相对提升 4.21%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果嵌入解耦&quot;,&quot;从众偏差&quot;,&quot;多任务排序&quot;]"><strong><a href="debias_fairness_industrial/cam2_conformity_aware_multi_task_ranking_model_for_large_scale_recommender_/">CAM2: Conformity-Aware Multi-Task Ranking Model for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用特征分区+任务无关辅助损失在十亿级视频排序中解耦“从众”与“真实兴趣”因果嵌入，离线 NE 降 0.139%，线上互动 +0.50%、DAU +0.21%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;因果解耦嵌入&quot;,&quot;对比学习样本增广&quot;]"><strong><a href="debias_fairness_industrial/disentangled_causal_embedding_with_contrastive_learning_for_recommender_sys/">Disentangled Causal Embedding With Contrastive Learning For Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用流行度加权双通道对比学习把交互解耦为兴趣与从众两类因果嵌入，OOD 干预测试集 HR@20 较 MACR 提升 93%，快手线上 EVTR +7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;自适应负采样&quot;,&quot;组公平性&quot;,&quot;双层优化&quot;]"><strong><a href="debias_fairness_industrial/fairly_adaptive_negative_sampling_for_recommendations/">Fairly Adaptive Negative Sampling for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次从负采样视角治理物品组公平：用组级 G-BCE 感知不公平、动量自适应调组采样分布并与重要性采样 Mixup，ML1M-2 上 Recall-Disp 降 90.97%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;DP-SGD&quot;,&quot;整数规划公平重排&quot;]"><strong><a href="debias_fairness_industrial/fairness_aware_differentially_private_collaborative_filtering/">Fairness-aware Differentially Private Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 DP-SGD 放大活跃/非活跃用户精度鸿沟（Matthew 效应），提出分离式用户-物品梯度扰动 + 整数规划公平重排的 DP-Fair 两阶段框架，在四项真实数据集上同时提升总效用并压降 F_U。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;数据增广&quot;,&quot;双层优化&quot;,&quot;群组公平性&quot;]"><strong><a href="debias_fairness_industrial/improving_recommendation_fairness_via_data_augmentation/">Improving Recommendation Fairness via Data Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 FDA 框架，用嵌入扰动生成跨组“伪交互”并以双层优化做数据增广，无需预定义公平指标即可将 DP/EO 不公平度降低 4%–12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦召回训练&quot;,&quot;成员推理攻击&quot;,&quot;本地差分隐私&quot;,&quot;参数容量再分配&quot;]"><strong><a href="debias_fairness_industrial/interaction_level_membership_inference_attack_against_federated_recommender/">Interaction-level Membership Inference Attack Against Federated Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次揭示联邦推荐中交互级成员推理攻击：服务器仅凭上传的物品嵌入迭代训练影子模型反推用户正样本，并提出公共参数约束正则以近乎零推荐损失防御。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;不变学习&quot;,&quot;解耦表征&quot;,&quot;流行度分布偏移&quot;]"><strong><a href="debias_fairness_industrial/invariant_collaborative_filtering_to_popularity_distribution_shift/">Invariant Collaborative Filtering to Popularity Distribution Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将流行度偏移重构为分布外泛化问题，用表示增广+距离相关解耦出不变偏好表征，在 5 个真实集与 4 类评测协议下稳定超越主流去偏基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;子模最大化&quot;,&quot;比例表示约束&quot;,&quot;偏置干预&quot;]"><strong><a href="debias_fairness_industrial/maximizing_submodular_functions_for_recommendation_in_the_presence_of_biase/">Maximizing Submodular Functions for Recommendation in the Presence of Biases</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>证明固定公平约束在子模推荐下可失效，提出无偏组校准配额的两阶段贪心；MovieLens离线模拟中14/31类型集合提升≥3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Max-min Fairness&quot;,&quot;Provider Fairness&quot;,&quot;Online Resource Allocation&quot;]"><strong><a href="debias_fairness_industrial/p_mmf_provider_max_min_fairness_re_ranking_in_recommender_system/">P-MMF: Provider Max-min Fairness Re-ranking in Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将提供商公平重排建模为MMF资源分配，在对偶空间动量在线求解，四数据集W1@10提升0.6–2.0%。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部11篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
