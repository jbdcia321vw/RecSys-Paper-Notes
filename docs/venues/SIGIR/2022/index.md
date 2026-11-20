---
hide:
  - toc
---

# 🏛️ SIGIR 2022 推荐系统论文解读

57 篇 SIGIR 2022 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×4 · 跨视图对比学习 ×3 · 去偏公平与工业架构 ×2 · 知识图谱推荐 ×2 · 离线强化学习 ×2 · 端到端生成式 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视图对比学习" aria-pressed="false">跨视图对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱推荐" aria-pressed="false">知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/训练目标&quot;,&quot;损失函数自动生成&quot;,&quot;强化学习搜索&quot;,&quot;奖励过滤&quot;]"><strong><a href="collaborative_filtering/autolossgen_automatic_loss_function_generation_for_recommender_systems/">AutoLossGen: Automatic Loss Function Generation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 RNN 控制器在基础算子空间上以 RL 从零搜索损失函数，配合代理测试与奖励过滤克服稀疏奖励，生成 MaxR/SumR 在 AUC 上稳定超过 MSE/BCE/Hinge/Focal。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;端到端训练范式&quot;,&quot;过参数化冗余利用&quot;,&quot;同伴协作权重移植&quot;,&quot;层重要性熵度量&quot;]"><strong><a href="collaborative_filtering/enhancing_top_n_item_recommendations_by_peer_collaboration/">Enhancing Top-N Item Recommendations by Peer Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 PCRec：两个同构推荐模型按层熵/范数度量冗余并互相移植权重，在推理零额外开销下稳定提升 Top-N 精度</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;归纳式嵌入&quot;,&quot;模板基向量&quot;,&quot;免重训练&quot;]"><strong><a href="collaborative_filtering/inmo_a_model_agnostic_and_scalable_module_for_inductive_collaborative_filte/">INMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用模板用户/物品作基向量、由交互集合线性生成用户/物品嵌入，为 MF/LightGCN 注入归纳能力并大幅压缩参数量；直推与两种归纳场景全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤&quot;,&quot;召回（双塔打分）&quot;,&quot;动量对比学习&quot;,&quot;标注偏置去噪&quot;]"><strong><a href="collaborative_filtering/mp2_a_momentum_contrast_approach_for_recommendation_with_pointwise_and_pair/">MP2: A Momentum Contrast Approach for Recommendation with Pointwise and Pairwise Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用三塔结构+动量物品塔平滑表示、以双塔差异度重加权逐点标签，联合逐点/成对损失缓解隐式反馈标注偏置，NDCG@5 相对最优基线提升 1.3–3.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图趋势滤波&quot;,&quot;自适应平滑&quot;,&quot;鲁棒协同过滤&quot;]"><strong><a href="graph_rec/graph_trend_filtering_networks_for_recommendations/">Graph Trend Filtering Networks for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 L1 图趋势滤波替换拉普拉斯等权平滑，得到边级自适应加权的嵌入传播，Full-Ranking 下 NDCG@20 最高相对提升 12.77%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;知识图谱聚合&quot;,&quot;门控融合&quot;]"><strong><a href="graph_rec/hakg_hierarchy_aware_knowledge_gated_network_for_recommendation/">HAKG: Hierarchy-Aware Knowledge Gated Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户-物品图与知识图谱同时嵌入双曲空间，用双曲关系传递聚合、层级锥角约束与双物品嵌入门控融合，Full-Ranking 下 nDCG@20 最高提升 15.43%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图神经网络&quot;,&quot;跨视图对比学习&quot;,&quot;过平滑缓解&quot;]"><strong><a href="graph_rec/hypergraph_contrastive_collaborative_filtering/">Hypergraph Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>参数化低秩超图学习刻画全局依赖 + 局部-全局跨视图 InfoNCE，缓解 GNN 过平滑与交互稀疏，Full-Ranking 下 NDCG@20 相对 LightGCN 提升 24.7%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱去噪&quot;,&quot;跨视图对比学习&quot;,&quot;图增广&quot;]"><strong><a href="graph_rec/knowledge_graph_contrastive_learning_for_recommendation/">Knowledge Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以KG结构一致性为去噪信号指导交互图增广，跨视图对比学习在三个数据集全库排序下Recall@20 提升约4-6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱图滤波&quot;,&quot;图去噪编码器&quot;]"><strong><a href="graph_rec/less_is_more_reweighting_important_spectral_graph_features_for_recommendati/">Less is More: Reweighting Important Spectral Graph Features for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>谱域分析发现仅少数平滑/粗糙图特征有效，用指数核带通滤波的 GDE 替代邻域聚合，5 数据集全库排序最优且比 LightGCN 快 12 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;kNN 物品语义图&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="graph_rec/multi_level_cross_view_contrastive_learning_for_knowledge_aware_recommender/">Multi-level Cross-view Contrastive Learning for Knowledge-aware Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>拆解用户-物品-实体图为协同/语义/结构三视图，做局部-全局跨视图对比学习，AUC 较 KGIN 提升 1.6–2.8 个百分点。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/端到端排序&quot;,&quot;知识图谱推荐&quot;,&quot;协同注意力&quot;,&quot;包围子图与门控融合&quot;]"><strong><a href="graph_rec/unify_local_and_global_information_for_top_n_recommendation/">Unify Local and Global Information for Top-$N$ Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双分支 KADM：协同注意力编码局部交互、关系感知 GNN 编码包围子图，门控融合后在 1:100 采样排序下 Recall@20 提升 3.1%~7.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;边特征图卷积&quot;,&quot;评论对齐对比学习&quot;]"><strong><a href="self_supervised_denoising/a_review_aware_graph_contrastive_learning_framework_for_recommendation/">A Review-aware Graph Contrastive Learning Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将评分+评论构建为边特征增强的用户-物品图，用评论调制邻居权重，并以节点/边判别双对比任务提供自监督信号；在5个5-core数据集上随机划分、MSE指标取得最高4.2%相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与表征学习&quot;,&quot;图对比学习&quot;,&quot;表示均匀性&quot;,&quot;噪声增强&quot;]"><strong><a href="self_supervised_denoising/are_graph_augmentations_necessary_simple_graph_contrastive_learning_for_rec/">Are Graph Augmentations Necessary? Simple Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>实验揭示InfoNCE带来的表示均匀性而非图增强是增益关键，提出嵌入空间加均匀噪声的SimGCL；三数据集全库排序下Recall@20较LightGCN最高提升25.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多兴趣解耦&quot;,&quot;隐式类别正交&quot;,&quot;对比学习去噪&quot;]"><strong><a href="self_supervised_denoising/improving_micro_video_recommendation_via_contrastive_multiple_interests/">Improving Micro-video Recommendation via Contrastive Multiple Interests</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用隐式正交类别解耦多兴趣，并以跨视图对比多兴趣损失抵御微视频自动播放的假阳性噪声，双数据集 Recall@20 提升 24%/1.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;记忆效应 (Memorization Effect)&quot;,&quot;元学习重加权&quot;]"><strong><a href="self_supervised_denoising/self_guided_learning_to_denoise_for_robust_recommendation/">Self-Guided Learning to Denoise for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>利用推荐模型早期“抗噪期”的记忆效应收集干净样本，以元学习方式学出自适应重加权函数指导后续训练，模型无关地提升 Recall@20 最多 15%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;数据分布自适应&quot;,&quot;模型补丁参数调制&quot;]"><strong><a href="sequential_rec/ada_ranker_a_data_distribution_adaptive_ranking_paradigm_for_sequential_rec/">Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 Neural Process 提取候选集分布 z，经 FiLM 输入调制与参数池补丁调制，把全局精排器即时改写为局部模型，NDCG 平均提升约 2.5%，新分布下增益达 15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图序列粗到细建模&quot;,&quot;学习式局部注意力&quot;,&quot;联合概率解码&quot;]"><strong><a href="sequential_rec/coarse_to_fine_sparse_sequential_recommendation/">Coarse-to-Fine Sparse Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>显式建模类目/卖家意图序列作为稠密先验，与物品序列双编码器融合并按联合概率解码，稀疏数据上 NDCG@5 相对提升 44.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;侧信息融合&quot;,&quot;解耦注意力&quot;,&quot;辅助属性预测&quot;]"><strong><a href="sequential_rec/decoupled_side_information_fusion_for_sequential_recommendation/">Decoupled Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将侧信息从输入层移至注意力层解耦计算，配合辅助属性预测器，在四个 Amazon/Yelp 数据集全库排序上稳定超越 SASRec/NOVA/ICAI-SR。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;CTR预估&quot;,&quot;时间周期建模&quot;,&quot;行为去噪&quot;,&quot;时间嵌入&quot;]"><strong><a href="sequential_rec/denoising_time_cycle_modeling_for_recommendation/">Denoising Time Cycle Modeling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>DiCycle 用绝对/相对时间核编码 + 门控去噪分别刻画 ATC/RTC 两种周期模式，CTR 预测在 LastFM/ML-1M/Books/工业集上 AUC 稳定超越时序基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Determinantal Point Process&quot;,&quot;集合似然损失&quot;,&quot;质量-多样性权衡&quot;]"><strong><a href="sequential_rec/determinantal_point_process_likelihoods_for_sequential_recommendation/">Determinantal Point Process Likelihoods for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 DPP 集合似然替换逐点/成对损失，把「下 T 项」建模为集合抽取，即插即用于 Caser/MARank/SASRec，Beauty Recall@3 最高 +37%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;判别式训练&quot;,&quot;ELECTRA 式替换检测&quot;,&quot;采样难负例&quot;]"><strong><a href="sequential_rec/elecrec_training_sequential_recommenders_as_discriminators/">ELECRec: Training Sequential Recommenders as Discriminators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>借鉴 ELECTRA 思想，将序列推荐从下一项生成改为真假判别训练，辅助生成器采样难负例，四数据集 Full-Ranking 下 HR@5 提升 38%–137%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Session-aware Recommendation&quot;,&quot;Temporal Self-Attention&quot;]"><strong><a href="sequential_rec/exploiting_session_information_in_bert_based_session_aware_sequential_recom/">Exploiting Session Information in BERT-based Session-aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>在 BERT4Rec 输入层以会话标记、会话段嵌入与时间感知自注意力三种轻量插件注入会话结构，仅增 0.1% 参数即在全库排序下取得最高 55% NDCG 提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Fastformer&quot;,&quot;细粒度用户建模&quot;,&quot;新闻推荐&quot;]"><strong><a href="sequential_rec/fum_fine_grained_and_fast_user_modeling_for_news_recommendation/">FUM: Fine-grained and Fast User Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户点击新闻拼接成一条长文档，用 Fastformer 线性复杂度捕获词级跨新闻交互，并融合新闻级粗粒度通路；MIND/Feeds 上 AUC 相对提升 2.2%–2.5%，推理仅 2.75s/1k 样本。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间多样性&quot;,&quot;顺序无关Transformer&quot;]"><strong><a href="sequential_rec/is_news_recommendation_a_sequential_recommendation_task/">Is News Recommendation a Sequential Recommendation Task?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>论证新闻推荐非标准序列任务，提出双序无关 Transformer 与时间多样性惩罚 TempRec，在 MIND/News 曝光排序稳定提升 AUC/nDCG。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;候选感知用户建模&quot;,&quot;单塔交互式匹配&quot;]"><strong><a href="sequential_rec/news_recommendation_with_candidate_aware_user_modeling/">News Recommendation with Candidate-aware User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用候选新闻反向引导用户建模，Candi-SelfAtt/Candi-CNN/Candi-Att 三路聚合，MIND 上 nDCG@5 相对提升 4.2%</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部12篇「序列与会话推荐」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端对话仿真/评测&quot;,&quot;用户模拟&quot;,&quot;话语重述生成&quot;]"><strong><a href="agent_conversational_rec/analyzing_and_simulating_user_utterance_reformulation_in_conversational_rec/">Analyzing and Simulating User Utterance Reformulation in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>基于用户研究提出重述序列生成任务，用类型引导 T5/BART 与阅读难度过滤提升模拟用户拟人度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;多层面用户建模&quot;,&quot;跨视图自监督对齐&quot;,&quot;Look-alike 用户&quot;]"><strong><a href="agent_conversational_rec/user_centric_conversational_recommendation_with_multi_aspect_user_modeling/">User-Centric Conversational Recommendation with Multi-Aspect User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将历史对话会话与相似用户引入 CRS，用多视图自监督对齐与时序 look-alike 选择器实现以用户为中心建模，全库排序下 HR@10 相对提升 21.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;社交与群组推荐&quot;,&quot;召回（组-物品全库排序）&quot;,&quot;超立方体表示&quot;,&quot;组偏好聚合&quot;,&quot;交集自监督&quot;]"><strong><a href="social_group_rec/thinking_inside_the_box_learning_hypercube_representations_for_group_recomm/">Thinking inside The Box: Learning Hypercube Representations for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用中心-偏移超立方体替代点向量表示群组，配内外双重距离与组交集自监督，全库排序下 Recall@10 平均超 GroupIM 14.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多关系图卷积&quot;,&quot;耦合RNN时序互影响&quot;]"><strong><a href="poi_spatiotemporal_rec/empowering_next_poi_recommendation_with_multi_relational_modeling/">Empowering Next POI Recommendation with Multi-Relational Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MEMO 用多关系 GCN + 关系专属自注意力聚合异构社交图，并以耦合 RNN 建模用户-POI 时序互影响；在三个 LBSN 数据集 Recall@10 相对 STAN 提升约 2.7%–3.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;轨迹流图&quot;,&quot;Graph Enhanced Transformer&quot;]"><strong><a href="poi_spatiotemporal_rec/getnext_trajectory_flow_map_enhanced_transformer_for_next_poi_recommendatio/">GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建用户无关的全局轨迹流图并用 GCN+转移动注意力图显式编码群体转移模式，融合 Transformer 序列建模，在 NYC/TKY/CA 全库排序上 Acc@5 提升约 11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长时行为序列建模&quot;,&quot;对抗训练与行为去噪&quot;]"><strong><a href="ctr_feature_interaction/adversarial_filtering_modeling_on_long_term_user_behavior_sequences_for_cli/">Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用分层聚合压缩重复行为、自注意力打分门选 top-k 行为，并以对抗训练（IEU 最小化、BSU 最大化剩余序列损失）剔除长序列噪声行为，工业集 AUC +0.0047、线上 CTR +4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征精炼&quot;,&quot;上下文感知表示&quot;,&quot;比特级门控&quot;]"><strong><a href="ctr_feature_interaction/enhancing_ctr_prediction_with_context_aware_feature_representation_learning/">Enhancing CTR Prediction with Context-Aware Feature Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>FRNet 通过比特级门控融合原始特征与互补特征，为每条样本生成上下文感知的特征表示，即插即用并显著提升多种 CTR 模型。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;属性树聚合&quot;,&quot;层次意图建模&quot;]"><strong><a href="ctr_feature_interaction/hien_hierarchical_intention_embedding_network_for_click_through_rate_predic/">HIEN: Hierarchical Intention Embedding Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用属性树自底向上聚合 + 层次注意力双向意图建模精修Embedding，作为即插即用输入模块，在阿里/Tmall/腾讯三个数据集上AUC较DG-ENN最高提升0.86%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构信息网络&quot;,&quot;Graph-Masked Transformer&quot;,&quot;一致性正则&quot;]"><strong><a href="ctr_feature_interaction/neighbour_interaction_based_click_through_rate_prediction_via_graph_masked_/">Neighbour Interaction based Click-Through Rate Prediction via Graph-masked Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 CTR 重建为异构信息图上 user-item 局部邻域连边预测，用四类交互图 + 头分组图掩码 Transformer 编码邻域，微信线上 CTR +21.9%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;概率校准&quot;,&quot;保序回归&quot;,&quot;自适应分箱&quot;]"><strong><a href="ctr_feature_interaction/posterior_probability_matters_doubly_adaptive_calibration_for_neural_predic/">Posterior Probability Matters: Doubly-Adaptive Calibration for Neural Predictions in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 AdaCalib，用后验统计引导的分段线性保序函数族做字段级概率校准，并以频率驱动的自适应分箱匹配字段取值，Field-RCE 相对降 10%~70%，线上 CVR +5.05%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR预估与特征交叉&quot;,&quot;召回&quot;,&quot;嵌入维度搜索&quot;,&quot;单次剪枝&quot;,&quot;混合维度嵌入&quot;]"><strong><a href="ctr_feature_interaction/single_shot_embedding_dimension_search_in_recommender_system/">Single-shot Embedding Dimension Search in Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SSEDS 用一次前向-反向的梯度显著性打分完成字段级嵌入维度剪枝，90% 参数下 AUC 反升，并在微信订阅号 A/B 提升约 4% CTR。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;set2list 交互&quot;,&quot;排列不变性&quot;]"><strong><a href="multi_task_reranking/multi_level_interaction_reranking_with_user_behavior_history/">Multi-Level Interaction Reranking with User Behavior History</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MIR 将候选集视为无序集合、用户历史视为时序列表，用 SLAttention 做 set2list 跨层交互并保证排列不变性，三数据集全指标超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;提示式学习&quot;,&quot;离线强化学习&quot;,&quot;会话序列推荐&quot;]"><strong><a href="rl_bandit_rec/rethinking_reinforcement_learning_for_recommendation_a_prompt_perspective/">Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将离线 RL 推荐从「状态-动作→Q 值」翻转为「状态-累计回报→动作」的提示式监督学习，全库排序下购买 HR@5 最高 +13.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端强化学习&quot;,&quot;状态编码器&quot;,&quot;可复现性研究&quot;]"><strong><a href="rl_bandit_rec/state_encoders_in_reinforcement_learning_for_recommendation_a_reproducibili/">State Encoders in Reinforcement Learning for Recommendation: A Reproducibility Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用去偏 SOFA 模拟器+ DQN 复现“注意力状态编码器最优”结论，发现该结论在 Yahoo! R3 成立但在 Coat 数据集失效，无单一最优编码器。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;基准可信度复检&quot;,&quot;折扣因子消融&quot;,&quot;环境模拟器评估&quot;]"><strong><a href="rl_bandit_rec/towards_validating_long_term_user_feedbacks_in_interactive_recommendation_s/">Towards Validating Long-Term User Feedbacks in Interactive Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>复检 RL 交互推荐基准，发现公共评论数据集长期效应微弱：贪心单步奖励模型 (γ=0) 反超 RL，beam 与 greedy 差距 &lt;1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/重排&quot;,&quot;离线强化学习&quot;,&quot;不确定性惩罚&quot;,&quot;长期满意度&quot;]"><strong><a href="rl_bandit_rec/value_penalized_q_learning_for_recommender_systems/">Value Penalized Q-Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出乘性不确定性惩罚 VPQ，用 Q 集成方差构造 $p$-mul 保守目标并配 critic 框架，在两数据集上为三种序列模型平均带来 6–9% 相对增益。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回/精排双阶段&quot;,&quot;开放基准与可复现性&quot;,&quot;Full-Ranking 评测协议&quot;]"><strong><a href="debias_fairness_industrial/bars_towards_open_benchmarking_for_recommender_systems/">BARS: Towards Open Benchmarking for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出含 7 项标准产物的开放基准流水线，固化数据划分与评测协议 ID，在 6 数据集 70+ 模型上完成 8000+ 次全库排序与 CTR 复现实验，揭示近年增益被系统性夸大。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Two-sided Fairness&quot;,&quot;Constrained Re-ranking&quot;,&quot;Greedy MILP Relaxation&quot;]"><strong><a href="debias_fairness_industrial/cpfair_personalized_consumer_and_producer_fairness_re_ranking_for_recommend/">CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用加性贪心重排把用户侧与物品侧公平偏差同时作为软惩罚，8 数据集上 mCPF 平均降 12.8%~46.9% 且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;用户公平&quot;,&quot;复现性&quot;]"><strong><a href="debias_fairness_industrial/experiments_on_generalizability_of_user_oriented_fairness_in_recommender_sy/">Experiments on Generalizability of User-Oriented Fairness in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复现并泛化评估用户公平重排 UFR：8 数据集 6 基础模型 2 分组，揭示领域/分组/基模型依赖与公平-效用权衡。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排解释&quot;,&quot;反事实解释&quot;,&quot;公平-效用权衡&quot;]"><strong><a href="debias_fairness_industrial/explainable_fairness_in_recommendation/">Explainable Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出反事实可解释公平框架 CEF，用最小特征扰动解释曝光不公平，并在三数据集擦除评测中取得更优公平-效用权衡。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;归一化系数 r&quot;,&quot;流行度去偏&quot;,&quot;LightGCN&quot;]"><strong><a href="debias_fairness_industrial/investigating_accuracy_novelty_performance_for_graph_based_collaborative_fi/">Investigating Accuracy-Novelty Performance for Graph-based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>理论证明 GNN 对称归一化是流行度偏置之源，提出零开销插件 r-AdjNorm，通过单一系数 r 控制归一化强度，在三个基准上不损精度的同时显著提升新颖性。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/排序策略&quot;,&quot;期望曝光&quot;,&quot;多边公平&quot;,&quot;Plackett-Luce&quot;]"><strong><a href="debias_fairness_industrial/joint_multisided_exposure_fairness_for_recommendation/">Joint Multisided Exposure Fairness for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出六类用户-物品双侧群体曝光公平指标，分解差异/相关性，并用可微 Plackett-Luce 随机排序优化 GG-F。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端生成式&quot;,&quot;自然语言用户画像&quot;,&quot;可审阅推荐 (Scrutability)&quot;,&quot;大语言模型&quot;]"><strong><a href="debias_fairness_industrial/on_natural_language_user_profiles_for_transparent_and_scrutable_recommendat/">On Natural Language User Profiles for Transparent and Scrutable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出以自然语言用户画像替代稠密向量作为推荐的唯一中间表示，分解为画像生成与 NL 推荐两步，并用 LaMDA 少样本提示做定性可行性验证。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据发布/预处理&quot;,&quot;隐私可控合成数据&quot;,&quot;Gumbel-Softmax 离散采样&quot;,&quot;隐私-效用权衡&quot;]"><strong><a href="debias_fairness_industrial/privacy_preserving_synthetic_data_generation_for_recommendation_systems/">Privacy-Preserving Synthetic Data Generation for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>UPC-SDG 用贡献度感知选择+Gumbel-Softmax 可微生成，按用户隐私偏好(k,γ)替换原始交互，兼顾数据/物品双层隐私与效用。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;Provider Fairness&quot;,&quot;Adversarial Debiasing&quot;]"><strong><a href="debias_fairness_industrial/profairrec_provider_fairness_aware_news_recommendation/">ProFairRec: Provider Fairness-aware News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出公平/偏置表征解耦、对抗学习与正交正则，在MIND上提升新闻提供者曝光公平且精度基本持平</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;供应商群体公平&quot;,&quot;Hellinger 距离&quot;,&quot;最大边际相关性&quot;]"><strong><a href="debias_fairness_industrial/regulating_group_exposure_for_item_providers_in_recommendation/">Regulating Group Exposure for Item Providers in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SIGIR 2022 短文，将供应商按敏感属性分组，提出基于 Hellinger 距离与 MMR 的贪心重排序，使推荐曝光贴合平台指定目标分布，在几乎不损失 NDCG 的前提下显著提升少数群体曝光与列表级多样性/新颖性。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部14篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
