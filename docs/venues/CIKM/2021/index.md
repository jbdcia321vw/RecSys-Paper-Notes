---
hide:
  - toc
---

# 🏛️ CIKM 2021 推荐系统论文解读

42 篇 CIKM 2021 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×3 · 时序点过程 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序点过程" aria-pressed="false">时序点过程<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;离散路径索引&quot;,&quot;束搜索&quot;]"><strong><a href="collaborative_filtering/deep_retrieval_learning_a_retrievable_structure_for_large_scale_recommendat/">Deep Retrieval: Learning A Retrievable Structure for Large-Scale Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出端到端学习的K×D路径索引与束搜索召回，在MovieLens/Amazon及线上亿级系统逼近暴力检索并显著优于ANN。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;排序蒸馏&quot;,&quot;隐式反馈&quot;]"><strong><a href="collaborative_filtering/dual_correction_strategy_for_ranking_distillation_in_top_n_recommender_syst/">Dual Correction Strategy for Ranking Distillation in Top-N Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用师生排名差异动态选取待蒸馏知识，并行做用户侧与物品侧排序校正，Top-5 Hit 相对 RRD 提升 3.0%–6.6%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/评分预测&quot;,&quot;矩阵补全&quot;,&quot;局部-全局核&quot;]"><strong><a href="collaborative_filtering/glocal_k_global_and_local_kernels_for_recommender_systems/">GLocal-K: Global and Local Kernels for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用局部RBF核预训练AE、全局卷积核微调，在纯评分矩阵上做矩阵补全；ML-1M RMSE 0.822，稀疏场景更稳。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;余弦对比损失&quot;,&quot;负采样比例&quot;,&quot;轻量基线&quot;]"><strong><a href="collaborative_filtering/simplex_a_simple_and_strong_baseline_for_collaborative_filtering/">SimpleX: A Simple and Strong Baseline for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出余弦对比损失 CCL 并配合大负采样比，用极简 MF+行为聚合架构在 11 个数据集上超越 29 个基线，NDCG@20 最高提升 48.5%，训练提速 10 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质三部图&quot;,&quot;图去噪子图采样&quot;,&quot;Gumbel-Softmax 可微离散采样&quot;]"><strong><a href="graph_rec/concept_aware_denoising_graph_neural_network_for_micro_video_recommendation/">Concept-Aware Denoising Graph Neural Network for Micro-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用标题/评论抽取概念构建用户-视频-概念三部图，GRU+Gumbel-Softmax 个性化去噪子图，UAUC 相对 GAT 提升 8.3%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱蒸馏&quot;,&quot;条件注意力&quot;]"><strong><a href="graph_rec/conditional_attention_networks_for_distilling_knowledge_graphs_in_recommend/">Conditional Graph Attention Networks for Distilling and Refining Knowledge Graphs in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>知识感知注意力蒸馏目标特定子图 + 局部条件注意力精炼，Leave-One-Out 采样排序 Hit@10 相对提升约 10%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;可微采样 Gumbel-Softmax&quot;,&quot;关系感知注意力聚合&quot;,&quot;知识图谱去噪&quot;]"><strong><a href="graph_rec/dskreg_differentiable_sampling_on_knowledge_graph_for_recommendation_with_r/">DSKReG: Differentiable Sampling on Knowledge Graph for Recommendation with Relational GNN</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 co-interact 关系图并用 Gumbel-Softmax 可微采样 top-K 相关邻居，端到端去噪 KG 噪声；全库排序下 NDCG@20 最高提升 37.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲空间&quot;,&quot;图卷积网络&quot;,&quot;Lorentz 模型&quot;]"><strong><a href="graph_rec/fully_hyperbolic_graph_convolution_network_for_recommendation/">Fully Hyperbolic Graph Convolution Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将嵌入、邻域聚合、激活校准与测地距离打分全部置于 Lorentz 双曲空间，避免切空间近似误差，全库排序 R@20 相对 LightGCN 提升 7%–9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图信号处理与低通滤波&quot;,&quot;闭式解免训练协同过滤&quot;]"><strong><a href="graph_rec/how_powerful_is_graph_convolution_for_recommendation/">How Powerful is Graph Convolution for Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用图信号处理证明 LightGCN 本质是低通滤波，导出无限维嵌入闭式解，并提出免训练 GF-CF 基线，Amazon-book 上 Recall@20 较 LightGCN 提升 72.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;神经微分方程 NODE&quot;,&quot;可学习层组合时间&quot;,&quot;高阶 ODE 求解器与稠密连接&quot;]"><strong><a href="graph_rec/lt_ocf_learnable_time_ode_based_collaborative_filtering/">LT-OCF: Learnable-Time ODE-based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将 LightGCN 的离散层传播改写为用户/商品双共演化神经常微分方程，并学习层组合时间点，在三个基准的全库排序评测上稳定超越 LightGCN。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;无穷层消息传递极限&quot;,&quot;约束损失&quot;,&quot;物品共现图&quot;]"><strong><a href="graph_rec/ultragcn_ultra_simplification_of_graph_convolutional_networks_for_recommend/">UltraGCN: Ultra Simplification of Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用约束损失直接逼近无穷层消息传递的收敛态，无显式传播、仅 75 轮收敛，Recall@20 较 DGCF 提升 61.4% 且训练加速 14 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;相关特征掩码 (CFM)&quot;,&quot;对比学习正则化&quot;,&quot;双塔召回&quot;]"><strong><a href="self_supervised_denoising/self_supervised_learning_for_large_scale_item_recommendations/">Self-supervised Learning for Large-scale Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出双塔召回上的多任务 SSL 框架与互信息引导的相关特征掩码 CFM，在 490M/1B 样本全库排序下 Recall@10 提升 6.1%/10.2%，线上 A/B 冷启切片 +4.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;连续时间编码&quot;,&quot;时序图协同注意力&quot;]"><strong><a href="sequential_rec/continuous_time_sequential_recommendation_with_temporal_graph_collaborative/">Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构造连续时间二分图，用时间核编码+协同注意力统一序列模式与跨用户时序协同信号，Recall@10 平均绝对提升 22.5%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;时序点过程&quot;,&quot;社交激励&quot;,&quot;事件级直接范式&quot;]"><strong><a href="sequential_rec/extracting_attentive_social_temporal_excitation_for_sequential_recommendati/">Extracting Attentive Social Temporal Excitation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用时序点过程在事件级直接建模好友行为的细粒度激励，STEN 在三数据集 LOO 全库排序下 Recall@10 最高提升 5.79%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双曲空间超图卷积&quot;,&quot;自监督预训练&quot;]"><strong><a href="sequential_rec/hyperbolic_hypergraphs_for_sequential_recommendation/">Hyperbolic Hypergraphs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用月度/季度/年度分层双曲超图卷积 + 三项自监督预训练（含超边链接预测）建模序列推荐，在 AMT/Goodreads 采样排序下 NDCG@1 最高提 31.61%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;时间切片图&quot;,&quot;时序点过程&quot;,&quot;双重动态表征&quot;]"><strong><a href="sequential_rec/learning_dual_dynamic_representations_on_time_sliced_user_item_interaction_/">Learning Dual Dynamic Representations on Time-Sliced User-Item Interaction Graphs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>按等长时间切片构建全局用户-物品二部图，用切片 GNN + 双侧 GRU 建模用户/物品双重动态，再以时序点过程辅助任务补偿离散化时间损失；1:100 采样排序下 NDCG@10 相对 SCoRe 提升 3.4%–10.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;组合嵌入&quot;,&quot;双注意力&quot;]"><strong><a href="sequential_rec/lightweight_self_attentive_sequential_recommendation/">Lightweight Self-Attentive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>LSAN 用商余数动态组合嵌入压缩物品表，并以卷积/自注意力双分支建模局部/全局偏好；Leave-One-Out+Full-Ranking 下 HR@20 最高较 SASRec 提升 99.65%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;高斯分布表征&quot;,&quot;Wasserstein 距离&quot;,&quot;冷启动&quot;]"><strong><a href="sequential_rec/modeling_sequences_as_distributions_with_uncertainty_for_sequential_recomme/">Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>CIKM 2021][序列与会话推荐] 用椭圆高斯分布刻画物品与序列不确定性，双 Transformer 分别建模均值/协方差，以 2-Wasserstein 距离排序，全库采样排序下 Recall@1 最高提升 32.4%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图协同训练&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/self_supervised_graph_co_training_for_session_based_recommendation/">Self-Supervised Graph Co-Training for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以物品视图/会话视图双图构建互补充自监督信号，用协同训练互传伪标签并加对抗散度约束，Tmall P@20 提升 8.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排(评分预测)&quot;,&quot;评论情感学习&quot;,&quot;交互融合网络&quot;]"><strong><a href="multimodal_rec/sifn_a_sentiment_aware_interactive_fusion_network_for_review_based_item_rec/">SIFN: A Sentiment-aware Interactive Fusion Network for Review-based Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用BERT编码评论并引入显式情感监督，结合融合+交互网络耦合身份与评论特征做评分预测；5个Amazon数据集随机划分MSE下平均相对CARP降低2.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;社交与群组推荐&quot;,&quot;端到端组决策预测/排序&quot;,&quot;群组隐式反馈&quot;,&quot;反向社会选择&quot;,&quot;隐私泄露&quot;]"><strong><a href="social_group_rec/deepgroup_representation_learning_for_group_recommendation_with_implicit_fe/">DeepGroup: Representation Learning for Group Recommendation with Implicit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>仅用群组隐式反馈学习置换不变群表征，预测新组决策并反推个人偏好，在合成群组数据上显著优于AGREE等基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;双尺度自监督对比学习&quot;,&quot;群组偏好聚合&quot;]"><strong><a href="social_group_rec/double_scale_self_supervised_hypergraph_learning_for_group_recommendation/">Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用用户级/群组级双层超图刻画群内与跨群组高阶交互，并以粗/细粒度节点丢弃构造双尺度自监督对比信号，缓解群组交互稀疏；在三个基准上 NDCG@20 相对最强基线最高提升约 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;元路径异质图神经网络&quot;,&quot;图互信息自监督学习&quot;]"><strong><a href="social_group_rec/social_recommendation_with_self_supervised_metagraph_informax_network/">Social Recommendation with Self-Supervised Metagraph Informax Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出元路径异质图编码 + 跨元路径注意力 + 子结构感知互信息自监督的 SMIN，在 Ciao/Epinions/Yelp 的 1:99 采样排序上较 NGCF+ 提升约 2.4% HR@10。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;Meta-Learning&quot;,&quot;上下文调制&quot;]"><strong><a href="cold_start_rec/cmml_contextual_modulation_meta_learning_for_cold_start_recommendation/">CMML: Contextual Modulation Meta Learning for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用上下文编码+混合上下文+调制网络替代 MAML 内循环梯度，纯前馈实现冷启动任务快速适配，精度持平/超越 s2Meta 而推理开销降 2k 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;零样本学习&quot;,&quot;跨模态重构&quot;,&quot;双塔协同训练&quot;]"><strong><a href="cold_start_rec/zero_shot_on_the_cold_start_problem_model_agnostic_interest_learning_for_re/">Zero Shot on the Cold-Start Problem: Model-Agnostic Interest Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将新用户冷启动类比零样本学习，用双自编码器跨模态重构+MMD 对齐属性与行为空间，为排序塔生成虚拟行为，线上 CTR 提升 13%–15%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;排序&quot;,&quot;市场自适应&quot;,&quot;MAML/Forking&quot;]"><strong><a href="cross_domain_multi_scenario/cross_market_product_recommendation/">Cross-Market Product Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 XMarket 与 FOREC：MAML 预训练市场无关 NMF，再分叉冻结底层并加市场头微调，在低资源市场 Top-N 推荐中稳定超越迁移基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;星型拓扑参数生成&quot;,&quot;分区归一化&quot;]"><strong><a href="cross_domain_multi_scenario/one_model_to_serve_all_star_topology_adaptive_recommender_for_multi_domain_/">One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>STAR 用共享中心 FCN 与域私有 FCN 的逐元素乘积生成各域专属网络，辅以分区归一化与域指示辅助网络，在阿里 19 域生产数据上 Overall AUC 0.6506、线上 CTR +8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景CTR&quot;,&quot;干预去偏&quot;,&quot;公平系数&quot;]"><strong><a href="cross_domain_multi_scenario/sar_net_a_scenario_aware_ranking_network_for_personalized_fair_recommendati/">SAR-Net: A Scenario-Aware Ranking Network for Personalized Fair Recommendation in Hundreds of Travel Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>双注意力跨场景兴趣+场景去偏专家+FC加权，20场景AUC 0.6997，线上CTR+5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;重排&quot;,&quot;异构行为序列&quot;,&quot;搜索推荐联合建模&quot;]"><strong><a href="cross_domain_multi_scenario/user_a_unified_information_search_and_recommendation_model_based_on_integra/">USER: A Unified Information Search and Recommendation Model based on Integrated Behavior Sequence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将搜索与推荐行为融合为异构序列，用统一排序框架同时服务两任务，统一数据下搜索 MAP 提升 3.49%、推荐 MAP 提升 71.96%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;精排&quot;,&quot;Neural ODE&quot;,&quot;Next-POI 序列预测&quot;]"><strong><a href="poi_spatiotemporal_rec/lightmove_a_lightweight_next_poi_recommendation_for_taxicab_rooftop_adverti/">LightMove: A Lightweight Next-POI Recommendation for Taxicab Rooftop Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用含跳步与自适应参数生成的 NODE 轻量预测出租车下一位置，全库排序、逐车时序切分下 Foursquare Hits@1 提升 23.6% 且参数减 78%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;周期兴趣建模&quot;,&quot;时空上下文切序列&quot;]"><strong><a href="poi_spatiotemporal_rec/st_pil_spatial_temporal_periodic_interest_learning_for_next_point_of_intere/">ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用星期/时段/geohash三重时空上下文切分历史轨迹构造周期兴趣序列，配合双层注意力融合长短兴趣，Leave-One-Out 全库排序下 Acc@1 提升约 70%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;超图神经网络&quot;,&quot;多模态序列建模&quot;]"><strong><a href="ctr_feature_interaction/click_through_rate_prediction_with_multi_modal_hypergraphs/">Click-Through Rate Prediction with Multi-Modal Hypergraphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用时间槽切分的用户/物品双超图 + 自监督兴趣超边做多模态 CTR 建模，三数据集 AUC 提升 3.18%–7.43%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;解耦自注意力&quot;,&quot;高阶特征交叉&quot;]"><strong><a href="ctr_feature_interaction/disentangled_self_attentive_neural_networks_for_click_through_rate_predicti/">Disentangled Self-Attentive Neural Networks for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将自注意力打分拆成白化成对项与一元项以解耦梯度，Criteo/Avazu 全量 AUC 上稳定超越 AutoInt 等 SOTA 且训练开销相当。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR特征交叉&quot;,&quot;精排&quot;,&quot;可复现评测基准&quot;,&quot;特征交叉网络对比&quot;,&quot;FuxiCTR&quot;]"><strong><a href="ctr_feature_interaction/open_benchmarking_for_click_through_rate_prediction/">Open Benchmarking for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 BARS-CTR 标准化基准，统一协议重评估 24 个 CTR 模型，揭示充分调参后模型差距远小于论文宣称。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;文案到 CTR 回归&quot;,&quot;弱监督语义相似检索&quot;,&quot;冷启动泛化&quot;]"><strong><a href="ctr_feature_interaction/tsi_an_ad_text_strength_indicator_using_text_to_ctr_and_semantic_ad_similar/">TSI: an Ad Text Strength Indicator using Text-to-CTR and Semantic-Ad-Similarity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用微调 BERT 做广告文案-to-CTR、弱监督 SBERT 检索语义邻居，按邻域相对 pCTR 判定文案强弱并给建议；冷启动 AUC +1.7%，P@1 0.93，线上采纳者 CTR +27%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;意外点击过滤&quot;,&quot;软标签交叉熵&quot;,&quot;概率校准&quot;]"><strong><a href="ctr_feature_interaction/unbiased_filtering_of_accidental_clicks_in_verizon_media_native_advertising/">Unbiased Filtering of Accidental Clicks in Verizon Media Native Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用辅助模型预测意外点击概率作为跳过的软标签、以软标签交叉熵训练 Offset，保持无偏校准并获线上 1.18% 收入提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/重排（排序融合）&quot;,&quot;Actor-Critic&quot;,&quot;冷启动 LTV&quot;,&quot;POC-MDP&quot;]"><strong><a href="rl_bandit_rec/reinforcement_learning_to_optimize_lifetime_value_in_cold_start_recommendat/">Reinforcement Learning to Optimize Lifetime Value in Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 POC-MDP 建模商品生命周期，Actor-Critic 预估冷启商品 LTV 并与 CTR 分数双路融合排序，淘宝线上 IPV/GMV 相对提升 8.67%/18.03%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;增益建模 (Uplift Modeling)&quot;,&quot;张量分解&quot;,&quot;回归断点设计&quot;]"><strong><a href="debias_fairness_industrial/causcf_causal_collaborative_filtering_for_recommendationeffect_estimation/">CausCF: Causal Collaborative Filtering for Recommendation Effect Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将矩阵分解扩展为用户-物品-干预三维张量分解以估计推荐增益，并用回归断点设计（RDD）替代 RCT 做无偏评估，ATE 误差降 13.8%、Uplift@10 提升 21.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/事后解释&quot;,&quot;反事实推理&quot;,&quot;可解释推荐评测&quot;,&quot;属性级解释&quot;]"><strong><a href="debias_fairness_industrial/counterfactual_explainable_recommendation/">Counterfactual Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以反事实扰动物品属性面生成低复杂度、高强度的推荐解释，并首创用户/模型双视角离线评测指标。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;工业级全链路架构与评测基准&quot;,&quot;离线评测与基准&quot;,&quot;统一数据抽象&quot;,&quot;GPU 加速 Top-K 全排序&quot;]"><strong><a href="debias_fairness_industrial/recbole_towards_a_unified_comprehensive_and_efficient_framework_for_recomme/">RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出统一 PyTorch 推荐库 RecBole：73 模型 / 28 数据集、六类原子文件与 Interaction 张量结构、四接口标准化评测、GPU 矩阵化 Full-Ranking 加速 13–27×。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;结构因果模型&quot;,&quot;反事实数据增广&quot;]"><strong><a href="debias_fairness_industrial/top_n_recommendation_with_counterfactual_user_preference_simulation/">Top-N Recommendation with Counterfactual User Preference Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 Pearl 结构因果模型建模 U→R→S 生成过程，以高斯策略干预推荐列表生成反事实排序样本，MIND 上 NDCG@10 提升 10.4%，冷门物品 HR@10 提升 151.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;对抗训练&quot;,&quot;流行度偏差&quot;]"><strong><a href="debias_fairness_industrial/understanding_the_effects_of_adversarial_personalized_ranking_optimization_/">Understanding the Effects of Adversarial Personalized Ranking Optimization Method on Recommendation Quality</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>揭示 APR 在提升准确率与鲁棒性时放大流行度偏差、降低新颖性与覆盖度，并以梯度更新理论和水玻璃现象验证。</p>
</dd>
</dl>
</section>
</div>
