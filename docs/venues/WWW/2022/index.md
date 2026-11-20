---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2022 推荐系统论文解读

38 篇 WWW (TheWebConf) 2022 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** WWW 2022 ×24 · WWW ×13 · 个性化商品搜索 ×2 · 协同过滤 ×2 · 去偏公平与可信推荐 ×2 · 变分自编码器 ×2 · 图协同过滤 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化商品搜索" aria-pressed="false">个性化商品搜索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分自编码器" aria-pressed="false">变分自编码器<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤" aria-pressed="false">图协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;在线知识蒸馏&quot;,&quot;异构目标一致性&quot;,&quot;多任务梯度归一化&quot;]"><strong><a href="collaborative_filtering/consensus_learning_from_heterogeneous_objectives_for_one_class_collaborativ/">Consensus Learning from Heterogeneous Objectives for One-Class Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为 OCCF 目标模型挂异构目标辅助头，用排名一致性生成动态共识做在线蒸馏，全库排序下 R@20 最高提升 30.7%，推理零额外开销</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;倒排多索引&quot;,&quot;重要性采样&quot;,&quot;变分自编码器&quot;]"><strong><a href="collaborative_filtering/fast_variational_autoencoder_with_inverted_multi_index_for_collaborative_fi/">Fast Variational AutoEncoder with Inverted Multi-Index for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用倒排多索引分解 softmax 概率，构造亚线性时间的无偏/低偏采样器 MIDX_Uni/MIDX_Pop，使 Mult-VAE 训练提速 6× 且 NDCG@50 反超全 softmax 基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲空间&quot;,&quot;几何正则化&quot;,&quot;图协同过滤&quot;]"><strong><a href="collaborative_filtering/hrcf_enhancing_collaborative_filtering_via_hyperbolic_geometric_regularizat/">HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用“根对齐 + 原点感知惩罚”把双曲 embedding 推离原点，缓解双曲图聚合的过平滑，3 个数据集的 Recall@20 较 HGCF 提升 1.6%–3.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;损失函数设计&quot;,&quot;用户兴趣边界&quot;]"><strong><a href="collaborative_filtering/learning_explicit_user_interest_boundary_for_recommendation/">Learning Explicit User Interest Boundary for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为每个用户学习可学习兴趣边界 $b_u$，令正负样本分别与边界做 pairwise，融合 pointwise 灵活性与 pairwise 排序性，Leave-One-Out + 1:100 采样排序下 Hit@1 平均提升 5.22%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;链接预测&quot;,&quot;度加权路径分数&quot;]"><strong><a href="collaborative_filtering/revisiting_neighborhood_based_link_prediction_for_collaborative_filtering/">Revisiting Neighborhood-based Link Prediction for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>WWW&#x27;22 图学习 Workshop：把推荐还原为二部图链接预测，提出仅 6 个可学习参数的 LinkProp 度加权三跳路径分数，全库排序下 Amazon-Book 的 Recall/NDCG@20 相对 GTN 提升 60%/70%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;双侧交互网络&quot;,&quot;共检索机制&quot;,&quot;直播推荐&quot;]"><strong><a href="collaborative_filtering/who_to_watch_next_two_side_interactive_networks_for_live_broadcast_recommen/">Who to Watch Next: Two-side Interactive Networks for Live Broadcast Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向直播推荐的三元组（用户-主播-商品）双侧交互网络，用 PNN+RNN 编码静态/动态特征、双注意力挖掘 AND 交互模式并按类目共检索剪枝，线上 ACTR 提升 8.11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图神经网络&quot;,&quot;个性化商品搜索&quot;,&quot;邻居特征交互&quot;]"><strong><a href="graph_rec/ihgnn_interactive_hypergraph_neural_network_for_personalized_product_search/">IHGNN: Interactive Hypergraph Neural Network for Personalized Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 user-query-product 三元超边构图，两步超图传播并显式建模邻居高阶特征交互，HR@10 最高相对提升 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;个性化商品搜索&quot;,&quot;跳跃图卷积&quot;,&quot;过平滑&quot;]"><strong><a href="graph_rec/modeling_user_behavior_with_graph_convolution_for_personalized_product_sear/">Modeling User Behavior with Graph Convolution for Personalized Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用全体用户短期行为构建连续行为图，以跳跃图卷积丰富商品表征并接入零注意力用户建模，8 个 Amazon 域 NDCG@10 较 ZAM 提升 7.8%~41.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;邻域对比学习&quot;,&quot;原型聚类 (EM)&quot;,&quot;图协同过滤&quot;]"><strong><a href="self_supervised_denoising/improving_graph_collaborative_filtering_with_neighborhood_enriched_contrast/">Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>NCL 用 GNN 偶数层输出与 K-means 原型分别构造结构/语义邻域对比对，Full-Ranking 下 Yelp Recall@10 较 LightGCN 提升 26%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/粗排&quot;,&quot;跨模型一致性&quot;,&quot;隐式反馈去噪&quot;,&quot;KL 散度正则&quot;]"><strong><a href="self_supervised_denoising/learning_robust_recommenders_through_cross_model_agreement/">Learning Robust Recommenders through Cross-Model Agreement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨模型预测一致性作弱监督信号，通过 KL 一致性约束+似然最大化联合去噪正负样本，在 4 个骨干上一致提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长短期兴趣解耦&quot;,&quot;自监督对比学习&quot;]"><strong><a href="sequential_rec/disentangling_long_and_short_term_interests_for_recommendation/">Disentangling Long and Short-Term Interests for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用兴趣代理做自监督对比，解耦长短期兴趣并注意力自适应融合，淘宝/快手 GAUC 提升超 0.01、NDCG@2 提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;端到端推荐决策&quot;,&quot;分层Transformer&quot;,&quot;MOBA Draft&quot;]"><strong><a href="sequential_rec/draftrec_personalized_draft_recommendation_for_winning_in_multi_player_onli/">DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层Transformer联合玩家偏好与队伍协同，阈值约束胜率最大化；LOL HR@1较SASRec提升12.1%，86.9%用户愿用。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;频域滤波&quot;,&quot;全MLP&quot;]"><strong><a href="sequential_rec/filter_enhanced_mlp_is_all_you_need_for_sequential_recommendation/">Filter-enhanced MLP is All You Need for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>FMLP-Rec 用 FFT 频域可学习滤波替代自注意力，以全 MLP 去噪建模序列；8 数据集 1:99 负采样排序，NDCG@10 较最佳基线提升 4.5%–7.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Intent Contrastive Learning&quot;,&quot;EM&quot;,&quot;False-Negative Mitigation&quot;]"><strong><a href="sequential_rec/intent_contrastive_learning_for_sequential_recommendation/">Intent Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用K-means意图原型与意图对比SSL及FNM，在四数据集全库排序下较SOTA提升7.47%–33.33%。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;意图-兴趣解耦&quot;,&quot;物品感知对比学习&quot;]"><strong><a href="sequential_rec/intent_interest_disentanglement_and_item_aware_intent_contrastive_learning_/">Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>意图-兴趣解耦与物品感知意图对比，Amazon 三域全库排序下 HR@20 最高相对提升 14.62%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（下次物品预测）&quot;,&quot;学习式数据增强 (Learning to Augment)&quot;,&quot;策略梯度 REINFORCE&quot;,&quot;休闲用户 / 核心用户鸿沟&quot;]"><strong><a href="sequential_rec/learning_to_augment_for_casual_user_recommendation/">Learning to Augment for Casual User Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 REINFORCE 学习核心用户序列的 Keep/Drop 编辑策略，合成贴近休闲用户行为的增强序列，配合元验证集奖励与 replay，在 4 个数据集、3 种序列模型上让休闲用户 NDCG@5 相对最优基线提升 6.6%–9.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;对比学习&quot;,&quot;反向流正则&quot;]"><strong><a href="sequential_rec/re4_learning_to_re_contrast_re_attend_re_construct_for_multi_interest_recom/">Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出反向流 Re4：以对比、重注意、重构三项正则约束多兴趣表征；Amazon/Gowalla 全库排序 NDCG@20 相对提升 42%/12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布嵌入&quot;,&quot;冷启动&quot;]"><strong><a href="sequential_rec/sequential_recommendation_via_stochastic_self_attention/">Sequential Recommendation via Stochastic Self-Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将物品建模为高斯分布并以负 2-Wasserstein 距离替代点积自注意力，配合正负样本距离正则，在 5 个 Amazon 域上全库排序显著提升且冷启动物品增益最大。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端&quot;,&quot;copy mechanism&quot;,&quot;pointer network&quot;]"><strong><a href="generative_rec/conditional_generation_net_for_medication_recommendation/">Conditional Generation Net for Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>COGNet 以 copy-or-predict 编码器-解码器逐药生成处方，用 visit/medication 两级软检索决定复制历史药或生成新药，在 MIMIC-III 全药库评测上 Jaccard 较 SafeDrug 提升 2.36%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;解释生成&quot;,&quot;比较式解释&quot;,&quot;IDF-BLEU&quot;,&quot;策略梯度&quot;]"><strong><a href="agent_conversational_rec/comparative_explanations_of_recommendations/">Comparative Explanations of Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 extract-and-refine 比较式解释生成框架，用 vMF 方向分布选句、梯度方向改写、IDF-BLEU 分层奖励策略梯度训练，使解释能还原推荐排序。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排/后置解释生成&quot;,&quot;抽取式解释&quot;,&quot;图注意力网络&quot;,&quot;整数线性规划&quot;]"><strong><a href="agent_conversational_rec/graph_based_extractive_explainer_for_recommendations/">Graph-based Extractive Explainer for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用属性作中介构建用户-物品-属性-句子异构图 GAT+DCN 打分，再以 ILP 联合优化相关性与冗余度抽取评论句作解释，BLEU-1 最高 +28.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;互正则化&quot;]"><strong><a href="cold_start_rec/mutually_regularized_dual_collaborative_variational_auto_encoder_for_recomm/">Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双协同 VAE，把 UAE 末层权重视为物品嵌入并以内容互正则，配合对称推断实现新物品免重训；三数据集全库排序优于 CVAE/CoVAE 等。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;RKHS 分布对齐&quot;,&quot;拓扑保持&quot;]"><strong><a href="cross_domain_multi_scenario/deep_unified_representation_for_heterogeneous_recommendation/">Deep Unified Representation for Heterogeneous Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 RKHS 分布对齐 + 协方差拓扑保持将多类异构物品映射入统一空间，Douban AUC 0.804，线上 CTR +3.67%、冷启新物品 CTR +27.22%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;差分隐私矩阵发布&quot;,&quot;异构跨域对齐&quot;]"><strong><a href="cross_domain_multi_scenario/differential_private_knowledge_transfer_for_privacy_preserving_cross_domain/">Differential Private Knowledge Transfer for Privacy-Preserving Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>两阶段隐私保护跨域推荐：源域用 JLT/SJLT 差分隐私发布评分矩阵，目标域用异构 HeteroCDR 对齐用户表征；留一法+99负采样下 HR@5 相对 ETL 提升 9.1%~11.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨场景多图 CSMG&quot;,&quot;图神经网络消息传递&quot;]"><strong><a href="cross_domain_multi_scenario/multi_graph_based_multi_scenario_recommendation_in_large_scale_online_video/">Multi-Graph based Multi-Scenario Recommendation in Large-scale Online Video Services</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨场景多图 CSMG 承载 i2i 转移边，MGFN 做场景内消息传递+跨场景融合，配跨场景负采样，线上新用户 CTR/VV 提升 0.63%/0.71%，跨场景视频激活 +25%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;AutoML 特征选择&quot;,&quot;Gumbel-Softmax&quot;]"><strong><a href="ctr_feature_interaction/autofield_automating_feature_selection_in_deep_recommender_systems/">AutoField: Automating Feature Selection in Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微控制器 + Gumbel-Softmax 双层优化自动筛选特征域，Avazu 砍掉 50% 域反而涨点并加速推理 4%–17%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排/机制设计&quot;,&quot;校准信息结构&quot;,&quot;GSP 拍卖&quot;,&quot;贝叶斯信息设计&quot;]"><strong><a href="ctr_feature_interaction/calibrated_click_through_auctions_an_information_design_approach/">Calibrated Click-Through Auctions: An Information Design Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 GSP 点击通过拍卖的信息披露问题建模为校准约束下的最优无偏 CTR 估计量设计，证明私密+相关信号可在对称环境中同时达成社会效率与近乎完全剩余榨取。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排/预拍卖&quot;,&quot;两阶段拍卖&quot;,&quot;Pre-Auction Score&quot;,&quot;GSP&quot;]"><strong><a href="ctr_feature_interaction/on_designing_a_two_stage_auction_for_online_advertising/">On Designing a Two-stage Auction for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 PAS 预拍卖打分，建模广告进入精排 Top-K 的概率，替代粗排 $b\\times ctr$ 贪心选择，在 Amazon Books 与工业拍卖日志上提升社会福利率与收入。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多反馈用户建模&quot;,&quot;强到弱注意力蒸馏&quot;,&quot;参与度多任务学习&quot;]"><strong><a href="multi_task_reranking/feedrec_news_feed_recommendation_with_various_user_feedbacks/">FeedRec: News Feed Recommendation with Various User Feedbacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用异构/同构 Transformer 与强到弱注意力统一建模点击、跳过、分享、点踩、读完、秒关六类反馈，并以点击-读完-时长多任务训练，AUC 相对最强基线提升约 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;双边匹配市场&quot;,&quot;双随机排序&quot;,&quot;PBM&quot;]"><strong><a href="multi_task_reranking/optimizing_rankings_for_recommendation_in_matching_markets/">Optimizing Rankings for Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双边匹配市场列表级联合排序，双随机策略优化社会福利，离线仿真匹配数显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Contextualized Transformer&quot;,&quot;Merged Cross-Attention&quot;,&quot;列表级多任务学习&quot;]"><strong><a href="multi_task_reranking/pear_personalized_re_ranking_with_contextualized_transformer_for_recommenda/">PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以特征级 MLP + 列表/历史双向交叉注意力 Transformer 重排 top-30 列表，并加列表级多任务监督，MicroVideo 与华为新闻日志上 gAUC@30 分别提升 1.07%/1.71%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;工具变量&quot;,&quot;因果解耦表征&quot;,&quot;搜索推荐联合建模&quot;]"><strong><a href="debias_fairness_industrial/a_model_agnostic_causal_learning_framework_for_recommendation_using_search_/">A Model-Agnostic Causal Learning Framework for Recommendation using Search Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用搜索查询作工具变量，将推荐物品嵌入解耦为因果/非因果两部分再融合，DIN/NRHUB 上 AUC 稳定提升约 0.7%–1.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端信息流分发（For You 召回+排序黑盒审计）&quot;,&quot;Sock-Puppet 算法审计&quot;,&quot;信息茧房与过滤气泡&quot;,&quot;Jaccard 信息流差异度量&quot;]"><strong><a href="debias_fairness_industrial/an_empirical_investigation_of_personalization_factors_on_tiktok/">An Empirical Investigation of Personalization Factors on TikTok</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用双生子傀儡账号审计 TikTok「For You」算法，量化关注/完播/点赞/语言/位置的个性化影响排序与噪声基线</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平与工业架构&quot;,&quot;评测与行为测试&quot;,&quot;行为测试 Behavioral Testing&quot;,&quot;流行度偏差与长尾&quot;,&quot;开源评测框架&quot;]"><strong><a href="debias_fairness_industrial/beyond_ndcg_behavioral_testing_of_recommender_systems_with_reclist/">Beyond NDCG: behavioral testing of recommender systems with RecList</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 NLP 行为测试迁移到推荐，用可插拔接口在隐空间做任务不变量、错误距离与切片偏差测试，揭示 HR@10 相近模型的巨大行为差异。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果结构学习&quot;,&quot;OOD 泛化&quot;,&quot;负采样&quot;]"><strong><a href="debias_fairness_industrial/causpref_causal_preference_learning_for_out_of_distribution_recommendation/">CausPref: Causal Preference Learning for Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微 DAG 学习不变用户偏好并配套反偏好负采样，在 5 类分布偏移下显著超越 NeuMF/LightGCN/IPS/DICE。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/排序打分&quot;,&quot;交叉配对排序损失&quot;,&quot;流行度去偏&quot;,&quot;动态难样本采样&quot;]"><strong><a href="debias_fairness_industrial/cross_pairwise_ranking_for_unbiased_item_recommendation/">Cross Pairwise Ranking for Unbiased Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>交叉配对损失抵消用户/物品曝光倾向，实现样本级无偏学习，Recall@20 较 DICE 提升 9.2%–18.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;模型训练后遗忘（召回/排序通用）&quot;,&quot;Machine Unlearning&quot;,&quot;Balanced Sharding&quot;,&quot;Adaptive Aggregation&quot;]"><strong><a href="debias_fairness_industrial/recommendation_unlearning/">Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 RecEraser：相似度平衡分片保留协同信息，注意力自适应聚合子模型；全库排序下高效精确遗忘且性能优于 SISA/GraphEraser。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（W2W 推荐列表重连）&quot;,&quot;随机游走隔离度&quot;,&quot;图边重连&quot;,&quot;吸收马尔可夫链&quot;]"><strong><a href="debias_fairness_industrial/rewiring_what_to_watch_next_recommendations_to_reduce_radicalization_pathwa/">Rewiring What-to-Watch-Next Recommendations to Reduce Radicalization Pathways</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 W2W 推荐图建模为 d-正则有向图，用吸收随机游走定义节点隔离度并提出 NP-难的 k-Rewiring，基于 Sherman-Morrison 增量更新的贪心算法在 YouTube/NELA-GT 上将最坏隔离度降低 40–80%。</p>
</dd>
</dl>
</section>
</div>
