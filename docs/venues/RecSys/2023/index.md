---
hide:
  - toc
---

# 🏛️ RecSys 2023 推荐系统论文解读

45 篇 RecSys 2023 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×2 · Contextual Bandit ×2 · 大语言模型推荐 ×2 · 离线评测协议 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">24</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测协议" aria-pressed="false">离线评测协议<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;兴趣胶囊动态路由&quot;,&quot;辅助任务联合学习&quot;,&quot;物品表征增广&quot;]"><strong><a href="collaborative_filtering/a_model_agnostic_framework_for_recommendation_via_interest_aware_item_embed/">A Model-Agnostic Framework for Recommendation via Interest-aware Item Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多兴趣胶囊网络退化为辅助任务并共享物品嵌入，用兴趣梯度增广 DIN/DIEN 精排，Amazon 双集 AUC 稳定提升</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;对抗训练&quot;,&quot;Sharpness-aware Minimization&quot;]"><strong><a href="collaborative_filtering/adversarial_collaborative_filtering_for_free/">Adversarial Collaborative Filtering for Free</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SharpCF 以轨迹损失隐式替代对抗训练，免二次反传；全库排序下均超 BPR 18.1%、APR 6.82%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;解耦表征&quot;,&quot;对比学习&quot;]"><strong><a href="collaborative_filtering/augmented_negative_sampling_for_collaborative_filtering/">Augmented Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>解耦难/易因子后受控扰动易因子合成负样本，并以增强增益采样，5 数据集全库排序显著超越 DNS/MixGCF/DENS。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;截断SVD&quot;,&quot;谱聚类与社区检测&quot;]"><strong><a href="collaborative_filtering/of_spiky_svds_and_music_recommendation/">Of Spiky SVDs and Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示截断 SVD 音乐嵌入空间中普遍存在的“尖峰”几何，证明其等价于变度社区（DCBM），并给出余弦/点积选择与嵌入稳定性评估的工业实践。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;平均池化嵌入&quot;,&quot;序统计量闭式分析&quot;,&quot;表征自洽性&quot;]"><strong><a href="collaborative_filtering/on_the_consistency_of_average_embeddings_for_item_recommendation/">On the Consistency of Average Embeddings for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 Consistency_k 期望精度指标与正态近似闭式解，证明理想 i.i.d. 嵌入下平均向量自洽，而 Deezer 真实嵌入的 Consistency 不足 14%/6%/2%，揭示工业平均池化表征的系统性失配。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣学习&quot;,&quot;兴趣感知难负采样&quot;,&quot;路由正则化&quot;]"><strong><a href="collaborative_filtering/rethinking_multi_interest_learning_for_candidate_matching_in_recommender_sy/">Rethinking Multi-Interest Learning for Candidate Matching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示多兴趣召回中均匀采样负例失效与路由坍缩两大训练缺陷，用兴趣感知难负采样与路由方差正则修复，Recall@20 相对提升最高 19.7%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Bayesian Surprise&quot;,&quot;Serendipity&quot;,&quot;LDA Topic Model&quot;]"><strong><a href="collaborative_filtering/topic_level_bayesian_surprise_and_serendipity_for_recommender_systems/">Topic-Level Bayesian Surprise and Serendipity for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 LDA 主题分布上的贝叶斯惊喜（KL 后验-先验散度）结合评分构成事后惊喜度，再用协作过滤找相似用户做惊喜推荐；Goodreads 4 用户 449 本书人工标注，AROW 惊喜检测 F1 56.5%、AROW+vbBLR 推荐 F1 37.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;全局新闻图&quot;,&quot;GGNN&quot;,&quot;实体图增强&quot;]"><strong><a href="graph_rec/going_beyond_local_global_graph_enhanced_personalized_news_recommendations/">Going Beyond Local: Global Graph-Enhanced Personalized News Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用跨用户全局新闻图(GGNN)与全局实体图增强新闻/候选表示，MIND-large AUC 69.54、nDCG@10 44.19，兼顾多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低比特量化&quot;,&quot;梯度直通估计器&quot;,&quot;Hessian 迹估计&quot;]"><strong><a href="graph_rec/hessian_aware_quantized_node_embeddings_for_recommendation/">Hessian-aware Quantized Node Embeddings for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用二阶 Hessian 迹修正 STE 的梯度失配，实现任意比特 GNN 节点量化嵌入，1-bit 下 Recall@50 较 HashGNN 平均提升约 16%，推理加速 3.6×。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;混合曲率流形&quot;,&quot;几何感知 Margin&quot;]"><strong><a href="graph_rec/knowledge_based_multiple_adaptive_spaces_fusion_for_recommendation/">Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习曲率的统一空间兼容双曲/欧氏/球面并注意力融合多子空间，配合几何感知 margin，在三个 KG 数据集采样排序下 HR@20 最高提升 1.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为序列推荐&quot;,&quot;跨关系对比学习&quot;,&quot;动态记忆网络&quot;]"><strong><a href="self_supervised_denoising/multi_relational_contrastive_learning_for_recommendation/">Multi-Relational Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多行为序列推荐框架 RCL，用关系感知图编码器 + 跨关系记忆网络 + 行为级 InfoNCE，全库 99 负采样下 HR@10 最高提升 24.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;等变对比学习&quot;,&quot;替换项检测&quot;,&quot;数据增广&quot;]"><strong><a href="sequential_rec/equivariant_contrastive_learning_for_sequential_recommendation/">Equivariant Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用生成器-条件判别器实现替换项检测损失，使序列编码器对侵入式增广等变、对温和增广不变；全库排序下 ML-1m/Yelp Recall@10 较 SASRec 提升 19.5%/10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;单纯曝光效应&quot;,&quot;重复消费建模&quot;]"><strong><a href="sequential_rec/ex2vec_characterizing_users_and_items_from_the_mere_exposure_effect/">Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ACT-R 基级激活调制用户-物品相对距离，再经倒 U 型兴趣函数打分；Deezer 新歌重复收听预测 Balanced Acc 64.27%，较最优基线相对提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;物品相关性加权损失&quot;,&quot;多未来物品评估协议&quot;]"><strong><a href="sequential_rec/integrating_item_relevance_in_training_loss_for_sequential_recommender_syst/">Integrating Item Relevance in Training Loss for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多未来物品评估协议与相关性加权损失替代单目标假设，SASRec 在 4 个数据集上 NDCG@10 提升约 1.2%–1.63%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负反馈建模&quot;,&quot;反事实模拟&quot;]"><strong><a href="sequential_rec/learning_from_negative_user_feedback_and_measuring_responsiveness_for_seque/">Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 not-to-recommend 损失将显式不喜欢与隐式跳过纳入序列召回训练，并用反事实模拟量化响应性，线上不喜欢率降 2.44%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;LLM item embedding&quot;,&quot;PCA initialization&quot;,&quot;BERT4Rec&quot;,&quot;prompt fine-tuning&quot;]"><strong><a href="sequential_rec/leveraging_large_language_models_for_sequential_recommendation/">Leveraging Large Language Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>三种LLM增强序列推荐：LLM物品嵌入+PCA初始化BERT4Rec在Beauty/Delivery Hero上NDCG@20提升15–20%，简单语义相似检索在Beauty具竞争力。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Buy-It-Again 复购推荐&quot;,&quot;生存分析&quot;,&quot;ARIMA 时序特征&quot;]"><strong><a href="sequential_rec/personalized_category_frequency_prediction_for_buy_it_again_recommendations/">Personalized Category Frequency prediction for Buy It Again recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出层级 PCIC 模型，用生存分析+ARIMA 特征预测用户复购品类，再在品类内按频率/新近度排名，NDCG@10 最高提升约 15%，线上单位购买量 +27.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双边序列匹配&quot;,&quot;自蒸馏&quot;,&quot;时间敏感共注意力&quot;]"><strong><a href="sequential_rec/reciprocal_sequential_recommendation/">Reciprocal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将双向推荐重构为双视角行为序列匹配任务，用时间敏感共注意力做微级匹配并以自蒸馏压回点积宏级匹配，采样排序下 HR@5 最高相对提升 83%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Next Basket Recommendation&quot;,&quot;Hawkes Process&quot;,&quot;时间感知权重&quot;]"><strong><a href="sequential_rec/time_aware_item_weighting_for_the_next_basket_recommendations/">Time-Aware Item Weighting for the Next Basket Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用物品级 Hawkes 强度函数刻画复购周期并叠加 kNN 邻域模块，在 Leave-One-Basket 全库排序下相对 SOTA 提升 P/R 约 3%–9%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Decoder-only Transformer&quot;,&quot;播放列表生成&quot;,&quot;在线动态量化&quot;]"><strong><a href="sequential_rec/track_mix_generation_on_music_streaming_services_using_transformers/">Track Mix Generation on Music Streaming Services using Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Deezer 用单隐层 Decoder-only Transformer 做“以歌生单”的混音歌单生成，借 ONNX 融合与动态量化落地，线上 A/B 听歌时长 +6.84%、新用户收藏 +20.26%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样交叉熵&quot;,&quot;Full-Ranking 评测&quot;]"><strong><a href="sequential_rec/turning_dross_into_gold_loss_is_bert4rec_really_better_than_sasrec/">Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 SASRec 与 BERT4Rec 的差距实为损失函数差异：统一为全库 softmax 交叉熵后 SASRec 更快更强，ML-1M 的 NDCG@10 提升 20.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Genre Spectrum&quot;,&quot;内容元数据嵌入&quot;]"><strong><a href="llm_rec/beyond_labels_leveraging_deep_learning_and_llms_for_content_metadata/">Beyond Labels: Leveraging Deep Learning and LLMs for Content Metadata</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 Genre Spectrum，用多标签监督把离散类型标签映射到连续隐空间，Mixup 增广后于 1.1M 影视数据 top-100 类型相似度低热度桶提升约 12 个百分点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;大语言模型推荐&quot;,&quot;召回/特征增强&quot;,&quot;异构行为融合&quot;,&quot;指令微调 LoRA&quot;,&quot;自然语言用户画像&quot;]"><strong><a href="llm_rec/heterogeneous_knowledge_fusion_a_novel_approach_for_personalized_recommenda/">Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ChatGPT 将美团外卖多维异构行为改写为自然语言知识，再以 LoRA 指令微调 ChatGLM-6B 输出类别/POI 偏好，离线 HR@5 提升 52.5%，冷启动在线 CTR +2.45%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;LLM4Rec&quot;,&quot;召回&quot;,&quot;合成查询数据增强&quot;,&quot;叙事驱动推荐&quot;,&quot;双塔检索&quot;]"><strong><a href="llm_rec/large_language_model_augmented_narrative_driven_recommendations/">Large Language Model Augmented Narrative Driven Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 175B LLM 从 Yelp 评论合成叙事查询，训练 110M 双塔/交叉编码器，Pointrec 上 NDCG@10 相对提升 7%+。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;长尾冷启动&quot;,&quot;数据模拟与检索增强&quot;]"><strong><a href="agent_conversational_rec/alleviating_the_long_tail_problem_in_conversational_recommender_systems/">Alleviating the Long-Tail Problem in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用网页属性模拟平衡 CRS 语料，配合 DMP/CCA 预训练与检索增强+标签平滑微调，ReDial 上 Tail-Coverage@50 相对提升 23.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;物品-属性图&quot;,&quot;多任务预训练&quot;,&quot;对齐-均匀性损失&quot;]"><strong><a href="cold_start_rec/multi_task_item_attribute_graph_pre_training_for_strict_cold_start_item_rec/">Multi-task Item-attribute Graph Pre-training for Strict Cold-start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>抽取细粒度物品属性构建物品-属性图，以三任务对齐-均匀性预训练实现严格冷启动物品全排序召回，较 UniSRec 最高提升 64%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;情境特征建模&quot;,&quot;Gumbel-Softmax去噪&quot;,&quot;MLP-Mixer&quot;]"><strong><a href="ctr_feature_interaction/deep_situation_aware_interaction_network_for_click_through_rate_prediction/">Deep Situation-Aware Interaction Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出“情境(situation)”概念，用 Gumbel-Softmax 去噪 + 三向 MLP-Mixer 融合行为类型/时空情境特征，美团外卖线上 CTR +2.70%、GMV +2.16%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;特征工程与 AutoML 模型搜索（精排前置）&quot;,&quot;基数归一化互信息 CardMI&quot;,&quot;3MR 特征排序&quot;,&quot;哈希交互特征流式估计&quot;]"><strong><a href="ctr_feature_interaction/outrank_speeding_up_automl_based_model_search_for_large_sparse_data_sets_wi/">OutRank: Speeding up AutoML-based Model Search for Large Sparse Data sets with Cardinality-aware Feature Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出基数归一化互信息 CardMI 与 3MR 非短视特征排序，流式哈希估计交互与冗余，在 1.5M 生产 CTR 数据上实现约 30% AutoML 加速且 F1 无损。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;辅助模型&quot;,&quot;上下文特征解耦&quot;,&quot;工业级部署&quot;]"><strong><a href="ctr_feature_interaction/unleash_the_power_of_context_enhancing_large_scale_recommender_systems_with/">Unleash the Power of Context: Enhancing Large-Scale Recommender Systems with Context-Based Prediction Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用一个不含物品特征的上下文辅助模型预测上下文 CTR，作为新特征注入主 CTR 模型，离线 RIG 最高 +5.58%、线上 RPM +0.97%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;用户生命周期建模&quot;,&quot;动态损失加权&quot;]"><strong><a href="multi_task_reranking/stan_stage_adaptive_network_for_multi_task_recommendation_by_learning_user_/">STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布采样刻画用户生命周期阶段向量，动态加权各任务损失，工业数据集 AUC 最高 +18.5%，线上停留时长 +3.05%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Bandit&quot;,&quot;实时机器学习平台&quot;]"><strong><a href="rl_bandit_rec/adaptex_a_self_service_contextual_bandit_platform/">AdaptEx: A Self-Service Contextual Bandit Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Expedia 自服务上下文 Bandit 平台，Sampler–Flink–Spark 实时闭环，支持 TS/BLR/RLS/IGW/级联 Bandit，把多臂试验从 ~10 周压缩到 1 周。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Sleeping Bandit&quot;,&quot;EXP3&quot;]"><strong><a href="rl_bandit_rec/adversarial_sleeping_bandit_problems_with_multiple_plays_algorithm_and_rank/">Adversarial Sleeping Bandit Problems with Multiple Plays: Algorithm and Ranking Application</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向在线候选生成的对抗 Sleeping Bandit 多臂选择算法，给出 O(kN^2√(T log T)) 遗憾界，但无实证实验。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/策略层&quot;,&quot;Deep Exploration&quot;,&quot;Epistemic Neural Network&quot;,&quot;Randomized Value Function&quot;]"><strong><a href="rl_bandit_rec/deep_exploration_for_recommendation_systems/">Deep Exploration for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>随机化值函数+认识论网络在用户生命周期内固定认知指数实现深度探索，高保真仿真中生命周期累计回报较最强短视基线提升 57.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;Diag-LinUCB&quot;,&quot;稀疏二部图&quot;,&quot;冷启动&quot;]"><strong><a href="rl_bandit_rec/online_matching_a_real_time_bandit_system_for_large_scale_recommendations/">Online Matching: A Real-time Bandit System for Large-scale Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>离线双塔聚类稀疏二部图剪枝+Diag-LinUCB实时分布式Bandit，YouTube线上A/B提升鲜内容发现与语料探索</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (11)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;位置偏置建模&quot;,&quot;因果去偏与离线评估&quot;,&quot;曝光概率校准&quot;]"><strong><a href="debias_fairness_industrial/a_probabilistic_position_bias_model_for_short_video_recommendation_feeds/">A Probabilistic Position Bias Model for Short-Video Recommendation Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Yule-Simon 滚动预算的生存函数给出闭式曝光概率，仅 3 参数的 C-3PO 刷新 NLL 并让离线-在线相关性较 DCG 提升 16%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;TU Matching&quot;,&quot;互惠推荐&quot;]"><strong><a href="debias_fairness_industrial/fast_and_examination_agnostic_reciprocal_recommendation_in_matching_markets/">Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Choo–Siow 可转移效用匹配做互惠推荐重排，摆脱检查函数依赖并用 (2d+2) 维内积实现 ANN 检索，匹配总量超越 Naive/Reciprocal 且与 SW 持平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据采集与反馈&quot;,&quot;评分通胀&quot;,&quot;随机对照实验&quot;]"><strong><a href="debias_fairness_industrial/interface_design_to_mitigate_inflation_in_recommender_systems/">Interface Design to Mitigate Inflation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用Piki音乐App自然实验与RCT证明评分通胀源于用户异质与个性化曝光，计时器界面可提升评分信息量。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（评分预测）&quot;,&quot;差分隐私&quot;,&quot;集合矩阵分解&quot;,&quot;公共物品特征&quot;]"><strong><a href="debias_fairness_industrial/private_matrix_factorization_with_public_item_features/">Private Matrix Factorization with Public Item Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出DP-CMF，将公共物品元数据/相似度/公开评分以集合矩阵分解融入用户级DP矩阵分解，在ML10M评分预测上显著缩小隐私-效用差距。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;表征中立性&quot;,&quot;冷启动新用户&quot;]"><strong><a href="debias_fairness_industrial/providing_previously_unseen_users_fair_recommendations_using_variational_au/">Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在 Mult-VAE 召回上切分隐状态 z/b 并用解析高斯 KL 解耦敏感属性，使用户不相交划分下的新用户也能获得公平推荐</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与工业架构&quot;,&quot;端到端评测基准&quot;,&quot;Shilling Attack&quot;,&quot;攻击者知识约束&quot;,&quot;可复现性&quot;]"><strong><a href="debias_fairness_industrial/recad_towards_a_unified_library_for_recommender_attack_and_defense/">RecAD: Towards A Unified Library for Recommender Attack and Defense</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一推荐攻击-防御基准库，标准化数据/模型/工作流三模块与双轨评估协议，揭示攻击者知识比例与检测失效现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/重排&quot;,&quot;最优传输&quot;,&quot;拥堵缓解&quot;,&quot;职位推荐&quot;]"><strong><a href="debias_fairness_industrial/recon_reducing_congestion_in_job_recommendation_using_optimal_transport/">ReCon: Reducing Congestion in Job Recommendation using Optimal Transport</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可微 Sinkhorn 最优传输与 CNE 联合训练，在职位推荐中同时优化拥堵与相关性，取得更优 Pareto 权衡。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;Explainable Recommendation&quot;,&quot;Adversarial Robustness&quot;]"><strong><a href="debias_fairness_industrial/stability_of_explainable_recommendation/">Stability of Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>实证揭示特征级可解释推荐器在参数随机/FGSM扰动下解释稳定性显著下降，采样排序评估显示对抗噪声危害更大。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;可复现性审计&quot;,&quot;GRU4Rec&quot;,&quot;离线评测协议&quot;]"><strong><a href="debias_fairness_industrial/the_effect_of_third_party_implementations_on_reproducibility/">The Effect of Third Party Implementations on Reproducibility</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>审计 GRU4Rec 六个第三方复现：特征缺失与 bug 使 recall@20 相对官方下降 13%–99%，训练慢 2–336 倍，呼吁复现须对齐官方实现。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差去偏&quot;,&quot;噪声观看校正&quot;,&quot;因果分解&quot;]"><strong><a href="debias_fairness_industrial/uncovering_user_interest_from_biased_and_noised_watch_time_in_video_recomme/">Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将观看时长因果分解为兴趣、时长偏差与噪声三项，用分时长 GMM + 敏感度控制校正生成去偏标签，KuaiRand/WeChat 上 GAUC 相对提升约 10%，线上 VV +8.35%。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部11篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
