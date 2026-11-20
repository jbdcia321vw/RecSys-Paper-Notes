---
hide:
  - toc
---

# 📖 IEEE TKDE 2022 推荐系统论文解读

10 篇 IEEE TKDE 2022 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDE ×8

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TNNLS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（评分预测）&quot;,&quot;幅值约束&quot;,&quot;球坐标重参数化&quot;]"><strong><a href="collaborative_filtering/magnitude_bounded_matrix_factorisation_for_recommender_systems/">Magnitude Bounded Matrix Factorisation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>通过球坐标重参数化约束用户/物品特征幅值，实现个体化预测边界与快速SGD，稀疏数据RMSE/F1多数领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱嵌入&quot;,&quot;元路径与嵌入传播&quot;,&quot;可解释推荐&quot;]"><strong><a href="graph_rec/a_survey_on_knowledge_graph_based_recommender_systems/">A Survey on Knowledge Graph-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以“嵌入/路径/统一”三范式归档 39 篇 KG 推荐工作，梳理 7 类场景数据集、可解释性来源与开放问题。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;字典学习&quot;,&quot;动态用户偏好&quot;,&quot;GRU+Attention&quot;]"><strong><a href="sequential_rec/modeling_dynamic_user_preference_via_dictionary_learning_for_sequential_rec/">Modeling Dynamic User Preference via Dictionary Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>S2PNM以共享字典把GRU序列编码译为动态偏好，与静态偏好同空间相加；Netflix时间切分RMSE较SVD++降2.18%，Baby Care NDCG@5较SASRec升6.10%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;社交影响扩散&quot;,&quot;多层图注意力&quot;]"><strong><a href="social_group_rec/diffnet_a_neural_influence_and_interest_diffusion_network_for_social_recomm/">DiffNet++: A Neural Influence and Interest Diffusion Network for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将社交图与兴趣图统一为异构图，用多层注意力递归扩散影响与兴趣，采样排序下 HR@10 较最强基线提升 21%+</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;模体GCN&quot;,&quot;对抗训练&quot;]"><strong><a href="social_group_rec/enhancing_social_recommendation_with_adversarial_graph_convolutional_networ/">Enhancing Social Recommendation with Adversarial Graph Convolutional Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>模体GCN生成替代邻居、重构去噪与情境注意力，对抗训练统一；Full-Ranking下NDCG@10提升6.75%–7.73%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;高阶社交传播&quot;,&quot;层级注意力聚合&quot;]"><strong><a href="social_group_rec/modelling_high_order_social_relations_for_item_recommendation/">Modelling High-Order Social Relations for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用多层 GCN 沿社交图递归传播用户嵌入以注入高阶好友影响，再用注意力聚合各层输出；在 Yelp/Douban 全库排序下 Recall@20 最高提升 22.28%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;自注意力群体聚合&quot;,&quot;中心性感知图卷积&quot;]"><strong><a href="social_group_rec/overcoming_data_sparsity_in_group_recommendation/">Overcoming Data Sparsity in Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用多头自注意力聚合成员表征、中心性感知多视图图卷积增强用户表征并联合优化，双重缓解偶发群体与用户交互稀疏，Hits@20 显著优于 SIGR/GroupSA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;自我网络&quot;,&quot;分层图池化&quot;,&quot;谱域图滤波&quot;]"><strong><a href="social_group_rec/understanding_wechat_user_preferences_and_wow_diffusion/">Understanding WeChat User Preferences and \&quot;Wow\&quot; Diffusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>基于微信看一看海量数据揭示好友圈层结构对\&quot;Wow\&quot;与点击行为的相反影响，提出谱域滤波+分层图池化的DiffuseGNN预测模型，AUC最高提升约2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;离散哈希编码&quot;,&quot;去噪自编码器&quot;,&quot;成对排序损失&quot;]"><strong><a href="cold_start_rec/deep_pairwise_hashing_for_cold_start_recommendation/">Deep Pairwise Hashing for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>DAE 内容预训练 + 成对排序离散哈希，将用户/物品映射为二值码用汉明距离检索；10% 稀疏冷启动下 Yelp MRR 较 DDL 提升 23%，在线打分加速 3.7–6.3 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据预处理/召回前净化&quot;,&quot;职业恶意用户检测&quot;,&quot;注意力度量学习&quot;]"><strong><a href="debias_fairness_industrial/detect_professional_malicious_user_with_metric_learning_in_recommender_syst/">Detect Professional Malicious User with Metric Learning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 HDAN 情感分与 LFM 评分构造“情感落差”画像筛候选，再用注意力度量学习聚类识别职业恶意用户，F-score 最高 0.987。</p>
</dd>
</dl>
</section>
</div>
