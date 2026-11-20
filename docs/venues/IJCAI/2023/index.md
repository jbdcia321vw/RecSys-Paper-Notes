---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2023 推荐系统论文解读

8 篇 IJCAI (RecSys Track) 2023 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI ×4 · IJCAI 2023 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;意图解耦表征&quot;,&quot;意图级对比学习&quot;,&quot;码率约简正则&quot;]"><strong><a href="graph_rec/intent_aware_recommendation_via_disentangled_graph_contrastive_learning/">Intent-aware Recommendation via Disentangled Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户-物品-概念图为基础，用概念软聚类语义基引导行为意图解耦，配合意图级对比学习与码率约简正则，在 ML-100k/ML-1M/MtBusiness 全库排序下 Recall@20 最高相对提升 8.52%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列行为检索&quot;,&quot;多类型行为融合&quot;,&quot;边信息解耦注意力&quot;]"><strong><a href="sequential_rec/a_survey_on_user_behavior_modeling_in_recommender_systems/">A Survey on User Behavior Modeling in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以长度/类型/边信息三轴重划 UBM 为四类范式，串联代表模型与线上收益，论证长序列检索与解耦边信息最具落地性</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;神经符号学习&quot;,&quot;Beta 分布嵌入&quot;]"><strong><a href="sequential_rec/sequential_recommendation_with_probabilistic_logical_reasoning/">Sequential Recommendation with Probabilistic Logical Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布做概率逻辑嵌入，将可闭运算的 AND/NOT 逻辑推理网络与 DNN 序列模型双路解耦融合，全库排序下各骨干稳定涨点。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;社交与群组推荐&quot;,&quot;召回与精排&quot;,&quot;去噪自监督&quot;,&quot;跨视图对齐&quot;]"><strong><a href="social_group_rec/denoised_self_augmented_learning_for_social_recommendation/">Denoised Self-Augmented Learning for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出双视图GCN与交互引导去噪自监督，在Ciao/Epinions/Yelp的Leave-One-Out全库排序上显著超越MHCN/SMIN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;增量学习&quot;,&quot;特征陈旧性&quot;,&quot;经验回放&quot;]"><strong><a href="ctr_feature_interaction/feature_staleness_aware_incremental_learning_for_ctr_prediction/">Feature Staleness Aware Incremental Learning for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示增量训练中特征嵌入陈旧问题，用加权最大覆盖采样+陈旧度正则实现自适应历史回放，四数据集AUC平均提升1.21%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排/拍卖分配&quot;,&quot;真实拍卖&quot;,&quot;自动出价&quot;,&quot;临界ROI&quot;]"><strong><a href="ctr_feature_interaction/truthful_auctions_for_automated_bidding_in_online_advertising/">Truthful Auctions for Automated Bidding in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向自动出价公共估值与私有预算/ROI，提出DSIC拍卖；合成实验收入/福利超一价二价且达最优90%+。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;实时策略推荐&quot;,&quot;工作联盟评分&quot;,&quot;深度强化学习&quot;,&quot;心理治疗对话&quot;]"><strong><a href="rl_bandit_rec/supervisorbot_nlp_annotated_real_time_recommendations_of_psychotherapy_trea/">SupervisorBot: NLP-Annotated Real-Time Recommendations of Psychotherapy Treatment Strategies with Deep Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用WAI量表与对话嵌入生成回合级评分，将会话视为用户、治疗话题视为物品，以DDPG/TD3/BCQ实时推荐心理治疗策略。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/端侧推理&quot;,&quot;联邦推荐&quot;,&quot;双重个性化&quot;]"><strong><a href="debias_fairness_industrial/dual_personalization_on_federated_recommendation/">Dual Personalization on Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>PFedRec 以端侧轻量模型、局部评分函数与物品嵌入后微调实现双重个性化，在 ML-100K/1M 上 HR@10 较 Fed 最优提升约 8–10%，并给出 LDP 隐私-效用分析。</p>
</dd>
</dl>
</section>
</div>
