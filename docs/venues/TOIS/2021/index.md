---
hide:
  - toc
---

# 📖 ACM TOIS 2021 推荐系统论文解读

4 篇 ACM TOIS 2021 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×4

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/训练采样&quot;,&quot;负采样&quot;,&quot;图扩散核&quot;,&quot;策略梯度&quot;]"><strong><a href="collaborative_filtering/cosam_an_efficient_collaborative_adaptive_sampler_for_recommendation/">CoSam: An Efficient Collaborative Adaptive Sampler for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在图扩散核上定义协同采样分布并以自适应随机走采样，再用乘性集成交互框架矫正采样偏置；Epinions 上 Pre@5 提升 61.7% 且比对抗采样快 20 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;交互式在线推荐(多轮对话)&quot;,&quot;Thompson Sampling&quot;,&quot;冷启动&quot;,&quot;属性问询策略&quot;]"><strong><a href="agent_conversational_rec/seamlessly_unifying_attributes_and_items_conversational_recommendation_for_/">Seamlessly Unifying Attributes and Items: Conversational Recommendation for Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>把物品与属性统一进同一臂空间，用上下文 Thompson Sampling 端到端决定「问属性还是推物品」，Yelp 上 SR@15 较 EAR 提升 69.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多模态推荐&quot;,&quot;重排&quot;,&quot;用户偏好建模&quot;,&quot;键值记忆网络&quot;]"><strong><a href="multimodal_rec/learning_to_respond_with_your_favorite_stickers_a_framework_of_unifying_mul/">Learning to Respond with Your Favorite Stickers: A Framework of Unifying Multi-Modality and User Preference in Multi-Turn Dialog</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在多轮对话表情包回复匹配中引入位置感知键值记忆与门控融合，10候选采样排序下 MAP/R10@1 显著超越 SRS 与 MRFN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;评测基准与复现审计&quot;,&quot;神经协同过滤&quot;,&quot;基线调优与早停&quot;,&quot;数据划分泄漏&quot;]"><strong><a href="debias_fairness_industrial/a_troubling_analysis_of_reproducibility_and_progress_in_recommender_systems/">A Troubling Analysis of Reproducibility and Progress in Recommender Systems Research</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>系统复现26篇神经CF论文中的12篇，用统一调优的KNN/线性基线证明11/12可被简单方法击败，揭示评测协议与基线调优的系统性缺陷。</p>
</dd>
</dl>
</section>
</div>
