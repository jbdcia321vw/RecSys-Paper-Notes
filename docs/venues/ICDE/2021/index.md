---
hide:
  - toc
---

# 🏛️ ICDE 2021 推荐系统论文解读

8 篇 ICDE 2021 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;用户邻域&quot;,&quot;归纳式表征&quot;]"><strong><a href="collaborative_filtering/explore_user_neighborhood_for_real_time_e_commerce_recommendation/">Explore User Neighborhood for Real-time E-commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用归纳式 UI 模型在线推理用户向量，实时检索相似用户邻域，与全局打分融合，实现工业级实时用户邻域召回；全库排序下平均提升 SASRec 8.9% HR@20，线上点击+2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多空间度量学习&quot;,&quot;球面黎曼优化&quot;]"><strong><a href="collaborative_filtering/multi_facet_recommender_networks_with_spherical_optimization/">Multi-Facet Recommender Networks with Spherical Optimization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用多面投影空间化解单度量空间的表征冲突，并以球面黎曼优化增强鲁棒性，6 数据集 HR/nDCG 最高提升 47%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;有向异构图&quot;,&quot;多视图嵌入传播&quot;,&quot;双成对损失&quot;]"><strong><a href="graph_rec/group_buying_recommendation_for_social_e_commerce/">Group-Buying Recommendation for Social E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建发起者/参与者/社交分享三类有向异构图，用视图内+跨视图 GCN 与双成对损失解团购推荐，采样排序下 Recall/NDCG 提升 2.69%–7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回排序&quot;,&quot;多行为推荐&quot;,&quot;跨行为关系建模&quot;,&quot;图消息传播&quot;]"><strong><a href="graph_rec/multi_behavior_enhanced_recommendation_with_cross_interaction_collaborative/">Multi-Behavior Enhanced Recommendation with Cross-Interaction Collaborative Relation Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GNMR 用类型特定消息编码 + 跨行为注意力聚合建模多行为依赖，在 HR/NDCG@10 上相对最优基线提升 4.6%–27.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;后置解释层 (Post-hoc Explanation)&quot;,&quot;本体建模 (OWL/DL)&quot;,&quot;SPARQL 推理与知识图谱&quot;]"><strong><a href="agent_conversational_rec/semantic_modeling_for_food_recommendation_explanations/">Semantic Modeling for Food Recommendation Explanations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 OWL 本体 FEO 与 SPARQL 推理为食物推荐生成情境/对比/反事实解释，以 3 个能力问题做任务级评测。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;会话知识图谱预训练&quot;,&quot;用户状态表征&quot;,&quot;对话交互表征&quot;]"><strong><a href="ctr_feature_interaction/improving_conversational_recommendation_system_by_pretraining_on_billions_s/">Improving Conversational Recommendation System by Pretraining on Billions Scale of Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 60 亿三元组会话知识图谱，用 GCN+TransE 联合预训练生成用户状态与对话交互表征，注入 DCN 做 CTR 排序，Alime 线上较 DCN 提升 4.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;黑盒 Top-K 输出/数据注入&quot;,&quot;跨域用户画像&quot;,&quot;强化学习策略梯度&quot;]"><strong><a href="debias_fairness_industrial/attacking_black_box_recommendations_via_copying_cross_domain_user_profiles/">Attacking Black-box Recommendations via Copying Cross-domain User Profiles</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>ICDE 2021｜可信推荐/数据投毒：用跨域真实画像复制、层次策略梯度选择与裁剪画像，在黑盒 Top-K 查询下攻击推荐，HR@20 最高提升 166.1%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;工业级全链路架构&quot;,&quot;会话推荐近邻方法&quot;,&quot;在线 A/B 延迟优化&quot;]"><strong><a href="debias_fairness_industrial/learnings_from_a_retail_recommendation_system_on_billions_of_interactions_a/">Learnings from a Retail Recommendation System on Billions of Interactions at bol.com</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>电商 ABO 系统实证：item-CF 共现打分+九路召回集成优于神经会话模型，且 p90 延迟 32→15ms 带来订单 +2.19%。</p>
</dd>
</dl>
</section>
</div>
