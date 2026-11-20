---
hide:
  - toc
---

# 🏛️ ECIR 2023 推荐系统论文解读

8 篇 ECIR 2023 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质图&quot;,&quot;Personalized PageRank&quot;,&quot;遗传算法&quot;]"><strong><a href="graph_rec/graph_based_recommendation_for_sparse_and_heterogeneous_user_interactions/">Graph-based Recommendation for Sparse and Heterogeneous User Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用异质交互图与遗传算法学习边权，PPR 全库排序；教育集 MRR@5 提升 6.7%，小样本保险集优于 NGCF。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;归纳式推荐&quot;,&quot;物品图卷积&quot;,&quot;Top-K 剪枝&quot;]"><strong><a href="graph_rec/item_graph_convolution_collaborative_filtering_for_inductive_recommendation/">Item Graph Convolution Collaborative Filtering for Inductive Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出物品图卷积协同过滤 IGCCF，用余弦物品图与 top-K 剪枝学习物品嵌入、以历史交互加权构造用户嵌入，实现免重训的归纳推荐，在四个数据集上优于 LightGCN 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/可解释重排&quot;,&quot;路径推理&quot;,&quot;可复现性基准&quot;,&quot;推荐公平性&quot;]"><strong><a href="graph_rec/knowledge_is_power_understanding_is_impact_utility_and_beyond_goals_explana/">Knowledge is Power, Understanding is Impact: Utility and Beyond Goals, Explanation Quality, and Fairness in Path Reasoning Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>复现 PGPR/CAFE/UCPR 三篇路径推理推荐，用 16 指标统一协议揭示其以效用与覆盖度换取惊喜度、多样性与可解释保真度。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排/CTR&quot;,&quot;多域推荐&quot;,&quot;图神经网络&quot;,&quot;域感知消息传递&quot;]"><strong><a href="cross_domain_multi_scenario/exploiting_graph_structured_cross_domain_representation_for_multi_domain_re/">Exploiting Graph Structured Cross-Domain Representation for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用域感知图卷积与全局图池化建模多域交互图，CTR 负采样评测在 Amazon 2–13 域稳定超越 MDR/MTL 基线。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;市场嵌入 (Market Embedding)&quot;,&quot;跨市场推荐 CMR&quot;,&quot;元学习替代方案&quot;]"><strong><a href="cross_domain_multi_scenario/market_aware_models_for_efficient_cross_market_recommendation/">Market-Aware Models for Efficient Cross-Market Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用市场嵌入 o_l 逐元素调制物品表征替代元学习，在 XMarket 上以 15% 训练时间取得或超越 MAML/FOREC 的 nDCG@10，全局建模下 6/7 市场夺冠。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;在线服务/召回&quot;,&quot;API-centric 平台&quot;,&quot;多域推荐系统&quot;]"><strong><a href="cross_domain_multi_scenario/uptrendz_api_centric_real_time_recommendations_in_multi_domain_settings/">Uptrendz: API-Centric Real-time Recommendations in Multi-Domain Settings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 API 为中心、schema 驱动的实时多域推荐平台，用系统级/物品级双层域划分与场景化算法配置实现一套服务支撑多业务，MovieLens-100k 与创业辅导两域实证可部署性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Filter Bubble&quot;,&quot;动态仿真&quot;,&quot;校准推荐&quot;]"><strong><a href="debias_fairness_industrial/evolution_of_filter_bubbles_and_polarization_in_news_recommendation/">Evolution of Filter Bubbles and Polarization in News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 4 万篇立场标注新闻与 500 个 Pew 仿真用户构建长程推荐闭环，揭示即使无说服效应，MF 反馈回路也会自发极化，且 SOTA 校准干预仅减缓不能阻断。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;差分隐私&quot;,&quot;安全多方计算&quot;]"><strong><a href="debias_fairness_industrial/privacy_preserving_fair_item_ranking/">Privacy-Preserving Fair Item Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次将EOAA公平重排与MPC+DP结合，在三个真实数据集上以NDCG≥0.8降低物品不公平。</p>
</dd>
</dl>
</section>
</div>
