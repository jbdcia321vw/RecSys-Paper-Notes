---
hide:
  - toc
---

# 🏛️ ECIR 2021 推荐系统论文解读

6 篇 ECIR 2021 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;评论建模&quot;,&quot;零注意力机制&quot;,&quot;相关性匹配&quot;]"><strong><a href="collaborative_filtering/a_zero_attentive_relevance_matching_networkfor_review_modeling_in_recommend/">A Zero Attentive Relevance Matching Network for Review Modeling in Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用相关性匹配替代语义共注意力，并以零注意力与真实评论辅助损失动态建模评论，在四个 Amazon 5-core 数据集上使 MSE 相对降低 0.8%–1.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;深度度量学习&quot;,&quot;Triplet Loss&quot;,&quot;会话编码器&quot;]"><strong><a href="sequential_rec/metric_learning_for_session_based_recommendations/">Metric Learning for Session-based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将会话推荐重构为共享嵌入空间的度量学习，用位置加权 Triplet/NCAS 损失与轻量编码器实现 SOTA，并给出与 pairwise LTR 的同构性分析与完整消融。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;聊天式用户建模&quot;,&quot;实体扩展&quot;,&quot;域特异词权重&quot;]"><strong><a href="agent_conversational_rec/you_get_what_you_chat_using_conversations_to_personalize_search_based_recom/">You Get What You Chat: Using Conversations to Personalize Search-based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首次以用户间聊天构建零成本用户模型，用 KL 混合与查询扩展重排实体搜索结果，NDCG@20 提升 2–4 点。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排/排序&quot;,&quot;因果推断&quot;,&quot;匹配估计器&quot;]"><strong><a href="debias_fairness_industrial/causality_aware_neighborhood_methods_for_recommender_systems/">Causality-Aware Neighborhood Methods for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用匹配估计器统一邻域推荐，免倾向性估计因果效应并稳健排序；半合成数据全库排序上 CP@10 相对最优基线提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦学习&quot;,&quot;BPR 成对排序&quot;,&quot;流行度偏置&quot;]"><strong><a href="debias_fairness_industrial/federank_user_controlled_feedback_with_federated_recommender_systems/">FedeRank: User Controlled Feedback with Federated Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个联邦 pairwise 排序推荐框架，用正反馈掩码让用户自主控制披露比例 π，在 Amazon/LibraryThing 上以 10% 披露反超集中式 BPR-MF 约 50%</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（评分预测）&quot;,&quot;元学习超网络&quot;,&quot;联邦隐私预算&quot;,&quot;可复现性研究&quot;]"><strong><a href="debias_fairness_industrial/robustness_of_meta_matrix_factorization_against_strict_privacy_constraints/">Robustness of Meta Matrix Factorization Against Strict Privacy Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>复现 MetaMF 并证明元学习是其抵御严格隐私预算的关键，提出 ΔMAE@β 度量，揭示高活跃用户反而对隐私收紧最敏感。</p>
</dd>
</dl>
</section>
</div>
