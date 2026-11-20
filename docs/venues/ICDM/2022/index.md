---
hide:
  - toc
---

# 🏛️ ICDM 2022 推荐系统论文解读

10 篇 ICDM 2022 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;概率嵌入 Beta Distribution&quot;,&quot;高阶互补推荐&quot;]"><strong><a href="graph_rec/towards_high_order_complementary_recommendation_via_logical_reasoning_netwo/">Towards High-Order Complementary Recommendation via Logical Reasoning Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Beta 概率嵌入与可微逻辑算子（投影/否定/交集）建模非对称互补关系，并以注意力聚合查询集合实现高阶互补推荐，Amazon 六类目 Hit@3 最高相对提升 155%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2022&quot;,&quot;序列推荐&quot;,&quot;召回/近邻检索&quot;,&quot;神经切线核&quot;,&quot;无限宽RNN&quot;]"><strong><a href="sequential_rec/beyond_double_ascent_via_recurrent_neural_tangent_kernel_in_sequential_reco/">Beyond Double Ascent via Recurrent Neural Tangent Kernel in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>观测模型宽度“双上升”，提出免训练 OverRec：以 RNTK 度量序列相似度并加权 kNN，逼近无限宽 RNN，在 Beauty/Toys 全库排序提升。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;动态子网络路由&quot;,&quot;权重切片超网&quot;,&quot;Gumbel-Softmax&quot;]"><strong><a href="sequential_rec/one_person_one_model_learning_compound_router_for_sequential_recommendation/">One Person, One Model—Learning Compound Router for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>CANet 用轻量路由器为每条用户序列在嵌入/隐藏/层深三维复合空间内路由子网络，权重切片共享超网，Gumbel-Softmax 端到端可微，全库排序下减算 55~65% 而 NDCG 仅降约 1.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;信息扩散级联&quot;,&quot;多视图注意力融合&quot;]"><strong><a href="social_group_rec/igniter_news_recommendation_in_microblogging_applications_extended_version/">IGNiteR: News Recommendation in Microblogging Applications (Extended Version)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用新闻转发级联训练影响力节点嵌入，并以语义/扩散/采纳曲线三视图注意力融合做微博客新闻推荐，采样排序下 NDCG@10 相对最强基线提升 2.3%–3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM WAIN 2022&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;团队组建&quot;,&quot;语义匹配&quot;,&quot;约束贪心&quot;]"><strong><a href="social_group_rec/ultra_a_data_driven_approach_for_recommending_team_formation_in_response_to/">ULTRA: A Data-driven Approach for Recommending Team Formation in Response to Proposal Calls</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>抽取 NSF 招标与学者画像做 SPECTER 语义匹配，按预算与技能互补约束贪心组队；70 条匹配中 65 条≥7 分，历史中标命中 54.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KAIS / ICDM 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;嵌入层（召回/精排通用）&quot;,&quot;Actor-Critic 强化学习&quot;,&quot;层级共享嵌入&quot;,&quot;流式推荐&quot;]"><strong><a href="cold_start_rec/autoassign_automatic_shared_embedding_assignment_in_streaming_recommendatio/">AutoAssign+: Automatic Shared Embedding Assignment in Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 actor-critic 智能体为流式推荐中的低频 ID 动态分配层级共享嵌入，AUC 最高 +5.93%，嵌入表参数降 20–31%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;时空轨迹与 POI 本地生活推荐&quot;,&quot;精排&quot;,&quot;图核网络&quot;,&quot;地理-序列一致性学习&quot;]"><strong><a href="poi_spatiotemporal_rec/kernel_based_substructure_exploration_for_next_poi_recommendation/">Kernel-based Substructure Exploration for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>地理消息传递图 + 随机游走图核序列图双模块，用 KL 一致性学习互蒸馏，AUC 较最优基线提升 3.3%/4.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户行为建模&quot;,&quot;字段对选择&quot;,&quot;注意力剪枝&quot;]"><strong><a href="ctr_feature_interaction/autoattention_automatic_field_pair_selection_for_attention_in_user_behavior/">AutoAttention: Automatic Field Pair Selection for Attention in User Behavior Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将全部 item/user/context 字段纳入注意力查询，学习字段对权重并剪枝自动选对，Alibaba/Tencent 用户加权 AUC 相对提升 2.19%/3.58%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;仿真环境/基准&quot;,&quot;Contextual Bandit&quot;,&quot;推荐数据集重参数化&quot;]"><strong><a href="rl_bandit_rec/simulated_contextual_bandits_for_personalization_tasks_from_recommendation_/">Simulated Contextual Bandits for Personalization Tasks from Recommendation Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将推荐数据集的用户反馈与内容特征重参数化为上下文赌博机状态/动作，用余弦奖励生成可训练个性化仿真环境，并在 MovieLens/IMDb 上验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;异构图神经网络&quot;,&quot;差分隐私&quot;,&quot;隐私预算双层优化&quot;]"><strong><a href="debias_fairness_industrial/heterogeneous_graph_neural_network_for_privacy_preserving_recommendation/">Heterogeneous Graph Neural Network for Privacy-Preserving Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HeteDP 用「语义感知特征加噪 + DP-SGD 拓扑重构 + 双层隐私预算分配」双阶段保护异构图推荐，LP 最高相对提升 37.2%，但节点分类在强隐私下明显掉点。</p>
</dd>
</dl>
</section>
</div>
