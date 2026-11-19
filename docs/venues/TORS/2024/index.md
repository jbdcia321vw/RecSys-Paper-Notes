---
hide:
  - toc
---

# 📖 ACM TORS 2024 推荐系统论文解读

8 篇 ACM TORS 2024 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TORS ×7 · 去偏公平与工业架构 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;解耦表征学习&quot;,&quot;元网络&quot;]"><strong><a href="graph_rec/disentangled_cascaded_graph_convolution_networks_for_multi_behavior_recomme/">Disentangled Cascaded Graph Convolution Networks for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>解耦因子块+元网络个性化迁移+行为级注意力，在Beibei/Tmall全库排序平均提升7.07%/9.00%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;使用场景隐式提问&quot;,&quot;T5 文本生成&quot;,&quot;众包数据标注&quot;]"><strong><a href="agent_conversational_rec/generating_usage_related_questions_for_preference_elicitation_in_conversati/">Generating Usage-related Questions for Preference Elicitation in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出以“商品使用场景”替代属性槽位的隐式偏好引导问题生成任务，构建众包三步标注数据集，对比两种模板与两种 T5 端到端模型，NSQG 在 BLEU-4/METEOR 与人工成对评测上全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;AutoML&quot;,&quot;搜索空间抽象化&quot;,&quot;特征交互搜索&quot;]"><strong><a href="ctr_feature_interaction/a_comprehensive_survey_on_automated_machine_learning_for_recommendations/">A Comprehensive Survey on Automated Machine Learning for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个按 DRS 组件（特征选择/嵌入/交叉/训练）细粒度划分的 AutoML 综述，提出“单→多组件、详细→抽象搜索空间、梯度主导”的三轴演进路线，系统对比各方法搜索空间量级与复杂性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线探索与推荐策略&quot;,&quot;Incentive-Aware Bandit&quot;,&quot;Randomized Pooling&quot;,&quot;Ex-Post Fairness&quot;]"><strong><a href="rl_bandit_rec/incentive_aware_recommender_systems_in_two_sided_markets/">Incentive-Aware Recommender Systems in Two-Sided Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出激励兼容的随机池化 Bandit 推荐，已知/未知机会成本下分别用 ARP/MARP 兼顾探索、公平与次线性 regret。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/曝光多样性&quot;,&quot;纵向用户研究&quot;,&quot;曝光多样性&quot;,&quot;隐式联想测试&quot;]"><strong><a href="debias_fairness_industrial/assessing_the_impact_of_music_recommendation_diversity_on_listeners_a_longi/">Assessing the Impact of Music Recommendation Diversity on Listeners: A Longitudinal Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>12 周 110 人纵向对照实验，用 SC-IAT 隐式联想与 Guttman 开放性量表量化高/低多样性音乐推荐对听众态度的长期塑造效应。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;重排&quot;,&quot;价值敏感设计&quot;,&quot;多利益相关者优化&quot;]"><strong><a href="debias_fairness_industrial/building_human_values_into_recommender_systems_an_interdisciplinary_synthes/">Building Human Values into Recommender Systems: An Interdisciplinary Synthesis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>22位跨学科作者综合：价值→指标→排序干预的七步落地流程、三类度量数据源与多利益相关者权衡防护，属立场综述而非实验论文。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估&quot;,&quot;分布性评估&quot;,&quot;曝光公平性&quot;]"><strong><a href="debias_fairness_industrial/distributionally_informed_recommender_system_evaluation/">Distributionally-Informed Recommender System Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出从点估计均值转向分布性评估，区分样本/子群/利益相关者三类分布与五类不确定性来源，并在MovieLens案例中揭示均值掩盖的性别公平与曝光公平差异。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与工业架构&quot;,&quot;离线评估与模型验证&quot;,&quot;邻域切片评估&quot;,&quot;子群公平性&quot;,&quot;Welch t 检验&quot;]"><strong><a href="debias_fairness_industrial/group_validation_in_recommender_systems_framework_for_multi_layer_performan/">Group Validation in Recommender Systems: Framework for Multi-layer Performance Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 PCC-KNN 将评分数据切成可重叠邻域，以损失差+Welch 单侧 t 检验筛出关键邻域，暴露被全局均值掩盖的小用户群失效。</p>
</dd>
</dl>
</section>
</div>
