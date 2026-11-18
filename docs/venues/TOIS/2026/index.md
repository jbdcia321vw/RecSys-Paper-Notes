---
hide:
  - toc
---

# 📖 ACM TOIS 2026 推荐系统论文解读

4 篇 ACM TOIS 2026 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×4

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 表征增强" aria-pressed="false">LLM 表征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布外泛化" aria-pressed="false">分布外泛化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级学习" aria-pressed="false">列表级学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果发现" aria-pressed="false">因果发现<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络推荐" aria-pressed="false">图神经网络推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;符号边重加权&quot;,&quot;LLM 表征增强&quot;,&quot;精度-多样性权衡&quot;]"><strong><a href="graph_rec/cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图对比学习&quot;,&quot;相似就诊检索&quot;]"><strong><a href="graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;因果发现&quot;,&quot;分布外泛化&quot;]"><strong><a href="cross_domain_multi_scenario/causal_invariant_cross_domain_out_of_distribution_recommendation/">Causal-Invariant Cross-Domain Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一建模跨域与单域双重分布偏移，用双层因果结构与 LLM 引导混杂因子发现实现跨域 OOD 推荐，HR@10 平均提升 6.34%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;流行度去偏&quot;,&quot;逆倾向加权&quot;,&quot;列表级学习&quot;]"><strong><a href="debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
</dl>
</section>
</div>
