---
hide:
  - toc
---

# 📖 ACM TKDD 2025 推荐系统论文解读

2 篇 ACM TKDD 2025 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDD ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="IPS 去偏" aria-pressed="false">IPS 去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层聚类" aria-pressed="false">双层聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排两阶段" aria-pressed="false">召回+精排两阶段<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="技能感知表征" aria-pressed="false">技能感知表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据异质性" aria-pressed="false">数据异质性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="标签分布学习" aria-pressed="false">标签分布学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义增强Transformer" aria-pressed="false">语义增强Transformer<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回+精排两阶段&quot;,&quot;语义增强Transformer&quot;,&quot;技能感知表征&quot;,&quot;标签分布学习&quot;]"><strong><a href="ctr_feature_interaction/jobformer_skill_aware_job_recommendation_with_semantic_enhanced_transformer/">JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用局部-全局注意力 Transformer 聚合邻居 JD 补全语义，并以技能分布引导两阶段召回+CTR排序，Recall@20 较 NRMS 提升 2.31%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;数据异质性&quot;,&quot;双层聚类&quot;,&quot;IPS 去偏&quot;]"><strong><a href="debias_fairness_industrial/exploring_and_exploiting_data_heterogeneity_in_recommendation/">Exploring and Exploiting Data Heterogeneity in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出双层聚类 BHE 分层挖掘推荐数据预测机制与协变量分布异质性，用于多子模型泛化与 IPS 去偏，Yelp/MovieLens 上 NDCG 提升 20%+。</p>
</dd>
</dl>
</section>
</div>
