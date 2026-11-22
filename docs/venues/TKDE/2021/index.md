---
hide:
  - toc
---

# 📖 IEEE TKDE 2021 推荐系统论文解读

5 篇 IEEE TKDE 2021 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDE ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;耦合张量分解&quot;,&quot;图正则SNMF&quot;,&quot;ADMM&quot;]"><strong><a href="collaborative_filtering/coupled_graphs_and_tensor_factorization_for_recommender_systems_and_communi/">Coupled Graphs and Tensor Factorization for Recommender Systems and Community Detection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 CGTF：PARAFAC 张量与对角缩放 SNMF 图耦合，ADMM 闭式补全缺失评分与图链接，40% 缺失下 NMSE 降 53.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样器设计&quot;,&quot;多行为反馈融合&quot;]"><strong><a href="collaborative_filtering/sampler_design_for_bayesian_personalized_ranking_by_leveraging_view_data/">Sampler Design for Bayesian Personalized Ranking by Leveraging View Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用电商浏览行为作购买与未观测之间的中间反馈，重构 BPR 负采样器与三元加权损失，在留一法全库排序下 HR 相对提升 35.6%/17.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图&quot;,&quot;元路径&quot;,&quot;注意力融合&quot;]"><strong><a href="graph_rec/deep_collaborative_filtering_with_multi_aspect_information_in_heterogeneous/">Deep Collaborative Filtering with Multi-Aspect Information in Heterogeneous Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元路径抽取方面级相似度，经 MLP 学习多视角隐因子并以注意力/自注意力融合，在 Leave-One-Out+99 负采样下 HR/NDCG 显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;级联预测&quot;]"><strong><a href="multi_task_reranking/learning_to_recommend_with_multiple_cascading_behaviors/">Learning to Recommend with Multiple Cascading Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>NMTR 以共享嵌入、分行为交互函数与级联预测建模多行为序关系，并在两电商数据集全库排序上显著超越单/多行为基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;评测基准/在线评估&quot;,&quot;Prequential Evaluation&quot;,&quot;ADWIN&quot;]"><strong><a href="debias_fairness_industrial/statistically_robust_evaluation_of_stream_based_recommender_systems/">Statistically Robust Evaluation of Stream-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出预序评估下的用户级 k 折与 ADWIN 自适应窗口，实现流式推荐在线成对统计检验；4 数据集上 Wilcoxon+bootstrap 兼顾 I/II 类错误。</p>
</dd>
</dl>
</section>
</div>
