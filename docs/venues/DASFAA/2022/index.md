---
hide:
  - toc
---

# 🏛️ DASFAA 2022 推荐系统论文解读

3 篇 DASFAA 2022 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2022&quot;,&quot;自监督对比与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;多视图对比学习&quot;,&quot;行为区分对比学习&quot;]"><strong><a href="self_supervised_denoising/multi_view_multi_behavior_contrastive_learning_in_recommendation/">Multi-view Multi-behavior Contrastive Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用行为级、视图级、行为优先级三类对比任务统一多行为多视图推荐，Tmall/CIKM 采样排序下 NDCG@10 相对提升 7.4%/6.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;类别感知会话图&quot;,&quot;目标类别意图迁移&quot;]"><strong><a href="sequential_rec/intention_adaptive_graph_neural_network_for_category_aware_session_based_re/">Intention Adaptive Graph Neural Network for Category-aware Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出类别感知会话推荐新任务 CSBR，构建含目标类别节点的异构会话图，用意图自适应 GAT 将与目标类别相关的历史交互迁移到目标类别域，P@20 最高提升 6.30%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;社交与群组推荐&quot;,&quot;精排/排序&quot;,&quot;图注意力&quot;,&quot;动态静态表示&quot;,&quot;物品相关图&quot;]"><strong><a href="social_group_rec/graph_neural_networks_with_dynamic_and_static_representations_for_social_re/">Graph Neural Networks with Dynamic and Static Representations for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>GNN-DSR用LSTM与图注意力建模用户/物品的动态-静态表示，并聚合社交图与物品相关图，在Ciao/Epinions/Delicious上提升评分预测与物品排序。</p>
</dd>
</dl>
</section>
</div>
