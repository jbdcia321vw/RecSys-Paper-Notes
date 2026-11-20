---
hide:
  - toc
---

# 📖 ACM TORS 2023 推荐系统论文解读

5 篇 ACM TORS 2023 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TORS ×3 · TORS 2023 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;单位一致性&quot;,&quot;张量补全&quot;]"><strong><a href="collaborative_filtering/tensor_completion_with_provable_consistency_and_fairness_guarantees_for_rec/">Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以单位一致性约束替代秩/范数优化，实现可证明唯一、共识排序与尺度公平的矩阵/张量补全；MovieLens-1M 扰动测试零违例。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;全链路(召回/排序/重排)&quot;,&quot;高阶连通性&quot;,&quot;综述与分类体系&quot;]"><strong><a href="graph_rec/a_survey_of_graph_neural_networks_for_recommender_systems_challenges_method/">A Survey of Graph Neural Networks for Recommender Systems: Challenges, Methods, and Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出阶段/场景/目标/应用四维分类法，围绕图构建、传播聚合、优化与效率四大挑战系统梳理 GNN 推荐，并开源论文-代码索引库</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;输出扰动&quot;,&quot;无负采样&quot;]"><strong><a href="self_supervised_denoising/selfcf_a_simple_framework_for_self_supervised_collaborative_filtering/">SelfCF: A Simple Framework for Self-supervised Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TORS 2023 自监督去噪：输出扰动替代负采样，四数据集全库排序下较 BUIR 平均提升 17.79%，训练快 2–4×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;图结构学习&quot;,&quot;异质图神经网络&quot;]"><strong><a href="social_group_rec/graph_learning_augmented_heterogeneous_graph_neural_network_for_social_reco/">Graph Learning Augmented Heterogeneous Graph Neural Network for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Graph Learner 在 u2u/u2i/i2i 三路异质全局图上学习并重构图结构，HGNN 捕获跨语义高阶信息，Ciao-5 RMSE 相对最优基线降 4.68%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;前门调整&quot;,&quot;未观测混杂&quot;]"><strong><a href="debias_fairness_industrial/deconfounded_causal_collaborative_filtering/">Deconfounded Causal Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以客观物品特征为前门中介，统一去除未观测混杂，并在 Amazon/Yelp 的 SKEW/RAND 采样排序上显著提升 nDCG@5。</p>
</dd>
</dl>
</section>
</div>
