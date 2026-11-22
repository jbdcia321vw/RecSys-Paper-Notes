---
hide:
  - toc
---

# 📖 ACM TKDD 2021 推荐系统论文解读

2 篇 ACM TKDD 2021 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;快照集成&quot;,&quot;记忆网络&quot;]"><strong><a href="collaborative_filtering/neuse_a_neural_snapshot_ensemble_method_for_collaborative_filtering/">NeuSE: A Neural Snapshot Ensemble Method for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>复用训练轨迹中的中间快照，用三层记忆网络为每个 user-item 对自适应生成集成权重，配合软标签 KL 训练，在 7 类 CF 骨架上取得最高 15.9% 相对提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排/评分预测&quot;,&quot;元图&quot;,&quot;因子分解机&quot;,&quot;异构信息网络&quot;,&quot;辅助信息融合&quot;]"><strong><a href="graph_rec/side_information_fusion_for_recommender_systems_over_heterogeneous_informat/">Side Information Fusion for Recommender Systems over Heterogeneous Information Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出元图相似度与“MF+FM”/HAF 融合异构辅助信息，在 4 个评分预测数据集上 RMSE 显著优于元路径基线。</p>
</dd>
</dl>
</section>
</div>
