---
hide:
  - toc
---

# 📖 ACM TOIS 2022 推荐系统论文解读

6 篇 ACM TOIS 2022 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×6

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
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回与排序&quot;,&quot;反射式缺失值填补&quot;,&quot;绑定权重自编码器&quot;,&quot;负采样去偏&quot;]"><strong><a href="collaborative_filtering/collaborative_reflection_augmented_autoencoder_network_for_recommender_syst/">Collaborative Reflection-Augmented Autoencoder Network for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以反射门显式填补未交互评分并联合自编码器去噪，在四个基准改善RMSE与Top-K排序，且免负采样。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;特征级注意力&quot;,&quot;物品相似度建模&quot;]"><strong><a href="collaborative_filtering/feature_level_attentive_icf_for_recommendation/">Feature-level Attentive ICF for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为 ICF 历史物品生成 d 维特征级注意力向量替代标量权重，跨物品同维 softmax 隐式融合物品级注意力，6 数据集 HR/NDCG@10 稳定提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;位置编码&quot;,&quot;会话图神经网络&quot;]"><strong><a href="sequential_rec/exploiting_positional_information_for_session_based_recommendation/">Exploiting Positional Information for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双位置编码DPE/LDPE与位置感知门控图网络PosRec，全库排序下在Yoochoose/Diginetica稳定超越SOTA约0.4%–2.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图超边嵌入&quot;,&quot;偶发群组推荐&quot;,&quot;社交同质性&quot;]"><strong><a href="social_group_rec/hierarchical_hyperedge_embedding_based_representation_learning_for_group_re/">Hierarchical Hyperedge Embedding-based Representation Learning for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HyperGroup 用社交图 GNN 增强个体偏好、把群组建模为超图超边做归纳式嵌入，全库排序下 Yelp HR@5 较最强基线提升 268%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 (MAML)&quot;,&quot;序列推荐&quot;,&quot;图扩散表征&quot;]"><strong><a href="cold_start_rec/learning_to_learn_a_cold_start_sequential_recommender/">Learning to Learn a Cold-start Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>metaCSR 用扩散表征+自注意力序列编码+MAML 元学习，仅凭 ID 与交互矩阵实现新用户少样本快速适配，冷启动 AUC 最高 +7.85%、MAP +29.62%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;曝光公平&quot;,&quot;最大流&quot;,&quot;多边公平&quot;]"><strong><a href="debias_fairness_industrial/a_graph_based_approach_for_mitigating_multi_sided_exposure_bias_in_recommen/">A Graph-based Approach for Mitigating Multi-sided Exposure Bias in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>FairMatch 以二部图最大流重排联合提升物品/供应商曝光公平与聚合多样性，在 Last.fm/MovieLens 上保持可接受精度。</p>
</dd>
</dl>
</section>
</div>
