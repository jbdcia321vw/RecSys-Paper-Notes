---
hide:
  - toc
---

# 📖 IEEE TMM 2023 推荐系统论文解读

3 篇 IEEE TMM 2023 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TMM ×3 · 多模态推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;多级图神经网络&quot;,&quot;超图&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/contrastive_multi_level_graph_neural_networks_for_session_based_recommendat/">Contrastive Multi-Level Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 L/G/H 三级图与跨层级 InfoNCE，将高阶超图监督注入成对会话表示，在 Tmall/Diginetica/Nowplaying 全库排序上多数指标领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;解耦表征&quot;,&quot;多模态注意力&quot;]"><strong><a href="multimodal_rec/disentangled_multimodal_representation_learning_for_recommendation/">Disentangled Multimodal Representation Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用距离相关正则解耦 ID/文本/视觉特征的 K 个因子，再以权重共享的多模态注意力建模用户对各因子模态偏好，Amazon 五域全库排序 Recall@20 最高提升 15.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跨模态检索&quot;,&quot;片段时序对齐&quot;,&quot;动态时间规整&quot;]"><strong><a href="multimodal_rec/video_to_music_recommendation_using_temporal_alignment_of_segments/">Video-to-Music Recommendation using Temporal Alignment of Segments</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用片段级自监督嵌入与序列对齐成本实现视频到音乐推荐，在 1,000 曲全库检索中 Mean Rank 由 118 降至 32。</p>
</dd>
</dl>
</section>
</div>
