---
hide:
  - toc
---

# 🏛️ ACM MM 2024 推荐系统论文解读

4 篇 ACM MM 2024 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;动态参数生成&quot;,&quot;语义码本量化&quot;]"><strong><a href="sequential_rec/semantic_codebook_learning_for_dynamic_recommendation_models/">Semantic Codebook Learning for Dynamic Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用语义序列主干+裁剪物品分支双路生成动态参数，并以语义码本量化去噪压缩搜索空间；Leave-One-Out/1:99 采样排序下 NDCG@10 最高提升约 10.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;属性驱动解耦表征&quot;,&quot;跨模态对比学习&quot;]"><strong><a href="multimodal_rec/attribute_driven_disentangled_representation_learning_for_multimodal_recomm/">Attribute-driven Disentangled Representation Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 price/brand/category/popularity 显式绑定到表征 chunk，用模态内分类、模态间对比与属性值预测实现属性级解耦，Recall@20 提升 4.8%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回/排序&quot;,&quot;图扩散&quot;,&quot;跨模态对比&quot;]"><strong><a href="multimodal_rec/diffmm_multi_modal_diffusion_model_for_recommendation/">DiffMM: Multi-Modal Diffusion Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用模态感知图扩散生成用户-物品图，并以跨模态对比对齐多模态与协同信号；Amazon 全库排序 R@20 相对 MMSSL 提升约 3.9%–5.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;模态失衡&quot;,&quot;知识蒸馏&quot;,&quot;反事实推断&quot;]"><strong><a href="multimodal_rec/modality_balanced_learning_for_multimedia_recommendation/">Modality-Balanced Learning for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出反事实知识蒸馏 CKD，用单模态教师 + 通用/特定双层蒸馏 + 反事实因果效应重加权，解决多模态推荐中模态欠优化，插拔式提升 6 个骨干网络 Recall@20 平均 3.4%–45.6%。</p>
</dd>
</dl>
</section>
</div>
