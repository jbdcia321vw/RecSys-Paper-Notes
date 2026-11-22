---
hide:
  - toc
---

# 🏛️ ACM MM 2021 推荐系统论文解读

5 篇 ACM MM 2021 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2021&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图结构学习&quot;,&quot;模态自适应图融合&quot;]"><strong><a href="multimodal_rec/mining_latent_structures_for_multimedia_recommendation/">Mining Latent Structures for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用模态感知图结构学习从多模态特征中挖掘物品-物品潜在语义图，经GCN注入ID嵌入，Clothing Recall@20 相对 SOTA 提升 12.5%</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回表征预训练&quot;,&quot;图预训练&quot;,&quot;多模态图 Transformer&quot;,&quot;掩码特征重构&quot;]"><strong><a href="multimodal_rec/pre_training_graph_transformer_with_multimodal_side_information_for_recomme/">Pre-training Graph Transformer with Multimodal Side Information for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用共购行为构建物品多模态图，以 MCNSampling 采样上下文邻居、多样性促进 Transformer 做结构与掩码特征双目标预训练，Recall@10 最高提升 15.4%，线上 60 万用户新播放 +6.80%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2021&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;视觉印象建模&quot;,&quot;记忆网络融合&quot;]"><strong><a href="multimodal_rec/why_do_we_click_visual_impression_aware_news_recommendation/">Why Do We Click: Visual Impression-aware News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以新闻卡片渲染快照为“视觉印象”，局部记忆注意力+全局门控融合，在 IM-MIND 上 AUC/NDCG@5 相对提升约 1.0%/3.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;互信息最大化&quot;]"><strong><a href="cold_start_rec/contrastive_learning_for_cold_start_recommendation/">Contrastive Learning for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将冷启动物表征学习化为 U-I 与 R-E 互信息下界最大化，双对比网络在全库排序下冷启动 R@10 最高提升 103.95%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;视觉去偏&quot;]"><strong><a href="debias_fairness_industrial/causalrec_causal_inference_for_visual_debiasing_in_visually_aware_recommend/">CausalRec: Causal Inference for Visual Debiasing in Visually-Aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用因果图中介干预与TIE反事实推断剔除视觉偏置，全库排序在8个Amazon数据集上NDCG最高提升53.6%。</p>
</dd>
</dl>
</section>
</div>
