---
hide:
  - toc
---

# 🏛️ ACM MM 2022 推荐系统论文解读

3 篇 ACM MM 2022 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;人格画像&quot;,&quot;多视角用户表征&quot;]"><strong><a href="multimodal_rec/personality_driven_social_multimedia_content_recommendation/">Personality-Driven Social Multimedia Content Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 MBTI 人格向量 + 多视角时间线特征构建用户表征，BPR 训练的多模态内容召回器，8:2 划分下 AUC 0.905、nDCG@10 提升 12%，线上降本 4.2 倍</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;动态图神经网络&quot;,&quot;混合度量学习&quot;]"><strong><a href="cross_domain_multi_scenario/ddghm_dual_dynamic_graph_with_hybrid_metric_training_for_cross_domain_seque/">DDGHM: Dual Dynamic Graph with Hybrid Metric Training for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双层动态图分别刻画域内/域间序列转移，用融合注意力门迁移知识，并以协同+对比双度量学习缓解稀疏，全库排序 HR@20 提升 4%~9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差&quot;,&quot;对抗学习&quot;]"><strong><a href="debias_fairness_industrial/dvr_micro_video_recommendation_optimizing_watch_time_gain_under_duration_bi/">DVR: Micro-Video Recommendation Optimizing Watch-Time-Gain under Duration Bias</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出去时长偏差的无偏指标 WTG/DCWTG 与模型无关的对抗去偏框架 DVR，在两个真实微视频数据集上 WTG@10 相对提升逾 300%。</p>
</dd>
</dl>
</section>
</div>
