---
hide:
  - toc
---

# 📖 IEEE TMM 2025 推荐系统论文解读

2 篇 IEEE TMM 2025 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TMM ×2 · 对比学习 ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="引导扩散模型" aria-pressed="false">引导扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="本地差分隐私原型" aria-pressed="false">本地差分隐私原型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;引导扩散模型&quot;,&quot;对比学习&quot;]"><strong><a href="multimodal_rec/knowledge_aware_diffusion_enhanced_multimedia_recommendation/">Knowledge-aware Diffusion-Enhanced Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用重启随机游走 Jaccard 注意力矩阵替代可学习注意力，并以用户嵌入引导扩散模型重建低噪知识图谱构建对比视图，三数据集 Recall@20 全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;联邦学习&quot;,&quot;本地差分隐私原型&quot;,&quot;对比学习&quot;]"><strong><a href="cross_domain_multi_scenario/federated_user_preference_modeling_for_privacy_preserving_cross_domain_reco/">Federated User Preference Modeling for Privacy-Preserving Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用评论+潜在正物品挖掘综合偏好，再以 LDP 原型经联邦聚合跨域迁移，四组 CDR 任务 HR@10 平均提升约 13%。</p>
</dd>
</dl>
</section>
</div>
