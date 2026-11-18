---
hide:
  - toc
---

# 📖 ACM TKDD 2026 推荐系统论文解读

2 篇 ACM TKDD 2026 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDD ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双流MLP对齐" aria-pressed="false">双流MLP对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现评测基准" aria-pressed="false">可复现评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;可复现评测基准&quot;]"><strong><a href="generative_rec/eval4direc_a_unified_and_systematic_evaluation_framework_for_diffusion_base/">Eval4DiRec: A Unified and Systematic Evaluation Framework for Diffusion-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个面向扩散推荐的开源统一评测框架，覆盖 5 场景 14 模型，标准化扩散推理配置并揭示噪声调度与步数的关键影响。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;双流MLP对齐&quot;]"><strong><a href="ctr_feature_interaction/dual_stream_mlp_is_all_you_need_for_ctr_prediction/">Dual-Stream MLP is All You Need for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用知识蒸馏把复杂显式交叉教师压进纯MLP双流学生，配合隐状态BN与预测对齐，三基准AUC全面SOTA且推理低延迟。</p>
</dd>
</dl>
</section>
</div>
