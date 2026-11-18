---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2026 推荐系统论文解读

4 篇 IJCAI (RecSys Track) 2026 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2026 ×2 · 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Draft-and-Verify" aria-pressed="false">Draft-and-Verify<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文老虎机" aria-pressed="false">上下文老虎机<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基础模型个性化" aria-pressed="false">基础模型个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性感知序列建模" aria-pressed="false">属性感知序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并行验证推理" aria-pressed="false">并行验证推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;降尺度卷积&quot;,&quot;属性感知序列建模&quot;]"><strong><a href="sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;生成式推荐&quot;,&quot;Draft-and-Verify&quot;,&quot;混合稀疏-稠密表征&quot;,&quot;并行验证推理&quot;]"><strong><a href="generative_rec/dav_gen_end_to_end_generative_retrieval_via_draft_and_verify/">DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成式检索重构为“起草-验证”双阶段单模型范式，以混合稀疏-稠密表征与复合损失统一召回与排序，工业搜索延迟降至70ms。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文老虎机&quot;,&quot;知识追踪&quot;,&quot;课程推荐&quot;]"><strong><a href="rl_bandit_rec/pace_a_personalized_adaptive_curriculum_engine_for_9_1_1_call_taker_trainin/">PACE: A Personalized Adaptive Curriculum Engine for 9-1-1 Call-taker Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用技能图信念传播+个体学习/遗忘动力学+Thompson Sampling 上下文老虎机为 9-1-1 调度员推荐训练场景，Z2H 缩短 19.50%、终局掌握度提升 10.95%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI-ECAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;联邦推荐&quot;,&quot;基础模型个性化&quot;,&quot;语义隐私边界&quot;]"><strong><a href="debias_fairness_industrial/a_survey_of_personalized_federated_foundation_models_for_privacy_preserving/">A Survey of Personalized Federated Foundation Models for Privacy-Preserving Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>综述提出“主动语义保人格”范式，用几何正交约束与语义感知聚合解耦全局知识与用户人格，并给出五类开放问题与未来路线。</p>
</dd>
</dl>
</section>
</div>
