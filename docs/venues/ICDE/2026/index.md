---
hide:
  - toc
---

# 🏛️ ICDE 2026 推荐系统论文解读

9 篇 ICDE 2026 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×2 · 时空轨迹与 POI 推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与 POI 推荐" aria-pressed="false">时空轨迹与 POI 推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;MoE 并行量化码本&quot;,&quot;推理路径自反思&quot;]"><strong><a href="generative_rec/reg4rec_reasoning_enhanced_generative_model_for_large_scale_recommendation_/">REG4Rec: Reasoning-Enhanced Generative Model for Large-Scale Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 并行量化码本构建无序多路径推理空间，配 GRPO 偏好对齐与自反思剪枝，在 Amazon 三域与阿里广告 5.7B 交互数据上 Full-Ranking R@5 相对提升 4.97%~16.59%，线上广告收入 +5.60%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;查询改写&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="llm_rec/relevance_matters_a_multi_task_and_multi_stage_large_language_model_approac/">Relevance Matters: A Multi-Task and Multi-Stage Large Language Model Approach for E-commerce Query Rewriting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将相关性打标任务注入电商查询改写，多任务 SFT + 查询级归一化 GRPO 融合奖励对齐，离线召回 67.94%、线上 UCVR +0.181%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;模态交换对比学习&quot;]"><strong><a href="multimodal_rec/cammsr_category_guided_attentive_mixture_of_experts_for_multimodal_sequenti/">CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用类别预测任务引导注意力 MoE 动态分配模态权重，并以模态交换对比学习对齐跨模态序列表征，Full-Ranking 下 NDCG@5 最高提升 15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;GRPO 行为对齐&quot;,&quot;自适应门控融合&quot;]"><strong><a href="multimodal_rec/gala_generative_aligned_learning_for_adaptive_multimodal_representation_in_/">GALA: Generative Aligned Learning for Adaptive Multimodal Representation in the Taobao Shangou Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三阶段流水线：行为三元组预训练 + GRPO 生成式行为对齐 + 自适应门控融合，离线 AUC +0.12/+0.20，线上订单量 +0.55%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;不确定性量化&quot;]"><strong><a href="cold_start_rec/user_adaptive_meta_learning_for_cold_start_medication_recommendation_with_u/">User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出两级元自适应（自自适应+同伴自适应）与不确定性过滤的 MetaDrug，在 MIMIC-III/AKI 冷启动患者上 PRAUC 提升约 2.5%-6.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;精排&quot;,&quot;持续学习&quot;,&quot;兴趣记忆检索&quot;,&quot;模型无关框架&quot;]"><strong><a href="poi_spatiotemporal_rec/efficient_model_agnostic_continual_learning_for_next_poi_recommendation/">Efficient Model-Agnostic Continual Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 GIRAM：以分布型兴趣记忆 + 上下文感知键编码 + cVAE 多键生成检索 + 一致性自适应融合，实现免重训的持续 Next-POI 推荐，三数据集 Acc@5 与 MRR 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;生成式端到端&quot;,&quot;quadkey 层次网格编码&quot;,&quot;可学习傅里叶位置编码&quot;,&quot;POI 转移嵌入对齐&quot;]"><strong><a href="poi_spatiotemporal_rec/geography_aware_large_language_models_for_next_poi_recommendation/">Geography-Aware Large Language Models for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 quadkey+可学习傅里叶位置编码(GCIM)与序列模型 POI 嵌入对齐(PAM)增强 Llama-2-7B，三数据集 Acc@1 相对提升 14.7%–16.7%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;触发式推荐&quot;,&quot;意图建模&quot;,&quot;成对损失&quot;]"><strong><a href="ctr_feature_interaction/cascading_relevance_driven_recommendation_network_for_ctr_prediction_in_tri/">Cascading Relevance-driven Recommendation Network for CTR Prediction in Trigger-Introduced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向触发式推荐(TIR)中 trigger-target 相关性建模不足，以门控交叉、级联注意力融合与类目辅助成对损失取得 AUC 相对提升 6.31%/7.18%，线上 CTR +3.87%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多模态融合&quot;,&quot;目标感知注意力&quot;,&quot;推理优化&quot;]"><strong><a href="ctr_feature_interaction/decoupled_multimodal_fusion_for_user_interest_modeling_in_click_through_rat/">Decoupled Multimodal Fusion for User Interest Modeling in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以解耦目标注意力融合ID与多模态相似度，兼顾细粒度交互与精排推理效率。</p>
</dd>
</dl>
</section>
</div>
