---
hide:
  - toc
---

# 🏛️ ICDM 2025 推荐系统论文解读

11 篇 ICDM 2025 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 生成式推荐 ×3 · 大语言模型推荐 ×2

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Agentic Copilot" aria-pressed="false">Agentic Copilot<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GDR 设计表征" aria-pressed="false">GDR 设计表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存" aria-pressed="false">KV缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文压缩" aria-pressed="false">上下文压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文预算" aria-pressed="false">上下文预算<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化专家" aria-pressed="false">个性化专家<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主动学习人格分类" aria-pressed="false">主动学习人格分类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="倒置强化学习" aria-pressed="false">倒置强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全空间 CVR 建模" aria-pressed="false">全空间 CVR 建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层神经架构搜索" aria-pressed="false">双层神经架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实推断" aria-pressed="false">反事实推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释偏好摘要" aria-pressed="false">可解释偏好摘要<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推理链" aria-pressed="false">可解释推理链<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图实体描述预训练" aria-pressed="false">图实体描述预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态提示生成" aria-pressed="false">多模态提示生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗域适应" aria-pressed="false">对抗域适应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小语言模型蒸馏" aria-pressed="false">小语言模型蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐遗忘" aria-pressed="false">推荐遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序用户画像" aria-pressed="false">时序用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模仿学习" aria-pressed="false">模仿学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力融合" aria-pressed="false">注意力融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流水线归因" aria-pressed="false">流水线归因<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱增强LLM" aria-pressed="false">知识图谱增强LLM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线 RL" aria-pressed="false">离线 RL<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构因果模型" aria-pressed="false">结构因果模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征解耦" aria-pressed="false">表征解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域推荐" aria-pressed="false">跨域推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆强化学习" aria-pressed="false">逆强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="遗忘偏置 (Unlearning Bias)" aria-pressed="false">遗忘偏置 (Unlearning Bias)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本 LLM 解释生成" aria-pressed="false">零样本 LLM 解释生成<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;上下文压缩&quot;,&quot;个性化专家&quot;,&quot;KV缓存&quot;]"><strong><a href="sequential_rec/efficient_sequential_recommendation_for_long_term_user_interest_via_persona/">Efficient Sequential Recommendation for Long Term User Interest Via Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习 token 将长历史分段压缩为个性化专家，配合定制注意力掩码与 KV 缓存，在 HSTU/HLLM 上以约 1/4 推理开销逼近全序列精度。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;时序用户画像&quot;,&quot;注意力融合&quot;]"><strong><a href="llm_rec/effectiveness_of_llms_in_temporal_user_profiling_for_recommendation/">Effectiveness of LLMs in Temporal User Profiling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 分别生成短期/长期自然语言画像，经 SBERT 编码与注意力融合做内容推荐，Movies&amp;TV 上 Recall@10 相对提升 17%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;小语言模型蒸馏&quot;,&quot;倒置强化学习&quot;,&quot;多模态提示生成&quot;]"><strong><a href="llm_rec/scaling_multimodal_search_and_recommendation_with_small_language_models_via/">Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Llama-3 合成 5200 万意图-提示对蒸馏进 104M nanoGPT，并以倒置强化学习实现长度/模态可控的多任务提示生成，性能逼近 8B 模型而显存仅 500MB。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;精排&quot;,&quot;可解释推荐&quot;,&quot;双层神经架构搜索&quot;,&quot;零样本 LLM 解释生成&quot;]"><strong><a href="agent_conversational_rec/bi_nas_towards_effective_and_personalized_explanation_for_recommender_syste/">Bi-NAS: Towards Effective and Personalized Explanation for Recommender Systems via Bi-Level Neural Architecture Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双层 NAS 自动搜索交叉注意力与特征交互算子，再以零样本 LLM 提示生成个性化解释，四数据集上推荐与解释指标双优。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;Agentic Copilot&quot;,&quot;主动学习人格分类&quot;,&quot;可解释推理链&quot;]"><strong><a href="agent_conversational_rec/persopilot_an_adaptive_ai_copilot_for_transparent_contextualized_persona_cl/">PersoPilot: An Adaptive AI-Copilot for Transparent Contextualized Persona Classification and Personalized Response Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双模式智能体 Copilot：PersoAgent 做上下文人格抽取与社区推荐，Analytic Tools 用 Phi-4-mini+TF-IDF 主动学习闭环做可解释人格分类，UniEval 四维均分 0.84–1.0。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;知识图谱增强LLM&quot;,&quot;可解释偏好摘要&quot;,&quot;图实体描述预训练&quot;]"><strong><a href="agent_conversational_rec/reasoning_over_user_preferences_knowledge_graph_augmented_llms_for_explaina/">Reasoning over User Preferences: Knowledge Graph-Augmented LLMs for Explainable Conversational Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>COMPASS 用图实体描述预训练+知识感知指令微调，让 LLM 生成可解释文本偏好摘要并以门控注入现有 CRS，ReDial/INSPIRED 上 HR@10 最高提升 44.76%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;上下文预算&quot;,&quot;流水线归因&quot;,&quot;GDR 设计表征&quot;]"><strong><a href="multimodal_rec/fuse_failure_aware_usage_of_subagent_evidence_for_multimodal_search_and_rec/">FUSE: Failure-aware Usage of Subagent Evidence for MultiModal Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用紧凑 GDR 替代原始图像提示，系统对比 7 种上下文预算策略并以五阶段归因层定位失败，Context Compression 在意图/路由/召回/NDCG@5 全面最优且 p95 延迟降 45%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域推荐&quot;,&quot;表征解耦&quot;,&quot;对抗域适应&quot;]"><strong><a href="cross_domain_multi_scenario/adversarial_alignment_and_disentanglement_for_cross_domain_ctr_prediction_w/">Adversarial Alignment and Disentanglement for Cross-Domain CTR Prediction with Domain-Encompassing Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 A²DCDR，用 DC-MMD 对抗对齐 + CLUB 互信息解耦 + 特征重构，保留域不变与有价值非对齐特征，四组 Amazon 场景 HR@10 最高 +6.96，线上 CTR +7.18%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;反事实推断&quot;,&quot;结构因果模型&quot;,&quot;全空间 CVR 建模&quot;]"><strong><a href="ctr_feature_interaction/on_predicting_post_click_conversion_rate_via_counterfactual_inference/">On Predicting Post-Click Conversion Rate via Counterfactual Inference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 SCM 对未点击样本做 do(C=1) 反事实推断生成转化伪标签，全空间训练 CVR 模型，离线 AUC 平均 +1.01%，线上 CVR +17.35%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;逆强化学习&quot;,&quot;模仿学习&quot;,&quot;离线 RL&quot;]"><strong><a href="rl_bandit_rec/beyond_static_llm_policies_imitation_enhanced_reinforcement_learning_for_re/">Beyond Static LLM Policies: Imitation-Enhanced Reinforcement Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 轨迹作次优示范，经对抗逆强化学习提取奖励并加权模仿，离线 RL 策略在 Steam/Amazon 上累积奖励较 BiLLP 提升 13%+。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;推荐遗忘&quot;,&quot;RAG&quot;,&quot;遗忘偏置 (Unlearning Bias)&quot;]"><strong><a href="debias_fairness_industrial/customized_retrieval_augmented_generation_with_llm_for_debiasing_recommenda/">Customized Retrieval-Augmented Generation with LLM for Debiasing Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将推荐遗忘重构为 RAG 检索端过滤问题，用三种检索策略实现用户级原子遗忘，平均加速 4.5× 且保留约 90% 原模型效用。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
