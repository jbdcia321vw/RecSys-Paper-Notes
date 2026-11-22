---
hide:
  - toc
---

# 🏛️ DASFAA 2026 推荐系统论文解读

19 篇 DASFAA 2026 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 大语言模型推荐 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;层次表征学习&quot;,&quot;自监督学习&quot;,&quot;药物推荐&quot;]"><strong><a href="graph_rec/self_supervised_hierarchical_representation_for_medication_recommendation/">Self-supervised Hierarchical Representation for Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 ICD 编码树做父节点聚合关系嵌入与码位位置编码，即插即用替换 one-hot 嵌入层，四个药物推荐基线的 Jaccard 提升 0.8%~5.7%</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;符号图卷积&quot;,&quot;负偏好过滤&quot;]"><strong><a href="graph_rec/towards_unified_modeling_for_positive_and_negative_preferences_in_sign_awar/">Towards Unified Modeling for Positive and Negative Preferences in Sign-Aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>单一符号图编码器统一传播正负偏好，配合负偏好过滤与多任务训练，全库排序下 Recall@10 最高提升 16.21%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;触发诱导推荐&quot;,&quot;协同对比学习&quot;,&quot;跨场景泛化&quot;]"><strong><a href="self_supervised_denoising/beyond_the_trigger_learning_collaborative_context_for_generalizable_trigger/">Beyond the Trigger: Learning Collaborative Context for Generalizable Trigger-Induced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 (user, trigger) 为复合条件，用同页共点击/共不点击作正样本、单点击作负样本构建对比任务，学场景泛化协同表征；新场景 AUC +0.50pt，线上 CTR +12.3%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;生成式端到端&quot;,&quot;Flow Matching&quot;,&quot;行为先验&quot;,&quot;确定性ODE推理&quot;]"><strong><a href="sequential_rec/flowrec_prior_informed_flow_matching_for_efficient_sequential_recommendatio/">FlowRec: Prior-Informed Flow Matching for Efficient Sequential Recommendation Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用行为先验替代高斯先验、直线流匹配替代扩散去噪，配合单步对齐损失实现确定性 ODE 生成，Full-Ranking 下 HR@5 相对提升最高 9.87%，推理提速约 6.7 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式召回&quot;,&quot;曝光偏差&quot;,&quot;检索增强&quot;]"><strong><a href="generative_rec/genplugin_a_plug_and_play_framework_for_long_tail_generative_recommendation/">GENPLUGIN: A Plug-and-Play Framework for Long-Tail Generative Recommendation with Exposure Bias Mitigation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出双编码器-共享解码器插件，以语义替换缓解曝光偏差、检索增强改善长尾，5 个 Amazon 数据集 H@10 最高提升 65%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;偏好优化&quot;,&quot;动态负采样&quot;]"><strong><a href="llm_rec/dynamicpo_dynamic_preference_optimization_for_recommendation/">DynamicPO: Dynamic Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示多负样本 DPO 的偏好优化坍缩现象，提出动态边界负样本选择与双边际动态 β 调整，即插即用提升 LLM 推荐 HitRatio@1。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;语义对齐&quot;,&quot;协同知识注入&quot;]"><strong><a href="llm_rec/enhancing_llm_based_recommendation_through_semantic_aligned_collaborative_k/">Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过对比对齐与预热投影将协同知识注入 LLM，三阶段训练使 Movie/Book 上 AUC 相对提升 2.6%/4.8%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端序列决策策略&quot;,&quot;层次强化学习 HMDP&quot;,&quot;LoRA 策略微调&quot;]"><strong><a href="llm_rec/hierllm_hierarchical_large_language_model_for_question_recommendation/">HierLLM: Hierarchical Large Language Model for Question Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「概念→题目」双层 LLM 策略将超大题库动作空间逐级收缩，并以策略梯度直接优化学习收益，显著缓解教育推荐冷启动与长程决策难题。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;Chain-of-Thought&quot;,&quot;偏好对齐&quot;,&quot;S-DPO&quot;]"><strong><a href="llm_rec/trackrec_iterative_alternating_feedback_with_chain_of_thought_via_preferenc/">TrackRec: Iterative Alternating Feedback with Chain-of-Thought via Preference Alignment for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RecCoT 生成器与验证器交替 S-DPO/指令微调，把 CoT 偏好摘要接入 CTR 主干特征域，公开集与亿级广告平台线上双增。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;课程强化学习&quot;,&quot;解释一致性&quot;]"><strong><a href="agent_conversational_rec/curr_rlcer_curriculum_reinforcement_learning_for_coherence_explainable_reco/">Curr-RLCER: Curriculum Reinforcement Learning For Coherence Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用三阶段课程强化学习（CTR→评分→解释生成）训练 LLM，并设计一致性奖励与评估方案，使解释与评分双向对齐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2025&quot;,&quot;社交与群组推荐&quot;,&quot;排序 (Top-K 全库排序)&quot;,&quot;关系型知识蒸馏 (RKD)&quot;,&quot;准确率-多样性权衡&quot;]"><strong><a href="social_group_rec/leave_no_one_behind_enhancing_diversity_while_maintaining_accuracy_in_socia/">Leave No One Behind: Enhancing Diversity While Maintaining Accuracy in Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用关系型知识蒸馏把「非社会推荐」的高多样性结构知识迁移到社会推荐学生模型，在 5 个骨干 × 3 个数据集上 Coverage@100 提升 2.7%–68.5%，Recall 波动 &lt;4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;乘积量化(PQ)&quot;,&quot;嵌入表压缩&quot;,&quot;对比学习&quot;]"><strong><a href="ctr_feature_interaction/a_universal_framework_for_compressing_embeddings_in_ctr_prediction/">A Universal Framework for Compressing Embeddings in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>MEC 用两阶段乘积量化压缩嵌入表，以流行度加权正则平衡码本分配、对比学习拉平码分布，在 Criteo/Avazu/华为工业集上压缩 47–450× 内存且 AUC 不降反升。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM 特征编码&quot;,&quot;自适应特征交互&quot;,&quot;多域知识蒸馏&quot;]"><strong><a href="ctr_feature_interaction/ufin_universal_feature_interaction_network_for_multi_domain_click_through_r/">UFIN: Universal Feature Interaction Network for Multi-Domain Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将特征文本化经 LLM 编码-解码为通用特征，再用 MoE+EulerNet 学习自适应交互阶数并多域蒸馏，跨平台 AUC 提升 0.1%–1.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;半自回归生成&quot;,&quot;在线知识蒸馏&quot;,&quot;个性化位置编码&quot;]"><strong><a href="multi_task_reranking/efficient_personalized_reranking_with_semi_autoregressive_generation_and_on/">Efficient Personalized Reranking with Semi-Autoregressive Generation and Online Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用半自回归分块生成做教师、在线蒸馏轻量打分网络做学生，并以用户画像门控与个性化位置编码深化用户-物品交互，重排精度领先且推理延迟减半。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;多任务学习与列表级重排&quot;,&quot;端到端生成式&quot;,&quot;任务级 LoRA-MoE&quot;,&quot;收敛感知采样调度&quot;]"><strong><a href="multi_task_reranking/generative_representational_learning_of_foundation_models_for_recommendatio/">Generative Representational Learning of Foundation Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建 3 表征+10 生成任务的统一推荐基座数据集，用任务级低秩专家 MoE、收敛感知采样调度与模型合并在 Mistral-7B 上统一多任务训练，13 任务平均排名最优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;分层强化学习&quot;,&quot;LLM 语义规划&quot;,&quot;过滤气泡缓解&quot;]"><strong><a href="rl_bandit_rec/llm_enhanced_reinforcement_learning_for_long_term_user_satisfaction_in_inte/">LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>LLM 做类别级语义规划、PPO 做物品级策略，KuaiSim 全会话仿真下 Tint/Rcum 显著超越 DNaIR 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端策略学习&quot;,&quot;RL exploration&quot;,&quot;Ranking alignment&quot;,&quot;Collaborative regularization&quot;]"><strong><a href="rl_bandit_rec/personalized_education_with_ranking_alignment_recommendation/">Personalized Education with Ranking Alignment Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 RAR，以学习目标对称差与推荐序列距离构造排序对齐损失，增强 RL 题目推荐探索效率；在 KSS 与四个 KT 模拟器上显著提升学习效果。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;安全审计(黑盒抽取+投毒攻击)&quot;,&quot;认知分布对齐&quot;,&quot;行为一致污染生成&quot;]"><strong><a href="debias_fairness_industrial/a_cognitive_distribution_and_behavior_consistent_framework_for_black_box_at/">A Cognitive Distribution and Behavior-Consistent Framework for Black-Box Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用位置衰减认知分布对齐蒸馏黑箱序列推荐模型，并融合协同先验与梯度双信号生成行为一致污染序列，黑箱攻击成功率0.93–0.98。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;去偏公平与工业系统&quot;,&quot;召回/精排&quot;,&quot;特征掩码&quot;,&quot;状态感知适配器&quot;,&quot;长尾冷启动&quot;]"><strong><a href="debias_fairness_industrial/adaf_2m_2_comprehensive_learning_and_responsive_leveraging_features_in_reco/">AdaF^2M^2: Comprehensive Learning and Responsive Leveraging Features in Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以特征掩码多前向训练与状态感知适配器缓解ID特征长尾依赖，线上活跃天数+0.212%、时长+0.442%</p>
</dd>
</dl>
</section>
</div>
