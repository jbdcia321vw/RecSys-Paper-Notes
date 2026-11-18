# 🧠 大语言模型推荐 (LLM4Rec)

**CIKM 2026** · **9** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [CIKM 2025 (6)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×8 · 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO" aria-pressed="false">GRPO<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习精炼" aria-pressed="false">GRPO 强化学习精炼<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Procrustes 投影" aria-pressed="false">Procrustes 投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Recall Ceiling" aria-pressed="false">Recall Ceiling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SPC 三难困境" aria-pressed="false">SPC 三难困境<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文感知特征选择" aria-pressed="false">上下文感知特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化稠密检索" aria-pressed="false">个性化稠密检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推荐" aria-pressed="false">免训练推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同信号注入" aria-pressed="false">协同信号注入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同视角内容理解" aria-pressed="false">协同视角内容理解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图传播表征细化" aria-pressed="false">图传播表征细化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图推理" aria-pressed="false">意图推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续推荐" aria-pressed="false">持续推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理增强" aria-pressed="false">推理增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品画像" aria-pressed="false">物品画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目录忠实性幻觉" aria-pressed="false">目录忠实性幻觉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="置信度校准" aria-pressed="false">置信度校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自蒸馏" aria-pressed="false">自蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评估协议" aria-pressed="false">评估协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID生成" aria-pressed="false">语义ID生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐步推理损失" aria-pressed="false">逐步推理损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="闭候选重排" aria-pressed="false">闭候选重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式推理" aria-pressed="false">隐式推理<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;置信度校准&quot;,&quot;目录忠实性幻觉&quot;]"><strong><a href="do_llm_recommenders_know_when_they_re_hallucinating_auditing_confidence_cal/">Do LLM Recommenders Know When They&#x27;re Hallucinating? Auditing Confidence Calibration in Catalog Faithfulness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>联合审计零样本 LLM 推荐的目录幻觉率与言语化置信度校准，揭示校准误差方向由目录而非模型决定，并给出匹配器验证协议。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;隐式推理&quot;,&quot;逐步推理损失&quot;,&quot;语义ID生成&quot;]"><strong><a href="do_recommenders_need_to_think_in_words_latent_reasoning_for_efficient_llm_b/">Do Recommenders Need to Think in Words? Latent Reasoning for Efficient LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用训练期逐步头监督替代推理期 CoT/隐式推理，HR@10 相对提升 4.0–5.0% 且延迟降低 18.3×</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;物品画像&quot;,&quot;上下文感知特征选择&quot;]"><strong><a href="profiling_what_matters_context_aware_item_profiles_from_large_scale_metadat/">Profiling What Matters: Context-Aware Item Profiles from Large-Scale Metadata for LLM Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将海量异构物品元数据结构化并训练轻量选择器，为每个用户-物品对动态生成上下文相关画像，显著提升 LLM 重排效果。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;自蒸馏&quot;,&quot;推理增强&quot;]"><strong><a href="selfdr_self_distillation_from_reasoning_for_llm_based_recommendation/">SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用下游推荐奖励训练同源 Reasoner 生成理由，构建推理增强教师，再自蒸馏出免推理的直接推荐学生，重排 HR@1 提升约 15%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;Recall Ceiling&quot;,&quot;评估协议&quot;,&quot;闭候选重排&quot;]"><strong><a href="the_recall_ceiling_of_llm_recommendation_reranking/">The Recall Ceiling of LLM Recommendation Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>证明闭候选重排器 NDCG 受自身窗口召回率硬上界约束，Oracle 协议高估 92–95%，九类 LLM 与监督重排器在真实检索下均无法显著超越 CF。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;LLM推荐&quot;,&quot;召回&quot;,&quot;意图推理&quot;,&quot;个性化稠密检索&quot;,&quot;GRPO&quot;]"><strong><a href="think_to_personalize_unifying_reasoning_and_retrieval_for_user_centric_pers/">Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 SFT+GRPO 两阶段让 LLM 显式推理用户历史生成意图增强查询并统一编码，Full-Ranking 下 Broad/LongTail 召回显著提升，线上订单量 +0.46%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同视角内容理解&quot;,&quot;GRPO 强化学习精炼&quot;]"><strong><a href="towards_comprehensible_recommendation_with_large_language_model_fine_tuning/">Towards Comprehensible Recommendation with Large Language Model Fine-tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 GRPO 两阶段对齐 LLM 生成协同视角的用户模式与推荐理由，并以推荐模型打分为奖励精炼，Full-Ranking 下 Recall@5 相对提升 14–18%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;持续推荐&quot;,&quot;SPC 三难困境&quot;,&quot;Procrustes 投影&quot;]"><strong><a href="tracer_balancing_stability_plasticity_cognitivity_trilemma_for_llm_enhanced/">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 LLM 增强持续推荐中的 SPC 三难困境，用 Procrustes 投影、置信门控凝聚与语义同步三模块协同破局，Full-Ranking 下 ACC 最高提升 14.38%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;免训练推荐&quot;,&quot;协同信号注入&quot;,&quot;图传播表征细化&quot;]"><strong><a href="training_free_llm_based_recommendation_with_post_llm_item_refinement_using_/">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 post-LLM 范式 CoRRe，用共购图方向细化与流行度幅度校准改造 LLM 物品表征，免训练即在三域 Amazon 上超越全部免训练基线并多数超越训练式方法。</p>
</dd>
</dl>
