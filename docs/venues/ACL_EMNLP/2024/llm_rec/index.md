---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**ACL / EMNLP (RecSys & CRS) 2024** · **9** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../2022/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2024 (1)](../../../ICDM/2024/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2022 (1)](../../../KDD/2022/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/llm_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×5 · 端到端生成式 ×4 · ACL ×3 · ACL 2024 ×3 · LLM4Rec ×2 · 指令微调 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指令微调" aria-pressed="false">指令微调<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;LLM4Rec&quot;,&quot;端到端生成式&quot;,&quot;RLHF/PPO 对齐&quot;,&quot;可控推荐&quot;]"><strong><a href="aligning_large_language_models_for_controllable_recommendations/">Aligning Large Language Models for Controllable Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>两阶段 SL+PPO 对齐 Llama2-7b，以 SASRec 标签增强与规则奖励实现类目/比例可控推荐；Leave-One-Out 生成式 top-k，TCP 95.8%、CPA 88.4%。</p>
</dd>
<dt data-paper-tags="[&quot;NAACL 2024 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;指令微调&quot;,&quot;掩码物品建模&quot;,&quot;贝叶斯个性化排序&quot;]"><strong><a href="aligning_large_language_models_with_recommendation_knowledge/">Aligning Large Language Models with Recommendation Knowledge</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 MIM/MLM/BPR 经典算子改写为自然语言辅助任务微调 LLM，配合去 user-ID 的精简推荐提示，在 Amazon 三域全库检索上超越 TIGER 达 19%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;高阶交互&quot;,&quot;全词嵌入&quot;]"><strong><a href="enhancing_high_order_interaction_awareness_in_llm_based_recommender_model/">Enhancing High-order Interaction Awareness in LLM-based Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用LightGCN随机特征传播生成图感知全词嵌入并注入T5，配合免训练重排；Leave-One-Out下直接与序列推荐均显著领先。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;召回/打分&quot;,&quot;提示工程&quot;,&quot;文本增强&quot;]"><strong><a href="llm_rec_personalized_recommendation_via_prompting_large_language_models/">LLM-Rec: Personalized Recommendation via Prompting Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用四类提示让LLM增强物品文本，拼接原始描述嵌入并以轻量MLP打分；在MovieLens-1M/Recipe的1:1000采样排序上NDCG@10相对MLP提升8.54%/21.72%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;理由蒸馏&quot;,&quot;提示蒸馏&quot;]"><strong><a href="rdrec_rationale_distillation_for_llm_based_recommendation/">RDRec: Rationale Distillation for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Llama-2-7b + CoT 从评论蒸馏「用户偏好/物品属性」双句理由，作为第四个生成任务注入 T5-small 的 POD 提示蒸馏框架；Leave-One-Out 生成式全库排序下 Top-N 提升 12–42%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;领域持续预训练&quot;,&quot;指令微调&quot;,&quot;文本化物品表征&quot;]"><strong><a href="recgpt_generative_pre_training_for_text_based_recommendation/">RecGPT: Generative Pre-training for Text-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个领域自适应全量预训练的 7B 推荐 LLM（20.5B token 推荐语料持续预训练 + 10 万指令微调），以语义相似度对齐解码结果，在 4 个基准上刷新评分预测与序列推荐 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;LLM4Rec&quot;,&quot;后验解释生成&quot;,&quot;可解释推荐&quot;,&quot;混合专家适配器&quot;,&quot;复现研究&quot;]"><strong><a href="reproducibility_study_of_xrec_large_language_models_for_explainable_recomme/">Reproducibility Study of \&quot;XRec: Large Language Models for Explainable Recommendation\&quot;</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 Llama-3 复现 XRec 并做 MoE 反事实消融，证实协同注入提分，但指出其并非全面超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;LLM推荐&quot;,&quot;评分预测/精排&quot;,&quot;自然语言用户画像&quot;,&quot;可审视推荐&quot;]"><strong><a href="transparent_and_scrutable_recommendations_using_natural_language_user_profi/">Transparent and Scrutable Recommendations Using Natural Language User Profiles</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>LLM从评论生成可编辑NL画像，仅以画像微调GPT-2评分预测；warm-start两数据集接近MF/PETER+并验证画像编辑可改推荐。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2024 Findings&quot;,&quot;大语言模型推荐 (LLM4Rec)&quot;,&quot;解释生成/推荐后处理&quot;,&quot;协同指令微调&quot;,&quot;Mixture-of-Experts Adapter&quot;]"><strong><a href="xrec_large_language_models_for_explainable_recommendation/">XRec: Large Language Models for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出模型无关 XRec，用 LightGCN 协同分词器与 MoE 适配器把协同信号注入 LLM 各层，在三数据集语义解释指标上显著优于 ID-based 基线且 USR≈1。</p>
</dd>
</dl>
