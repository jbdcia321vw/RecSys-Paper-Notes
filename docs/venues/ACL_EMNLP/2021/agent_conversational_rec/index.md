---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**ACL / EMNLP (RecSys & CRS) 2021** · **9** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../2022/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (3)](../../2023/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (8)](../../../AAAI/2026/agent_conversational_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/agent_conversational_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/agent_conversational_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDE 2021 (1)](../../../ICDE/2021/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2022 (1)](../../../KDD/2022/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (6)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/agent_conversational_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/agent_conversational_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/agent_conversational_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** EMNLP 2021 ×4 · 端到端生成式 ×4 · EMNLP ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮对话式推荐（端到端生成）&quot;,&quot;树状知识图谱推理&quot;,&quot;对话动作（Dialog Act）序列化&quot;]"><strong><a href="cr_walker_tree_structured_graph_reasoning_and_dialog_acts_for_conversationa/">CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 walker cell 在 KG 上做两跳树状推理，并把推理树前序线性化为对话动作来约束 GPT-2 生成；ReDial 覆盖率 14.7、知识 F1 47.7，人工评测四项全面超越人类参考答案。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;知识图谱多跳推理&quot;,&quot;强化学习策略搜索&quot;,&quot;路径片段融合&quot;]"><strong><a href="crfr_improving_conversational_recommender_systems_via_flexible_fragments_re/">CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用强化学习在物品/概念双知识图谱上做显式多跳推理，以柔性路径片段而非完整路径增强推荐与回复生成，ReDial R@10 相对提升 10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;系统与工具链(端到端)&quot;,&quot;模块化解耦框架&quot;,&quot;三子任务联合建模&quot;,&quot;统一评测协议&quot;]"><strong><a href="crslab_an_open_source_toolkit_for_building_conversational_recommender_syste/">CRSLab: An Open-Source Toolkit for Building Conversational Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个开源 CRS 工具链：将推荐/对话/策略三子任务统一形式化，提供四级解耦数据流、6 数据集、18 模型与全栈自动评测接口。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;双语平行语料&quot;,&quot;跨语言对话推荐&quot;]"><strong><a href="durecdial_2_0_a_bilingual_parallel_corpus_for_conversational_recommendation/">DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个中英平行对话推荐语料，定义单语/多语/跨语五任务，验证英文数据可反向增益中文推荐。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回+重排&quot;,&quot;批评到偏好改写 (Critique Interpretation)&quot;,&quot;少样本提示与观点检索&quot;]"><strong><a href="it_doesn_t_look_good_for_a_date_transforming_critiques_into_preferences_for/">It doesn&#x27;t look good for a date: Transforming Critiques into Preferences for Conversational Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GPT-3 few-shot 把自由批评改写为正向偏好，再用 USE 检索 / BERT 重排评论论据；检索 Precision@1 提升 124%，重排 nDCG@1 提升 19–59%。</p>
</dd>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;多子目标规划&quot;,&quot;知识图谱&quot;,&quot;知识增强生成&quot;]"><strong><a href="kers_a_knowledge_enhanced_framework_for_recommendation_dialog_systems_with_/">KERS: A Knowledge-Enhanced Framework for Recommendation Dialog Systems with Multiple Subgoals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以子目标规划筛选知识，并用顺序注意力、知识门控与词典增强抑制噪声，DuRecDial 上 F1 与知识 F1 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;神经模板槽位填充&quot;,&quot;充分多注意力物品选择&quot;,&quot;新物品零样本&quot;]"><strong><a href="learning_neural_templates_for_recommender_dialogue_system/">Learning Neural Templates for Recommender Dialogue System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>解耦对话生成与物品推荐，用 [ITEM] 槽位模板 + 富信息注意力选择器填充，ReDial 上 ReR@1 与 Dist-4 大幅刷新 SOTA 并支持新物品零样本推荐。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;可解释推荐&quot;,&quot;个性化文本生成&quot;,&quot;多任务学习&quot;]"><strong><a href="personalized_transformer_for_explainable_recommendation/">Personalized Transformer for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用“上下文预测”辅助任务把用户/物品 ID 映射到解释词空间，使 2 层无预训练 Transformer 兼顾个性化解释生成与评分预测，BLEU-4 超微调 BERT 与 NETE 约 24–28%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL Findings 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式（对话推荐+回复生成）&quot;,&quot;情感感知评论检索&quot;,&quot;评论注意力解码器&quot;]"><strong><a href="revcore_review_augmented_conversational_recommendation/">RevCore: Review-augmented Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用情感一致评论检索增强REDIAL对话推荐画像与回复生成；R@1 6.1、Dist-3 0.558。</p>
</dd>
</dl>
