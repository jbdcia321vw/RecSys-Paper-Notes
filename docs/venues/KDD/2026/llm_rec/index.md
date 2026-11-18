# 🧠 大语言模型推荐 (LLM4Rec)

**KDD 2026** · **7** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [KDD 2025 (10)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×6 · 生成式推荐 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习" aria-pressed="false">GRPO 强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KG-RAG" aria-pressed="false">KG-RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不变性学习" aria-pressed="false">不变性学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基准审计" aria-pressed="false">基准审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体强化学习" aria-pressed="false">多智能体强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多轮 Grounding" aria-pressed="false">多轮 Grounding<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多重图消息传递" aria-pressed="false">多重图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理期干预" aria-pressed="false">推理期干预<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="教师学生蒸馏" aria-pressed="false">教师学生蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方面解耦" aria-pressed="false">方面解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="画像外推" aria-pressed="false">画像外推<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱传播" aria-pressed="false">知识图谱传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义画像生成" aria-pressed="false">语义画像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式意图生成" aria-pressed="false">隐式意图生成<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;LLM推荐&quot;,&quot;生成式推荐&quot;,&quot;方面解耦&quot;,&quot;多重图消息传递&quot;]"><strong><a href="copersona_collaborative_persona_graphs_for_robust_llm_personalization/">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用方面级多重人物图+双分支推理补全稀疏用户画像，Amazon六域生成式评测全面超越DEP/DPL。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语义画像生成&quot;,&quot;知识图谱传播&quot;]"><strong><a href="enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026 TSMO&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;隐式意图生成&quot;,&quot;教师学生蒸馏&quot;]"><strong><a href="improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;多轮 Grounding&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="mgfrec_towards_reinforced_reasoning_recommendation_with_multiple_groundings/">MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将单步 grounding 扩展为多轮物品空间接地并引入用户 Agent 反馈，用 GRPO 训练 LLM 推理推荐，Full-Ranking 下 H@5 最高提升约 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;KG-RAG&quot;,&quot;MoE&quot;,&quot;多智能体强化学习&quot;]"><strong><a href="mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;画像外推&quot;,&quot;不变性学习&quot;]"><strong><a href="proex_a_unified_framework_leveraging_large_language_model_with_profile_extr/">ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用四步 CoT 为每个用户/物品生成多面画像并做环境不变性融合，在三个数据集上对 6 个基模型平均提升 5%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;基准审计&quot;,&quot;推理期干预&quot;]"><strong><a href="whose_name_comes_up_ii_benchmarking_and_intervention_based_auditing_of_llm_/">Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 LLMScholarBench，联合审计 22 个 LLM 在 5 类学者推荐任务、3 类推理期干预下的技术质量与社会代表性权衡。</p>
</dd>
</dl>
