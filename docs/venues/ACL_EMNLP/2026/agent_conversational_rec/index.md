---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**ACL / EMNLP (RecSys & CRS) 2026** · **9** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2021 (9)](../../2021/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../2022/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (3)](../../2023/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (8)](../../../AAAI/2026/agent_conversational_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/agent_conversational_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/agent_conversational_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDE 2021 (1)](../../../ICDE/2021/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2022 (1)](../../../KDD/2022/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (6)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/agent_conversational_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/agent_conversational_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/agent_conversational_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** 生成式推荐 ×5 · ACL 2026 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;贝叶斯偏好推断&quot;,&quot;主动提问&quot;,&quot;决策支持&quot;]"><strong><a href="decisive_guiding_user_decisions_with_optimal_preference_elicitation_from_un/">Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将决策解耦为文档客观打分矩阵与贝叶斯偏好推断，用决策感知信息增益主动提问，Top-1 准确率最高提升约 20%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026 Findings&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;长期记忆基准&quot;,&quot;遗忘感知评估&quot;]"><strong><a href="from_recall_to_forgetting_benchmarking_long_term_memory_for_personalized_ag/">From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 Memora 长周期记忆基准与 FAMA 遗忘感知指标，揭示 LLM 与记忆智能体在记忆突变下频繁复用失效记忆。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;树搜索推理&quot;,&quot;分层偏好优化&quot;]"><strong><a href="harpo_hierarchical_agentic_reasoning_for_user_aligned_conversational_recomm/">HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将对话推荐重构为多维质量驱动的结构化决策，用分层偏好奖励引导树搜索推理，ReDial R@10 达 29.8%（+18.7%）。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;Generate-then-Match&quot;,&quot;冷启动推理&quot;]"><strong><a href="intent_driven_semantic_id_generation_for_grounded_conversational_news_recom/">Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 直接生成三层语义 ID 前缀再模糊匹配新闻池，结构上消除幻觉，冷启动 L1 达 18.0%，延迟 85ms。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;不确定性蒸馏&quot;,&quot;DPO&quot;]"><strong><a href="mirroring_users_towards_building_preference_aligned_user_simulator_with_use/">Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成 EKB 决策过程作澄清、按认知不确定性蒸馏并拒绝采样去噪，微调 3B 模拟器在 8 域上超越 GPT-5 与教师模型。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;情境对话推荐&quot;,&quot;评分标准强化学习&quot;,&quot;异构负例DPO&quot;]"><strong><a href="re2a_situated_conversational_recommendation_via_rubric_based_preference_rea/">Re2A: Situated Conversational Recommendation via Rubric-based Preference Reasoning and Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将情境对话推荐重构为“先推理后对齐”两阶段，用自动评分标准引导偏好状态推理并以异构负例 DPO 实现用户-场景双重对齐，SIMMC 2.1 上 Hit@1 提升 8.17pt。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP Findings&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;MCP 工具编排&quot;,&quot;模糊意图&quot;,&quot;LLM-as-a-Judge&quot;]"><strong><a href="rectoolbench_benchmarking_recommendation_specific_tool_orchestration_under_/">RecToolBench: Benchmarking Recommendation-Specific Tool Orchestration under Fuzzy User Intent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个基于 MCP 的模糊意图推荐工具编排基准，1288 条四级难度可执行任务，揭示语法合法≠推荐成功的执行-结果鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;结构化上下文建模&quot;,&quot;蒙特卡洛树搜索&quot;]"><strong><a href="towards_effective_structured_context_modeling_for_conversational_recommende/">Towards Effective Structured Context Modeling for Conversational Recommender Systems via Dual-node Monte Carlo Tree Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将对话上下文建模为双节点 MCTS 树，ELNode 搜索追问动作、EXNode 精炼检索查询，Redial/OpenDialKG 上 SR 平均提升 9.07%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;场景转移估计&quot;,&quot;贝叶斯逆向推理&quot;]"><strong><a href="where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conve/">Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SiPeR 框架，用场景转移估计解决“去哪”与贝叶斯逆向推理解决“要什么”，在 SIMMC 2.1 与 SCREEN 上 R@1 平均提升约 10.9%/10.6%。</p>
</dd>
</dl>
