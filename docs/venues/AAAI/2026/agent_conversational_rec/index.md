---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**AAAI (RecSys Track) 2026** · **8** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2022 (1)](../../2022/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (9)](../../../ACL_EMNLP/2021/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (3)](../../../ACL_EMNLP/2023/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../../ACL_EMNLP/2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/agent_conversational_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/agent_conversational_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDE 2021 (1)](../../../ICDE/2021/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2022 (1)](../../../KDD/2022/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (6)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/agent_conversational_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/agent_conversational_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/agent_conversational_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** AAAI 2026 ×7 · 端到端生成式 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Query Complexity Routing&quot;,&quot;Agentic Task Planning&quot;,&quot;Memory Filtering&quot;]"><strong><a href="adaptjobrec_enhancing_conversational_career_recommendation_through_an_llm_p/">AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用查询复杂度路由 + 少样本记忆过滤 + 异步嵌套任务规划器改造 LLM Agent，在沃尔玛真实求职场景实现延迟降低 53.3% 且推荐精度提升。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;仿真评测（LLM 用户模拟器）&quot;,&quot;画像缺陷诊断&quot;,&quot;多轮双向演化&quot;]"><strong><a href="diagnostic_guided_dynamic_profile_optimization_for_llm_based_user_simulator/">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用专用小模型诊断画像缺陷、通用 LLM 施治，迭代优化用户画像并与序列推荐器多轮双向演化，模拟保真度较最优基线提升 14.6%–18.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;知识树序列化&quot;,&quot;检索增强生成&quot;,&quot;软提示学习&quot;]"><strong><a href="enhancing_conversational_recommender_systems_with_tree_structured_knowledge/">Enhancing Conversational Recommender Systems with Tree-Structured Knowledge and Pretrained Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 RAG 式对话专属知识树序列化 KG 供 PLM 结构推理，并用辅助任务显式注入协同偏好，INSPIRED 上 MRR@10 相对提升 19.7%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;Theory of Mind&quot;,&quot;意图推理&quot;,&quot;谄媚偏差&quot;]"><strong><a href="rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm_based_conve/">RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个面向对话推荐场景的人类标注 ToM 基准，从认知推断与行为预测两维评估 LLM，揭示细粒度意图与谄媚偏差瓶颈</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Chain-of-User-Thought&quot;,&quot;GUI Agent&quot;,&quot;潜在意图推理&quot;]"><strong><a href="smartagent_chain_of_user_thought_for_embodied_personalized_agent_in_cyber_w/">SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 COUT 三阶用户思维链范式与两阶段训练的 GUI 智能体 SmartAgent，在自建 SmartSpot 基准上实现具身操作与个性化推荐全链路推理。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;分层负反馈&quot;,&quot;虚拟试穿&quot;,&quot;多智能体协作&quot;]"><strong><a href="styletailor_towards_personalized_fashion_styling_via_hierarchical_negative_/">StyleTailor: Towards Personalized Fashion Styling via Hierarchical Negative Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个统一个性化服装设计、购物推荐与虚拟试穿的协作智能体框架，以三级分层负反馈闭环迭代优化，在自建 64 样本基准上 Style Consistency 相对基线提升 39.4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;MLLM Agent&quot;,&quot;GRPO/RL&quot;,&quot;用户模拟&quot;,&quot;LLM4Rec&quot;]"><strong><a href="vragent_r1_boosting_video_recommendation_with_mllm_based_agents_via_reinfor/">VRAgent-R1: Boosting Video Recommendation with MLLM-based Agents via Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双智能体VRAgent-R1：IP Agent渐进理解视频，US Agent以CoT+GRPO模拟用户并重排；MicroLens-100K NDCG@10提升6.0%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;负反馈建模&quot;,&quot;多智能体协同&quot;,&quot;渐进式强化学习&quot;]"><strong><a href="when_top_ranked_recommendations_fail_modeling_multi_granular_negative_feedb/">When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三智能体协同建模多粒度负反馈，S-GRPO 渐进奖励实现可解释视频推荐，线上快滑率降 9.4%</p>
</dd>
</dl>
