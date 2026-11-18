---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**ECIR 2026** · **3** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Reranking" aria-pressed="false">LLM Reranking<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-a-Judge" aria-pressed="false">LLM-as-a-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Reference-Free 仿真" aria-pressed="false">Reference-Free 仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双智能体交互" aria-pressed="false">双智能体交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="外部推荐器适配" aria-pressed="false">外部推荐器适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据合成/仿真训练" aria-pressed="false">数据合成/仿真训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="角色掩码损失" aria-pressed="false">角色掩码损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM Reranking&quot;,&quot;外部推荐器适配&quot;]"><strong><a href="improving_conversational_recommendation_with_contextual_adaptation_of_exter/">Improving Conversational Recommendation with Contextual Adaptation of External Recommenders and LLM-based Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用轻量实体级序列推荐器产出候选、LLM 做上下文适配重排，ReDial/INSPIRED 上 H@5 平均提升 54%/25%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;数据合成/仿真训练&quot;,&quot;Reference-Free 仿真&quot;,&quot;角色掩码损失&quot;,&quot;双智能体交互&quot;]"><strong><a href="interplay_training_independent_simulators_for_reference_free_conversational/">Interplay: Training Independent Simulators for Reference-Free Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>训练独立用户/推荐双模拟器，以目标属性替代目标物品消除 oracle 泄漏，8B 模型 Recall@1 超 32B 基线 15 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;用户模拟&quot;,&quot;LLM-as-a-Judge&quot;]"><strong><a href="usersimcrs_v2_simulation_based_evaluation_for_conversational_recommender_sy/">UserSimCRS v2: Simulation-Based Evaluation for Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>升级对话推荐用户模拟评测工具包，引入 LLM 模拟器、统一数据格式与 LLM-as-a-Judge 指标，案例揭示模拟器间系统排序显著分歧。</p>
</dd>
</dl>
  </div>
</div>
