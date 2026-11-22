---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**WSDM 2023** · **2** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [WSDM 2021 (1)](../../2021/agent_conversational_rec/) · [WSDM 2025 (2)](../../2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (8)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (9)](../../../ACL_EMNLP/2021/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (3)](../../../ACL_EMNLP/2023/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../../ACL_EMNLP/2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/agent_conversational_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/agent_conversational_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDE 2021 (1)](../../../ICDE/2021/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2022 (1)](../../../KDD/2022/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (6)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/agent_conversational_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/agent_conversational_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/agent_conversational_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端对话评测（仿真）&quot;,&quot;用户模拟器&quot;,&quot;议程驱动交互模型&quot;]"><strong><a href="usersimcrs_a_user_simulation_toolkit_for_evaluating_conversational_recommen/">UserSimCRS: A User Simulation Toolkit for Evaluating Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>模块化议程驱动用户模拟器工具箱，新增满意度预测、人格/情境建模与条件式自然语言生成，仅用 8 条标注对话即可黑盒评测 IAI MovieBot，但仅有定性案例无定量基准。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;变分推断&quot;,&quot;知识图谱补全&quot;]"><strong><a href="variational_reasoning_over_incomplete_knowledge_graphs_for_conversational_r/">Variational Reasoning over Incomplete Knowledge Graphs for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将对话特定子图建模为离散隐变量，用变分贝叶斯同时补全不完整 KG 并按上下文选知识；REDIAL 上 Recall@1 相对提升 11.8%。</p>
</dd>
</dl>
