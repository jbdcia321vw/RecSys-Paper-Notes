---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**CIKM 2024** · **2** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [CIKM 2022 (1)](../../2022/agent_conversational_rec/) · [CIKM 2023 (1)](../../2023/agent_conversational_rec/) · [CIKM 2025 (3)](../../2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (8)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (9)](../../../ACL_EMNLP/2021/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (3)](../../../ACL_EMNLP/2023/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../../ACL_EMNLP/2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDE 2021 (1)](../../../ICDE/2021/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2022 (1)](../../../KDD/2022/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (6)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2021 (2)](../../../RecSys/2021/agent_conversational_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/agent_conversational_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/agent_conversational_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/agent_conversational_rec/)

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
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;记忆增强 LLM&quot;,&quot;实体记忆库&quot;,&quot;冷启动&quot;]"><strong><a href="memocrs_memory_enhanced_sequential_conversational_recommender_systems_with_/">MemoCRS: Memory-enhanced Sequential Conversational Recommender Systems with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用实体记忆库与通用记忆增强 LLM，建模跨会话偏好连续性；时序切分下 MRR@20 最高提升 38.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端策略学习&quot;,&quot;离线强化学习&quot;,&quot;用户模拟器&quot;,&quot;以用户为中心偏好建模&quot;]"><strong><a href="reformulating_conversational_recommender_systems_as_tri_phase_offline_polic/">Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 CRS 重构为「用户建模—离线策略—独立可控仿真评估」三阶段解耦范式，Yelp SR@15 达 0.415 且 AT 降至 11.62。</p>
</dd>
</dl>
