---
hide:
  - toc
---

# 🗣️ 对话推荐与智能体仿真

**ACL / EMNLP (RecSys & CRS) 2023** · **3** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2024 (3)](../../2024/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../2026/agent_conversational_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/agent_conversational_rec/) · [CIKM 2024 (2)](../../../CIKM/2024/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2023 (1)](../../../KDD/2023/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/agent_conversational_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/agent_conversational_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/agent_conversational_rec/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2023 (1)](../../../WWW/2023/agent_conversational_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** ACL ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;检索增强生成&quot;,&quot;方面学习&quot;,&quot;可解释推荐&quot;]"><strong><a href="explainable_recommendation_with_personalized_review_retrieval_and_aspect_le/">Explainable Recommendation with Personalized Review Retrieval and Aspect Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>ERRA 以训练集评论为检索语料、用 Sentence-BERT+Faiss 检索个性化句子并结合 top-n 方面掩码注入 Transformer，联合打分与解释生成；8:1:1 划分下 TripAdvisor RMSE 降 3.3%、BLEU-4 升 15.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;主观偏好消歧&quot;,&quot;多模态推荐基准&quot;]"><strong><a href="multimodal_recommendation_dialog_with_subjective_preference_a_new_challenge/">Multimodal Recommendation Dialog with Subjective Preference: A New Challenge and Benchmark</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>构建 12K 主观偏好购物对话基准 SURE，提出偏好消歧/区域理解/多模态推荐三任务与 MRA 基线，揭示现有模型远未解决。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回/生成式端到端&quot;,&quot;树结构推理&quot;,&quot;知识图谱&quot;]"><strong><a href="trea_tree_structure_reasoning_schema_for_conversational_recommendation/">TREA: Tree-Structure Reasoning Schema for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多层级推理树显式建模实体因果链，配合分支隔离/语义对齐损失，ReDial/TG-ReDial 全实体 R@50 提升 1.7%/41.0%。</p>
</dd>
</dl>
