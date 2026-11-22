---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**IJCAI (RecSys Track) 2026** · **2** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/multimodal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2021 (3)](../../../ACM_MM/2021/multimodal_rec/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/multimodal_rec/) · [ACM MM 2023 (5)](../../../ACM_MM/2023/multimodal_rec/) · [ACM MM 2024 (3)](../../../ACM_MM/2024/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (4)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2021 (1)](../../../CIKM/2021/multimodal_rec/) · [CIKM 2024 (5)](../../../CIKM/2024/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2022 (1)](../../../KDD/2022/multimodal_rec/) · [KDD 2024 (1)](../../../KDD/2024/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (4)](../../../KDD/2026/multimodal_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/multimodal_rec/) · [RecSys 2024 (2)](../../../RecSys/2024/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/multimodal_rec/) · [SIGIR 2024 (1)](../../../SIGIR/2024/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2021 (1)](../../../TMM/2021/multimodal_rec/) · [IEEE TMM 2023 (2)](../../../TMM/2023/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (2)](../../../TMM/2026/multimodal_rec/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/multimodal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/multimodal_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/multimodal_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** IJCAI 2026 ×2 · 多模态推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;多模态推荐&quot;,&quot;端到端解释生成&quot;,&quot;多兴趣画像&quot;,&quot;偏好锚定推理&quot;,&quot;多模态聚类&quot;]"><strong><a href="joint_multi_modal_multi_interest_profiling_and_preference_grounded_reasonin/">Joint Multi-Modal Multi-Interest Profiling and Preference-Grounded Reasoning for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>PRIME以兴趣感知聚类、多模态多兴趣画像与偏好锚定检索增强可解释推荐；时间戳切分下BLEURT较Reason4Rec提升7.4%–10.2%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多教师在线蒸馏&quot;,&quot;模态不平衡&quot;]"><strong><a href="three_minds_one_student_online_multi_teacher_knowledge_distillation_for_mul/">Three Minds, One Student: Online Multi-Teacher Knowledge Distillation for Multimodal Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多模态推荐重构为“三教师-单学生”在线蒸馏，在排序级与表征级选择性迁移 ID/图/文偏好，Full-Ranking 上最高相对提升 7.22%。</p>
</dd>
</dl>
