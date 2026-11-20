---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**ICLR / NeurIPS / ICML (RecSys) 2026** · **3** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../2025/multimodal_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/multimodal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/multimodal_rec/) · [ACM MM 2023 (5)](../../../ACM_MM/2023/multimodal_rec/) · [ACM MM 2024 (3)](../../../ACM_MM/2024/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2024 (5)](../../../CIKM/2024/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [KDD 2022 (1)](../../../KDD/2022/multimodal_rec/) · [KDD 2024 (1)](../../../KDD/2024/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/multimodal_rec/) · [RecSys 2024 (2)](../../../RecSys/2024/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/multimodal_rec/) · [SIGIR 2024 (1)](../../../SIGIR/2024/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2023 (2)](../../../TMM/2023/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/multimodal_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/multimodal_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** NeurIPS ×3 · 多模态推荐 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;互补信号&quot;,&quot;流形增强&quot;]"><strong><a href="can_complementary_signals_bridge_similarity_islands_manifold_augmented_grap/">Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 MAGE，用离线 Soft Conceptor 残差注入 + 在线非对称双流 GNN 桥接互补品“相似性孤岛”，四个 Amazon 数据集 Full-Ranking 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;文本瓶颈&quot;,&quot;可控推荐&quot;,&quot;多模态内容理解&quot;]"><strong><a href="controllable_and_content_based_recommendations/">Controllable and Content-Based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用多模态 LLM 将图像/音频/视频条目转述为文本画像，经标签监督与 InfoNCE 对齐冻结 CF 骨干，实现可读可编辑的文本瓶颈控制。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;工具增强推理&quot;,&quot;强化微调&quot;]"><strong><a href="intentlens_grounding_underspecified_multimodal_queries_for_recommendation_v/">IntentLens: Grounding Underspecified Multimodal Queries for Recommendation via Tool-Augmented Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MLLM 工具增强推理将欠指定图文查询接地为结构化意图，再以轻量双塔排序，构建两个新基准并显著超越 MLLM 与检索基线。</p>
</dd>
</dl>
