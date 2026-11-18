---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**WSDM 2025** · **2** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态去噪" aria-pressed="false">多模态去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态图学习" aria-pressed="false">多模态图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="最优传输" aria-pressed="false">最优传输<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波去噪" aria-pressed="false">频域滤波去噪<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;频域滤波去噪&quot;,&quot;多模态图学习&quot;]"><strong><a href="spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;多模态去噪&quot;,&quot;知识蒸馏&quot;,&quot;最优传输&quot;]"><strong><a href="teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
</dl>
  </div>
</div>
