---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**CIKM 2026** · **2** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [CIKM 2023 (3)](../../2023/collaborative_filtering/) · [CIKM 2024 (4)](../../2024/collaborative_filtering/) · [CIKM 2025 (2)](../../2025/collaborative_filtering/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICDE 2023 (1)](../../../ICDE/2023/collaborative_filtering/) · [ICDE 2024 (2)](../../../ICDE/2024/collaborative_filtering/) · [ICDM 2023 (1)](../../../ICDM/2023/collaborative_filtering/) · [ICDM 2024 (1)](../../../ICDM/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/collaborative_filtering/) · [KDD 2023 (1)](../../../KDD/2023/collaborative_filtering/) · [KDD 2024 (1)](../../../KDD/2024/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2023 (7)](../../../RecSys/2023/collaborative_filtering/) · [RecSys 2024 (4)](../../../RecSys/2024/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2023 (6)](../../../SIGIR/2023/collaborative_filtering/) · [SIGIR 2024 (4)](../../../SIGIR/2024/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/collaborative_filtering/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/collaborative_filtering/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2023 (1)](../../../TORS/2023/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2023 (1)](../../../WSDM/2023/collaborative_filtering/) · [WSDM 2024 (3)](../../../WSDM/2024/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/collaborative_filtering/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;显式负反馈&quot;]"><strong><a href="dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Transformer Scaling Law&quot;,&quot;特征异构归一化&quot;,&quot;多域统一建模&quot;]"><strong><a href="transretrieval_scaling_up_transformer_based_retrieval_for_industrial_recomm/">TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用加权平均聚合+目标Token压缩+位置式域嵌入，让Transformer在召回阶段实现对数线性Scaling，线上收入+2.53%。</p>
</dd>
</dl>
