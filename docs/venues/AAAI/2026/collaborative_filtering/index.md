# 🤝 协同过滤与双塔召回

**AAAI (RecSys Track) 2026** · **2** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (1)](../../2025/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** AAAI 2026 ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩子空间共享" aria-pressed="false">低秩子空间共享<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层对比学习" aria-pressed="false">分层对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="半监督矩阵完成" aria-pressed="false">半监督矩阵完成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="正样本扩充" aria-pressed="false">正样本扩充<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="泛化界" aria-pressed="false">泛化界<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正样本扩充&quot;,&quot;分层对比学习&quot;]"><strong><a href="crops_improving_dense_retrieval_with_cross_perspective_positive_samples_in_/">CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用查询改写/推荐流/LLM 三源扩充正样本，配分层标签与 H-InfoNCE，快手搜索线上 CTR +0.87%、改写率 -0.65%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;半监督矩阵完成&quot;,&quot;低秩子空间共享&quot;,&quot;泛化界&quot;]"><strong><a href="generalization_bounds_for_semi_supervised_matrix_completion_with_distributi/">Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出隐式反馈采样分布与评分矩阵共享低秩子空间的半监督矩阵完成范式，证明误差可解耦为两项独立界，并在 ML-100K/Douban/Yelp 上验证。</p>
</dd>
</dl>
