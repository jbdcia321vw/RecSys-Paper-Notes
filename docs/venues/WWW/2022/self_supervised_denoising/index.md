---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪

**WWW (TheWebConf) 2022** · **2** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (4)](../../2023/self_supervised_denoising/) · [WWW (TheWebConf) 2024 (3)](../../2024/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (4)](../../2026/self_supervised_denoising/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/self_supervised_denoising/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/self_supervised_denoising/) · [CIKM 2021 (1)](../../../CIKM/2021/self_supervised_denoising/) · [CIKM 2022 (2)](../../../CIKM/2022/self_supervised_denoising/) · [CIKM 2023 (2)](../../../CIKM/2023/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [CIKM 2026 (1)](../../../CIKM/2026/self_supervised_denoising/) · [DASFAA 2022 (1)](../../../DASFAA/2022/self_supervised_denoising/) · [DASFAA 2023 (2)](../../../DASFAA/2023/self_supervised_denoising/) · [DASFAA 2026 (1)](../../../DASFAA/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICDE 2022 (1)](../../../ICDE/2022/self_supervised_denoising/) · [ICDE 2023 (1)](../../../ICDE/2023/self_supervised_denoising/) · [ICDE 2024 (3)](../../../ICDE/2024/self_supervised_denoising/) · [ICDM 2021 (1)](../../../ICDM/2021/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/self_supervised_denoising/) · [KDD 2021 (1)](../../../KDD/2021/self_supervised_denoising/) · [KDD 2022 (1)](../../../KDD/2022/self_supervised_denoising/) · [KDD 2023 (2)](../../../KDD/2023/self_supervised_denoising/) · [KDD 2024 (3)](../../../KDD/2024/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2023 (1)](../../../RecSys/2023/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2021 (2)](../../../SIGIR/2021/self_supervised_denoising/) · [SIGIR 2022 (4)](../../../SIGIR/2022/self_supervised_denoising/) · [SIGIR 2023 (6)](../../../SIGIR/2023/self_supervised_denoising/) · [SIGIR 2024 (1)](../../../SIGIR/2024/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/self_supervised_denoising/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/self_supervised_denoising/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/self_supervised_denoising/) · [ACM TORS 2023 (1)](../../../TORS/2023/self_supervised_denoising/) · [WSDM 2021 (1)](../../../WSDM/2021/self_supervised_denoising/) · [WSDM 2022 (1)](../../../WSDM/2022/self_supervised_denoising/) · [WSDM 2023 (1)](../../../WSDM/2023/self_supervised_denoising/) · [WSDM 2024 (1)](../../../WSDM/2024/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;邻域对比学习&quot;,&quot;原型聚类 (EM)&quot;,&quot;图协同过滤&quot;]"><strong><a href="improving_graph_collaborative_filtering_with_neighborhood_enriched_contrast/">Improving Graph Collaborative Filtering with Neighborhood-enriched Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>NCL 用 GNN 偶数层输出与 K-means 原型分别构造结构/语义邻域对比对，Full-Ranking 下 Yelp Recall@10 较 LightGCN 提升 26%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/粗排&quot;,&quot;跨模型一致性&quot;,&quot;隐式反馈去噪&quot;,&quot;KL 散度正则&quot;]"><strong><a href="learning_robust_recommenders_through_cross_model_agreement/">Learning Robust Recommenders through Cross-Model Agreement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨模型预测一致性作弱监督信号，通过 KL 一致性约束+似然最大化联合去噪正负样本，在 4 个骨干上一致提升。</p>
</dd>
</dl>
