# 🔄 自监督对比学习与交互去噪

**SIGIR 2026** · **2** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [SIGIR 2025 (1)](../../2025/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sinkhorn 可微投影" aria-pressed="false">Sinkhorn 可微投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不可能负样本" aria-pressed="false">不可能负样本<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="批采样" aria-pressed="false">批采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户表征学习" aria-pressed="false">用户表征学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应数据增强" aria-pressed="false">自适应数据增强<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自适应数据增强&quot;,&quot;Sinkhorn 可微投影&quot;]"><strong><a href="asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;批采样&quot;,&quot;不可能负样本&quot;,&quot;用户表征学习&quot;]"><strong><a href="don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别地理约束下 in-batch 负样本中的“不可能负样本”问题，提出区域同质批采样 RCBS，仅改批构造即显著提升用户表征与线上指标。</p>
</dd>
</dl>
