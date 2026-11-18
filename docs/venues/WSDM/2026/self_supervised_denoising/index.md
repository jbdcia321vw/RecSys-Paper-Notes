# 🔄 自监督对比学习与交互去噪

**WSDM 2026** · **1** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [WSDM 2025 (2)](../../2025/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散数据增强" aria-pressed="false">扩散数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负采样" aria-pressed="false">难负采样<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散数据增强&quot;,&quot;难负采样&quot;]"><strong><a href="simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
</dl>
