---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪

**CIKM 2023** · **2** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [CIKM 2025 (3)](../../2025/self_supervised_denoising/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/self_supervised_denoising/) · [DASFAA 2023 (2)](../../../DASFAA/2023/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICDE 2023 (1)](../../../ICDE/2023/self_supervised_denoising/) · [ICDE 2024 (3)](../../../ICDE/2024/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2023 (2)](../../../KDD/2023/self_supervised_denoising/) · [KDD 2024 (3)](../../../KDD/2024/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2023 (1)](../../../RecSys/2023/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2023 (6)](../../../SIGIR/2023/self_supervised_denoising/) · [SIGIR 2024 (1)](../../../SIGIR/2024/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/self_supervised_denoising/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/self_supervised_denoising/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [ACM TORS 2023 (1)](../../../TORS/2023/self_supervised_denoising/) · [WSDM 2023 (1)](../../../WSDM/2023/self_supervised_denoising/) · [WSDM 2024 (1)](../../../WSDM/2024/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2023 (4)](../../../WWW/2023/self_supervised_denoising/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

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
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;随机播放&quot;,&quot;会话推荐&quot;,&quot;转移增广&quot;]"><strong><a href="muse_music_recommender_system_with_shuffle_play_recommendation_enhancement/">MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对音乐随机播放会话唯一转移率过高的问题，提出转移频次引导的会话增广与细粒度匹配，在 MSSD 全库排序上稳定超越 12 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;超图推荐&quot;,&quot;一致性感知增强&quot;,&quot;购物篮推荐&quot;]"><strong><a href="robust_basket_recommendation_via_noise_tolerated_graph_contrastive_learning/">Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>BNCL 以双行为图/超图与一致性感知对比学习抑制购物篮噪声，全库排序下 R@60 最高 21.10。</p>
</dd>
</dl>
