---
hide:
  - toc
---

# 📍 时空轨迹与 POI 本地生活推荐

**ICDE 2026** · **2** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/poi_spatiotemporal_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/poi_spatiotemporal_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/poi_spatiotemporal_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/poi_spatiotemporal_rec/) · [KDD 2024 (1)](../../../KDD/2024/poi_spatiotemporal_rec/) · [KDD 2025 (1)](../../../KDD/2025/poi_spatiotemporal_rec/) · [KDD 2026 (3)](../../../KDD/2026/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [SIGIR 2021 (1)](../../../SIGIR/2021/poi_spatiotemporal_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/poi_spatiotemporal_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/poi_spatiotemporal_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../../SIGIR/2026/poi_spatiotemporal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/poi_spatiotemporal_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/poi_spatiotemporal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/poi_spatiotemporal_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

🔥 **高频主题：** 时空轨迹与 POI 推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与 POI 推荐" aria-pressed="false">时空轨迹与 POI 推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;精排&quot;,&quot;持续学习&quot;,&quot;兴趣记忆检索&quot;,&quot;模型无关框架&quot;]"><strong><a href="efficient_model_agnostic_continual_learning_for_next_poi_recommendation/">Efficient Model-Agnostic Continual Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 GIRAM：以分布型兴趣记忆 + 上下文感知键编码 + cVAE 多键生成检索 + 一致性自适应融合，实现免重训的持续 Next-POI 推荐，三数据集 Acc@5 与 MRR 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;生成式端到端&quot;,&quot;quadkey 层次网格编码&quot;,&quot;可学习傅里叶位置编码&quot;,&quot;POI 转移嵌入对齐&quot;]"><strong><a href="geography_aware_large_language_models_for_next_poi_recommendation/">Geography-Aware Large Language Models for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 quadkey+可学习傅里叶位置编码(GCIM)与序列模型 POI 嵌入对齐(PAM)增强 Llama-2-7B，三数据集 Acc@1 相对提升 14.7%–16.7%</p>
</dd>
</dl>
