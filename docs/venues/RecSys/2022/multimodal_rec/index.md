---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**RecSys 2022** · **2** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [RecSys 2024 (2)](../../2024/multimodal_rec/) · [RecSys 2025 (9)](../../2025/multimodal_rec/) · [RecSys 2026 (2)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/multimodal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/multimodal_rec/) · [ACM MM 2023 (5)](../../../ACM_MM/2023/multimodal_rec/) · [ACM MM 2024 (3)](../../../ACM_MM/2024/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2024 (5)](../../../CIKM/2024/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2022 (1)](../../../KDD/2022/multimodal_rec/) · [KDD 2024 (1)](../../../KDD/2024/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/multimodal_rec/) · [SIGIR 2024 (1)](../../../SIGIR/2024/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2023 (2)](../../../TMM/2023/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/multimodal_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/multimodal_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×2

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
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;音频内容分析&quot;,&quot;情绪条件推荐&quot;,&quot;工业级部署&quot;]"><strong><a href="flow_moods_recommending_music_by_moods_on_deezer/">Flow Moods: Recommending Music by Moods on Deezer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 Flow Moods 工业系统：将六类情绪作为显式上下文，融合专业策展标注、musicnn 音频嵌入与随机森林情绪分类器，对协同过滤检索结果做情绪硬过滤与兜底，已在千万级用户上线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;可复用自注意力&quot;,&quot;异构实体序列建模&quot;,&quot;穿搭生成&quot;]"><strong><a href="reusable_self_attention_recommender_systems_in_fashion_industry_application/">Reusable Self-Attention Recommender Systems in Fashion Industry Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>Zalando 用单一可复用 Transformer 统一穿搭排序、会话推荐与穿搭生成，靠异构实体/交互共享嵌入+时间衰减会话+目标掩码，线上 A/B 留存最高 +30%、会话互动 +130%。</p>
</dd>
</dl>
