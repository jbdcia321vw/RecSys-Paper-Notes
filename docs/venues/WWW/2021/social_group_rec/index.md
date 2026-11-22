---
hide:
  - toc
---

# 👥 社交与群组推荐

**WWW (TheWebConf) 2021** · **2** 篇论文解读

> **方向说明**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (2)](../../2023/social_group_rec/) · [WWW (TheWebConf) 2024 (2)](../../2024/social_group_rec/) · [WWW (TheWebConf) 2025 (2)](../../2025/social_group_rec/) · [WWW (TheWebConf) 2026 (3)](../../2026/social_group_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/social_group_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/social_group_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/social_group_rec/) · [CIKM 2021 (3)](../../../CIKM/2021/social_group_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/social_group_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/social_group_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/social_group_rec/) · [CIKM 2025 (2)](../../../CIKM/2025/social_group_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/social_group_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/social_group_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/social_group_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/social_group_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/social_group_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (1)](../../../ICLR_NeurIPS_ICML/2023/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/social_group_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/social_group_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/social_group_rec/) · [KDD 2024 (1)](../../../KDD/2024/social_group_rec/) · [RecSys 2025 (3)](../../../RecSys/2025/social_group_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/social_group_rec/) · [SIGIR 2021 (2)](../../../SIGIR/2021/social_group_rec/) · [SIGIR 2022 (1)](../../../SIGIR/2022/social_group_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/social_group_rec/) · [IEEE TKDE 2022 (5)](../../../TKDE/2022/social_group_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/social_group_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/social_group_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/social_group_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/social_group_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/social_group_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;多关系图神经网络&quot;,&quot;上下文调制&quot;,&quot;物品吸引力建模&quot;]"><strong><a href="dual_side_deep_context_aware_modulation_for_social_recommendation/">Dual Side Deep Context-aware Modulation for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>DICER 用多关系 RGNN 得到图增强的用户/物品表征作为“深度上下文”，双向调制历史交互以提取用户兴趣与物品吸引力，Ciao/Epinion 上 Recall@10 相对 DiffNet++ 提升 9.6%/3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;自监督互信息最大化&quot;]"><strong><a href="self_supervised_multi_channel_hypergraph_convolutional_network_for_social_r/">Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用三角形模体诱导多通道超图卷积建模高阶社交关系，并以层级互信息最大化自监督补偿聚合损失，全库排序下 NDCG@10 最高提升 21.3%。</p>
</dd>
</dl>
