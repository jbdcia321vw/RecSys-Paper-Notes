---
hide:
  - toc
---

# 👥 社交与群组推荐

**SIGIR 2025** · **2** 篇论文解读

> **方向说明**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

📌 **同领域跨会议浏览：** [SIGIR 2022 (1)](../../2022/social_group_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/social_group_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/social_group_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/social_group_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/social_group_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/social_group_rec/) · [CIKM 2025 (2)](../../../CIKM/2025/social_group_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/social_group_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/social_group_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/social_group_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/social_group_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (1)](../../../ICLR_NeurIPS_ICML/2023/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/social_group_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/social_group_rec/) · [KDD 2024 (1)](../../../KDD/2024/social_group_rec/) · [RecSys 2025 (3)](../../../RecSys/2025/social_group_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/social_group_rec/) · [IEEE TKDE 2022 (5)](../../../TKDE/2022/social_group_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/social_group_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/social_group_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/social_group_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/social_group_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/social_group_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/social_group_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/social_group_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/social_group_rec/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/social_group_rec/)

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
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;多模态好友推荐&quot;,&quot;模态感知成对匹配&quot;,&quot;全局模态注意力&quot;]"><strong><a href="frog_effective_friend_recommendation_in_online_games_via_modality_aware_use/">FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向在线游戏好友推荐，用模态感知双向匹配+局部/全局双路注意力建模用户偏好，离线 HR/NDCG 最高提升 15.82%/14.59%，线上 accRate 提升 33.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图不变学习&quot;,&quot;社交去噪&quot;,&quot;对抗环境生成&quot;]"><strong><a href="invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
</dl>
