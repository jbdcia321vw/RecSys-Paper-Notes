---
hide:
  - toc
---

# 👥 社交与群组推荐

**SIGIR 2021** · **2** 篇论文解读

> **方向说明**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

📌 **同领域跨会议浏览：** [SIGIR 2022 (1)](../../2022/social_group_rec/) · [SIGIR 2025 (2)](../../2025/social_group_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/social_group_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/social_group_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/social_group_rec/) · [CIKM 2021 (3)](../../../CIKM/2021/social_group_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/social_group_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/social_group_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/social_group_rec/) · [CIKM 2025 (2)](../../../CIKM/2025/social_group_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/social_group_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/social_group_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/social_group_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/social_group_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/social_group_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (1)](../../../ICLR_NeurIPS_ICML/2023/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/social_group_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/social_group_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/social_group_rec/) · [KDD 2024 (1)](../../../KDD/2024/social_group_rec/) · [RecSys 2025 (3)](../../../RecSys/2025/social_group_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/social_group_rec/) · [IEEE TKDE 2022 (5)](../../../TKDE/2022/social_group_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/social_group_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/social_group_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/social_group_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/social_group_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/social_group_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/social_group_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/social_group_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/social_group_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/social_group_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/social_group_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;一致性邻居采样&quot;,&quot;关系注意力&quot;]"><strong><a href="consisrec_enhancing_gnn_for_social_recommendation_via_consistent_neighbor_a/">ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出一致性查询+邻居采样+关系注意力，抑制社交图中上下文级与关系级不一致噪声，在 Ciao/Epinions 评分预测上 RMSE 相对降 2.79%/1.74%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;好友圈分组&quot;,&quot;个性化搜索&quot;]"><strong><a href="group_based_personalized_search_by_integrating_search_behaviour_and_friend_/">Group based Personalized Search by Integrating Search Behaviour and Friend Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>FNPS 用“关系圈+行为圈”双通道好友分组与掩码交叉注意力构建群体画像，在微信搜索日志重排上 MAP 较原始排序 +3.22%。</p>
</dd>
</dl>
