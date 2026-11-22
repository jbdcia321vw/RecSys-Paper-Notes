---
hide:
  - toc
---

# 📍 时空轨迹与 POI 本地生活推荐

**SIGIR 2022** · **2** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [SIGIR 2021 (1)](../../2021/poi_spatiotemporal_rec/) · [SIGIR 2023 (2)](../../2023/poi_spatiotemporal_rec/) · [SIGIR 2024 (3)](../../2024/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../2026/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/poi_spatiotemporal_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/poi_spatiotemporal_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/poi_spatiotemporal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/poi_spatiotemporal_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/poi_spatiotemporal_rec/) · [KDD 2024 (1)](../../../KDD/2024/poi_spatiotemporal_rec/) · [KDD 2025 (1)](../../../KDD/2025/poi_spatiotemporal_rec/) · [KDD 2026 (3)](../../../KDD/2026/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/poi_spatiotemporal_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/poi_spatiotemporal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/poi_spatiotemporal_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

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
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多关系图卷积&quot;,&quot;耦合RNN时序互影响&quot;]"><strong><a href="empowering_next_poi_recommendation_with_multi_relational_modeling/">Empowering Next POI Recommendation with Multi-Relational Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MEMO 用多关系 GCN + 关系专属自注意力聚合异构社交图，并以耦合 RNN 建模用户-POI 时序互影响；在三个 LBSN 数据集 Recall@10 相对 STAN 提升约 2.7%–3.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;轨迹流图&quot;,&quot;Graph Enhanced Transformer&quot;]"><strong><a href="getnext_trajectory_flow_map_enhanced_transformer_for_next_poi_recommendatio/">GETNext: Trajectory Flow Map Enhanced Transformer for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建用户无关的全局轨迹流图并用 GCN+转移动注意力图显式编码群体转移模式，融合 Transformer 序列建模，在 NYC/TKY/CA 全库排序上 Acc@5 提升约 11%。</p>
</dd>
</dl>
