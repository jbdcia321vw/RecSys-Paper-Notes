---
hide:
  - toc
---

# 📍 时空轨迹与 POI 本地生活推荐

**IEEE TKDE 2024** · **2** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [IEEE TKDE 2023 (1)](../../2023/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/poi_spatiotemporal_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/poi_spatiotemporal_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/poi_spatiotemporal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/poi_spatiotemporal_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/poi_spatiotemporal_rec/) · [KDD 2024 (1)](../../../KDD/2024/poi_spatiotemporal_rec/) · [KDD 2025 (1)](../../../KDD/2025/poi_spatiotemporal_rec/) · [KDD 2026 (3)](../../../KDD/2026/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [SIGIR 2021 (1)](../../../SIGIR/2021/poi_spatiotemporal_rec/) · [SIGIR 2022 (2)](../../../SIGIR/2022/poi_spatiotemporal_rec/) · [SIGIR 2023 (2)](../../../SIGIR/2023/poi_spatiotemporal_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../../SIGIR/2026/poi_spatiotemporal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/poi_spatiotemporal_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

🔥 **高频主题：** TKDE ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;贝叶斯图增广&quot;,&quot;多视图注意力&quot;,&quot;鲁棒推荐&quot;]"><strong><a href="bayes_enhanced_multi_view_attention_networks_for_robust_poi_recommendation/">Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用贝叶斯后验引导的节点复制图增广加三视图注意力建模个人/语义/距离依赖，在签到缺失与噪声下 R@5 最高提升 16.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;图结构学习&quot;,&quot;层次原型&quot;,&quot;对比多视图融合&quot;]"><strong><a href="bi_level_graph_structure_learning_for_next_poi_recommendation/">Bi-Level Graph Structure Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 EM 聚类原型构造 POI–原型层次图，并配自适应成对结构学习与多视图对比融合；全库排序下 Acc@20 最高相对提升 13.56%。</p>
</dd>
</dl>
