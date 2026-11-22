---
hide:
  - toc
---

# 📍 时空轨迹与 POI 本地生活推荐

**SIGIR 2024** · **3** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [SIGIR 2021 (1)](../../2021/poi_spatiotemporal_rec/) · [SIGIR 2022 (2)](../../2022/poi_spatiotemporal_rec/) · [SIGIR 2023 (2)](../../2023/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../2026/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/poi_spatiotemporal_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/poi_spatiotemporal_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/poi_spatiotemporal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/poi_spatiotemporal_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/poi_spatiotemporal_rec/) · [KDD 2024 (1)](../../../KDD/2024/poi_spatiotemporal_rec/) · [KDD 2025 (1)](../../../KDD/2025/poi_spatiotemporal_rec/) · [KDD 2026 (3)](../../../KDD/2026/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/poi_spatiotemporal_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/poi_spatiotemporal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/poi_spatiotemporal_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端生成式&quot;,&quot;行程推荐&quot;,&quot;重复抑制&quot;,&quot;解码采样策略&quot;]"><strong><a href="analyzing_and_mitigating_repetitions_in_trip_recommendation/">Analyzing and Mitigating Repetitions in Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次剖析行程推荐中POI重复生成的两大成因，提出循环感知预测器（引导矩阵+漂移惩罚+置信度自适应采样）在Flickr四城上兼顾精度与去重。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;异构图基准数据集&quot;,&quot;选址推荐&quot;,&quot;全库排序评测&quot;]"><strong><a href="opensiterec_an_open_dataset_for_site_recommendation/">OpenSiteRec: An Open Dataset for Site Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个公开选址推荐数据集，用 5 类实体 9 类关系的异构图刻画 4 座国际都市，并在 5-core + 全库排序协议下基准测试 16 个推荐模型。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;生成式端到端&quot;,&quot;语义ID量化&quot;,&quot;强化微调GRPO&quot;,&quot;熵坍缩&quot;]"><strong><a href="refine_poi_reinforcement_fine_tuned_large_language_models_for_next_point_of/">Refine-POI: Reinforcement Fine-Tuned Large Language Models for Next Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用层次自组织映射生成拓扑保序语义ID，并以GRPO列表级奖励替代SFT标签模仿，配合KL-Cov熵正则恢复有据推理，Acc@5最高提升12.1%。</p>
</dd>
</dl>
