---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**WSDM 2026** · **3** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [WSDM 2023 (2)](../../2023/sequential_rec/) · [WSDM 2024 (3)](../../2024/sequential_rec/) · [WSDM 2025 (5)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性注意力&quot;,&quot;持续学习&quot;]"><strong><a href="capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;前瞻预测&quot;,&quot;直播推荐&quot;]"><strong><a href="foresight_prediction_enhanced_live_streaming_recommendation/">Foresight Prediction Enhanced Live-Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将直播切片经 LLM 语义量化成 Sid 序列，用 Transformer 编解码预测下一段内容，作为前瞻特征注入多任务精排，线上曝光 +0.46%、送礼 +2.48%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;侧信息融合&quot;,&quot;频域去噪&quot;,&quot;时间跨度划分&quot;]"><strong><a href="time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
</dd>
</dl>
