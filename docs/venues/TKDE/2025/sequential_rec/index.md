# ⏱️ 序列与会话推荐

**IEEE TKDE 2025** · **3** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [IEEE TKDE 2026 (2)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** TKDE ×3 · 可解释推荐 ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪先验物品生成" aria-pressed="false">伪先验物品生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层强化学习" aria-pressed="false">分层强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向时序数据增强" aria-pressed="false">双向时序数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="概率嵌入" aria-pressed="false">概率嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱路径推理" aria-pressed="false">知识图谱路径推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识增强微调" aria-pressed="false">知识增强微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逻辑合取算子" aria-pressed="false">逻辑合取算子<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;知识图谱路径推理&quot;,&quot;分层强化学习&quot;,&quot;可解释推荐&quot;]"><strong><a href="explainable_session_based_recommendation_via_path_reasoning/">Explainable Session-based Recommendation via Path Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出分层强化学习框架 PR4SR，用会话级智能体选起点、路径级智能体做知识图谱推理，配合多目标与路径中点奖励，为 5 种 SR 模型提供可解释路径并提升精度。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向时序数据增强&quot;,&quot;伪先验物品生成&quot;,&quot;知识增强微调&quot;]"><strong><a href="improving_sequential_recommendations_via_bidirectional_temporal_data_augmen/">Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>BARec 用双向时序增强预训练生成保偏好伪先验物品，并以 KL 知识增强微调对齐原/增广序列分布，全库排序下 Beauty Recall@5 提升 16.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;概率嵌入&quot;,&quot;逻辑合取算子&quot;,&quot;可解释推荐&quot;]"><strong><a href="pattern_wise_transparent_sequential_recommendation/">Pattern-wise Transparent Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多尺度滑窗切出点级/联合级模式，以概率嵌入+逻辑合取建模模式，KL散度+距离权重实现可解释且超越黑盒的序列推荐。</p>
</dd>
</dl>
