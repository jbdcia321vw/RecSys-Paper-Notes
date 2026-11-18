# 🕸️ 图神经网络与知识图谱推荐

**ACM TOIS 2026** · **2** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [ACM TOIS 2025 (2)](../../2025/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

🔥 **高频主题：** TOIS ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 表征增强" aria-pressed="false">LLM 表征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络推荐" aria-pressed="false">图神经网络推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相似就诊检索" aria-pressed="false">相似就诊检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="符号边重加权" aria-pressed="false">符号边重加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精度-多样性权衡" aria-pressed="false">精度-多样性权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图对比学习" aria-pressed="false">超图对比学习<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;符号边重加权&quot;,&quot;LLM 表征增强&quot;,&quot;精度-多样性权衡&quot;]"><strong><a href="cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图对比学习&quot;,&quot;相似就诊检索&quot;]"><strong><a href="hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
</dl>
