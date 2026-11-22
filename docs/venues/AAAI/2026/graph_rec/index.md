---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**AAAI (RecSys Track) 2026** · **3** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2021 (2)](../../2021/graph_rec/) · [AAAI (RecSys Track) 2022 (1)](../../2022/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../2024/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/graph_rec/) · [CIKM 2022 (5)](../../../CIKM/2022/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2021 (1)](../../../DASFAA/2021/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2021 (2)](../../../ICDE/2021/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/graph_rec/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/graph_rec/) · [KDD 2022 (1)](../../../KDD/2022/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/graph_rec/) · [RecSys 2022 (1)](../../../RecSys/2022/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/graph_rec/) · [SIGIR 2022 (7)](../../../SIGIR/2022/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/graph_rec/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/graph_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (3)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/graph_rec/) · [WSDM 2022 (7)](../../../WSDM/2022/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2021 (4)](../../../WWW/2021/graph_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/graph_rec/)

🔥 **高频主题：** AAAI 2026 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;检索增强生成&quot;,&quot;图注意力网络&quot;,&quot;空间相关性回灌&quot;]"><strong><a href="graph_neural_field_with_spatial_correlation_augmentation_for_hrtf_personali/">Graph Neural Field with Spatial-Correlation Augmentation for HRTF Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以受试者检索图 + GAT 编解码生成个性化 HRTF，再用方向图 GNN 做空间一致性回灌，3 方向测量下 LSD 降至 3.60 dB。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;端到端序列决策&quot;,&quot;学习路径推荐&quot;,&quot;GraphRAG&quot;,&quot;判别学习&quot;]"><strong><a href="graphrag_induced_dual_knowledge_structure_graphs_for_personalized_learning_/">GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 TextGrad+EDU-GraphRAG 自动合成先决/相似双知识结构图，再用 PPO 三智能体生成学习路径，仿真协议下最长路径提升约 29%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;动态图推荐&quot;,&quot;任务感知子图检索&quot;]"><strong><a href="task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
</dl>
