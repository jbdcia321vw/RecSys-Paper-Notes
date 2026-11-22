---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**WSDM 2022** · **7** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [WSDM 2021 (1)](../../2021/graph_rec/) · [WSDM 2023 (2)](../../2023/graph_rec/) · [WSDM 2024 (3)](../../2024/graph_rec/) · [WSDM 2025 (3)](../../2025/graph_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/graph_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/graph_rec/) · [CIKM 2022 (5)](../../../CIKM/2022/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2021 (1)](../../../DASFAA/2021/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [DASFAA 2026 (2)](../../../DASFAA/2026/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2021 (2)](../../../ICDE/2021/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/graph_rec/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/graph_rec/) · [KDD 2022 (1)](../../../KDD/2022/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/graph_rec/) · [RecSys 2022 (1)](../../../RecSys/2022/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2021 (3)](../../../SIGIR/2021/graph_rec/) · [SIGIR 2022 (7)](../../../SIGIR/2022/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/graph_rec/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/graph_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (3)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WWW (TheWebConf) 2021 (4)](../../../WWW/2021/graph_rec/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/graph_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">6</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图表征学习&quot;,&quot;分布式训练系统&quot;]"><strong><a href="graph4rec_a_universal_toolkit_with_graph_neural_networks_for_recommender_sy/">Graph4Rec: A Universal Toolkit with Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 GNN4Rec 训练统一为五段式流水线，配分布式图引擎与参数服务器，在 ICF/UCF/U2I 召回协议下系统对比 8 类 GNN 并给出工程实践建议。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2020&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;离散哈希码学习&quot;,&quot;直通估计器STE&quot;]"><strong><a href="learning_to_hash_with_graph_neural_networks_for_recommender_systems/">Learning to Hash with Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HashGNN 端到端联合学习 GNN 连续表征与二值哈希码，融合链接重建与 Hamming 排序损失，并以 Guidance STE 消除梯度放大；全库排序 HR@50 较 GraphSage 提升 4–10%，检索提速 2.5–3.6 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;洛伦兹双曲几何&quot;,&quot;知识图谱推荐&quot;,&quot;注意力聚合&quot;]"><strong><a href="modeling_scale_free_graphs_with_hyperbolic_geometry_for_knowledge_aware_rec/">Modeling Scale-free Graphs with Hyperbolic Geometry for Knowledge-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户-物品-KG 三分图嵌入洛伦兹双曲流形，用异构解耦传播与知识感知注意力建模无标度层级结构，Full-Ranking 下 Recall@20 提升 3.6–15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;跨层互关系注意力&quot;]"><strong><a href="multi_behavior_graph_neural_networks_for_recommender_system/">Multi-Behavior Graph Neural Networks for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MBRec 以多通道行为编码、跨行为自注意力与跨层互关系融合建模多类型交互，三数据 1:99 采样排序下 HR@10 超最强基线 4.7%–6.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;贝叶斯图神经网络&quot;,&quot;图生成式数据增强&quot;]"><strong><a href="preference_and_concurrence_aware_bayesian_graph_neural_networks_for_recomme/">Preference and Concurrence Aware Bayesian Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用用户偏好、物品共现与度约束的 PECO 图生成器增强 BGNN，四数据集 Recall@20 较 Node-Copy 最高提升 1.44%。</p>
</dd>
<dt data-paper-tags="[&quot;ISWC 2020&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/CTR 打分&quot;,&quot;规则学习 (Rule Learning)&quot;,&quot;知识图谱嵌入规则过滤 (RotatE)&quot;,&quot;冷启动推荐&quot;]"><strong><a href="rule_guided_graph_neural_networks_for_recommender_systems/">Rule-Guided Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用自动规则学习为 KG 上的 GraphSAGE 式聚合提供显式长程语义与采样先验，Dianping-Food AUC 相对最优基线提升 4.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;场景图&quot;,&quot;层次图注意力&quot;,&quot;BPR&quot;]"><strong><a href="scenerec_scene_based_graph_neural_networks_for_recommender_systems/">SceneRec: Scene-Based Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>把场景形式化为共现类目集合，构建物品-类目-场景三层图并用场景注意力聚合物品表征，JD 四数据集 NDCG@10 提升 5.9%–25.6%。</p>
</dd>
</dl>
