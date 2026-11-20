---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**IEEE TKDE 2023** · **4** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [IEEE TKDE 2024 (2)](../../2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../2025/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/graph_rec/) · [SIGIR 2024 (8)](../../../SIGIR/2024/graph_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/graph_rec/) · [SIGIR 2026 (3)](../../../SIGIR/2026/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

🔥 **高频主题：** TKDE ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;反事实去偏&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="causal_inference_for_knowledge_graph_based_recommendation/">Causal Inference for Knowledge Graph based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用因果图拆解 KG 推荐中结构混淆与属性捷径偏差，以 do 干预+反事实推断实现去偏属性偏好建模，Full-Ranking 下 Recall@20 提升 3.79%~6.66%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱注意力&quot;,&quot;偏置随机游走&quot;,&quot;门控上下文融合&quot;]"><strong><a href="contextualized_graph_attention_network_for_recommendation_with_item_knowled/">Contextualized Graph Attention Network for Recommendation with Item Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户特定图注意力聚合 KG 局部邻居、偏置随机游走+GRU 抽取非局部高阶上下文、物品特定注意力建模历史交互；FM/ML/BC 6:2:2 划分全库排序下 HR@20 较最强基线提升 8.3%/1.0%/5.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;哈希学习 (Learning to Hash)&quot;,&quot;汉明空间图卷积&quot;,&quot;高阶相似度传播&quot;]"><strong><a href="hs_gcn_hamming_spatial_graph_convolutional_networks_for_recommendation/">HS-GCN: Hamming Spatial Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在汉明空间直接做图卷积传播，显式建模一阶与高阶汉明相似度，7 个数据集全库排序 HR@50 平均超最强哈希基线 26.2%、逼近实值 GNN 且训练/推理提速 2.4×/4.9×。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;评论图构建&quot;,&quot;个性化图池化&quot;,&quot;类型感知图注意力&quot;]"><strong><a href="learning_hierarchical_review_graph_representations_for_recommendation/">Learning Hierarchical Review Graph Representations for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为每个用户/物品单独构建带词序边类型的评论图，用类型感知图注意力与个性化图池化学习层级语义表征，6 个 Amazon/Yelp 数据集上 MSE 较最强基线 DAML 平均再降 1.14%</p>
</dd>
</dl>
