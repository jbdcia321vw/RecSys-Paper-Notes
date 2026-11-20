---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐

**SIGIR 2024** · **8** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [SIGIR 2023 (4)](../../2023/graph_rec/) · [SIGIR 2025 (2)](../../2025/graph_rec/) · [SIGIR 2026 (3)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/graph_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/graph_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ECIR 2023 (3)](../../../ECIR/2023/graph_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/graph_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICDM 2023 (2)](../../../ICDM/2023/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/graph_rec/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/graph_rec/) · [KDD 2023 (3)](../../../KDD/2023/graph_rec/) · [KDD 2024 (5)](../../../KDD/2024/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/graph_rec/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/graph_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/graph_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/graph_rec/) · [ACM TORS 2024 (1)](../../../TORS/2024/graph_rec/) · [ACM TWEB 2023 (1)](../../../TWEB/2023/graph_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/graph_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/graph_rec/) · [WWW (TheWebConf) 2024 (7)](../../../WWW/2024/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

🔥 **高频主题：** 结构位置编码 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构位置编码" aria-pressed="false">结构位置编码<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;特征过相关&quot;,&quot;过平滑&quot;,&quot;自适应正则&quot;]"><strong><a href="afdgcf_adaptive_feature_de_correlation_graph_collaborative_filtering_for_re/">AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 GNN-CF 中被忽视的“特征过相关”问题并证明其与过平滑成正比，提出层级自适应的列相关惩罚正则，模型无关地提升 4 类骨干模型最多 14.98%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;持续图学习&quot;,&quot;图提示微调&quot;,&quot;流式推荐&quot;]"><strong><a href="gpt4rec_graph_prompt_tuning_for_streaming_recommendation/">GPT4Rec: Graph Prompt Tuning for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将演化交互图解耦为多视图，用节点/结构/视图三级轻量图提示替代回放与模型扩张，实现流式推荐持续学习，R@20 最高相对提升 4.16%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;参数高效嵌入&quot;,&quot;元嵌入组合&quot;,&quot;矩阵伪逆闭式更新&quot;]"><strong><a href="lightweight_embeddings_for_graph_collaborative_filtering/">Lightweight Embeddings for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可学习稀疏分配矩阵替代固定哈希映射，在扩展图上联合优化元嵌入码本，实现 44× 参数压缩下接近全量嵌入的精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;线性化核注意力&quot;,&quot;结构位置编码&quot;,&quot;度数重加权掩码&quot;]"><strong><a href="masked_graph_transformer_for_large_scale_recommendation/">Masked Graph Transformer for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>纯 Transformer 全节点注意力 + SimRFs 线性核 + 可学习相对度数掩码，Full-Ranking 下 Recall@20 最高提升 8.62%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;负反馈感知超图卷积&quot;,&quot;Transformer Hawkes Process&quot;]"><strong><a href="nfarec_a_negative_feedback_aware_recommender_model/">NFARec: A Negative Feedback-Aware Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以反馈极性引导超图卷积消息通路并用 Transformer Hawkes 过程辅助预测下一交互情感极性，五数据集全库排序 R@20 最高提升 6.6%、N@20 提升 14.2%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表示学习&quot;,&quot;符号图&quot;,&quot;图Transformer&quot;]"><strong><a href="sigformer_sign_aware_graph_transformer_for_recommendation/">SIGformer: Sign-aware Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用Transformer直接建模整张符号图，以符号谱/路径编码融合正负反馈，五数据集Recall@20最高提升15.61%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;注意力采样&quot;,&quot;结构位置编码&quot;]"><strong><a href="transgnn_harnessing_the_collaborative_power_of_transformers_and_graph_neura/">TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>交替堆叠 Transformer 与 GNN 层，用注意力采样+三种结构位置编码互相扩感受野与注入拓扑，全库排序下 Recall@20 最高提升 21%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多项式图滤波&quot;,&quot;免训练协同过滤&quot;,&quot;低通滤波&quot;]"><strong><a href="turbo_cf_matrix_decomposition_free_graph_filtering_for_fast_recommendation/">Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式图滤波器替代特征分解实现免训练免分解的协同过滤，全库排序下精度持平SOTA而耗时降至亚秒级。</p>
</dd>
</dl>
