---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪

**SIGIR 2023** · **6** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [SIGIR 2021 (2)](../../2021/self_supervised_denoising/) · [SIGIR 2022 (4)](../../2022/self_supervised_denoising/) · [SIGIR 2024 (1)](../../2024/self_supervised_denoising/) · [SIGIR 2025 (1)](../../2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../2026/self_supervised_denoising/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/self_supervised_denoising/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/self_supervised_denoising/) · [CIKM 2021 (1)](../../../CIKM/2021/self_supervised_denoising/) · [CIKM 2022 (2)](../../../CIKM/2022/self_supervised_denoising/) · [CIKM 2023 (2)](../../../CIKM/2023/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [CIKM 2026 (1)](../../../CIKM/2026/self_supervised_denoising/) · [DASFAA 2022 (1)](../../../DASFAA/2022/self_supervised_denoising/) · [DASFAA 2023 (2)](../../../DASFAA/2023/self_supervised_denoising/) · [DASFAA 2026 (1)](../../../DASFAA/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICDE 2022 (1)](../../../ICDE/2022/self_supervised_denoising/) · [ICDE 2023 (1)](../../../ICDE/2023/self_supervised_denoising/) · [ICDE 2024 (3)](../../../ICDE/2024/self_supervised_denoising/) · [ICDM 2021 (1)](../../../ICDM/2021/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/self_supervised_denoising/) · [KDD 2021 (1)](../../../KDD/2021/self_supervised_denoising/) · [KDD 2022 (1)](../../../KDD/2022/self_supervised_denoising/) · [KDD 2023 (2)](../../../KDD/2023/self_supervised_denoising/) · [KDD 2024 (3)](../../../KDD/2024/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2023 (1)](../../../RecSys/2023/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/self_supervised_denoising/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/self_supervised_denoising/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/self_supervised_denoising/) · [ACM TORS 2023 (1)](../../../TORS/2023/self_supervised_denoising/) · [WSDM 2021 (1)](../../../WSDM/2021/self_supervised_denoising/) · [WSDM 2022 (1)](../../../WSDM/2022/self_supervised_denoising/) · [WSDM 2023 (1)](../../../WSDM/2023/self_supervised_denoising/) · [WSDM 2024 (1)](../../../WSDM/2024/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2022 (2)](../../../WWW/2022/self_supervised_denoising/) · [WWW (TheWebConf) 2023 (4)](../../../WWW/2023/self_supervised_denoising/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/self_supervised_denoising/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;方向性扩散噪声&quot;,&quot;线性图 Transformer&quot;]"><strong><a href="a_directional_diffusion_graph_transformer_for_recommendation/">A Directional Diffusion Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以各向异性方向高斯噪声替代等方向噪声做嵌入级扩散，并用条件线性 Transformer 反向去噪，全库排序 Recall@20 相对 SOTA 提升约 3–4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回（全库排序）&quot;,&quot;变分图重构&quot;,&quot;聚类感知对比学习&quot;]"><strong><a href="generative_contrastive_graph_learning_for_recommendation/">Generative-Contrastive Graph Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以变分图重构采样替代数据增广生成对比视图，联合节点/簇级对比，在三个数据集 Full-Ranking 下 NDCG@20 较 LightGCN 提升 8.61%–28.17%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;Graph Transformer&quot;,&quot;Masked Autoencoding&quot;,&quot;Rationale Discovery&quot;]"><strong><a href="graph_transformer_for_recommendation/">Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GFormer 用拓扑感知图 Transformer 参数化发现协同 rationale 子图，并以 rationale-aware 掩码自编码替代随机对比增广；全库排序下 Recall@20 较 SGL 提升 5.2%–7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为对齐&quot;,&quot;KL散度去噪&quot;,&quot;隐变量推断&quot;]"><strong><a href="improving_implicit_feedback_based_recommendation_through_multi_behavior_ali/">Improving Implicit Feedback-Based Recommendation through Multi-Behavior Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户真实偏好设为隐变量，用 KL 散度对齐点击/购买两条行为链并交替去噪，三数据集 Recall@20 相对最优基线提升 0.5%–6.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与排序表征学习&quot;,&quot;知识图谱剪枝&quot;,&quot;对比去噪&quot;,&quot;伯努利掩码&quot;]"><strong><a href="knowledge_refined_denoising_network_for_robust_recommendation/">Knowledge-refined Denoising Network for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KRDN 用伯努利掩码剪枝 KG 任务无关三元组，并以协同/知识信号分歧对比式剔除噪声交互；全库排序下 NDCG@20 较 SOTA 提升 4.6%–12.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;梯度冲突矫正&quot;]"><strong><a href="multi_behavior_self_supervised_learning_for_recommendation/">Multi-behavior Self-supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为感知GNN+双层自判别对比+梯度混合矫正，Leave-One-Out全库排序下Recall@10平均提升约19%</p>
</dd>
</dl>
