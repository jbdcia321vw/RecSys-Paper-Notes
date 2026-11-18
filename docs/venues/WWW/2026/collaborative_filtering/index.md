---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**WWW (TheWebConf) 2026** · **5** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2025 (7)](../../2025/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/)

🔥 **高频主题：** WWW 2026 ×4

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Generate-then-Rank" aria-pressed="false">Generate-then-Rank<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Item-based CF" aria-pressed="false">Item-based CF<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="k-DPP 多样性采样" aria-pressed="false">k-DPP 多样性采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Target-Aware 蒸馏" aria-pressed="false">Target-Aware 蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Top-K 损失函数" aria-pressed="false">Top-K 损失函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Trigger-Target 负采样" aria-pressed="false">Trigger-Target 负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分位数回归" aria-pressed="false">分位数回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布鲁棒优化" aria-pressed="false">分布鲁棒优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分解因果自注意力" aria-pressed="false">分解因果自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="渐进剪枝" aria-pressed="false">渐进剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏嵌入" aria-pressed="false">稀疏嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;k-DPP 多样性采样&quot;]"><strong><a href="diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;稀疏嵌入&quot;,&quot;渐进剪枝&quot;,&quot;线性自编码器&quot;]"><strong><a href="efficient_learning_of_sparse_representations_from_interactions/">Efficient Learning of Sparse Representations from Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 ELSA 线性自编码器训练中渐进剪枝学习高维稀疏物品嵌入，实现最高 100× 压缩仅损 2.5% nDCG，并派生可解释语义分段。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Target-Aware 蒸馏&quot;,&quot;Generate-then-Rank&quot;,&quot;分解因果自注意力&quot;]"><strong><a href="grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Item-based CF&quot;,&quot;Trigger-Target 负采样&quot;,&quot;目标注意力&quot;]"><strong><a href="pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Top-K 损失函数&quot;,&quot;分位数回归&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
</dl>
  </div>
</div>
