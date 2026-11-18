# 🤝 协同过滤与双塔召回

**SIGIR 2026** · **6** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [SIGIR 2025 (3)](../../2025/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Bloom Index 特征过滤" aria-pressed="false">Bloom Index 特征过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GPU 向量检索" aria-pressed="false">GPU 向量检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推荐" aria-pressed="false">免训练推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔检索" aria-pressed="false">双塔检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双意图空间表征重构" aria-pressed="false">双意图空间表征重构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线学习" aria-pressed="false">在线学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐-均匀性优化" aria-pressed="false">对齐-均匀性优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入坍缩" aria-pressed="false">嵌入坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间衰减打分" aria-pressed="false">时间衰减打分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型化服务架构" aria-pressed="false">模型化服务架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="正交约束" aria-pressed="false">正交约束<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相关性蒸馏" aria-pressed="false">相关性蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏缩放" aria-pressed="false">稀疏缩放<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负样本挖掘" aria-pressed="false">难负样本挖掘<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔模型&quot;,&quot;在线学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双意图空间表征重构&quot;,&quot;对齐-均匀性优化&quot;]"><strong><a href="diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以原型意图与分布意图重构表征，用对齐-均匀性+粗细粒度匹配联合优化，Amazon-book R@20 相对提升 10.6%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正交约束&quot;,&quot;嵌入坍缩&quot;,&quot;稀疏缩放&quot;]"><strong><a href="ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 Item-ID 投影层上施加 Stiefel 流形正交约束，抑制稀疏缩放下的嵌入坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;GPU 向量检索&quot;,&quot;Bloom Index 特征过滤&quot;,&quot;模型化服务架构&quot;]"><strong><a href="silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom Index 与 Int8 融合 ANN 协同设计，吞吐提升 23.7×、成本效率提升 13.35×。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;相关性蒸馏&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;免训练推荐&quot;,&quot;时间衰减打分&quot;,&quot;新闻推荐&quot;]"><strong><a href="zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
</dl>
