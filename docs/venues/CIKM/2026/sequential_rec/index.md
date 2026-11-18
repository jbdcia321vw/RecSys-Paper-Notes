# ⏱️ 序列与会话推荐

**CIKM 2026** · **6** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [CIKM 2025 (6)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** MoE ×2 · 序列推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义融合" aria-pressed="false">LLM 语义融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效短视适配" aria-pressed="false">参数高效短视适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图谱结构信号" aria-pressed="false">图谱结构信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比蒸馏" aria-pressed="false">对比蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="归一化流" aria-pressed="false">归一化流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拉普拉斯位置编码" aria-pressed="false">拉普拉斯位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件变分自编码器" aria-pressed="false">条件变分自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力校准" aria-pressed="false">注意力校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组件级嫁接" aria-pressed="false">组件级嫁接<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为线索路由" aria-pressed="false">行为线索路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="角度相似度打分" aria-pressed="false">角度相似度打分<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;角度相似度打分&quot;,&quot;参数高效短视适配&quot;]"><strong><a href="closing_the_long_short_view_gap_in_sequential_recommendation_without_cached/">Closing the Long-Short View Gap in Sequential Recommendation without Cached History</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用角度相似度打分+softmax1 修正长视训练几何缺陷，再仅微调 Bias/LayerNorm 实现无缓存短视推理，短窗下反超全序列基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;拉普拉斯位置编码&quot;,&quot;图谱结构信号&quot;]"><strong><a href="enriching_sequential_recommendation_with_graph_laplacian_positional_embeddi/">Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用物品共现图归一化拉普拉斯特征向量冻结替换 SASRec 可学习位置嵌入，四基准多数指标提升且在线开销近零。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;注意力校准&quot;,&quot;MoE&quot;]"><strong><a href="from_overlooked_to_explored_recovering_item_relations_via_mixture_of_perspe/">From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示自注意力相似性偏置导致异构关系被系统性忽略，提出 K 个视角透镜（亲和/对比双视图）校准注意力，7 个基准全库排序平均提升 3–11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;组件级嫁接&quot;,&quot;对比蒸馏&quot;]"><strong><a href="god_enhancing_generalization_via_deep_grafting_for_sequential_recommendatio/">GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「嫁接」构造教师条件混合源模型，对学生嵌入表/编码器做组件级蒸馏，Full-Ranking 下最高相对提升 13.92%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;归一化流&quot;,&quot;条件变分自编码器&quot;,&quot;LLM 语义融合&quot;]"><strong><a href="lsf_sr_latent_semantic_fusion_for_sequential_recommendation_via_flow_based_/">LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用归一化流增强的 CVAE 将 LLM 语义作为条件概率融合进 ID 协同嵌入，Full-Ranking 下 Recall@20 最高提升 12.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;行为线索路由&quot;]"><strong><a href="routerec_behavior_guided_sparse_routing_for_sequential_recommendation/">RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用会话行为线索（节奏/焦点/记忆/流行度）驱动分层稀疏 MoE 路由，六数据集 18 指标平均排名 1.61 显著优于次优 4.11。</p>
</dd>
</dl>
