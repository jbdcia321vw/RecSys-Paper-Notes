# ⚖️ 多任务学习与列表级重排

**CIKM 2026** · **5** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [CIKM 2025 (4)](../../2025/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 多任务学习 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Anchor Token" aria-pressed="false">Anchor Token<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Ranking Transformer" aria-pressed="false">Ranking Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务依赖图" aria-pressed="false">任务依赖图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务边界掩码" aria-pressed="false">任务边界掩码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩图生成" aria-pressed="false">低秩图生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可控多目标推荐" aria-pressed="false">可控多目标推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态证据累积" aria-pressed="false">多模态证据累积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列列表重排" aria-pressed="false">序列列表重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型合并" aria-pressed="false">模型合并<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式预训练" aria-pressed="false">生成式预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线策略优化" aria-pressed="false">离线策略优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏MoE" aria-pressed="false">稀疏MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一主干" aria-pressed="false">统一主干<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应渐进掩码" aria-pressed="false">自适应渐进掩码<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;任务依赖图&quot;,&quot;低秩图生成&quot;,&quot;自适应渐进掩码&quot;]"><strong><a href="personalized_task_dependency_graphs_for_mitigating_signal_erosion_in_multi_/">Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用低秩个性化任务依赖图+硬因果掩码GCN替代刚性漏斗，缓解深层稀疏任务信号侵蚀，离线AUC最高+1.45%，线上CVR+1.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;模型合并&quot;,&quot;可控多目标推荐&quot;]"><strong><a href="recsoup_efficient_model_merging_approach_for_controllable_multi_objective_r/">RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用剪枝+可学习门控做任务向量合并，实现推理时可控多目标推荐，HV 提升 8.5% 且延迟快 165–228 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;序列列表重排&quot;,&quot;离线策略优化&quot;,&quot;多模态证据累积&quot;]"><strong><a href="sequential_multimodal_evidence_optimization_for_product_media_ranking_in_e_/">Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将商品媒体排序建模为证据累积序列决策，两阶段解耦效用学习与生存加权自回归策略，离线 DR-CVR +5.4%、MSC −15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Ranking Transformer&quot;,&quot;稀疏MoE&quot;,&quot;生成式预训练&quot;]"><strong><a href="sort_a_systematically_optimized_ranking_transformer_for_industrial_scale_re/">SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以请求为中心组织样本、局部注意力+查询剪枝、生成式预训练与稀疏MoE重构工业精排Transformer，线上GMV+8.65%且延迟降62%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Anchor Token&quot;,&quot;任务边界掩码&quot;,&quot;统一主干&quot;]"><strong><a href="task_blind_no_more_multi_task_information_flow_in_unified_ranking_backbones/">Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多任务信息流下沉至统一排序主干，以共享/私有 Anchor Token 逐层共演化，离线七任务全胜、线上时长+3.0%、P99 延迟降约30%。</p>
</dd>
</dl>
