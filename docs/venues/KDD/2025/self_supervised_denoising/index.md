# 🔄 自监督对比学习与交互去噪

**KDD 2025** · **3** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [KDD 2026 (1)](../../2026/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="FastSHAP" aria-pressed="false">FastSHAP<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Shapley Value" aria-pressed="false">Shapley Value<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交互价值评估" aria-pressed="false">交互价值评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务对比学习" aria-pressed="false">多任务对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大规模图引擎" aria-pressed="false">大规模图引擎<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构图表征" aria-pressed="false">异构图表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据剪枝/重训" aria-pressed="false">数据剪枝/重训<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离散码向量量化" aria-pressed="false">离散码向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="虚拟邻居增强" aria-pressed="false">虚拟邻居增强<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;离散码向量量化&quot;,&quot;虚拟邻居增强&quot;]"><strong><a href="enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;异构图表征&quot;,&quot;多任务对比学习&quot;,&quot;大规模图引擎&quot;]"><strong><a href="omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;数据剪枝/重训&quot;,&quot;Shapley Value&quot;,&quot;FastSHAP&quot;,&quot;交互价值评估&quot;]"><strong><a href="shapley_value_driven_data_pruning_for_recommender_systems/">Shapley Value-driven Data Pruning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 FastSHAP 实时估计交互对训练损失的边际贡献，按 Shapley 值剪枝低效用交互并重训，四数据集 Recall/NDCG 平均提升 3.72%。</p>
</dd>
</dl>
