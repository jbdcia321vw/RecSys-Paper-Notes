---
hide:
  - toc
---

# 🏛️ ICDE 2024 推荐系统论文解读

14 篇 ICDE 2024 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 联邦推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/排序&quot;,&quot;Softmax Loss&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Bilateral Softmax Loss&quot;]"><strong><a href="collaborative_filtering/bsl_understanding_and_improving_softmax_loss_for_recommendation/">BSL: Understanding and Improving Softmax Loss for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用KL-DRO解释Softmax Loss的负样本鲁棒性与方差公平性，提出正负双侧Log-Expectation-Exp的BSL，一行代码在四数据集三骨干上提升Recall/NDCG。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;k-DPP&quot;,&quot;集合级排序准则&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="collaborative_filtering/learning_k_determinantal_point_processes_for_personalized_ranking/">Learning k-Determinantal Point Processes for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 k-DPP 集合概率比较替代 BPR 的点对比较，把相关性排序与多样性排序统一进同一优化准则，全库排序下 Recall@5 相对最优基线提升 8–11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;子图学习&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="graph_rec/knowledge_enhanced_recommendation_with_user_centric_subgraph_network/">Knowledge-Enhanced Recommendation with User-Centric Subgraph Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KUCNet 为每个用户-物品对构建U-I子图，用注意力GNN与PPR剪枝的用户中心计算图实现个性化、归纳式推荐，全库排序下Last-FM Recall@20较KGIN提升23.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;Graph Information Bottleneck&quot;,&quot;Mixhop 图对比学习&quot;]"><strong><a href="self_supervised_denoising/graph_augmentation_for_recommendation/">Graph Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 GIB 正则化的可学习图增强器生成去噪对比视图，配合 Mixhop 混阶编码器缓解过平滑，Recall@20 较 NCL 提升 5.9%–8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;生成式视图增强&quot;,&quot;变分自编码器&quot;,&quot;元优化两步训练&quot;]"><strong><a href="self_supervised_denoising/meta_optimized_joint_generative_and_contrastive_learning_for_sequential_rec/">Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 VAE Seq2Seq 生成器替代手工增强生成对比视图，双 ELBO 导出互信息项并落地 InfoNCE，元优化两步训练自适应视图，全库排序 HR@10 最高提升 20.8%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;序列推荐&quot;,&quot;序列去噪&quot;,&quot;自增强&quot;,&quot;全局关系编码&quot;]"><strong><a href="self_supervised_denoising/ssdrec_self_augmented_sequence_denoising_for_sequential_recommendation/">SSDRec: Self-Augmented Sequence Denoising for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ICDE 2024 自监督交互去噪 | 序列增强后再去噪以缓解过度/不足去噪(OUPs)，Full-Ranking 提升 SASRec 平均 23%–110%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;文本嵌入白化&quot;,&quot;冷启动推荐&quot;]"><strong><a href="sequential_rec/are_id_embeddings_necessary_whitening_pre_trained_text_embeddings_for_effec/">Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings for Effective Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出ZCA白化预训练文本嵌入并融合全/松弛白化表示，在4个Amazon/食谱数据集全库排序上超越ID+文本基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式&quot;,&quot;语义ID量化&quot;,&quot;LLM指令微调&quot;,&quot;生成式检索&quot;]"><strong><a href="generative_rec/adapting_large_language_models_by_integrating_collaborative_semantics_for_r/">Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RQ-VAE 层级语义索引 + Sinkhorn 均匀映射消除索引冲突，LLaMA-7B 多任务对齐微调实现全库生成式检索，全排序平均提升 25.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;大语言模型推荐&quot;,&quot;召回/表征增强&quot;,&quot;LLM 图增强&quot;,&quot;去噪鲁棒训练&quot;]"><strong><a href="llm_rec/llmrec_large_language_models_with_graph_augmentation_for_recommendation/">LLMRec: Large Language Models with Graph Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>LLM 从自然语言侧增强 U-I 交互边、物品属性与用户画像，并以噪声剪枝+MAE 鲁棒化，全库排序 R@10 提升 13.95%/4.88%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跨模态对比学习&quot;,&quot;迁移推荐&quot;]"><strong><a href="multimodal_rec/multi_modality_is_all_you_need_for_transferable_recommender_systems/">Multi-Modality is All You Need for Transferable Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出纯多模态可迁移序列推荐 PMMRec，以跨模态下一项对齐和自监督去噪，在 4 源 10 目标数据集全库排序上最高提升 34.77%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;严格冷启动&quot;,&quot;冻结图传播&quot;,&quot;多模态知识图谱&quot;]"><strong><a href="cold_start_rec/firzen_firing_strict_cold_start_items_with_frozen_heterogeneous_and_homogen/">Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用冻结协作知识图与冻结同构图将热启动物品的协同信号单向传播给严格冷启动物品，全库排序下 Beauty 冷启动 M@20 提升 22%，热启动持平 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端联邦训练&quot;,&quot;联邦推荐&quot;,&quot;模型异构&quot;,&quot;知识蒸馏&quot;]"><strong><a href="debias_fairness_industrial/hetefedrec_federated_recommender_systems_with_model_heterogeneity/">HeteFedRec: Federated Recommender Systems with Model Heterogeneity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次让联邦推荐客户端训练不同维度物品嵌入，用填充聚合、双任务学习、去相关正则和关系蒸馏融合知识，采样NDCG@20提升1.9%–20.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练&quot;,&quot;联邦推荐&quot;,&quot;模型知识产权&quot;,&quot;预测蒸馏&quot;]"><strong><a href="debias_fairness_industrial/hide_your_model_a_parameter_transmission_free_federated_recommender_system/">Hide Your Model: A Parameter Transmission-free Federated Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出免参数传输联邦推荐框架，仅交换扰动预测分数，兼顾平台模型知识产权与用户数据隐私，通信降至KB级。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦推荐安全/训练阶段&quot;,&quot;投毒攻击&quot;,&quot;流行物品嵌入&quot;]"><strong><a href="debias_fairness_industrial/preventing_the_popular_item_embedding_based_attack_in_federated_recommendat/">Preventing the Popular Item Embedding Based Attack in Federated Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出模型无关、无需先验的联邦推荐投毒攻击 PIECK 与双正则防御，全库排序下 ER@10 显著提升且 HR 稳定。</p>
</dd>
</dl>
</section>
</div>
