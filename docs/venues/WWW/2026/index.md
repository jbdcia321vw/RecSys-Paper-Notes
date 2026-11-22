---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2026 推荐系统论文解读

98 篇 WWW (TheWebConf) 2026 论文解读，覆盖 16 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** WWW 2026 ×85 · 生成式推荐 ×15 · WWW ×13 · 去偏公平与可信推荐 ×11 · 大语言模型推荐 ×10 · 多模态推荐 ×4 · 对比学习 ×4 · 冷启动 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">32</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">24</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏注意力" aria-pressed="false">稀疏注意力<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO" aria-pressed="false">GRPO<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;k-DPP 多样性采样&quot;]"><strong><a href="collaborative_filtering/diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;稀疏嵌入&quot;,&quot;渐进剪枝&quot;,&quot;线性自编码器&quot;]"><strong><a href="collaborative_filtering/efficient_learning_of_sparse_representations_from_interactions/">Efficient Learning of Sparse Representations from Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 ELSA 线性自编码器训练中渐进剪枝学习高维稀疏物品嵌入，实现最高 100× 压缩仅损 2.5% nDCG，并派生可解释语义分段。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图 ANN 检索&quot;,&quot;多关系度量学习&quot;,&quot;GPU 并行计算&quot;]"><strong><a href="collaborative_filtering/gpu_accelerated_multi_relational_parallel_graph_retrieval_for_web_scale_rec/">GPU-accelerated Multi-relational Parallel Graph Retrieval for Web-scale Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>百度 GMP-GR 用自协变损失把欧氏排序规律注入多行为 DNN 度量，并配 GPU 分层并行图检索，Cov 较 NANN 提升 20%+、吞吐翻倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Target-Aware 蒸馏&quot;,&quot;Generate-then-Rank&quot;,&quot;分解因果自注意力&quot;]"><strong><a href="collaborative_filtering/grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Item-based CF&quot;,&quot;Trigger-Target 负采样&quot;,&quot;目标注意力&quot;]"><strong><a href="collaborative_filtering/pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Top-K 损失函数&quot;,&quot;分位数回归&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="collaborative_filtering/talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;]"><strong><a href="graph_rec/topkgat_a_top_k_objective_driven_architecture_for_recommendation/">TopKGAT: A Top-K Objective-Driven Architecture for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TopKGAT: A Top-K Objective-Driven Architecture for Recommendation</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图结构注入&quot;,&quot;对比学习&quot;,&quot;事件序列建模&quot;]"><strong><a href="self_supervised_denoising/beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;训练加速/数据侧&quot;,&quot;Coreset Selection&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="self_supervised_denoising/efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;扩散增广&quot;,&quot;非对称对比&quot;]"><strong><a href="self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;目标锚定对比学习&quot;,&quot;不变风险最小化&quot;]"><strong><a href="self_supervised_denoising/rmbrec_robust_multi_behavior_recommendation_towards_target_behaviors/">RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以目标行为为锚做用户侧对比对齐（RRM）+ 跨行为风险方差正则（ORM/REx），在 Taobao/TMall/Beibei 的 Full-Ranking 协议下 HR@10 最高提升 16.3%，且强抗噪。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长序列建模&quot;]"><strong><a href="sequential_rec/blossomrec_block_level_fused_sparse_attention_mechanism_for_sequential_reco/">BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双路块级稀疏注意力（LTIS 块选择 + STIS 幂律掩码）经门控融合，长序列训练提速 3.2×、显存降至 1/11，精度持平或超越 SASRec。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回 (Next-Item 全库排序)&quot;,&quot;推理轨迹&quot;,&quot;层次原型监督&quot;,&quot;自适应早停&quot;]"><strong><a href="sequential_rec/dtrec_learning_dynamic_reasoning_trajectories_for_sequential_recommendation/">DTRec: Learning Dynamic Reasoning Trajectories for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>层次原型软监督（由粗到细）+三指标自适应停机，让推理轨迹方向与深度双动态，NDCG@10 最高 +24.5%、推理步数省 41.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;LLM 语义嵌入&quot;,&quot;行为校准&quot;]"><strong><a href="sequential_rec/generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Legendre 正交多项式&quot;,&quot;Hyena 长卷积&quot;,&quot;次二次复杂度&quot;]"><strong><a href="sequential_rec/hyena_operator_for_fast_sequential_recommendation/">Hyena Operator for Fast Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Legendre 正交多项式参数化 Hyena 长卷积核，线性复杂度建模超长用户序列，XLong 上 NDCG@10 提升 17.65% 且训练最高 6× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;残差向量量化码本&quot;,&quot;多兴趣解耦学习&quot;,&quot;LLM 序列推荐&quot;]"><strong><a href="sequential_rec/hymirec_a_hybrid_multi_interest_learning_framework_for_llm_based_sequential/">HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差码本压缩长历史、轻量模型抽粗粒度多兴趣、LLM精炼兴趣，配合窗口目标+匈牙利匹配解耦多兴趣，全库排序下显著超越LLM基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;交叉注意力&quot;,&quot;长度外推&quot;]"><strong><a href="sequential_rec/make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;长序列建模&quot;]"><strong><a href="sequential_rec/mixture_of_sequence_theme_aware_mixture_of_experts_for_long_sequence_recomm/">Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示长序列“会话跳跃”现象，用主题码本路由+多尺度专家融合做子序列抽取，AUC 平均提升 0.68% 且 FLOPs 更低</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时序子图&quot;,&quot;兴趣演化建模&quot;]"><strong><a href="sequential_rec/modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多行为序列建模&quot;,&quot;图注意力&quot;]"><strong><a href="sequential_rec/multi_behavior_sequential_modeling_with_transition_aware_graph_attention_ne/">Multi-Behavior Sequential Modeling with Transition-Aware Graph Attention Network for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TGA 用 item/类目/邻居三视角稀疏转移图 + 行为对参数化图注意力替代 Transformer，线性复杂度建模多行为序列，Taobao AUC 0.7454 且训练推理快 5.8 倍，线上 CVR +1.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;冷启动基准&quot;]"><strong><a href="sequential_rec/vk_lsvd_a_large_scale_industrial_dataset_for_short_video_recommendation/">VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>发布 407 亿交互、千万用户、两千万短视频的工业级开放数据集，配全局时间切分与冷启动竞赛基准。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;会话层次建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="generative_rec/beyond_the_flat_sequence_hierarchical_and_preference_aware_generative_recom/">Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用会话层次预训练+偏好引导稀疏注意力打破 HSTU 扁平序列假设，APPGallery 全库 AUC 0.8929、线上 eCPM +1.99%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式召回&quot;,&quot;偏差放大&quot;,&quot;级联推理&quot;,&quot;语义 ID&quot;]"><strong><a href="generative_rec/bringing_reasoning_to_generative_recommendation_through_the_lens_of_cascade/">Bringing Reasoning to Generative Recommendation Through the Lens of Cascaded Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示生成式推荐的 token 级流行度偏差放大问题，用渐进历史编码与查询锚定并行推理实现去偏，在 TIGER/LETTER/SETRec 上同时提升准确率与多样性</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;生成式推荐&quot;,&quot;掩码离散扩散&quot;,&quot;语义ID并行解码&quot;]"><strong><a href="generative_rec/diffgrm_diffusion_based_generative_recommendation_model/">DiffGRM: Diffusion-based Generative Recommendation Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用掩码离散扩散替代自回归解码 SID，配合 OPQ 解耦分词、难度感知相干加噪与置信度并行去噪，NDCG@10 提升 6.9%–15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐&quot;,&quot;连续Token分词&quot;,&quot;条件扩散生成&quot;]"><strong><a href="generative_rec/diffusion_generative_recommendation_with_continuous_tokens/">Diffusion Generative Recommendation with Continuous Tokens</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 σ-VAE 连续分词器替代量化码本，并以条件扩散+离散散度损失生成用户偏好，四数据集 Full-Ranking 上 HR@10 平均超 TIGER 11.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;Human Priors&quot;,&quot;Multi-Head Decoding&quot;]"><strong><a href="generative_rec/don_t_waste_it_guiding_generative_recommenders_with_structured_human_priors/">Don&#x27;t Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-Head Decoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以零初始化残差适配头将物品/时序/事件/图先验注入HSTU/HLLM端到端训练，三域Recall与NDCG显著提升并兼顾多样性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回/生成式端到端&quot;,&quot;语义ID (Semantic ID)&quot;,&quot;正交残差量化&quot;]"><strong><a href="generative_rec/dos_dual_flow_orthogonal_semantic_ids_for_recommendation_in_meituan/">DOS: Dual-Flow Orthogonal Semantic IDs for Recommendation in Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用户-物品双流共享码本对齐生成空间，正交残差量化保语义；离线 AUC 0.8763、Hit@10 提升 32.3%，线上收入 +1.15%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID解码&quot;,&quot;长短期兴趣路由&quot;]"><strong><a href="generative_rec/dualgr_generative_retrieval_with_long_and_short_term_interests_modeling/">DualGR: Generative Retrieval with Long and Short-Term Interests Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双分支长短期路由+桶内搜索式SID解码+曝光感知NTP损失，快手双列场景在线HR@1000达19.53%，A/B视频播放+0.527%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;自回归序列生成&quot;,&quot;课程学习&quot;]"><strong><a href="generative_rec/generative_regression_based_watch_time_prediction_for_short_video_recommend/">Generative Regression Based Watch Time Prediction for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>把观看时长回归重构为自回归 token 序列生成，配动态分位词表与 CLEM 训练策略，KuaiRec MAE 降 3.36%，线上视频消费时长 +0.129%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回&quot;,&quot;推测解码&quot;,&quot;语义 ID 哈希校验&quot;,&quot;自回归草稿头&quot;]"><strong><a href="generative_rec/nezha_a_zero_sacrifice_and_hyperspeed_decoding_architecture_for_generative_/">NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模型内自回归草稿头替代独立草稿模型，并用语义 ID 哈希集做无模型校验，在零精度损失下把 GR 解码加速约 10 倍，已上線淘宝搜索广告首页。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;Semantic ID&quot;,&quot;Continued Pre-training&quot;,&quot;Generative Retrieval&quot;]"><strong><a href="generative_rec/plum_adapting_pre_trained_language_models_for_industrial_scale_generative_r/">PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以多模态+共现对比增强的 SIDv2、域内持续预训练与生成式召回 SFT 把 LLM 接入 YouTube 十亿级视频召回，样本效率提升且线上 CTR 最高 +4.96%。</p>
</dd>
</dl>
<p><a href="generative_rec/">查看全部12篇「生成式推荐与离散语义 ID」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;无用户嵌入&quot;,&quot;对比学习增强&quot;]"><strong><a href="llm_rec/alphafree_recommendation_free_from_users_ids_and_gnns/">AlphaFree: Recommendation Free from Users, IDs, and GNNs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语言表征替代ID、在线推断用户偏好、以相似物品增强+对比对齐替代GNN，实现无用户/无ID/无GNN的轻量推荐，内存降69%、冷启动提升26–38%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;推理蒸馏&quot;,&quot;对比自蒸馏&quot;]"><strong><a href="llm_rec/from_reasoning_llms_to_bert_a_two_stage_distillation_framework_for_search_r/">From Reasoning LLMs to BERT: A Two-Stage Distillation Framework for Search Relevance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>美团搜索广告两阶段框架：领域自适应推理LLM做教师标注，CRSD自蒸馏将推理能力内化进6层BERT，线上AdCTR+0.91%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排（下一物品预测）&quot;,&quot;Item-aware Attention&quot;,&quot;双掩码注意力&quot;,&quot;LoRA 微调&quot;]"><strong><a href="llm_rec/beyong_tokens_item_aware_attention_for_llm_based_recommendation/">From Token to Item: Enhancing Large Language Models for Recommendation via Item-aware Attention Mechanism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以物品内/物品间双掩码拆解 LLM 因果注意力，纯靠结构改造建模物品级协同，Amazon 三数据集 Full-Ranking 下 Prec@10 相对最优基线提升 6.80%~71.00%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;时间间隔预测&quot;,&quot;上下文提示&quot;,&quot;零样本推理&quot;]"><strong><a href="llm_rec/is_more_context_always_better_examining_llm_reasoning_capability_for_time_i/">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 GPT-4o/Claude/Gemini 在复购时间间隔预测上的零样本能力，发现 LLM 仅胜统计基线、全面落后 XGBoost，且中等上下文最优、高上下文反降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;语义推理&quot;,&quot;群体隐式兴趣&quot;]"><strong><a href="llm_rec/iterative_semantic_reasoning_from_individual_to_group_interests_for_generat/">Iterative Semantic Reasoning from Individual to Group Interests for Generative Recommendation with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 双向 CoT 推理物品/用户语义，构建个体显式兴趣图与相似用户群隐式兴趣图，并以对比蒸馏迭代对齐，直接推荐 H@10 最高提升 24.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;学习者状态建模&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="llm_rec/personalized_learning_path_planning_with_goal_driven_learner_state_modeling/">Personalized Learning Path Planning through Goal-Driven Learner State Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Pxplore 以四维结构化学习者状态+SFT/GRPO 强化学习训练解决学习路径规划中的目标对齐与长视界问题，真实用户学习增益 +28.28%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;偏好漂移&quot;,&quot;区域级知识编辑&quot;,&quot;区域专属 LoRA&quot;]"><strong><a href="llm_rec/raie_region_aware_incremental_preference_editing_with_lora_for_llm_based_re/">RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结 LLM 主干，用球面 k-means 构建偏好区域并为每区配专属 LoRA，通过 Update/Expand/Add 三种区域级编辑与置信路由实现抗遗忘的增量偏好适配。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;规则感知奖励塑形&quot;,&quot;自适应引导回放&quot;]"><strong><a href="llm_rec/taosr_agrl_adaptive_guided_reinforcement_learning_framework_for_e_commerce_/">TaoSR-AGRL: Adaptive Guided Reinforcement Learning Framework for E-commerce Search Relevance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用规则感知稠密奖励塑形与自适应引导回放缓解 GRPO 奖励稀疏，淘宝搜索相关性 Macro-F1 提升 2.28pt 并上线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;思维链蒸馏&quot;,&quot;LoRA 专家融合&quot;]"><strong><a href="llm_rec/thinkrec_thinking_based_recommendation_via_llm/">ThinkRec: Thinking-based Recommendation via LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以合成推理链激活 LLM 的 System 2 思维，并用实例级 LoRA 专家融合适配异构用户，AUC 平均提升 7.96%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;协同信号对齐&quot;,&quot;软标签NTP&quot;]"><strong><a href="llm_rec/token_level_collaborative_alignment_for_llm_based_generative_recommendation/">Token-level Collaborative Alignment for LLM-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将CF物品级logits投影为token级分布并与one-hot融合成软标签，以soft-NTP损失显式对齐LLM生成，即插即用提升多骨干推荐效果。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM Agent&quot;,&quot;替代互补关系推理&quot;]"><strong><a href="agent_conversational_rec/agentdr_dynamic_recommendation_with_implicit_item_item_relations_via_llm_ba/">AgentDR: Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将全库排序委托给传统推荐工具，用 LLM Agent 做个性化工具加权与替代/互补关系重排，全排序平均提升约两倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;Agentic Benchmark&quot;,&quot;Checklist-driven Evaluation&quot;,&quot;Personalization&quot;]"><strong><a href="agent_conversational_rec/agenticshop_benchmarking_agentic_product_curation_for_personalized_web_shop/">AgenticShop: Benchmarking Agentic Product Curation for Personalized Web Shopping</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个开放网络个性化商品策展基准，用真实购买史构建叙事画像与六维清单，LLM-as-a-judge 可验证评估，揭示现有智能体策展得分仅 15–38%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM多智能体协作&quot;,&quot;动态编排&quot;,&quot;协同过滤&quot;]"><strong><a href="agent_conversational_rec/multi_agent_collaborative_filtering_orchestrating_users_and_items_for_agent/">Multi-Agent Collaborative Filtering: Orchestrating Users and Items for Agentic Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将相似用户与相关物品实例化为LLM智能体，由中心编排器动态招募并下发个性化指令，多轮讨论聚合协同信号，三域HR@10最高提升17.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;指令遵循推荐&quot;,&quot;多轮强化学习&quot;]"><strong><a href="agent_conversational_rec/recoworld_building_simulated_environments_for_agentic_recommender_systems/">RecoWorld: Building Simulated Environments for Agentic Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出双视角LLM仿真环境，用户模拟器在多轮交互中发出反思指令，驱动指令遵循推荐器优化留存，并给出三种历史建模方案与多智能体群体仿真蓝图。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;可解释推荐&quot;,&quot;鲁棒性评测基准&quot;,&quot;行为扰动&quot;]"><strong><a href="agent_conversational_rec/robustexplain_evaluating_robustness_of_llm_based_explanation_agents_for_rec/">RobustExplain: Evaluating Robustness of LLM-Based Explanation Agents for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以五类用户行为扰动×五级严重度、四维一致性指标评测 LLM 解释智能体鲁棒性，7B–70B 模型仅约 0.51 分，大模型高约 8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;经济成瘾理论&quot;,&quot;用户模拟&quot;,&quot;GRPO&quot;]"><strong><a href="agent_conversational_rec/unveiling_and_simulating_short_video_addiction_behaviors_via_economic_addic/">Unveiling and Simulating Short-Video Addiction Behaviors via Economic Addiction Theory</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用经济学成瘾理论量化短视频成瘾曲线，并以 M2A+GRPO 训练 LLM 用户模拟器 AddictSim，会话级 MAE 较 PPO 降 11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;数据预处理/训练集增强&quot;,&quot;偏见检测与去偏&quot;,&quot;文本引导图像编辑&quot;,&quot;公平性&quot;]"><strong><a href="multimodal_rec/biasedit_a_training_free_bias_detect_and_edit_framework_for_learning_fair_v/">BiasEdit: A Training-Free Bias-Detect-and-Edit Framework for Learning Fair Visual Classifiers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>无训练偏见检测与编辑框架，通过视觉-语言属性提取+统计检验检测未知偏见属性，再用文本引导编辑生成真实偏见冲突样本，在100%偏见数据集上实现SOTA去偏。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;组感知聚合&quot;,&quot;偏好引导蒸馏&quot;]"><strong><a href="multimodal_rec/multimodal_enhanced_federated_recommendation_a_group_wise_fusion_approach/">Multimodal-enhanced Federated Recommendation: A Group-wise Fusion Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多模态表征学习卸载至服务器，以组感知聚合+偏好引导蒸馏把多模态语义注入联邦推荐，五数据集全库排序 HR@50 最高提升约 30%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;参数高效微调&quot;,&quot;用户分群个性化&quot;]"><strong><a href="multimodal_rec/personalized_parameter_efficient_fine_tuning_of_foundation_models_for_multi/">Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>按兴趣分群并为每群挂载独立 PEFT 模块，使 CLIP 编码出用户条件化的物品表征，配合群内难负采样，NDCG@20 最高提升 15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态交互分解&quot;,&quot;自适应融合&quot;]"><strong><a href="multimodal_rec/prism_personalized_recommendation_via_information_synergy_module/">PRISM: Personalized Recommendation via Information Synergy Module</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 显式分解模态唯一/冗余/协同信息并以用户偏好动态重加权，即插即用提升三类序列骨干，Yelp N@10 超最强基线 10.91%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;社交与群组推荐&quot;,&quot;事后解释 (Post-hoc Explainability)&quot;,&quot;多视图图神经网络&quot;,&quot;信息增益与条件熵&quot;,&quot;协同效应子图&quot;]"><strong><a href="social_group_rec/explaining_synergistic_effects_in_social_recommendations/">Explaining Synergistic Effects in Social Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出协同子图理论与条件熵掩码优化，从事后归因角度提取多视图社交推荐中的协同信息增益子图并生成路径解释，ACM 上 SIS 提升 71%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;分布式推荐&quot;,&quot;用户验证共识&quot;,&quot;链接预测&quot;]"><strong><a href="social_group_rec/from_aggregation_to_selection_user_validated_distributed_social_recommendat/">From Aggregation to Selection: User-Validated Distributed Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将分布式社交推荐从模型聚合转为用户驱动的算法选择+验证者多数共识投票，提出 Acc@K 共识评测，四数据集 Acc@2 平均提升 4.51%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;参数高效用户建模&quot;,&quot;重叠社区检测&quot;,&quot;自监督对比正则&quot;]"><strong><a href="social_group_rec/pulse_socially_aware_user_representation_modeling_toward_parameter_efficien/">PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用社区结构与社交邻居物品信号替代显式用户嵌入，参数降 50% 且 NDCG@20 提升 8-9%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;LLM Reasoning&quot;,&quot;GRPO&quot;,&quot;Cold-Start&quot;]"><strong><a href="cold_start_rec/llm_reasoning_for_cold_start_item_recommendation/">LLM Reasoning for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 Netflix 冷启动重排任务上系统对比结构化推理、软自洽与 SFT/GRPO 混合微调，推理微调模型在 Discovery 指标上超越线上生产排序模型最高 8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;长尾序列推荐&quot;]"><strong><a href="cold_start_rec/tail_aware_data_augmentation_for_long_tail_sequential_recommendation/">Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性物品相似度构建候选集，设计 T-Substitute/T-Insert 两个尾部感知算子并在表征层 Mixup，实现头部与尾部双赢的长尾序列推荐数据增强。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排/端到端生成式&quot;,&quot;Sharpness-Aware Minimization&quot;,&quot;LLM4Rec&quot;,&quot;长尾物品去偏&quot;]"><strong><a href="cold_start_rec/taming_the_long_tail_efficient_item_wise_sharpness_aware_minimization_for_l/">Taming the Long Tail: Efficient Item-wise Sharpness-Aware Minimization for LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>实证区分 LRS 的“先验长尾”与“数据长尾”，提出物品级锐度感知最小化 EISAM，尾部 NDCG@10 平均提升 8.90%，训练开销仅 +5.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;跨场景统计特征张量&quot;,&quot;稳定性感知模型更新&quot;]"><strong><a href="cold_start_rec/trinity_a_scenario_aware_recommendation_framework_for_large_scale_cold_star/">Trinity: A Scenario-Aware Recommendation Framework for Large-Scale Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向十亿级产品迁移的“特征-架构-更新”三位一体冷启动框架，Copilot 新场景 AUC 0.564→0.726、COPC 0.13→0.95，线上时长 +5.61%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域因果建模&quot;,&quot;部分标签学习&quot;,&quot;门控特征融合&quot;]"><strong><a href="cross_domain_multi_scenario/causality_enhancement_for_cross_domain_recommendation/">Causality Enhancement for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将跨域推荐重构为因果图，用相似度启发式构造因果感知数据并推导无偏部分标签因果损失，作为模型无关插件在四个域与线上 A/B 全面涨点</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;LoRA 方向分解&quot;,&quot;个性化聚合权重&quot;]"><strong><a href="cross_domain_multi_scenario/fedecider_an_llm_based_framework_for_federated_cross_domain_recommendation/">FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>联邦跨域 LLM 推荐中，服务器只广播 LoRA 更新的方向分量以消除尺度噪声，客户端用可学习个性化权重做数据感知融合，Goodreads 平均 Hit@5 提升 3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端生成式&quot;,&quot;Flow Matching&quot;,&quot;高斯混合先验&quot;,&quot;多域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/gaussian_mixture_flow_matching_with_domain_alignment_for_multi_domain_seque/">Gaussian Mixture Flow Matching with Domain Alignment for Multi-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双掩码单骨干 Transformer 解耦域不变/域特定意图，再以域对齐先验引导高斯混合流匹配生成多峰转移轨迹，JD/Amazon 1:999 采样排序下 NDCG@5 最高提升 44%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;数据重生成&quot;,&quot;对比解码&quot;,&quot;序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;ODE 免训练&quot;]"><strong><a href="cross_domain_multi_scenario/s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;LLM 语义增强&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/semacdr_llm_powered_transferable_semantics_for_cross_domain_sequential_reco/">SemaCDR: LLM-Powered Transferable Semantics for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成领域无关语义并与领域特定表征对比对齐，配合跨域行为融合与自适应加权，在 Amazon 六个跨域序列场景 HR@10 最高提升 14.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域公平性&quot;,&quot;信息论增益重分配&quot;]"><strong><a href="cross_domain_multi_scenario/the_double_edged_sword_of_knowledge_transfer_diagnosing_and_curing_fairness/">The Double-Edged Sword of Knowledge Transfer: Diagnosing and Curing Fairness Pathologies in Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论揭示跨域推荐中“差异迁移”与“信息增益不均”两类公平性病理，提出模型无关的 CDFA 框架，在 Tenrec QB/QK 上 UGF 最高降 98.97% 且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;模型合并 (Model Merging)&quot;,&quot;LoRA 混合域交叉训练&quot;]"><strong><a href="cross_domain_multi_scenario/weaverec_an_llm_based_cross_domain_sequential_recommendation_framework_with/">WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「源域+目标域」混合数据交叉训练多个 LoRA 再权重平均合并，既避免直接数据合并的梯度冲突，又让合并成员均强于目标域，实现零额外推理开销的跨域序列推荐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;精排&quot;,&quot;长尾学习&quot;,&quot;层级语义树&quot;,&quot;Gumbel-Softmax&quot;]"><strong><a href="poi_spatiotemporal_rec/adaptive_location_hierarchy_learning_for_long_tailed_mobility_prediction/">Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出架构无关插件 ALOHA：LLM+CoT 自动构建城市定制四级语义层级，配合 Gumbel 扰动与节点自适应权重的层级损失，在 6 城数据集上对 RNN/GNN/Transformer/Diffusion 四类骨干一致提升，M@1 最高 +16.59%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;冷启动表征&quot;,&quot;对比学习&quot;]"><strong><a href="ctr_feature_interaction/coins_semantic_ids_enhanced_cold_item_representation_for_click_through_rate/">COINS: Semantic Ids Enhanced Cold Item Representation for Click-through Rate Prediction in E-commerce Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RQ-OPQ 语义 ID 做门控双向迁移与残差对比学习，强化电商搜索冷启动物品表征；日志级 AUC 冷启 +0.93pp，线上订单量 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;延迟反馈建模&quot;,&quot;复购感知双分支&quot;,&quot;在线流式训练&quot;]"><strong><a href="ctr_feature_interaction/delayed_feedback_modeling_for_post_click_gross_merchandise_volume_predictio/">Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个后点击GMV延迟反馈基准TRACE，提出复购感知双分支READER，在线流式训练下ACC相对提升2.19%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;字段级语义增强&quot;,&quot;LLM知识蒸馏&quot;,&quot;LoRA自监督微调&quot;]"><strong><a href="ctr_feature_interaction/field_matters_a_lightweight_llm_enhanced_method_for_ctr_prediction/">Field Matters: A Lightweight LLM-enhanced Method for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出字段级 LLM 增强范式 LLaCTR：用自监督字段-特征微调蒸馏字段语义，再以 KL 对齐与字段交互矩阵增强特征表示与特征交叉，6 个骨干模型平均 AUC 相对提升 2.24%，训练成本比现有 LLM 增强方法低 10–100 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;生成式用户意图&quot;,&quot;语义ID兴趣群组&quot;,&quot;召回排序一致性&quot;]"><strong><a href="ctr_feature_interaction/genci_generative_modeling_of_user_interest_shift_via_cohort_based_intent_le/">GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用RQ-VAE语义ID+NTP生成兴趣群组，再以分层候选感知注意力注入精排，弥合召回-排序意图鸿沟，三数据集AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;级联延迟反馈&quot;,&quot;净转化率&quot;,&quot;在线持续学习&quot;]"><strong><a href="ctr_feature_interaction/modeling_cascaded_delay_feedback_for_online_net_conversion_rate_prediction_/">Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个淘宝净转化率流式基准 CASCADE，提出 CVR-RFR 级联+分阶段去偏+延迟感知排序损失的 TESLA，NetCVR 的 RI-AUC 绝对提升 12.41%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;统一Transformer骨干&quot;,&quot;混合参数化&quot;,&quot;KV缓存&quot;]"><strong><a href="ctr_feature_interaction/onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;空间外部性&quot;,&quot;机制设计&quot;,&quot;近似算法&quot;]"><strong><a href="ctr_feature_interaction/online_advertising_with_spatial_interactions/">Online Advertising with Spatial Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次将广告位视为度量空间点，用空间折扣函数建模邻近广告互斥外部性，给出 1/18 近似真实机制、2D 欧氏 PTAS 及乘积距离模型的 m^{1-ε} 硬度。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长期行为序列建模&quot;]"><strong><a href="ctr_feature_interaction/unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多任务与列表级重排&quot;,&quot;重排&quot;,&quot;约束多目标优化&quot;,&quot;LLM 智能体编排&quot;,&quot;演化算法&quot;]"><strong><a href="multi_task_reranking/llms_as_orchestrators_constraint_compliant_multi_agent_optimization_for_rec/">LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双种群演化搜索把公平/卖家覆盖/新品曝光当作硬约束，LLM 每 10 代调度探索-利用配比 α，配合 ε-松弛保证 100% 可行性与 HV 提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;梯度协调对比学习&quot;,&quot;难度感知模型路由&quot;]"><strong><a href="multi_task_reranking/not_all_candidates_are_created_equal_a_heterogeneity_aware_approach_to_pre_/">Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-ranking in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>针对粗排异构候选的梯度冲突与算力错配，提出 GHCL 分组对比损失与难度感知双模型路由，AUC 全面超越 SOTA 且线上降本 6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;多目标 Pareto 优化&quot;,&quot;Look-ahead 序列重排&quot;,&quot;沉浸式交互推荐&quot;]"><strong><a href="multi_task_reranking/stcrank_spatio_temporal_collaborative_ranking_for_interactive_recommender_s/">STCRank: Spatio-temporal Collaborative Ranking for Interactive Recommender System at Kuaishou E-shop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向快手电商沉浸式交互推荐的时空协同排序框架，用标签/权重重构消解 vtr-cvr-sdr 目标干扰，并以看前式序列级重排破解时序贪心陷阱，线上 A/B 全链路 IPV +9.65%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;粗排&quot;,&quot;扩散生成规划&quot;,&quot;自动出价&quot;,&quot;跨渠道预算分配&quot;]"><strong><a href="rl_bandit_rec/ahbid_an_adaptable_hierarchical_bidding_framework_for_cross_channel_adverti/">AHBid: An Adaptable Hierarchical Bidding Framework for Cross-Channel Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用扩散模型生成跨渠道预算/约束目标轨迹、结合约束执行与实时控制出价，离线与线上 A/B 收益提升 13.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/批量流量分配&quot;,&quot;Contextual Bandit&quot;,&quot;Constrained LP&quot;]"><strong><a href="rl_bandit_rec/banditlp_large_scale_stochastic_optimization_for_personalized_recommendatio/">BanditLP: Large-Scale Stochastic Optimization for Personalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>神经 Thompson 采样与十亿级 LP 约束动作选择耦合，LinkedIn 邮件营销 A/B 收入 +3.08%、退订 -1.51%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;出价与预算分配&quot;,&quot;LLM 上下文学习&quot;,&quot;GRPO 微调&quot;,&quot;双阶段智能体&quot;]"><strong><a href="rl_bandit_rec/dara_few_shot_budget_allocation_in_online_advertising_via_in_context_decisi/">DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双阶段 LLM 智能体分离少样本规划与精细优化，配合动态参考策略 GRPO-Adaptive 微调，在真实与合成广告环境中显著降低边际 ROI 方差。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;库存约束&quot;,&quot;离线策略学习&quot;,&quot;上下文赌博机&quot;]"><strong><a href="rl_bandit_rec/off_policy_learning_with_limited_supply/">Off-Policy Learning with Limited Supply</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次形式化库存受限下的离线策略学习，提出按“相对奖励差”分配稀缺物品的 OPLS，在合成与 KuaiRec 上稳定超越贪心 OPL。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;流行度偏差&quot;,&quot;提示工程&quot;]"><strong><a href="debias_fairness_industrial/bridging_semantic_understanding_and_popularity_bias_with_llms/">Bridging Semantic Understanding and Popularity Bias with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将流行度偏差拆解为物品侧与用户侧语义，用结构化指令提示引导 RecLLM 双侧去偏，在 LtC/MRMC 与 MRR/F1 上同时超越多样性提示基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;生成式端到端（智能体记忆更新链路）&quot;,&quot;记忆投毒攻击&quot;,&quot;对抗文本触发器&quot;]"><strong><a href="debias_fairness_industrial/drunkagent_stealthy_memory_corruption_in_llm_powered_recommender_agents/">DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个面向 LLM 智能体推荐系统的黑盒记忆投毒攻击：用贪心文本搜索生成低困惑度触发器，再以提示注入策略阻断目标物品智能体的记忆更新，在 Amazon 三数据集、AgentCF/RAG/SEQ 三类受害系统上把目标物品 HR@1 推高 3–7 倍，且整体推荐性能几乎不变。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;特征工程/离线特征选择&quot;,&quot;特征重要性估计&quot;,&quot;梯度归因去偏&quot;,&quot;平滑基线&quot;]"><strong><a href="debias_fairness_industrial/fairfs_addressing_deep_feature_selection_biases_for_recommender_system/">FairFS: Addressing Deep Feature Selection Biases for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别深度特征选择的三类偏差（层偏差/基线偏差/近似偏差），以平滑基线与聚合近似实现无偏特征重要性估计，AUC 提升最高 0.35%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/后处理&quot;,&quot;稀疏自编码器 SAE&quot;,&quot;流行度去偏&quot;,&quot;神经元转向&quot;]"><strong><a href="debias_fairness_industrial/from_insight_to_intervention_interpretable_neuron_steering_for_controlling_/">From Insight to Intervention: Interpretable Neuron Steering for Controlling Popularity Bias in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用稀疏自编码器定位并转向“流行度神经元”，后处理式调控曝光公平，nDCG 几乎无损且可解释。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;用户活跃度重加权&quot;]"><strong><a href="debias_fairness_industrial/identifying_and_upweighting_power_niche_users_to_mitigate_popularity_bias_i/">Identifying and Upweighting Power-Niche Users to Mitigate Popularity Bias in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别高活跃-小众偏好用户并对其在BPR损失中双重加权，在三个基准上实现性能与流行度去偏的帕累托双赢。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦持续学习&quot;,&quot;原型迁移&quot;,&quot;自蒸馏&quot;]"><strong><a href="debias_fairness_industrial/learning_evolving_preferences_a_federated_continual_framework_for_user_cent/">Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>联邦持续推荐框架 FCUCR：时间感知自蒸馏抗遗忘 + 跨用户原型迁移促协同，四数据集全库排序 HR@10 最高提升 44.0%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果去偏&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="debias_fairness_industrial/mclmr_a_model_agnostic_causal_learning_framework_for_multi_behavior_recomme/">MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用后门干预+MoE自适应聚合+偏差感知对比学习，作为即插即用框架增强多行为推荐，三数据集全库排序显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;神经符号推理&quot;,&quot;前门调整&quot;]"><strong><a href="debias_fairness_industrial/modeling_endogenous_logic_causal_neuro_symbolic_reasoning_model_for_explain/">Modeling Endogenous Logic: Causal Neuro-Symbolic Reasoning Model for Explainable Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用前门调整把多行为链内源逻辑做成可解释因果中介，三路神经逻辑推理在三个数据集 Full-Ranking 上 HR@10 提升 20%+。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;后处理向量投影&quot;]"><strong><a href="debias_fairness_industrial/post_hoc_popularity_bias_correction_in_gnn_based_collaborative_filtering/">Post-hoc Popularity Bias Correction in GNN-based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在预训练 GNN 嵌入上以交互级流行度分数构造流行度方向向量并做向量投影剔除，免重训即显著提升长尾与整体排序效果（KuaiRec Recall@20 +186.4%）。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;联邦持续个性化&quot;,&quot;软提示与原型锚定&quot;,&quot;差分隐私&quot;]"><strong><a href="debias_fairness_industrial/prototype_aligned_federated_soft_prompts_for_continual_web_personalization/">Prototype-Aligned Federated Soft-Prompts for Continual Web Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结骨干上注入双时间尺度软提示，长期提示锚定DP联邦原型库，八基准NDCG@10平均+2.9%且降低遗忘与适应步数。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部14篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
