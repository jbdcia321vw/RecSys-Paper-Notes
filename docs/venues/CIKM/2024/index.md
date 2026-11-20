---
hide:
  - toc
---

# 🏛️ CIKM 2024 推荐系统论文解读

40 篇 CIKM 2024 论文解读，覆盖 14 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×4 · 端到端生成式 ×4 · 大语言模型推荐 ×3 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;混排与供给调控（策略层）&quot;,&quot;双稳健学习 (DRL)&quot;,&quot;增益建模 Uplift / CATE&quot;,&quot;广告供给个性化&quot;]"><strong><a href="collaborative_filtering/ads_supply_personalization_via_doubly_robust_learning/">Ads Supply Personalization via Doubly Robust Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 RCT 已知倾向分构造双稳健伪结局，轻量随机森林即可在长周期广告供给密度调控上把 CATE 估计与线上变现效率同时拉升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤&quot;,&quot;反馈分图对比学习&quot;,&quot;宏观聚类建模&quot;]"><strong><a href="collaborative_filtering/feedback_reciprocal_graph_collaborative_filtering/">Feedback Reciprocal Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按用户反馈将交互图拆为 I&amp;F/I&amp;U 双图分别做 GCF，用冲动性互惠对比与宏观聚类建模交换信息，全库排序下 Recall@20 平均提升 6.28%，淘宝 A/B GMV +1.61%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样偏差校正&quot;,&quot;排名损失 (WARP/LambdaRank)&quot;]"><strong><a href="collaborative_filtering/improved_estimation_of_ranks_for_learning_item_recommenders_with_negative_s/">Improved Estimation of Ranks for Learning Item Recommenders with Negative Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将采样指标的二项校正迁移进 WARP/LambdaRank 训练，把串行 F-Rank 排名估计换成可批量并行校正估计，MSD 上 WARP Recall@20 相对提升 16.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Triplet Shapley&quot;,&quot;负采样与三元组重加权&quot;]"><strong><a href="collaborative_filtering/interpretable_triplet_importance_for_personalized_ranking/">Interpretable Triplet Importance for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用三元组 Shapley 值可解释地度量 BPR 三元组重要性，蒙特卡洛+控制变量稳定估计并指导重采样，六数据集 R@20 平均提升约 9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;脑电信号&quot;,&quot;量子认知&quot;]"><strong><a href="graph_rec/quantum_cognition_inspired_eeg_based_recommendation_via_graph_neural_networ/">Quantum Cognition-Inspired EEG-based Recommendation via Graph Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以滑窗、量子认知塌缩和连续/干扰双图 GCN 学习 EEG 实时思维，MBD 采样 P@10 较 BPR 最高提升 95.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;结构化行列式点过程 SDPP&quot;,&quot;集合级优化准则&quot;,&quot;多样性建模&quot;]"><strong><a href="sequential_rec/a_universal_sets_level_optimization_framework_for_next_set_recommendation/">A Universal Sets-level Optimization Framework for Next Set Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用结构化行列式点过程 (SDPP) 将时序集合整体视为元素并设计集合级条件似然，联合共现表征与类别感知多样性核，在四个数据集上 Recall/NDCG 提升 6–25% 且同时提升多样性。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性循环单元&quot;,&quot;并行扫描 CUDA 核&quot;]"><strong><a href="sequential_rec/behavior_dependent_linear_recurrent_units_for_efficient_sequential_recommen/">Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以行为依赖的实数线性循环单元 BD-LRU + Triton 并行扫描核，同时达成强性能、快训练与低推理成本，Full-Ranking 下 ML-1M HR@10 提升 7.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;医疗处方推荐&quot;,&quot;意图对比学习&quot;]"><strong><a href="sequential_rec/contrastive_learning_on_medical_intents_for_sequential_prescription_recomme/">Contrastive Learning on Medical Intents for Sequential Prescription Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ARCI 用跨就诊 Transformer 抽取多条“时间路径”，并以意图对比学习正则化多头注意力，在 MIMIC-III/AKI 全库排序上命中率较 SOTA 提升 5–7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;自校正点过程&quot;,&quot;行列式多样性&quot;,&quot;指针式序列解码&quot;]"><strong><a href="sequential_rec/divnet_diversity_aware_self_correcting_sequential_recommendation_networks/">DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将整页集体推荐改写为带因果掩码的指针式序列选择，用自校正解码建模物品相互激发，并以相似度行列式注入多样性；线上 page CTR +8.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;属性感知文本编码&quot;,&quot;多向量匹配 MaxSim&quot;,&quot;零样本迁移&quot;]"><strong><a href="sequential_rec/mars_matching_attribute_aware_representations_for_text_based_sequential_rec/">MARS: Matching Attribute-aware Representations for Text-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用属性级多向量文本编码 + 逐属性 MaxSim 匹配替代单一序列向量，在全库排序下 R@10 最高提升 24.43%，零样本迁移平均提升 13.11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;近似最大内积搜索&quot;,&quot;交叉熵损失近似&quot;,&quot;硬负采样&quot;]"><strong><a href="sequential_rec/rece_reduced_cross_entropy_loss_for_large_catalogue_sequential_recommenders/">RECE: Reduced Cross-Entropy Loss for Large-Catalogue Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 GPU 友好的 LSH 分桶+MIPS 近似筛选高梯度负类，将全库 cross-entropy 的 logits 张量切片计算，峰值显存最多降 12 倍且质量持平或超越。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;文本侧信息融合&quot;]"><strong><a href="sequential_rec/sequence_level_semantic_representation_fusion_for_recommender_systems/">Sequence-level Semantic Representation Fusion for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>频域互滤波实现文本-ID 序列级语义融合，等价于上下文卷积；全库排序下 ML-1M/OR 提升 14%/38%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列均匀性&quot;,&quot;物品频次课程学习&quot;,&quot;多维时间建模&quot;]"><strong><a href="sequential_rec/unirec_a_dual_enhancement_of_uniformity_and_frequency_in_sequential_recomme/">UniRec: A Dual Enhancement of Uniformity and Frequency in Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用序列时间均匀性与物品频次构建双向增强分支，配合多维时间混合注意力，在 4 个数据集采样排序协议下 NDCG@10 相对最强基线提升 2.2%–8.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端召回&quot;,&quot;生成式标识 GID&quot;,&quot;内容-协同对齐&quot;]"><strong><a href="generative_rec/content_based_collaborative_generation_for_recommender_systems/">Content-Based Collaborative Generation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ColaRec 用 LightGCN 层次聚类构造协同 GID，T5 序列到序列直接生成物品标识，并以物品索引任务+对比损失对齐内容与协同空间；四数据集 8:1:1 全库排序 Recall@20 提升 3.5%–6.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;可学习物品分词&quot;,&quot;RQ-VAE&quot;,&quot;协同正则&quot;,&quot;多样性正则&quot;]"><strong><a href="generative_rec/learnable_item_tokenization_for_generative_recommendation/">Learnable Item Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CIKM 2024｜生成式推荐与离散语义ID：LETTER 以 RQ-VAE 层级码、协同对比对齐和码本多样性正则学习物品标识符，并加排序引导生成损失；三数据集 LOO 全库排序 R@10 较 TIGER/LC-Rec 提升 4.7%–10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;多行为序列建模&quot;,&quot;语义ID分词器&quot;,&quot;位置路由MoE&quot;]"><strong><a href="generative_rec/multi_behavior_generative_recommendation/">Multi-Behavior Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多行为序列推荐重构为「先预测行为、再预测物品」的生成式两步解码，配合均衡语义ID与位置路由稀疏MoE，全库排序相对提升31%~70%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;LLM推荐&quot;,&quot;召回+精排&quot;,&quot;互补知识图&quot;,&quot;E-E-I权重决策&quot;]"><strong><a href="llm_rec/breaking_the_barrier_utilizing_large_language_models_for_industrial_recomme/">Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CIKM 2024[LLM推荐] 用 LLM 离线构建互补知识图，结合 U2E2I 召回与 E-E-I 精排校准；在支付宝三场景 AUC 与线上转化显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;跨模态融合&quot;,&quot;GATE 门控&quot;,&quot;LoRA 两阶段训练&quot;]"><strong><a href="llm_rec/collaborative_cross_modal_fusion_with_large_language_model_for_recommendati/">Collaborative Cross-modal Fusion with Large Language Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以混合 Prompt 单 token 承载双模态，用 GATE 维度级门控融合 LLM 语义与 CF 协同嵌入，CTR 全量 AUC 相对 MF 提升 56.21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;评分预测/精排&quot;,&quot;LoRA 插件&quot;,&quot;ID-内容对比对齐&quot;]"><strong><a href="llm_rec/enhancing_content_based_recommendation_via_large_language_model/">Enhancing Content-based Recommendation via Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CIKM 2024｜LLM4Rec：LoRA 跨域语义插件经 DARE 合并，再用 ID-内容对比对齐做评分预测；Amazon 11 域 8:1:1 划分下 MSE 显著优于 RGCL。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;上下文思维链 (In-context CoT)&quot;,&quot;知识蒸馏&quot;]"><strong><a href="llm_rec/large_language_models_enhanced_collaborative_filtering/">Large Language Models Enhanced Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>离线微调 RecGen-LLaMA 生成 CoT 推理构建 in-context 样例库，在线用轻量 Transformer 解码器蒸馏出“世界知识+推理引导的协同过滤特征”注入任意骨干模型，CTR 上 AUC RelaImpr 最高 7.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;记忆增强 LLM&quot;,&quot;实体记忆库&quot;,&quot;冷启动&quot;]"><strong><a href="agent_conversational_rec/memocrs_memory_enhanced_sequential_conversational_recommender_systems_with_/">MemoCRS: Memory-enhanced Sequential Conversational Recommender Systems with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用实体记忆库与通用记忆增强 LLM，建模跨会话偏好连续性；时序切分下 MRR@20 最高提升 38.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端策略学习&quot;,&quot;离线强化学习&quot;,&quot;用户模拟器&quot;,&quot;以用户为中心偏好建模&quot;]"><strong><a href="agent_conversational_rec/reformulating_conversational_recommender_systems_as_tri_phase_offline_polic/">Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 CRS 重构为「用户建模—离线策略—独立可控仿真评估」三阶段解耦范式，Yelp SR@15 达 0.415 且 AT 降至 11.62。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回与排序&quot;,&quot;表征对齐&quot;,&quot;对比学习&quot;,&quot;视觉语言预训练&quot;]"><strong><a href="multimodal_rec/alignrec_aligning_and_training_in_multimodal_recommendations/">AlignRec: Aligning and Training in Multimodal Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多模态推荐目标拆解为内容内、内容-类别、用户-物品三重对齐，配合两阶段解耦训练与中间评估协议，全库排序 R@20 相对提升 4.9%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;缺失模态插补&quot;,&quot;图特征传播&quot;,&quot;Personalized PageRank&quot;]"><strong><a href="multimodal_rec/do_we_really_need_to_drop_items_with_missing_modalities_in_multimodal_recom/">Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用免训练插补替代“丢弃缺失模态物品”，提出 NeighMean/MultiHop/PersPageRank 三种图感知插补，在 Amazon 三域 Recall@20 最高提升 20.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;语义感知对比学习&quot;,&quot;SimTier 相似度分桶&quot;,&quot;工业系统&quot;]"><strong><a href="multimodal_rec/enhancing_taobao_display_advertising_with_multimodal_representations_challe/">Enhancing Taobao Display Advertising with Multimodal Representations: Challenges, Approaches and Insights</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>淘宝展示广告两阶段多模态框架：语义感知对比学习预训练 + SimTier 相似度分桶与 MAKE 多轮解耦集成，离线 GAUC +1.25%，线上 CTR +3.5%、新广告 CTR +6.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;召回&quot;,&quot;跨模态对齐&quot;,&quot;PLM 文本知识&quot;,&quot;组合用药推荐&quot;]"><strong><a href="multimodal_rec/natural_language_assisted_multi_modal_medication_recommendation/">Natural Language-Assisted Multi-modal Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将组合用药推荐建模为患者-药物双模态对齐与相似度检索，用 PLM 抽取病历与药物文本知识，Jaccard 平均提升 4.72%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;端到端生成式&quot;,&quot;视频摘要&quot;,&quot;视觉语言模型 (VLM)&quot;,&quot;偏好条件生成&quot;]"><strong><a href="multimodal_rec/personalized_video_summarization_by_multimodal_video_understanding/">Personalized Video Summarization by Multimodal Video Understanding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将长视频文本化后按 genre 偏好做 T5 语义打分选场景，UserPrefSum F1 26.8 较 UniVTG 提升 13.6%，免训练、推理常数时间</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;扩散模型去噪&quot;,&quot;社交关系重校准&quot;]"><strong><a href="social_group_rec/recdiff_diffusion_model_for_social_recommendation/">RecDiff: Diffusion Model for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在隐空间用扩散模型对社交图嵌入做多步加噪-去噪，任务感知训练剔除虚假社交边，Yelp Full-Ranking Recall@20 达 0.0597。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态图增强&quot;,&quot;用户模态对比学习&quot;]"><strong><a href="cold_start_rec/gume_graphs_and_user_modalities_enhancement_for_long_tail_multimodal_recomm/">GUME: Graphs and User Modalities Enhancement for Long-Tail Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多模态相似度给用户-物品图补边提升长尾连通性，并构造显式交互/扩展兴趣双用户模态做互信息增强与双层对齐去噪，四域 Amazon 全库排序 Recall@20 最高 +3.82%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;显式兴趣迁移&quot;,&quot;负迁移抑制&quot;,&quot;场景选择器&quot;]"><strong><a href="cross_domain_multi_scenario/exit_an_explicit_interest_transfer_framework_for_cross_domain_recommendatio/">EXIT: An EXplicit Interest Transfer Framework for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用兴趣组合标签显式监督跨域兴趣迁移概率，配合场景选择器抑制负迁移，美团首页线上 CTCVR +1.23%、GTV +3.65%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;大语言模型零样本知识推理&quot;,&quot;分层元网络&quot;,&quot;Prompt 工程&quot;]"><strong><a href="cross_domain_multi_scenario/llm4msr_an_llm_enhanced_paradigm_for_multi_scenario_recommendation/">LLM4MSR: An LLM-Enhanced Paradigm for Multi-Scenario Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>冻结 LLM 以场景级/用户级 Prompt 零样本推理跨场景知识与个性化偏好，经分层元网络生成元层注入多场景 CTR 骨干，AUC 最高相对提升 40% 且推理仅 52ms。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;对比正则&quot;]"><strong><a href="ctr_feature_interaction/retrieval_oriented_knowledge_for_click_through_rate_prediction/">Retrieval-Oriented Knowledge for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出ROK，将样本级检索聚合表征蒸馏为参数化知识库，免在线检索，在Tmall/Taobao/Alipay上AUC最高相对提升34.96%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;超长行为序列&quot;,&quot;层次聚类&quot;,&quot;目标注意力&quot;]"><strong><a href="ctr_feature_interaction/twin_v2_scaling_ultra_long_user_behavior_sequence_modeling_for_enhanced_ctr/">TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced CTR Prediction at Kuaishou</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>层次聚类压缩生命周期行为，簇感知目标注意力提升快手 CTR 与推荐多样性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;粗排&quot;,&quot;Pareto优化&quot;,&quot;遗忘曲线&quot;]"><strong><a href="multi_task_reranking/pareto_based_multi_objective_recommender_system_with_forgetting_curve/">Pareto-based Multi-Objective Recommender System with Forgetting Curve</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Ebbinghaus 遗忘曲线建模快滑负反馈，并以 Pareto 求解器自适应权衡粗排一致性与 recency，KuaiRec 与微信视频号 A/B 验证，GMV +1.45%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/端到端策略学习&quot;,&quot;Offline RL&quot;,&quot;Reward Shaping&quot;]"><strong><a href="rl_bandit_rec/roler_effective_reward_shaping_in_offline_reinforcement_learning_for_recomm/">ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用免训练 kNN 奖励整形与簇内距离不确定性惩罚修正 world model 奖励估计，在 KuaiRec 全观测模拟器上累积奖励较 DORL 提升 62.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;嵌入维度搜索&quot;,&quot;Soft Actor-Critic&quot;,&quot;流式推荐&quot;]"><strong><a href="rl_bandit_rec/scalable_dynamic_embedding_size_search_for_streaming_recommendation/">Scalable Dynamic Embedding Size Search for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 SAC 采样 Power-Law 分布参数，在任意内存预算约束下动态增减用户/物品嵌入维度，并以均值池化状态泛化到新用户/物品。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;内容侧攻击/精排&quot;,&quot;对抗文本改写&quot;,&quot;排名提升损失&quot;,&quot;黑盒替代模型&quot;]"><strong><a href="debias_fairness_industrial/adversarial_text_rewriting_for_text_aware_recommender_systems/">Adversarial Text Rewriting for Text-aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 ATR 文本改写攻击，以排名提升损失和两阶段微调攻破文本感知推荐系统，黑盒可行且改写文本较自然。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/精排共用特征底座&quot;,&quot;NeighborHash 哈希表&quot;,&quot;批量点查 KV 存储&quot;,&quot;版本一致性协议&quot;]"><strong><a href="debias_fairness_industrial/an_enhanced_batch_query_architecture_in_real_time_recommendation/">An Enhanced Batch Query Architecture in Real-time Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 NeighborHash（邻域探测+内联偏移）替代传统哈希，配 NVMe 分层 KV 与版本强一致滚动更新，批量点查吞吐近随机访存上界 90%，线上 CTR +2.2%~6.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;云端协同训练/召回表征&quot;,&quot;联邦学习&quot;,&quot;正则化经验风险最小化&quot;,&quot;局部差分隐私&quot;]"><strong><a href="debias_fairness_industrial/efficient_and_robust_regularized_federated_recommendation/">Efficient and Robust Regularized Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将联邦推荐重构为凸 RERM 问题，提出 RFRec/RFRecF，通信轮次降至 O(log 1/ε)、兼具鲁棒性与差分隐私。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/序列排序&quot;,&quot;模型水印&quot;,&quot;自回归OOD序列&quot;]"><strong><a href="debias_fairness_industrial/watermarking_recommender_systems/">Watermarking Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AOW 自回归 OOD 水印，为序列推荐植入黑盒可验证版权序列；四数据集 Full-Ranking 达 100% R@1，兼顾效用与抗蒸馏/微调。</p>
</dd>
</dl>
</section>
</div>
