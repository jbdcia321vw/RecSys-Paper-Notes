---
hide:
  - toc
---

# 📖 IEEE TKDE 2026 推荐系统论文解读

15 篇 IEEE TKDE 2026 论文解读，覆盖 7 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDE ×12 · TKDE 2026 ×3 · 多模态推荐 ×2 · 大语言模型推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Krylov 子空间&quot;,&quot;多项式图滤波&quot;,&quot;免训练推荐&quot;]"><strong><a href="collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;矩阵分解&quot;,&quot;archi2vec 架构编码&quot;,&quot;迁移性估计&quot;]"><strong><a href="collaborative_filtering/pre_trained_model_recommendation_for_downstream_fine_tuning/">Pre-Trained Model Recommendation for Downstream Fine-tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将预训练模型选择建模为 model-task 隐空间矩阵分解，配合 archi2vec 架构图编码与 CLIP 代理冷启动，O(1) 排序取得 SOTA 迁移性估计。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图压缩&quot;,&quot;二分结构损失&quot;,&quot;梯度匹配&quot;]"><strong><a href="graph_rec/democratic_recommendation_with_user_and_item_representatives_produced_by_gr/">Democratic Recommendation with User and Item Representatives Produced by Graph Condensation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用图压缩生成用户/物品代表，并以二分结构损失 BSL 约束压缩图，80% 数据量下 P@20 仅降 8.5% 且训练快于 LightGCN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;上下文注意力&quot;,&quot;堆叠多头注意力&quot;]"><strong><a href="sequential_rec/an_adaptive_context_aware_and_stacked_attention_network_based_recommendatio/">An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users&#x27; Temporal Preference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用上下文物品注意力+堆叠多头用户注意力捕捉用户时变偏好，Leave-One-Out+100负采样排序，三数据集HR/NDCG/MRR@10全面超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;数据增强&quot;,&quot;自监督辅助任务&quot;,&quot;扩散/LLM 生成&quot;]"><strong><a href="sequential_rec/data_augmentation_for_sequential_recommendation_a_survey/">Data Augmentation for Sequential Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首篇以数据增强为主角的序列推荐综述，提出原则×对象×目的三维分类与双训练范式形式化，三集 Full-Ranking 复现最高提升 66.0%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣聚类&quot;,&quot;稳定性-可塑性解耦&quot;]"><strong><a href="sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;推理时计算 (Test-Time Scaling)&quot;,&quot;隐式思维链 (Latent CoT)&quot;,&quot;推理位置编码&quot;]"><strong><a href="sequential_rec/think_before_recommend_unleashing_the_latent_reasoning_power_for_sequential/">Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出首个推理时计算框架 ReaRec，将末位隐状态自回归回灌实现隐式多步推理，配合 ERL/PRL 两种轻量学习策略，全库排序下平均提升 7.49%、推理延迟仅增 3.51%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;持续预训练&quot;,&quot;学习率调度&quot;,&quot;跨域行为序列&quot;]"><strong><a href="llm_rec/large_language_model_empowered_recommendation_meets_all_domain_continual_pr/">Large Language Model Empowered Recommendation Meets All-domain Continual Pre-Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>CPRec 用全领域持续预训练 + Warmup-Stable-Annealing 调度，把 LLM 开放世界语义平滑适配到协同域，全库未交互采样排序下 HR@1 平均提升 7.4%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;审慎对齐&quot;,&quot;多步推理&quot;]"><strong><a href="llm_rec/reason4rec_deliberative_user_preference_alignment_of_large_language_models_/">Reason4Rec: Deliberative User Preference Alignment of Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出审慎推荐任务与三专家多步推理框架，用评论监督偏好蒸馏-匹配-预测，评分预测与排序全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;小波分解&quot;,&quot;模态融合&quot;]"><strong><a href="multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图特征插值&quot;,&quot;缺失模态填补&quot;]"><strong><a href="multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;感知差距指数&quot;,&quot;谱图理论&quot;,&quot;连边推荐&quot;]"><strong><a href="social_group_rec/quantifying_and_minimizing_perception_gap_in_social_networks/">Quantifying and Minimizing Perception Gap in Social Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出连续意见下的感知差距指数 PGI，用谱图理论刻画网络连通性与社群结构对局部-全局认知偏差的影响，并证明连边推荐最小化 PGI 的不可近似性，给出近最优贪心启发式。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;显式反馈去偏&quot;,&quot;风险上界分解&quot;]"><strong><a href="debias_fairness_industrial/bpl_bias_adaptive_preference_distillation_learning_for_recommender_system/">BPL: Bias-adaptive Preference Distillation Learning for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以风险上界三项分解为骨架，用亲和度自适应的双蒸馏（可靠性过滤自蒸馏+置信惩罚偏好蒸馏）同时提升 factual 与 counterfactual 测试，Yahoo!R3 RCT MSE 降 22.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平/可信隐私与工业架构&quot;,&quot;精排/端到端生成式&quot;,&quot;后门攻击&quot;,&quot;触发器检测&quot;]"><strong><a href="debias_fairness_industrial/exploring_backdoor_attack_and_defense_for_llm_empowered_recommendations/">Exploring Backdoor Attack and Defense for LLM-empowered Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次揭示 LLM4Rec 的标题触发器后门：1% 投毒即可让 ASR≈100%，并提出 LLM 毒物扫描器 P-Scanner 将 ASR 压至 0 且几乎不损 H@1。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（数据投毒/画像注入）&quot;,&quot;Sharpness-Aware Minimization&quot;,&quot;投毒攻击迁移性&quot;]"><strong><a href="debias_fairness_industrial/sharpness_aware_poisoning_enhancing_transferability_of_injective_attacks_on/">Sharpness-Aware Poisoning: Enhancing Transferability of Injective Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用锐度感知最小化在代理模型邻域内逼近最坏受害模型，把投毒攻击从 min-min 扩展为 min-max-min 三层优化，显著提升跨模型迁移性与防御逃逸能力。</p>
</dd>
</dl>
</section>
</div>
