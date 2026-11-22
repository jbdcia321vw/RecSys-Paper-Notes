---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2026 推荐系统论文解读

56 篇 AAAI (RecSys Track) 2026 论文解读，覆盖 14 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI 2026 ×49 · 生成式推荐 ×7 · 去偏公平与可信推荐 ×6 · 大语言模型推荐 ×4 · 端到端生成式 ×4 · 语义ID ×4 · 可解释推荐 ×3 · AAAI ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索增强生成" aria-pressed="false">检索增强生成<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="算法审计" aria-pressed="false">算法审计<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正样本扩充&quot;,&quot;分层对比学习&quot;]"><strong><a href="collaborative_filtering/crops_improving_dense_retrieval_with_cross_perspective_positive_samples_in_/">CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用查询改写/推荐流/LLM 三源扩充正样本，配分层标签与 H-InfoNCE，快手搜索线上 CTR +0.87%、改写率 -0.65%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;半监督矩阵完成&quot;,&quot;低秩子空间共享&quot;,&quot;泛化界&quot;]"><strong><a href="collaborative_filtering/generalization_bounds_for_semi_supervised_matrix_completion_with_distributi/">Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出隐式反馈采样分布与评分矩阵共享低秩子空间的半监督矩阵完成范式，证明误差可解耦为两项独立界，并在 ML-100K/Douban/Yelp 上验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;检索增强生成&quot;,&quot;图注意力网络&quot;,&quot;空间相关性回灌&quot;]"><strong><a href="graph_rec/graph_neural_field_with_spatial_correlation_augmentation_for_hrtf_personali/">Graph Neural Field with Spatial-Correlation Augmentation for HRTF Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以受试者检索图 + GAT 编解码生成个性化 HRTF，再用方向图 GNN 做空间一致性回灌，3 方向测量下 LSD 降至 3.60 dB。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;端到端序列决策&quot;,&quot;学习路径推荐&quot;,&quot;GraphRAG&quot;,&quot;判别学习&quot;]"><strong><a href="graph_rec/graphrag_induced_dual_knowledge_structure_graphs_for_personalized_learning_/">GraphRAG-Induced Dual Knowledge Structure Graphs for Personalized Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 TextGrad+EDU-GraphRAG 自动合成先决/相似双知识结构图，再用 PPO 三智能体生成学习路径，仿真协议下最长路径提升约 29%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;动态图推荐&quot;,&quot;任务感知子图检索&quot;]"><strong><a href="graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回（负采样增强/训练阶段）&quot;,&quot;扩散负采样&quot;,&quot;可控硬度负样本&quot;]"><strong><a href="self_supervised_denoising/adaptive_diffusion_based_augmentation_for_recommendation/">Adaptive Diffusion-based Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用扩散前向加噪过程把正样本渐进腐蚀为“可控硬度”负样本，理论定位正负转换点 t* 并按用户打分自适应选步，作为即插即用模块普遍提升 CF 与序列推荐。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;重排&quot;,&quot;群组画像&quot;,&quot;群组感知强化学习&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="self_supervised_denoising/g_ubs_towards_robust_understanding_of_implicit_feedback_via_group_aware_use/">G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 群组画像 + 群组感知 GRPO 强化学习为个体隐式反馈去噪，构建 IF-VR 多模态基准，推理准确率提升 14.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;困难-噪声混淆&quot;,&quot;LLM 辅助信号&quot;,&quot;对比学习&quot;]"><strong><a href="self_supervised_denoising/hard_vs_noise_resolving_hard_noisy_sample_confusion_in_recommender_systems_/">Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 语义/逻辑双相关性信号区分困难样本与噪声样本，配合目标对齐与图对比抑制幻觉，全库排序下 Recall@20 平均提升约 45%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长尾推荐&quot;,&quot;混合意图&quot;,&quot;对比约束损失&quot;]"><strong><a href="sequential_rec/bid_farewell_to_seesaw_towards_accurate_long_tail_session_based_recommendat/">Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出混合意图学习与意图约束损失的双约束即插即用框架 HID，在三个数据集上同时提升会话推荐准确率与长尾覆盖，打破 see-saw 困境。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;紧迫度感知&quot;,&quot;时间位置编码&quot;]"><strong><a href="sequential_rec/driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域建模&quot;,&quot;跨会话信息&quot;,&quot;频域一致性损失&quot;]"><strong><a href="sequential_rec/exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;位置前馈网络&quot;]"><strong><a href="sequential_rec/hymoerec_hybrid_mixture_of_experts_for_sequential_recommendation/">HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用共享+稀疏双分支混合专家替换均匀 PFFN，配合自适应融合与负载均衡，在 ML-1M/Beauty 全库排序上稳定超越 Mamba4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;排序&quot;,&quot;双曲图卷积&quot;,&quot;Mamba-MoE&quot;]"><strong><a href="sequential_rec/hyperbolic_enhanced_mixture_of_experts_mamba_for_sequential_recommendation/">Hyperbolic-Enhanced Mixture-of-Experts Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双曲GCN+VGAE重构物品图，RoPE-MoE Mamba动态建模；四数据集HR/NDCG最高提升6.12%，但划分/排序协议未明。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;小波变换&quot;]"><strong><a href="sequential_rec/wavelet_enhanced_adaptive_frequency_filter_for_sequential_recommendation/">Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用上下文动态调制频域滤波器并叠加 Haar 小波高频增强，在 4 个基准 Full-Ranking 下 HR@20 最高提升 11.4%，长序列下训练开销低于 BSARec/SLIME4Rec。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;行为词表&quot;,&quot;可解释推荐&quot;]"><strong><a href="generative_rec/behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_b/">Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用VQ-VAE将用户/物品行为离散化为宏观兴趣+微观意图的行为词表，经多级文本监督与语义对齐正则注入冻结LLM，零样本解释推荐SOTA</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;投机解码&quot;,&quot;归纳式推荐&quot;,&quot;语义ID&quot;]"><strong><a href="generative_rec/inductive_generative_recommendation_via_retrieval_based_speculation/">Inductive Generative Recommendation via Retrieval-based Speculation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将投机解码的草稿-验证范式迁移至生成式推荐，用归纳式检索器起草候选、GR 模型做目标感知验证，使语义 ID 生成模型具备新物品推荐能力。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回/生成式端到端&quot;,&quot;地理语义ID&quot;,&quot;G-DPO&quot;,&quot;层次量化&quot;]"><strong><a href="generative_rec/llm_aligned_geographic_item_tokenization_for_local_life_recommendation/">LLM-Aligned Geographic Item Tokenization for Local-Life Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>LGSID 以 G-DPO 将地理距离注入 LLM，层次量化生成地理感知 SID；快手工业数据上 AUC 最高 +4.17，TIGER Hit@10 相对 +30.83%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;残差扩散&quot;,&quot;捆绑推荐&quot;]"><strong><a href="generative_rec/modeling_item_level_dynamic_variability_with_residual_diffusion_for_bundle_/">Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差扩散在潜空间重建捆绑包item级表征，模型无关地适配B-I归属动态变化，Recall/NDCG最高提升约23%，训练开销仅增4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;跨模态量化&quot;,&quot;语义ID&quot;,&quot;多方位对齐&quot;]"><strong><a href="generative_rec/multi_aspect_cross_modal_quantization_for_generative_recommendation/">Multi-Aspect Cross-modal Quantization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 RQ-VAE 量化各层注入跨模态对比与重建对齐，并在 GR 训练中加隐式/显式对齐，Full-Ranking 下 HR@10 最高提升约 9.4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;统一物品分词&quot;,&quot;MoE 码本量化&quot;,&quot;互信息校准&quot;]"><strong><a href="generative_rec/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi_domain/">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TokenMoE+码本+互信息校准的统一物品分词框架 UniTok，单模型跨 10 域训练，NDCG@10 最高提升 51.89%，参数量降 9.63×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;渐进式DPO&quot;]"><strong><a href="llm_rec/align_3_gr_unified_multi_level_alignment_for_llm_based_generative_recommend/">Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 token/行为/偏好三级对齐框架，双端 SCID 分词 + 渐进式 SP/RF-DPO，Instruments 上 R@10 提升 17.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICWSM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;可解释推荐&quot;,&quot;用户研究&quot;,&quot;情境化解释&quot;,&quot;Uses and Gratifications&quot;]"><strong><a href="llm_rec/contextualizing_recommendation_explanations_with_llms_a_user_study/">Contextualizing Recommendation Explanations with LLMs: A User Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>预注册在线实验(N=759)+访谈(N=30)对比无解释/通用解释/情境化解释，情境化解释显著提升认知需求与观看意愿，但对功利/情感需求增益有限。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排/生成式端到端&quot;,&quot;LoRA 权重合并&quot;,&quot;任务向量&quot;,&quot;冷启动泛化&quot;]"><strong><a href="llm_rec/reccocktail_a_generalizable_and_efficient_framework_for_llm_based_recommend/">RecCocktail: A Generalizable and Efficient Framework for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以“基酒+配料”双 LoRA 在权重空间线性合并，并用无标签测试样本的熵最小化自适应定权，一次训练即可即插即用地兼顾域内精度与冷启动泛化。</p>
</dd>
<dt data-paper-tags="[&quot;AIES&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;Persona 偏置审计&quot;,&quot;系统提示干预&quot;]"><strong><a href="llm_rec/same_question_different_answers_auditing_persona_dependent_variation_in_six/">Same Question, Different Answers: Auditing Persona-Dependent Variation in Six LLMs on Smoking-Cessation Advice</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 12 类人设×4 问题×6 模型共 2,880 条回答审计 LLM 戒烟建议的人设偏置，并提出“事实下限”系统提示干预。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Query Complexity Routing&quot;,&quot;Agentic Task Planning&quot;,&quot;Memory Filtering&quot;]"><strong><a href="agent_conversational_rec/adaptjobrec_enhancing_conversational_career_recommendation_through_an_llm_p/">AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用查询复杂度路由 + 少样本记忆过滤 + 异步嵌套任务规划器改造 LLM Agent，在沃尔玛真实求职场景实现延迟降低 53.3% 且推荐精度提升。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;仿真评测（LLM 用户模拟器）&quot;,&quot;画像缺陷诊断&quot;,&quot;多轮双向演化&quot;]"><strong><a href="agent_conversational_rec/diagnostic_guided_dynamic_profile_optimization_for_llm_based_user_simulator/">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用专用小模型诊断画像缺陷、通用 LLM 施治，迭代优化用户画像并与序列推荐器多轮双向演化，模拟保真度较最优基线提升 14.6%–18.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;知识树序列化&quot;,&quot;检索增强生成&quot;,&quot;软提示学习&quot;]"><strong><a href="agent_conversational_rec/enhancing_conversational_recommender_systems_with_tree_structured_knowledge/">Enhancing Conversational Recommender Systems with Tree-Structured Knowledge and Pretrained Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 RAG 式对话专属知识树序列化 KG 供 PLM 结构推理，并用辅助任务显式注入协同偏好，INSPIRED 上 MRR@10 相对提升 19.7%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;Theory of Mind&quot;,&quot;意图推理&quot;,&quot;谄媚偏差&quot;]"><strong><a href="agent_conversational_rec/rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm_based_conve/">RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个面向对话推荐场景的人类标注 ToM 基准，从认知推断与行为预测两维评估 LLM，揭示细粒度意图与谄媚偏差瓶颈</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Chain-of-User-Thought&quot;,&quot;GUI Agent&quot;,&quot;潜在意图推理&quot;]"><strong><a href="agent_conversational_rec/smartagent_chain_of_user_thought_for_embodied_personalized_agent_in_cyber_w/">SmartAgent: Chain-of-User-Thought for Embodied Personalized Agent in Cyber World</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 COUT 三阶用户思维链范式与两阶段训练的 GUI 智能体 SmartAgent，在自建 SmartSpot 基准上实现具身操作与个性化推荐全链路推理。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;分层负反馈&quot;,&quot;虚拟试穿&quot;,&quot;多智能体协作&quot;]"><strong><a href="agent_conversational_rec/styletailor_towards_personalized_fashion_styling_via_hierarchical_negative_/">StyleTailor: Towards Personalized Fashion Styling via Hierarchical Negative Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个统一个性化服装设计、购物推荐与虚拟试穿的协作智能体框架，以三级分层负反馈闭环迭代优化，在自建 64 样本基准上 Style Consistency 相对基线提升 39.4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;MLLM Agent&quot;,&quot;GRPO/RL&quot;,&quot;用户模拟&quot;,&quot;LLM4Rec&quot;]"><strong><a href="agent_conversational_rec/vragent_r1_boosting_video_recommendation_with_mllm_based_agents_via_reinfor/">VRAgent-R1: Boosting Video Recommendation with MLLM-based Agents via Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双智能体VRAgent-R1：IP Agent渐进理解视频，US Agent以CoT+GRPO模拟用户并重排；MicroLens-100K NDCG@10提升6.0%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;负反馈建模&quot;,&quot;多智能体协同&quot;,&quot;渐进式强化学习&quot;]"><strong><a href="agent_conversational_rec/when_top_ranked_recommendations_fail_modeling_multi_granular_negative_feedb/">When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三智能体协同建模多粒度负反馈，S-GRPO 渐进奖励实现可解释视频推荐，线上快滑率降 9.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;多模态推荐与内容理解&quot;,&quot;端侧个性化推理&quot;,&quot;联邦学习&quot;,&quot;视觉-语言融合&quot;,&quot;混合专家(MoE)&quot;]"><strong><a href="multimodal_rec/federated_vision_language_recommendation_with_personalized_fusion/">Federated Vision-Language-Recommendation with Personalized Fusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 FedVLR 双层融合框架：服务器端用多算子生成多视图，客户端以用户私有历史驱动 MoE 路由做个性化融合，在 7 个数据集上显著超越联邦基线并逼近集中式模型。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;离散语义分词&quot;,&quot;RQ-VAE&quot;]"><strong><a href="cold_start_rec/motorec_sparse_regularized_multimodal_tokenization_for_cold_start_recommend/">MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用稀疏正则化 RQ-VAE 将多模态特征离散化为组合语义码，配合稀有度放大与多源图编码，冷启动物品 N@20 最高提升 12.58%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;变分自编码器VAE&quot;,&quot;多视图解耦&quot;,&quot;混合专家MoE&quot;]"><strong><a href="cold_start_rec/m_2vae_multi_modal_multi_view_variational_autoencoder_for_cold_start_item_r/">M²VAE: Multi-Modal Multi-View Variational Autoencoder for Cold-start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 PoE 抽跨模态共性视图、用户感知层次 MoE 融合特有视图，解耦+共现对比免预训练生成新物品表征，HR@10 最高提升 27.6%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;动态路由&quot;]"><strong><a href="cross_domain_multi_scenario/from_ids_to_semantics_a_generative_framework_for_cross_domain_recommendatio/">From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用领域自适应语义ID分词+对称动态路由的生成式跨域推荐框架，Leave-Last-Out全库排序下全面超越SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;多场景推荐&quot;,&quot;超图学习&quot;,&quot;参数分裂&quot;]"><strong><a href="poi_spatiotemporal_rec/multifaceted_scenario_aware_hypergraph_learning_for_next_poi_recommendation/">Multifaceted Scenario-Aware Hypergraph Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出场景拆分式多视图超图与自适应参数分裂，在 NYC/TKY/Gowalla 上按场景提升 Next-POI 排序。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多模态LLM聚类&quot;,&quot;终身行为序列&quot;,&quot;HSTU&quot;]"><strong><a href="ctr_feature_interaction/dmgin_how_multimodal_llms_enhance_large_recommendation_models_for_lifelong_/">DMGIN: How Multimodal LLMs Enhance Large Recommendation Models for Lifelong User Post-click Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用MLLM离线聚类店铺压缩终身后点击序列，组内统计+Transformer与组间HSTU保留兴趣演化；工业与Amazon AUC/GAUC领先，线上CTR+4.7%、RPM+2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;假负例重标注&quot;,&quot;部分最优传输&quot;]"><strong><a href="ctr_feature_interaction/idefn_identifying_unclicked_space_false_negatives_via_relaxed_partial_optim/">IdeFN: Identifying Unclicked Space False Negatives via Relaxed Partial Optimal Transport for Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用松弛部分最优传输为未点击样本软匹配正样本并自适应重标注假负例，在 Ali-CCP/KuaiRand-Pure 上 AUC 提升约 2.0%、KS 提升 4.6–6.5%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;序列长度条件化&quot;,&quot;注意力温度调制&quot;]"><strong><a href="ctr_feature_interaction/length_adaptive_interest_network_for_balancing_long_and_short_sequence_mode/">Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将序列长度作为条件信号注入提示与注意力温度，缓解短序列用户注意力极化，五骨干平均AUC最高+1.15%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Dueling Bandit&quot;,&quot;反馈增强&quot;,&quot;置信界校准&quot;]"><strong><a href="rl_bandit_rec/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augment/">Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用增强置信界把相似/依赖观测注入无模型 Dueling Bandit，给出样本复杂度与遗憾权衡，在推荐、多目标优化与 LLM 响应优化上显著降遗憾。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (15)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;因果去混杂&quot;,&quot;反事实曝光干预&quot;,&quot;多样性推荐&quot;]"><strong><a href="debias_fairness_industrial/diversity_recommendation_via_causal_deconfounding_of_co_purchase_relations_/">Diversity Recommendation via Causal Deconfounding of Co-purchase Relations and Counterfactual Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用去混杂非对称共购图 UACR 与反事实曝光提权插件，在 LightGCN 上同时提升多样性与准确率，F_β 最优且训练开销仅高 12%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/事后干预&quot;,&quot;稀疏自编码器&quot;,&quot;单义神经元&quot;,&quot;可解释推荐&quot;]"><strong><a href="debias_fairness_industrial/extracting_interaction_aware_monosemantic_concepts_in_recommender_systems/">Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预测感知 SAE 从双塔嵌入中提取单义神经元，在 ML1M/Last.FM 上实现高纯度概念发现与事后可控干预。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦召回训练/隐私遗忘&quot;,&quot;属性遗忘&quot;,&quot;对抗训练&quot;,&quot;梯度重构攻击&quot;]"><strong><a href="debias_fairness_industrial/fedau2_attribute_unlearning_for_user_level_federated_recommender_systems_wi/">FedAU2: Attribute Unlearning for User-Level Federated Recommender Systems with Adaptive and Robust Adversarial Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用户级联邦推荐属性遗忘：SUT 自适应扰动预算稳定对抗训练 + DSVAE 双随机掩蔽梯度泄漏，BAcc 降 26.4% 而 NDCG@10 仅降 4.5%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;路径积分归因&quot;,&quot;反事实保真度&quot;]"><strong><a href="debias_fairness_industrial/fidelity_aware_recommendation_explanations_via_stochastic_path_integration/">Fidelity-Aware Recommendation Explanations via Stochastic Path Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将路径积分归因引入推荐解释，用随机基线采样适配稀疏二值交互，在三种模型三数据集上刷新反事实保真度基准。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;成员推断攻击&quot;,&quot;知识蒸馏&quot;,&quot;隐私泄露审计&quot;]"><strong><a href="debias_fairness_industrial/membership_inference_attack_against_large_language_model_based_recommendati/">Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「异构双阶段知识蒸馏」构造参考模型放大成员/非成员分布差异，融合置信度-熵-损失-隐层向量做成员推断，跨 4 数据集 3 类 LLM 全面超越影子模型 MIA</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;参数高效微调&quot;,&quot;向量量化&quot;]"><strong><a href="debias_fairness_industrial/plug_and_play_parameter_efficient_tuning_of_embeddings_for_federated_recomm/">Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结预训练全量物品嵌入、仅微调并上传压缩嵌入（LoRA/Hash/RQ-VAE），在联邦推荐中同时降通信与提精度。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;序贯决策智能体（离线仿真）&quot;,&quot;递推持续性环&quot;,&quot;因果损伤归因&quot;,&quot;机制可解释性评测&quot;]"><strong><a href="debias_fairness_industrial/recon_ipsundrum_an_inspectable_recurrent_persistence_loop_agent_with_affect/">ReCoN-Ipsundrum: An Inspectable Recurrent Persistence Loop Agent with Affect-Coupled Control and Mechanism-Linked Consciousness Indicator Assays</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 ReCoN 状态机上外挂感官显著性递推环与可选 affect 代理，用熟悉度控制的走廊偏好、探索性玩耍、疼痛尾部与因果损伤四类探针，把“持续性—偏好稳定—结构化扫描”拆成可分离的机制链路。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/服务端黑盒查询&quot;,&quot;成员推理攻击&quot;,&quot;混合推荐系统隐私&quot;,&quot;相对成员度量&quot;]"><strong><a href="debias_fairness_industrial/reference_recommendation_based_membership_inference_attack_against_hybrid_b/">Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>针对混合推荐系统提出参考推荐相对成员度量 MIA，仅需两次黑盒查询，ASR 达 0.93 且比影子训练快 93 倍。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;观看时长去偏&quot;,&quot;条件分位数变换&quot;,&quot;贝叶斯证据融合&quot;]"><strong><a href="debias_fairness_industrial/relative_advantage_debiasing_for_watch_time_prediction_in_short_video_recom/">Relative Advantage Debiasing for Watch-Time Prediction in Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以用户/视频 ID 为伞因子做条件分位数变换，把带偏观看时长映射为无偏相对优势标签，两阶段解耦估计与偏好学习，KuaiRand 与十亿级工业集全面超越 D2Q/CQE。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/策略层（交互式推荐）&quot;,&quot;物品生命周期建模&quot;,&quot;层级强化学习&quot;]"><strong><a href="debias_fairness_industrial/revisiting_fairness_aware_interactive_recommendation_item_lifecycle_as_a_co/">Revisiting Fairness-aware Interactive Recommendation: Item Lifecycle as a Control Knob</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>实证短视频物品呈压缩三段式生命周期，以 STL+iTransformer 阶段判别器驱动双层 RL 分阶段分配曝光，在 KuaiSim 模拟器上交互长度 +10.5%、累积奖励 +13.1% 且公平性同步改善。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部15篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
