---
hide:
  - toc
---

# ✨ 生成式推荐与离散语义 ID

**KDD 2026** · **23** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [KDD 2021 (1)](../../2021/generative_rec/) · [KDD 2023 (1)](../../2023/generative_rec/) · [KDD 2024 (2)](../../2024/generative_rec/) · [KDD 2025 (4)](../../2025/generative_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/generative_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2024 (3)](../../../CIKM/2024/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (15)](../../../CIKM/2026/generative_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/generative_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/generative_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/generative_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/generative_rec/) · [RecSys 2024 (4)](../../../RecSys/2024/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/generative_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (20)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (6)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2021 (1)](../../../WWW/2021/generative_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/generative_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (12)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×15 · 语义ID ×5 · 端到端生成式 ×2 · 约束解码 ×2 · 语义ID量化 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义 ID" aria-pressed="false">生成式推荐与离散语义 ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束解码" aria-pressed="false">约束解码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;生成式推荐&quot;,&quot;训练-推理对齐&quot;,&quot;Beam Search&quot;,&quot;前缀级排序优化&quot;]"><strong><a href="apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;图结构解码&quot;,&quot;似然陷阱&quot;]"><strong><a href="breaking_the_likelihood_trap_consistent_generative_recommendation_with_grap/">Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用有向无环图解码器扩展非自回归生成空间，并以 Gumbel-Softmax 打通评估器梯度，破解似然陷阱，快手线上 Views +0.78%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回&quot;,&quot;时间感知多目标预测&quot;,&quot;条件引导稀疏注意力&quot;]"><strong><a href="climber_pilot_a_non_myopic_generative_recommendation_model_towards_better_i/">Climber-Pilot: A Non-Myopic Generative Recommendation Model Towards Better Instruction-Following</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以时间感知多目标预测 TAMIP 缓解单步推理短视，用条件引导稀疏注意力 CGSA 实现指令遵循召回，网易云线上 Like Rate +4.24%、延迟仅 +0.32ms</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端推荐&quot;,&quot;协同记忆库&quot;,&quot;潜在上下文压缩&quot;,&quot;目标感知检索&quot;]"><strong><a href="collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;归一化流&quot;,&quot;连续生成式回归&quot;]"><strong><a href="flowtime_towards_continuous_generative_watch_time_prediction_via_flow_based/">FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出连续生成式回归新范式，用一步VAE+归一化流个性化先验建模多峰观看时长分布，KuaiRec MAE降0.676%、线上播放时长+1.044%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID构建&quot;,&quot;生成式检索&quot;,&quot;工业数据集&quot;]"><strong><a href="forge_forming_semantic_identifiers_for_generative_retrieval_in_industrial_d/">FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统化拆解 SID 构建全流程（多模态特征/量化编码/冲突后处理），提出免训练的 Embedding HitRate 与 Gini 系数代理指标，淘宝线上交易量 +0.35%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;语义ID量化&quot;,&quot;RSPO 列表级强化学习&quot;,&quot;动态Beam服务&quot;]"><strong><a href="generative_recommendation_for_large_scale_advertising/">Generative Recommendation for Large-Scale Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>快手广告 GR4AD：UA-SID 语义ID + LazyAR 惰性自回归解码 + VSL/RSPO 在线价值对齐与动态 Beam 服务，线上 A/B 广告收入 +4.28%、吞吐 +117%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID量化&quot;,&quot;分层位置编码&quot;]"><strong><a href="hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;Token级信用分配&quot;,&quot;残差策略优化&quot;,&quot;语义ID&quot;]"><strong><a href="hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式训练&quot;,&quot;动态哈希嵌入表&quot;,&quot;分布式训练系统&quot;]"><strong><a href="mtgenrec_an_efficient_distributed_training_system_for_generative_recommenda/">MTGenRec: An Efficient Distributed Training System for Generative Recommendation Models in Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>KDD 2026 工业系统论文：以动态哈希嵌入表、自动合表、两阶段ID去重与动态序列批处理，将GRM训练吞吐提升1.6–2.4×，线上PV_CTR提升1.31%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式召回/排序&quot;,&quot;动态语义 ID&quot;,&quot;多目标强化学习对齐&quot;]"><strong><a href="onelive_dynamically_unified_generative_framework_for_live_streaming_recomme/">OneLive: Dynamically Unified Generative Framework for Live-Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向直播的动态统一生成式框架：IA 动态 Tokenizer + 时间感知门控注意力 + Sequential MTP 推理加速 + 多目标 GRPO 对齐，离线 HR@128 相对提升 16.70%，已在快手双端上线。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;并行码本向量量化&quot;,&quot;对比对齐&quot;,&quot;多类别App推荐&quot;]"><strong><a href="pcr_ca_parallel_codebook_representations_with_contrastive_alignment_for_mul/">PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用并行码本VQ-AE替代RQ-VAE层级量化建模多类别App语义，配合用户/物品双层对比对齐与双流注意力，微软商店线上CTR+10.52%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;统一生成式检索&quot;,&quot;结果条件生成&quot;]"><strong><a href="pinrec_unified_generative_retrieval_for_pinterest_recommender_systems/">PinRec: Unified Generative Retrieval for Pinterest Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>PinRec以连续Embedding生成式检索统一Pinterest多场景，预训练-微调+结果条件生成，在1M负样本Recall@10与在线A/B取得搜索收藏+3.88%等收益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;GRPO 强化学习&quot;,&quot;SID-语言对齐&quot;]"><strong><a href="reasoning_over_semantic_ids_enhances_generative_recommendation/">Reasoning over Semantic IDs Enhances Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过教师模型合成富化语料强化 SID-语言对齐，再用 GRPO 结果驱动强化学习激发 SID 推理，三数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;逐步潜在推理&quot;,&quot;语义ID码本优化&quot;]"><strong><a href="s_2_gr_stepwise_semantic_guided_reasoning_in_latent_space_for_generative_re/">S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在每级 SID 码生成前插入受码本聚类语义对比监督的思考 token，实现逐步可验证潜在推理，工业短视频 A/B 显著提升时长。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;生成式端到端&quot;,&quot;LLM-as-Judge&quot;,&quot;策略优化 GRPO&quot;,&quot;多目标奖励融合&quot;]"><strong><a href="s_grec_personalized_semantic_aware_generative_recommendation_with_asymmetri/">S-GRec: Personalized Semantic-Aware Generative Recommendation with Asymmetric Advantage</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>离线 LLM 语义裁判 PSJ 产出可解释方面奖励，非对称优势 A2PO 与业务奖励对齐，线上 GMV +1.19% 且服务期零 LLM 推理。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;扩散模型&quot;,&quot;个性化偏好对齐&quot;]"><strong><a href="sell_it_before_you_make_it_revolutionizing_e_commerce_with_personalized_ai_/">Sell It Before You Make It: Revolutionizing E-Commerce with Personalized AI-Generated Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出PerFusion，用个性化奖励模型与组级偏好优化对齐扩散模型，实现电商AIGI个性化文生图；离线MAP/GAUC与线上CTR/CVR显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式召回&quot;,&quot;语义ID&quot;,&quot;强化学习解码&quot;]"><strong><a href="spend_search_where_it_pays_value_guided_structured_sampling_and_optimizatio/">Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以价值引导预算解码与兄弟组 GRPO 缓解概率-奖励错配，在 Amazon 与微信线上提升 HR/NDCG/GMV。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端解码/重排&quot;,&quot;约束解码&quot;,&quot;原始-对偶在线优化&quot;]"><strong><a href="stochastic_primal_dual_decoding_for_multiobjective_generative_recommender_s/">Stochastic Primal-Dual Decoding for Multiobjective Generative Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在自回归生成推荐推理期插入原始-对偶解码层，按约束松弛动态调权辅助目标，离线Pareto全面占优，线上辅助目标+5.44%且消费不降。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回/端到端&quot;,&quot;扩散推荐&quot;,&quot;硬负采样&quot;,&quot;自由能最大化&quot;]"><strong><a href="towards_a_tri_view_diffusion_framework_for_recommendation/">Towards A Tri-View Diffusion Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>从热力学视角用 Helmholtz 自由能统一扩散推荐的“能量最大化”与经典模型的“熵最小化”，配合各向异性去噪器与接受-拒绝 Gumbel 硬负采样，全库排序下平均超越 DiffRec 约 12%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;不确定性建模&quot;,&quot;GRPO偏好优化&quot;,&quot;置信度对齐&quot;]"><strong><a href="uncertainty_aware_generative_recommendation/">Uncertainty-aware Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成置信度与样本难度转化为GRPO奖励与梯度权重，并显式对齐置信度token，稳定训练并支持风险感知推荐。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;模型扩展律&quot;]"><strong><a href="understanding_generative_recommendation_with_semantic_ids_from_a_model_scal/">Understanding Generative Recommendation with Semantic IDs from a Model-scaling View</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 SID 式生成推荐在编码器/量化器/RS 三轴扩展均快速饱和，归因于 SID 语义容量瓶颈，并证明 Text-based GR 具备更优扩展性且 CF 能力随规模提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;约束解码&quot;,&quot;稀疏矩阵&quot;,&quot;语义ID&quot;]"><strong><a href="vectorizing_the_trie_efficient_constrained_decoding_for_llm_based_generativ/">Vectorizing the Trie: Efficient Constrained Decoding for LLM-based Generative Retrieval on Accelerators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将前缀树约束展平为静态 CSR 稀疏转移矩阵，把不规则树遍历转为全向量化稀疏算子，实现 O(1) I/O 的严格约束解码，线上延迟仅 0.033ms/步。</p>
</dd>
</dl>
