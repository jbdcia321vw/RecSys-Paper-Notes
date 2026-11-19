---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐 (Cold-Start & Long-Tail Recommendation)

> **研究范畴**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

当前方向共收录 **174** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 大语言模型推荐 ×20 · 生成式推荐 ×16 · 去偏公平与可信推荐 ×10 · 多模态推荐 ×9 · 语义ID ×7 · 对比学习 ×4 · 新闻推荐 ×4 · RQ-VAE ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">98</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">47</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">26</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">20</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动推荐" aria-pressed="false">冷启动推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prior-Fitted Network" aria-pressed="false">Prior-Fitted Network<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性量化" aria-pressed="false">不确定性量化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="严格冷启动" aria-pressed="false">严格冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲几何" aria-pressed="false">双曲几何<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排+重排" aria-pressed="false">召回+精排+重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业数据集" aria-pressed="false">工业数据集<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式检索" aria-pressed="false">生成式检索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱推荐" aria-pressed="false">知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾推荐" aria-pressed="false">长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本跨域推荐" aria-pressed="false">零样本跨域推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;简历补全&quot;,&quot;生成对抗对齐&quot;]"><strong><a href="../../venues/AAAI/2024/llm_rec/enhancing_job_recommendation_through_llm_based_generative_adversarial_netwo/">Enhancing Job Recommendation through LLM-based Generative Adversarial Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用交互历史引导 LLM 补全简历，并以 GAN 对齐少样本用户低质简历表征，三数据集 map@5 平均提升约 7.9%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端召回与排序&quot;,&quot;掩码域建模&quot;,&quot;域偏好建模&quot;]"><strong><a href="../../venues/AAAI/2024/cross_domain_multi_scenario/multi_domain_recommendation_to_attract_users_via_domain_preference_modeling/">Multi-Domain Recommendation to Attract Users via Domain Preference Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 DRIP，用掩码域建模统一学习多对多跨域映射并分解域级/物品级偏好；Amazon 双平台 Top-K 排序下显著优于 CDR 基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Temporal Feature Shift&quot;]"><strong><a href="../../venues/AAAI/2024/cold_start_rec/temporally_and_distributionally_robust_optimization_for_cold_start_recommen/">Temporally and Distributionally Robust Optimization for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出时间分布鲁棒优化 TDRO，以最坏组因子与漂移因子重加权特征提取器，在全库排序上稳定提升冷启动物品推荐。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;可解释推荐&quot;,&quot;推荐透明度与数据能动性&quot;,&quot;K-12 AI 素养教育&quot;]"><strong><a href="../../venues/AAAI/2025/debias_fairness_industrial/an_xai_social_media_platform_for_teaching_k_12_students_ai_driven_profiling/">An XAI Social Media Platform for Teaching K-12 Students AI-Driven Profiling, Clustering, and Engagement-Based Recommending</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向 K-12 的 XAI 社交媒体教学平台，实时可视化数据采集、画像聚类与推荐机制，209 名学生实测揭示三类使用画像。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;经验贝叶斯先验&quot;,&quot;Thompson采样&quot;,&quot;非平稳在线学习&quot;]"><strong><a href="../../venues/AAAI/2025/cold_start_rec/bayescns_a_unified_bayesian_approach_to_address_cold_start_and_non_stationa/">BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用经验贝叶斯先验网络+Thompson采样在线更新，统一解决物品冷启动与非平稳漂移，线上A/B新物品曝光+10.60%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导特征适配&quot;,&quot;参数高效微调&quot;]"><strong><a href="../../venues/AAAI/2025/multimodal_rec/befa_a_general_behavior_driven_feature_adapter_for_multimedia_recommendatio/">BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为特征引导的轻量适配器解耦-过滤-重构预训练内容特征，修复信息漂移与遗漏，全库排序下平均提升约 9-11%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;上下文组合老虎机&quot;,&quot;神经UCB聚类&quot;,&quot;单调超臂网络&quot;]"><strong><a href="../../venues/AAAI/2025/rl_bandit_rec/neural_combinatorial_clustered_bandits_for_recommendation_systems/">Neural Combinatorial Clustered Bandits for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双神经网络+上下文聚类替代组合老虎机的精确优化 oracle，实现 $\tilde{O}(\tilde{d}\sqrt{T})$ 遗憾并显著降低推荐累积遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长尾推荐&quot;,&quot;混合意图&quot;,&quot;对比约束损失&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/bid_farewell_to_seesaw_towards_accurate_long_tail_session_based_recommendat/">Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出混合意图学习与意图约束损失的双约束即插即用框架 HID，在三个数据集上同时提升会话推荐准确率与长尾覆盖，打破 see-saw 困境。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;半监督矩阵完成&quot;,&quot;低秩子空间共享&quot;,&quot;泛化界&quot;]"><strong><a href="../../venues/AAAI/2026/collaborative_filtering/generalization_bounds_for_semi_supervised_matrix_completion_with_distributi/">Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出隐式反馈采样分布与评分矩阵共享低秩子空间的半监督矩阵完成范式，证明误差可解耦为两项独立界，并在 ML-100K/Douban/Yelp 上验证。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;投机解码&quot;,&quot;归纳式推荐&quot;,&quot;语义ID&quot;]"><strong><a href="../../venues/AAAI/2026/generative_rec/inductive_generative_recommendation_via_retrieval_based_speculation/">Inductive Generative Recommendation via Retrieval-based Speculation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将投机解码的草稿-验证范式迁移至生成式推荐，用归纳式检索器起草候选、GR 模型做目标感知验证，使语义 ID 生成模型具备新物品推荐能力。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;离散语义分词&quot;,&quot;RQ-VAE&quot;]"><strong><a href="../../venues/AAAI/2026/cold_start_rec/motorec_sparse_regularized_multimodal_tokenization_for_cold_start_recommend/">MoToRec: Sparse-Regularized Multimodal Tokenization for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用稀疏正则化 RQ-VAE 将多模态特征离散化为组合语义码，配合稀有度放大与多源图编码，冷启动物品 N@20 最高提升 12.58%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Dueling Bandit&quot;,&quot;反馈增强&quot;,&quot;置信界校准&quot;]"><strong><a href="../../venues/AAAI/2026/rl_bandit_rec/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augment/">Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用增强置信界把相似/依赖观测注入无模型 Dueling Bandit，给出样本复杂度与遗憾权衡，在推荐、多目标优化与 LLM 响应优化上显著降遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/服务端黑盒查询&quot;,&quot;成员推理攻击&quot;,&quot;混合推荐系统隐私&quot;,&quot;相对成员度量&quot;]"><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/reference_recommendation_based_membership_inference_attack_against_hybrid_b/">Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>针对混合推荐系统提出参考推荐相对成员度量 MIA，仅需两次黑盒查询，ASR 达 0.93 且比影子训练快 93 倍。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;文本画像对齐&quot;,&quot;零样本推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/easyrec_simple_yet_effective_language_models_for_recommendation/">EasyRec: Simple yet Effective Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成用户/物品画像并以对比学习对齐协同信号，训练轻量文本编码器实现零样本推荐，Full-Ranking 下 Recall@20 最高提升 32%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;生成式推荐&quot;,&quot;统一物品分词器&quot;,&quot;课程学习码本&quot;,&quot;零样本跨域推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/generative_rec/recbase_generative_foundation_model_pretraining_for_zero_shot_recommendatio/">RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用课程学习增强的 RQ-VAE 将跨域物品统一离散为层级概念 ID，从零预训练 1.5B 自回归推荐基座，在 8 个未见数据集零样本 AUC 全面超越 7B LLM。</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2025&quot;,&quot;生成式推荐&quot;,&quot;有限标量量化(FSQ)&quot;,&quot;零样本跨域推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/generative_rec/recgpt_a_foundation_model_for_sequential_recommendation/">RecGPT: A Foundation Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>纯文本驱动+FSQ统一分词+混合双向因果注意力的序列推荐基础模型，零样本跨域显著超越10%数据微调的ID基线。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;指令微调&quot;,&quot;强化学习画像精炼&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/reclm_recommendation_instruction_tuning/">RecLM: Recommendation Instruction Tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以两轮协同指令微调+RL画像精炼为LLM注入协同信号，作为即插即用画像增强器在零样本冷启动下带来最高 200%+ 的 Recall 提升。</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;内容基推荐&quot;,&quot;主动学习冷启动&quot;,&quot;稠密语义嵌入&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/cold_start_rec/scholar_inbox_personalized_paper_recommendations_for_scientists/">Scholar Inbox: Personalized Paper Recommendations for Scientists</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向科研论文的开放获取推荐平台，以显式点赞/点踩训练加权逻辑回归，结合 Scholar Maps 与主动学习破解冷启动，在 80 万评分数据集上 GTE-Large 达 AUC 86.75。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语义编码器&quot;,&quot;评测基准&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_l/">Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建 BLaIR 基准，用 11 个 LLM 作语义编码器评测序列推荐、协同过滤与复杂查询商品搜索，揭示其排名与 MTEB 几乎不相关。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;分阶段兴趣演化&quot;,&quot;分组感知对齐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/hsuga_llm_enhanced_recommendation_with_hierarchical_semantic_understanding_/">HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分阶段可编辑兴趣演化替代长序列一次性摘要，并按用户活跃度分组调节语义对齐强度，三数据集全库排序稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;扩散模型生成式迁移&quot;,&quot;多模态大表征模型&quot;,&quot;边侧用户分布学习&quot;]"><strong><a href="../../venues/ACM_MM/2025/cross_domain_multi_scenario/leveraging_multimodal_data_and_side_users_for_diffusion_cross_domain_recomm/">Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 抽取多模态特征做初步对齐，再以边侧用户+重叠用户两阶段扩散生成目标域冷启动用户向量，Amazon 三任务 MAE 最高降 13.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;冷启动&quot;,&quot;端侧推理&quot;,&quot;参数编辑&quot;,&quot;原型聚类&quot;,&quot;端云协同&quot;]"><strong><a href="../../venues/ACM_MM/2025/cold_start_rec/tackling_device_data_distribution_real_time_shift_via_prototype_based_param/">Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端神经适配器免反传生成参数编辑矩阵，多原型聚类+跨层知识迁移实现端侧实时分布偏移适配，1s 内完成个性化且 NDCG@5 显著超越微调。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;微视频推荐数据集&quot;,&quot;端到端视频编码&quot;,&quot;多模态基准评测&quot;]"><strong><a href="../../venues/CIKM/2025/multimodal_rec/a_content_driven_micro_video_recommendation_dataset_at_scale/">A Content-Driven Micro-Video Recommendation Dataset at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布十亿级含原始视频模态的微视频推荐数据集 MicroLens，并系统基准 IDRec/VIDRec/VideoRec 三类范式，揭示端到端视频编码优于冻结特征的关键结论。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;多语言相关性判别&quot;,&quot;任务自适应预训练&quot;,&quot;翻译增强&quot;]"><strong><a href="../../venues/CIKM/2025/llm_rec/alibaba_international_e_commerce_product_search_competition_dcuragons_team_/">Alibaba International E-commerce Product Search Competition DcuRAGONs Team Technical Report</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 Gemma-3-12B 为骨干，用翻译增强+任务自适应预训练+类别感知查询分组划分，在阿里国际电商多语言相关性竞赛夺冠（F1_avg≈89.1）。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉画像生成&quot;,&quot;PhotoSet Transformer&quot;,&quot;OPQ 量化&quot;]"><strong><a href="../../venues/CIKM/2025/multimodal_rec/bilisting_modality_alignment_for_listings/">BiListing: Modality Alignment for Listings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成视觉画像 + PhotoSet Transformer 聚合多图，将 CLIP 图文对齐扩展到多图多文本的房源实体，线上 NDCG +0.425%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;多语言LLM&quot;,&quot;自蒸馏去噪&quot;,&quot;检索式查询增强&quot;]"><strong><a href="../../venues/CIKM/2025/cold_start_rec/csrm_llm_embracing_multilingual_llms_for_cold_start_relevance_matching_in_e/">CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>多语言 LLM 联合翻译任务、检索式查询增强与多轮自蒸馏，解决新兴市场零标注相关性匹配，离线 F1 0.965、线上缺陷率降 45.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;文本协同过滤&quot;,&quot;缩放定律&quot;,&quot;通用物品表征&quot;]"><strong><a href="../../venues/CIKM/2025/llm_rec/exploring_the_upper_limits_of_text_based_collaborative_filtering_using_larg/">Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 TCF 物品编码器从 125M 一路放大到 175B，系统检验缩放极限、通用表征、对 IDCF 的竞争力与零样本迁移四项命题。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;认知不确定性&quot;,&quot;集成CVAE&quot;,&quot;Sinkhorn散度&quot;]"><strong><a href="../../venues/CIKM/2025/cold_start_rec/harnessing_light_for_cold_start_recommendations_leveraging_epistemic_uncert/">Harnessing Light for Cold-Start Recommendations: Leveraging Epistemic Uncertainty to Enhance Performance in User-Item Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用集成 CVAE 生成暖化嵌入，并以 Sinkhorn 散度版 PaiDE 估计认知不确定性做优化，在冷启动 CTR 上稳定超越 CVAR 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多模态融合&quot;]"><strong><a href="../../venues/CIKM/2025/cold_start_rec/multi_modal_adaptive_mixture_of_experts_for_cold_start_recommendation/">Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双层 MoE（模态专属专家+动态门控融合）替代拼接/平均融合，在 Amazon 三域冷启动全库排序上 Recall@10 最高提升 16.17%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;非负矩阵分解&quot;,&quot;患者画像聚类&quot;,&quot;多源数据融合&quot;]"><strong><a href="../../venues/CIKM/2025/collaborative_filtering/patient_clustering_via_integrated_profiling_of_clinical_and_digital_data/">Patient Clustering via Integrated Profiling of Clinical and Digital Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用掩码约束非负低秩分解统一融合诊断码与门户浏览/搜索行为，同时产出可解释软聚类与患者嵌入，聚类与推荐双任务全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;EASE 闭式解&quot;,&quot;侧信息增强&quot;,&quot;冷物品先验&quot;]"><strong><a href="../../venues/CIKM/2025/cold_start_rec/shallow_autoencoding_recommender_with_cold_start_handling_via_side_features/">Shallow AutoEncoding Recommender with Cold Start Handling via Side Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户/物品侧特征拼入 EASE 输入矩阵，并用内容相似度矩阵作冷物品权重先验，闭式解统一处理双冷启动。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;知识图谱去噪&quot;,&quot;流行度自适应融合&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/spark_adaptive_low_rank_knowledge_graph_modeling_in_hybrid_geometric_spaces/">SPARK: Adaptive Low-Rank Knowledge Graph Modeling in Hybrid Geometric Spaces for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 低秩去噪 KG、SVD 初始化双曲/欧氏并行 GNN，并按物品流行度自适应融合多源信号，长尾推荐 R@10 最高提升 10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Folding-In&quot;,&quot;热启动&quot;,&quot;UltraGCN&quot;,&quot;低秩近似&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/ultra_fast_warm_start_solution_for_graph_recommendations/">Ultra Fast Warm Start Solution for Graph Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 SVD Folding-In 的闭式解迁移到 UltraGCN，用一次矩阵-向量乘精确更新热用户嵌入，比 SGD 微调快 30 倍且质量更优、覆盖率更高。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;合成查询生成&quot;,&quot;构造式相关性标签&quot;]"><strong><a href="../../venues/CIKM/2026/cold_start_rec/castle_contrastive_and_seed_guided_training_for_cold_start_natural_language/">CASTLE: Contrastive and Seed-Guided Training for Cold-Start Natural Language Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预订会话对比房源对+用户调研种子查询联合生成合成查询与构造式相关性标签，KL 1.01 对齐真实用户，9.2× 优于基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;物品索引&quot;,&quot;向量量化&quot;,&quot;流式推荐&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/merge_next_generation_item_indexing_paradigm_for_large_scale_streaming_reco/">MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以并查集动态生成聚类替代 VQ 盲分配，配合占用监控与由细到粗合并构建分层码本，工业 A/B 显著提升曝光多样性与互动。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;对话数据增强&quot;,&quot;知识图谱实体建模&quot;,&quot;对话-实体匹配&quot;]"><strong><a href="../../venues/CIKM/2026/agent_conversational_rec/on_mitigating_data_sparsity_in_conversational_recommender_systems/">On Mitigating Data Sparsity in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DACRS 用 LLM 两阶段对话增强 + KG 实体替换与相似度约束 + 对话引导注意力融合，缓解对话与实体双重稀疏，ReDial R@50 提升 11.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;流模型 (Normalizing Flow)&quot;,&quot;语义ID&quot;,&quot;乘积量化&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/tlow_flow_based_item_tokenizer_for_recommendation/">Tlow: Flow-based Item Tokenizer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可逆流模型将语义嵌入变换为标准正态潜空间后做PQ独立分词，配合码本引导对齐，微信线上新物品CTR提升11.64%</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图结构学习&quot;,&quot;非活跃用户冷启动&quot;,&quot;模仿学习&quot;]"><strong><a href="../../venues/DASFAA/2024/social_group_rec/learning_social_graph_for_inactive_user_recommendation/">Learning Social Graph for Inactive User Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 U2U 删噪边 + U2C 连活跃兴趣簇的知识驱动图结构学习与模仿学习，重塑社交图以服务非活跃用户；采样排序下 Flickr NDCG@10 相对提升 129.58%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图预训练&quot;,&quot;边信息融合&quot;,&quot;冷启动推荐&quot;]"><strong><a href="../../venues/DASFAA/2025/graph_rec/genet_unleashing_the_power_of_side_information_for_recommendation_via_hyper/">GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将异构边信息统一构建为超图，通过超边预测与双层自监督对比预训练，再以 LightGCN/GRU 微调，Top-N 与序列推荐最高提升 38%，冷启动提升 175%+。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;用户画像预处理&quot;,&quot;提供方公平性&quot;,&quot;校准多样性&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/how_to_diversify_any_personalized_recommender/">How to Diversify any Personalized Recommender?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出用户画像级预处理：用逻辑回归识别用户代表类别、userKNN 生成候选，再按 λ% 增删交互，使任意推荐器在保持精度的同时提升多样性与少数类曝光公平。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;多语种句子编码器&quot;,&quot;零样本跨语言迁移&quot;,&quot;领域适配&quot;]"><strong><a href="../../venues/ECIR/2025/cold_start_rec/news_without_borders_domain_adaptation_of_multilingual_sentence_embeddings_/">News Without Borders: Domain Adaptation of Multilingual Sentence Embeddings for Cross-lingual News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 PolyNews 多语新闻语料，以 DAE+MT 领域适配 LaBSE 得 NaSE，冻结嵌入+晚期点击融合在 ZS-XLT 冷启动新闻推荐上达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;新闻推荐&quot;,&quot;语言模型编码器&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/ECIR/2025/llm_rec/revisiting_language_models_in_neural_news_recommender_systems/">Revisiting Language Models in Neural News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 MIND 上统一复现 8 种 LM 作为新闻编码器，发现更大模型不必然更优，但显著利好冷启动用户。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;自然语言用户画像&quot;,&quot;学术文献推荐基准&quot;]"><strong><a href="../../venues/ECIR/2026/llm_rec/scinup_natural_language_user_interest_profiles_for_scientific_literature_re/">SciNUP: Natural Language User Interest Profiles for Scientific Literature Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建千名作者的自然语言兴趣画像学术推荐基准，系统对比稀疏/稠密/LLM重排，RRF 融合达 NDCG@10 0.4481。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;文本嵌入白化&quot;,&quot;冷启动推荐&quot;]"><strong><a href="../../venues/ICDE/2024/sequential_rec/are_id_embeddings_necessary_whitening_pre_trained_text_embeddings_for_effec/">Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings for Effective Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出ZCA白化预训练文本嵌入并融合全/松弛白化表示，在4个Amazon/食谱数据集全库排序上超越ID+文本基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;严格冷启动&quot;,&quot;冻结图传播&quot;,&quot;多模态知识图谱&quot;]"><strong><a href="../../venues/ICDE/2024/cold_start_rec/firzen_firing_strict_cold_start_items_with_frozen_heterogeneous_and_homogen/">Firzen: Firing Strict Cold-Start Items with Frozen Heterogeneous and Homogeneous Graphs for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用冻结协作知识图与冻结同构图将热启动物品的协同信号单向传播给严格冷启动物品，全库排序下 Beauty 冷启动 M@20 提升 22%，热启动持平 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;子图学习&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="../../venues/ICDE/2024/graph_rec/knowledge_enhanced_recommendation_with_user_centric_subgraph_network/">Knowledge-Enhanced Recommendation with User-Centric Subgraph Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KUCNet 为每个用户-物品对构建U-I子图，用注意力GNN与PPR剪枝的用户中心计算图实现个性化、归纳式推荐，全库排序下Last-FM Recall@20较KGIN提升23.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;异构交互建模&quot;,&quot;多头自注意力&quot;,&quot;预测上下文采样&quot;]"><strong><a href="../../venues/ICDE/2025/cold_start_rec/all_in_one_heterogeneous_interaction_modeling_for_cold_start_rating_predict/">All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HIRE 用纯 MHSA 在采样的预测上下文上端到端学习用户/物品/属性三层异构交互，三种冷启动场景下 Precision/NDCG/MAP 平均超 12 个基线 0.21/0.29/0.22。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;归一化流&quot;,&quot;神经过程&quot;,&quot;冷启动跨域迁移&quot;]"><strong><a href="../../venues/ICDE/2025/cross_domain_multi_scenario/personalized_multi_interest_modeling_for_cross_domain_recommendation_to_col/">Personalized Multi-Interest Modeling for Cross-Domain Recommendation to Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用归一化流增强神经过程，将单峰高斯先验转为多峰分布以建模用户多兴趣，并引入偏好池捕获跨用户共同偏好，五场景 MAE 平均降约 45%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;不确定性量化&quot;]"><strong><a href="../../venues/ICDE/2026/cold_start_rec/user_adaptive_meta_learning_for_cold_start_medication_recommendation_with_u/">User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出两级元自适应（自自适应+同伴自适应）与不确定性过滤的 MetaDrug，在 MIMIC-III/AKI 冷启动患者上 PRAUC 提升约 2.5%-6.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2026&quot;,&quot;生成式推荐&quot;,&quot;双曲几何&quot;,&quot;残差量化语义ID&quot;,&quot;长尾推荐&quot;]"><strong><a href="../../venues/ICDM/2026/generative_rec/hyprq_vae_hyperbolic_item_indexing_for_long_tail_aware_generative_recommend/">HypRQ-VAE: Hyperbolic Item Indexing for Long-Tail-Aware Generative Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 RQ-VAE 语义 ID 量化搬进 Poincaré 双曲球，用 Möbius 加减替代欧氏残差，长尾 Hit@10 提升 52.7%，全库排序验证。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;重排/推荐策略分配&quot;,&quot;双边公平&quot;,&quot;Price of Fairness&quot;,&quot;冷启动误估&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/user_item_fairness_tradeoffs_in_recommendations/">User-Item Fairness Tradeoffs in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>把用户-物品双边公平建成带物品公平约束的极小极大优化，解析证明「多样性红利」与「冷启动误估放大」，并在 arXiv 原型系统上实测验证。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;经验贝叶斯&quot;,&quot;协变量调制先验&quot;,&quot;矩阵分解&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/collaborative_filtering/covariate_moderated_empirical_bayes_matrix_factorization/">Covariate-moderated Empirical Bayes Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用协变量调制的经验贝叶斯先验统一矩阵分解，MovieLens 100K 上 RMSE 全面优于 EBMF/MFAI/VAE 系基线。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图信号频域分解&quot;,&quot;谱对比正则&quot;,&quot;冷启动鲁棒性&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/multimodal_rec/structured_spectral_reasoning_for_frequency_adaptive_multimodal_recommendat/">Structured Spectral Reasoning for Frequency-Adaptive Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多模态图信号分解为等能量频带，用频带掩码一致性训练、低秩超谱算子与谱对比正则实现频率自适应融合，全库排序下冷启动 Recall@20 提升约 10%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;冷启动&quot;,&quot;召回/主动elicitation&quot;,&quot;AI Persona Prior&quot;,&quot;Bayesian Experimental Design&quot;,&quot;闭式后验更新&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/cold_start_rec/adaptive_querying_with_ai_persona_priors/">Adaptive Querying with AI Persona Priors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 人物画像构建有限混合贝叶斯先验，实现闭式后验更新与自适应提问，在冷启动用户/题目场景下优于 CAT/IRT 基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;智能体推荐&quot;,&quot;能力画像匹配&quot;,&quot;长尾一次性监督&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/agent_conversational_rec/agentselect_benchmark_for_narrative_query_to_agent_recommendation/">AgentSelect: Benchmark for Narrative Query-to-Agent Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将异构评测产物统一为叙事查询到可部署智能体配置的推荐基准，揭示长尾一次性监督下内容感知匹配的必要性。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;约束感知决策&quot;,&quot;可持续饮食推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/cold_start_rec/eating_for_a_sustainable_planet_personalized_sustainable_diet_recommendatio/">Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将可持续饮食推荐重构为约束感知决策问题，用可学习用户约束阈值替代偏好建模，在自建 SusDiet 上兼顾偏好精度与四维可持续性。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;黑盒隐私评测&quot;,&quot;成员推断攻击&quot;,&quot;无影子攻击&quot;]"><strong><a href="../../venues/IJCAI/2024/debias_fairness_industrial/shadow_free_membership_inference_attacks_recommender_systems_are_more_vulne/">Shadow-Free Membership Inference Attacks: Recommender Systems Are More Vulnerable Than You Thought</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出无影子成员推断攻击：空画像热门锚点+自适应用户距离比较，黑盒免训练，3.7秒取得高TPR低FPR。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;三部图元路径&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/KDD/2024/graph_rec/consistency_and_discrepancy_based_contrastive_tripartite_graph_learning_for/">Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可达/不可达元路径导出一致性与差异性系数，经 GCN 极限理论免训练预计算，配合 CD 对比损失解决三部图推荐极端冷启动。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态对比学习&quot;,&quot;关系感知预训练&quot;]"><strong><a href="../../venues/KDD/2024/cold_start_rec/larp_language_audio_relational_pre_training_for_cold_start_playlist_continu/">LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出语言-音频三阶段对比预训练 LARP，在无重叠冷启动歌单续写的全库排序中显著超越多模态基线。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;MAML 元学习&quot;,&quot;Meta Embedding&quot;,&quot;工业级个性化&quot;]"><strong><a href="../../venues/KDD/2024/cold_start_rec/limaml_personalization_of_deep_recommender_models_via_meta_learning/">LiMAML: Personalization of Deep Recommender Models via Meta Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将推荐网络解耦为 Meta/Global 双块，仅对 Meta Block 做 MAML 元学习并离线蒸馏成定长 meta embedding，实现百亿参数模型的少样本个性化在线服务，线上 CTR 提升 2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;知识图谱推荐&quot;,&quot;健康行为干预&quot;]"><strong><a href="../../venues/KDD/2024/graph_rec/nudgerank_digital_algorithmic_nudging_for_personalized_health/">NudgeRank: Digital Algorithmic Nudging for Personalized Health</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可扩展异构知识图谱+注意力图卷积实现百万级人群个性化健康提醒，离线Precision@3提升45%，线上日步数+6.17%、MVPA+7.61%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;超网络生成特征图&quot;,&quot;元学习暖启动&quot;]"><strong><a href="../../venues/KDD/2024/cold_start_rec/warming_up_cold_start_ctr_prediction_by_learning_item_specific_feature_inte/">Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用超网络为每个新物品生成专属特征交互图，配合可证任意阶交互的定制 GNN 与元学习暖启动，在 MovieLens/Taobao 冷启动与三阶段暖启动 AUC 上全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;离线用户嵌入&quot;,&quot;超长用户序列&quot;,&quot;漏斗摘要编码器&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/dv365_extremely_long_user_history_modeling_at_instagram/">DV365: Extremely Long User History Modeling at Instagram</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以离线用户嵌入替代端到端长序列建模，用多切片+漏斗摘要将 4 万~7 万条历史压缩为 58×256 嵌入，15 个下游模型 NE 平均提升 0.4%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;冷启动偏好迁移&quot;,&quot;Classifier-Free Guidance&quot;]"><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/exploring_preference_guided_diffusion_model_for_cross_domain_recommendation/">Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用源域交互历史构造偏好引导信号，逐步注入扩散逆向过程生成目标域冷启动用户表征，三场景 MAE 最高降 32.60%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据生成/基准构建&quot;,&quot;概率生成模型&quot;,&quot;长尾分布混合&quot;,&quot;可控合成数据&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/flexible_generation_of_preference_data_for_recommendation_analysis/">Flexible Generation of Preference Data for Recommendation Analysis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HYDRA 以用户-物品匹配、物品流行度、用户活跃度三因子混合分布生成可控合成偏好数据，复现真实基准的度分布与算法排序。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;数据侧优化/核心集选择&quot;,&quot;最优传输&quot;,&quot;梯度范数界&quot;,&quot;组级选择&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/goracs_group_level_optimal_transport_guided_coreset_selection_for_llm_based/">GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用最优传输+梯度范数构造测试损失上界代理目标，配合贪心初始化-交换精炼两阶段算法做组级核心集选择，20%训练开销超越全量微调。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;兴趣单元聚类&quot;,&quot;分层序列建模&quot;,&quot;限库存推荐&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/iu4rec_interest_unit_based_product_organization_and_recommendation_for_e_co/">IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>闲鱼C2C限库存场景下以兴趣单元(IU)重构商品组织与两阶段推荐，IU-Boosted网络离线AUC+1.91%、线上CTR+2.90%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;元学习&quot;,&quot;流行度感知任务切分&quot;,&quot;在线流式训练&quot;]"><strong><a href="../../venues/KDD/2025/cold_start_rec/online_item_cold_start_recommendation_with_popularity_aware_meta_learning/">Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>按流行度固定切分元学习任务并配冷启动增强器，流式在线场景下冷启动召回相对 SOTA 提升 11%–74%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;用户专属特征图&quot;,&quot;向量量化码本&quot;,&quot;场景感知 GLU&quot;]"><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/perscen_learning_personalized_interaction_pattern_and_scenario_preference_f/">PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用用户专属特征图 + VQ 码本蒸馏场景感知偏好 + 渐进式 GLU 融合，在 KuaiRand/Alimama 四场景召回上全面超越 ADIN/SASS/M5。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;可微聚类&quot;,&quot;条件召回&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/synergizing_implicit_and_explicit_user_interests_a_multi_embedding_retrieva/">Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pinterest 工业级多嵌入召回框架，用可微聚类建模隐式兴趣、条件召回建模显式兴趣，线上 HF Repins +1.09%、A-Pincepts +0.81%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;评测基准&quot;,&quot;心智世界&quot;,&quot;多模态知识推理&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/the_mental_world_of_large_language_models_in_recommendation_a_benchmark_on_/">The Mental World of Large Language Models in Recommendation: A Benchmark on Association, Personalization, and Knowledgeability</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 LRWorld 基准（38K 样本/23M tokens），从关联、个性化、知识性三尺度十因素 31 任务系统评测数十个 LLM，揭示其在神经嵌入检索上的致命短板。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲图神经网络&quot;,&quot;层次聚类与探索利用&quot;,&quot;语义-协同对齐&quot;]"><strong><a href="../../venues/KDD/2026/collaborative_filtering/breaking_information_cocoons_a_hyperbolic_framework_for_balancing_explorati/">Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>洛伦兹双曲空间对齐 LLM 语义与协同信号，并用 Dasgupta 代价驱动的无超参层次聚类实现用户可调探索-利用权衡，效用最高 +5.49%、多样性 +11.39%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;冷启动POI预测&quot;,&quot;因果推断&quot;,&quot;时空图神经网络&quot;]"><strong><a href="../../venues/KDD/2026/poi_spatiotemporal_rec/causalpoi_spatio_temporal_graph_based_causal_modeling_for_cold_start_poi_ch/">CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用功能交互图+结构对齐的干预/对照图做冷启动POI签到量预测与因果提升估计，四区域RMSE最高降57.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;伪标签纠正&quot;,&quot;多模态内容相似&quot;]"><strong><a href="../../venues/KDD/2026/cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID构建&quot;,&quot;生成式检索&quot;,&quot;工业数据集&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/forge_forming_semantic_identifiers_for_generative_retrieval_in_industrial_d/">FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统化拆解 SID 构建全流程（多模态特征/量化编码/冲突后处理），提出免训练的 Embedding HitRate 与 Gini 系数代理指标，淘宝线上交易量 +0.35%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026 TSMO&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;隐式意图生成&quot;,&quot;教师学生蒸馏&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Prior-Fitted Network&quot;,&quot;免更新推理&quot;,&quot;合成先验&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/one_sequential_recommendation_model_pretrained_from_synthetic_priors_predic/">One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 hDCSBM 合成先验预训练 PFN，以支持集条件化实现免更新推理的序列推荐，五基准平均超最优基线 7.53%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;冷启动&quot;,&quot;召回/精排/重排全链路&quot;,&quot;内容冷启动&quot;,&quot;探索与去偏&quot;,&quot;工业级度量框架&quot;]"><strong><a href="../../venues/KDD/2026/cold_start_rec/pinequalizer_full_funnel_content_exploration_and_debiasing_system_at_pinter/">PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向工业级搜索/推荐全漏斗的新内容冷启动系统，以去偏替代纯探索，配合三层可扩展度量框架，线上新鲜内容曝光提升350%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;不确定性量化&quot;,&quot;UCB 探索&quot;,&quot;风险规避降权&quot;]"><strong><a href="../../venues/KDD/2026/rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;单分支权重共享&quot;,&quot;多模态对齐&quot;,&quot;缺失模态鲁棒性&quot;]"><strong><a href="../../venues/RecSys/2024/cold_start_rec/a_multimodal_single_branch_embedding_network_for_recommendation_in_cold_sta/">A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用权重共享的单分支网络统一编码交互与多模态内容，Item/User-SiBraR 在 Full-Ranking 协议下物品冷启动 nDCG@10 最高相对提升 70%，并显著缓解缺失模态与模态鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;精排&quot;,&quot;Semantic ID&quot;,&quot;RQ-VAE&quot;,&quot;SentencePiece 子词哈希&quot;,&quot;冷启动泛化&quot;]"><strong><a href="../../venues/RecSys/2024/generative_rec/better_generalization_with_semantic_ids_a_case_study_in_ranking_for_recomme/">Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 RQ-VAE 把冻结内容向量压成 8 层语义 ID，再以 SPM 子词哈希替代 YouTube 随机哈希视频 ID，冷启与长尾切片 CTR 提升且整体质量不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;条件检索 Conditional Retrieval&quot;,&quot;双塔 + In-batch 负采样&quot;,&quot;特征交叉&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/bootstrapping_conditional_retrieval_for_user_to_item_recommendations/">Bootstrapping Conditional Retrieval for User-to-Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品侧 topic 元数据为普通 user-item 互动样本合成条件，在用户塔内做 user×condition 早期特征交叉，无需条件专属训练数据即可支持条件检索；Pinterest 全量上线带来 +0.26% WAU，服务成本仅为过滤式双塔的 1/4。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/候选集排序&quot;,&quot;新闻推荐数据集&quot;,&quot;Beyond-Accuracy 评估&quot;,&quot;性别导向偏置&quot;]"><strong><a href="../../venues/RecSys/2024/debias_fairness_industrial/eb_nerd_a_large_scale_dataset_for_news_recommendation/">EB-NeRD: A Large-Scale Dataset for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>发布百万用户丹麦新闻推荐数据集与 RecSys&#x27;24 Challenge 基准，通过打乱位置的曝光日志揭示流行度偏置与性别导向差异。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;长尾查询&quot;,&quot;LLM 数据增强&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/embedding_based_retrieval_for_long_tail_search_queries_in_ecommerce/">Embedding based retrieval for long tail search queries in ecommerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Best Buy 用 LLM 合成查询、领域 RoBERTa 预训练、q2q→q2p 微调与权重融合优化双塔长尾召回，线上转化 +3%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元数据对齐正则&quot;,&quot;闭式 EASE 变体&quot;,&quot;语义特征嵌入&quot;]"><strong><a href="../../venues/RecSys/2024/cold_start_rec/marec_metadata_alignment_for_cold_start_recommendation/">MARec: Metadata Alignment for cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用一条 item-item 相似度对齐正则把元数据嵌入注入 EASE/VAE 骨干，冷启 hr@10 提升 8.4%–53.8%，训练快两个数量级且热启不掉点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;Prompt Tuning&quot;,&quot;对比学习去偏&quot;]"><strong><a href="../../venues/RecSys/2024/cold_start_rec/prompt_tuning_for_item_cold_start_recommendation/">Prompt Tuning for Item Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以高价值正反馈（pinnacle feedback）替代内容描述作为软提示，配合物品级个性化提示网络与双提示增强损失，在 4 个数据集 leave-one-out 采样排序上显著超越 SOTA，并落地亿级短视频平台 A/B 实验。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;场景关系图 SRG&quot;,&quot;动态冷启动&quot;,&quot;自适应专家混合 AEM&quot;]"><strong><a href="../../venues/RecSys/2024/cross_domain_multi_scenario/scene_wise_adaptive_network_for_dynamic_cold_start_scenes_optimization_in_c/">Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用场景关系图+用户感知相似度注意力为动态冷启动场景迁移先验，并以可微门控 MoE 自适应分配专家，线上 CTR +5.64%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;通用文本嵌入&quot;,&quot;零样本迁移&quot;,&quot;有效维度&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/do_we_really_need_specialization_evaluating_generalist_text_embeddings_for_/">Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>实证通用文本嵌入(GTE)零样本即超越领域微调模型，并以PCA有效维度揭示空间利用率与压缩收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;RQ-VAE&quot;,&quot;前缀 n-gram 参数化&quot;]"><strong><a href="../../venues/RecSys/2025/generative_rec/enhancing_embedding_representation_stability_in_recommendation_systems_with/">Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 Semantic ID prefix-ngram 分层前缀参数化，用语义聚类碰撞替代随机哈希，缓解 ID 漂移与长尾稀疏，Meta 线上 NE 提升 0.15% 且 A/A 方差降 43%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;评测与元评估&quot;,&quot;LLM-as-Evaluator&quot;,&quot;Serendipity&quot;,&quot;用户模拟&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/exploring_the_potential_of_llms_for_serendipity_evaluation_in_recommender_s/">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 模拟用户对推荐结果做惊喜度（serendipity）打分，在淘宝与 MovieLens 两个真实用户调研数据集上做元评估，最优配置 Pearson 相关达 21.5%，显著超越传统代理指标。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;物品中心探索&quot;,&quot;Beta 后验过滤&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;内容初始化&quot;,&quot;有界 delta 校正&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;零样本跨域迁移&quot;,&quot;域语义偏置&quot;,&quot;序列模式聚类&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;流行度偏置&quot;,&quot;向量模长重标定&quot;,&quot;物品公平性&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/on_inherited_popularity_bias_in_cold_start_item_recommendation/">On Inherited Popularity Bias in Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示生成式冷启动模型继承热模型流行度偏置，提出基于向量模长重标定的免训练后处理去偏方法，在三个多媒体数据集上显著提升物品公平性与曝光多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;主题偏置&quot;,&quot;公平性评估&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/reading_between_the_lines_a_study_of_thematic_bias_in_book_recommender_syst/">Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 BERTopic 为 Book-Crossing 打主题标签，构建数据-推荐-用户三阶段主题偏置评估框架，揭示长尾小众读者被主流主题同质化的现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;冷热阈值&quot;,&quot;p-core 过滤&quot;,&quot;评估协议&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/recommendation_is_a_dish_better_served_warm/">Recommendation Is a Dish Better Served Warm</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统量化冷启动阈值：物品侧训练交互扫描+用户侧推理历史截断，揭示 6–15 条交互的冷热分界并批判 p-core 过滤的任意性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;社交图种子扩散&quot;,&quot;KNN 邻居扩展&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Two-Tower&quot;,&quot;内容特征召回&quot;,&quot;ANN 分层索引&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/suggest_complement_inspire_story_of_two_tower_recommendations_at_allegro_co/">Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以内容特征双塔为统一底座，仅改查询塔与索引逻辑即复用至相似/互补/灵感三类召回，两年线上 A/B 验证 CTR 与 GMV 双增。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;增量向量索引&quot;,&quot;双哈希变更分类&quot;,&quot;控制器式对话编排&quot;]"><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/conversational_recommendation_over_live_e_commerce_catalogues_with_self_ref/">Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向实时电商商品库的自刷新检索式对话购物助手，以双哈希增量同步替代全量重建，同步开销降至全量 1.8%–12.3%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;多模态大模型&quot;,&quot;代理嵌入对齐&quot;]"><strong><a href="../../venues/RecSys/2026/cold_start_rec/idproxy_ctr_prediction_with_multimodal_llms_for_cold_start_recommendation_a/">IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MLLM 生成代理 ID 嵌入，经粗到细两阶段对齐工业 CTR 排序器，新笔记在线 AUC 提升约 2 倍于全局流量。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;反事实数据增强&quot;,&quot;长尾罕见药推荐&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/improving_rare_medication_recommendation_with_counterfactual_data_augmentat/">Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 反事实数据增强 + 病历摘要指令微调 + 逐药自回归生成，缓解罕见药长尾失效，Group-1 Jaccard 最高 +30.2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;稀疏嵌入&quot;,&quot;多模态内容表征&quot;]"><strong><a href="../../venues/RecSys/2026/cold_start_rec/learning_sparse_representations_of_multimodal_content_for_enhanced_cold_ite/">Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多模态内容冷启动表征改为 top-K 稀疏嵌入，并用 α-entmax 预稀疏激活诱导锐化与去噪，冷启动 NDCG@20 提升 16.6%–75.5% 且存储更低。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔语义检索&quot;,&quot;混合召回分数校准&quot;,&quot;多语言检索&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/multilingual_semantic_retrieval_for_apple_music_search/">Multilingual Semantic Retrieval for Apple Music Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>305M 多语双塔 + 课程式多目标微调，配合分位数校准混合召回，离线 Hit@10 相对提升 69%，线上尾部查询 CR 提升 7.93% 且无结果率降 86%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;自适应层次聚类&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/not_all_branches_are_equal_adaptive_semantic_id_construction_for_generative/">Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将语义 ID 重构为树路径，用轮廓系数自适应分支+概率质量约束构建 AST，五域 Amazon 全库排序平均相对提升 13.08%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;冷启动路由&quot;,&quot;LambdaRank&quot;,&quot;信息流生态&quot;]"><strong><a href="../../venues/RecSys/2026/cold_start_rec/predicting_custom_feed_returns_for_new_bluesky_posts_a_prospective_study/">Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将新帖作为查询、信息流作为候选的冷启动路由任务，构建 collect-first/label-later 基准，LambdaRank 达 cR@10 0.7361。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;概率用户聚类&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;课程学习&quot;,&quot;流行度去偏&quot;,&quot;连续训练&quot;]"><strong><a href="../../venues/RecSys/2026/cold_start_rec/soft_curriculum_learning_for_optimizing_fresh_and_generalized_recommendatio/">Soft Curriculum Learning for Optimizing Fresh and Generalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用损失退火与图内权重替代刚性数据过滤，在连续训练中打破流行度反馈环，线上冷启动互动 +5.70%、满意度 +0.54% 且吞吐无损。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;向量量化&quot;,&quot;在线解码&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/tokens_are_all_you_need_dual_purpose_semantic_ids_for_achieving_llm_level_i/">Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RQ-VAE 语义ID 同时承担协同身份与内容重建双职责，以 SiDec 在线解码替代稠密向量 I/O，检索 HitRate@100 提升至 0.2910 且吞吐较原始稠密嵌入快 20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;缩放律&quot;,&quot;多Token预测&quot;,&quot;冷启动语义塔&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/towards_generalizable_and_efficient_large_scale_generative_recommenders/">Towards Generalizable and Efficient Large-Scale Generative Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成式推荐骨干从2M扩至1B，用偏移幂律缩放律诊断任务天花板，并以MTP、采样softmax与语义塔解决缓存陈旧与冷启动，1M用户影子MRR全线提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;零观测唤醒&quot;,&quot;逐维门控&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排/候选排序&quot;,&quot;多语言新闻推荐&quot;,&quot;跨语言迁移&quot;]"><strong><a href="../../venues/SIGIR/2024/cold_start_rec/mind_your_language_a_multilingual_dataset_for_cross_lingual_news_recommenda/">MIND Your Language: A Multilingual Dataset for Cross-lingual News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>构建14语言平行新闻推荐数据集xMIND，系统评测零样本/少样本跨语言迁移，揭示mPLM仍显著掉点。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;冷启动引导&quot;,&quot;User Simulation&quot;,&quot;Preference Elicitation&quot;,&quot;Off-Policy Evaluation&quot;]"><strong><a href="../../venues/SIGIR/2024/agent_conversational_rec/minimizing_live_experiments_in_recommender_systems_user_simulation_to_evalu/">Minimizing Live Experiments in Recommender Systems: User Simulation to Evaluate Preference Elicitation Policies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用反事实鲁棒的用户状态/会话生成器与生产级模拟器，在 YouTube Music 新用户引导场景中替代小样本在线实验，准确预测策略上线指标并正确排序候选策略。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态缺失&quot;,&quot;不变学习&quot;,&quot;新物品推荐&quot;]"><strong><a href="../../venues/SIGIR/2024/cold_start_rec/multimodality_invariant_learning_for_multimedia_based_new_item_recommendati/">Multimodality Invariant Learning for Multimedia-Based New Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>面向多模态新物品推荐，用跨模态对齐与循环 Mixup 构造缺失模态环境，IRM 学习不变用户偏好；新物品全库排序下 NDCG@20 最高提升约 17.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Transformer&quot;,&quot;Scaling Law&quot;,&quot;特征编码&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/scaling_sequential_recommendation_models_with_transformers/">Scaling Sequential Recommendation Models with Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用特征编码替代物品ID嵌入，使SRT参数与目录规模解耦，在Amazon全量数据上验证Transformer序列推荐的缩放律与预训练微调收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱构建&quot;,&quot;跨类型资源推荐&quot;,&quot;模型血缘追踪&quot;]"><strong><a href="../../venues/SIGIR/2025/graph_rec/benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;神经重排&quot;,&quot;长尾学生表征增强&quot;,&quot;习题推荐&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;贴纸检索&quot;,&quot;用户建模&quot;]"><strong><a href="../../venues/SIGIR/2025/multimodal_rec/u_sticker_a_large_scale_multi_domain_user_sticker_dataset_for_retrieval_and/">U-Sticker: A Large-Scale Multi-Domain User Sticker Dataset for Retrieval and Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建含 22K 用户、370K 贴纸、8.3M 消息的多领域多语言贴纸数据集，支撑个性化贴纸检索与用户建模。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Aspect-Conditioned Scoring&quot;,&quot;异构图消息传递&quot;,&quot;作者谱系&quot;]"><strong><a href="../../venues/SIGIR/2026/graph_rec/aspect_aware_content_based_recommendations_for_mathematical_research_papers/">Aspect-Aware Content-Based Recommendations for Mathematical Research Papers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向数学论文的 aspect 条件异构图推荐，构建 GoldRiM/SilverRiM 数据集，AchGNN 在 Full-Ranking 下 MRR 相对 GraphCL 提升 26%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;复购节奏建模&quot;,&quot;集合注意力&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以日历时间二值信号+共享多尺度时序卷积建模物品复购节奏，再用 ISAB 建模跨物品依赖，在四个杂货数据集与千万级生产系统上取得显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;批采样&quot;,&quot;不可能负样本&quot;,&quot;用户表征学习&quot;]"><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别地理约束下 in-batch 负样本中的“不可能负样本”问题，提出区域同质批采样 RCBS，仅改批构造即显著提升用户表征与线上指标。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;知识缺口估计&quot;,&quot;选择性提示增强&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/filling_the_gaps_selective_knowledge_augmentation_for_llm_recommenders/">Filling the Gaps: Selective Knowledge Augmentation for LLM Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 CKP 比较式知识探测量化 LLM 对物品的协同知识缺口，仅对知识贫乏物品选择性注入属性与参考物品，免训练提升 Recall@1 与长尾覆盖。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自适应门控融合&quot;,&quot;双层对齐&quot;,&quot;长尾物品&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习消除空间结构冲突，全库排序下尾部长尾 H@10 最高提升 427%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;冷启动&quot;,&quot;生成式推荐&quot;,&quot;模型编辑&quot;,&quot;语义ID&quot;]"><strong><a href="../../venues/SIGIR/2026/cold_start_rec/genrecedit_adapting_model_editing_for_generative_recommendation_with_cold_s/">GenRecEdit: Adapting Model Editing for Generative Recommendation with Cold-Start Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 NLP 模型编辑迁移至生成式推荐，用逐位置定位-编辑+One-One 触发注入冷启动物品 SID 模式，仅需重训 9.5% 时间即大幅缓解冷启动坍缩。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;RAG&quot;,&quot;共购关系&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/itemrag_item_based_retrieval_augmented_generation_for_llm_based_recommendat/">ItemRAG: Item-Based Retrieval-Augmented Generation for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 RAG 从粗粒度用户历史检索转向细粒度物品级检索，融合共购与语义信息，标准与冷启动场景全面超越用户级 RAG 基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;Prior-Fitted Network&quot;,&quot;In-Context Learning&quot;,&quot;合成数据先验&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用结构因果先验合成点击流预训练 PFN，单次前向即可零样本适配新域序列推荐，8 个基准 Full-Ranking 上零样本 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;RQ-VAE码本坍缩&quot;,&quot;生成式检索&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Snapchat 工业级 SID 落地：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，线上短视频分享 +4.39%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;α-entmax 稀疏对比学习&quot;,&quot;内容空间知识蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>抛弃 CF 对齐、纯内容空间 item-item 相似度建模，用 α-entmax 稀疏化 sampled softmax 并支持内容空间蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测协议与数据审计&quot;,&quot;数据划分&quot;,&quot;时间泄漏&quot;,&quot;冷启动暴露&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/splitlight_an_exploratory_toolkit_for_recommender_systems_datasets_and_spli/">SplitLight: An Exploratory Toolkit for Recommender Systems Datasets and Splits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>开源数据集与划分审计工具箱，量化时间泄漏、冷启动暴露、重复消费与分布漂移，揭示预处理决策可使 NDCG@10 变动 1.2%–60%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回/表征增强&quot;,&quot;LVLM 适配&quot;,&quot;模态解耦 LoRA&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/structural_and_disentangled_adaptation_of_large_vision_language_models_for_/">Structural and Disentangled Adaptation of Large Vision Language Models for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态内结构软教师对齐 LVLM 跨模态表征，并以专家化门控 LoRA 解耦梯度流，即插即用提升多模态与序列推荐，长尾增益最高 18.70%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;用户表征初始化&quot;,&quot;语义聚类&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/well_begun_is_half_done_training_free_and_model_agnostic_semantically_guara/">Well Begun is Half Done: Training-Free and Model-Agnostic Semantically Guaranteed User Representation Initialization for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用交互物品模态特征+聚类全局语义免训练初始化用户表征，弥合用户-物品语义鸿沟，六模型四数据集Recall@5最高提升36.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;免训练推荐&quot;,&quot;时间衰减打分&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;多智能体强化学习&quot;,&quot;负迁移缓解&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/SIGIR-AP/2025/cross_domain_multi_scenario/marco_a_cooperative_knowledge_transfer_framework_for_personalized_cross_dom/">MARCO: A Cooperative Knowledge Transfer Framework for Personalized Cross-domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多源跨域推荐建模为协作式多智能体强化学习，每源域一个 Agent 估计贡献并加熵正则，冷启动 MAE 最高降 32.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回+精排两阶段&quot;,&quot;语义增强Transformer&quot;,&quot;技能感知表征&quot;,&quot;标签分布学习&quot;]"><strong><a href="../../venues/TKDD/2025/ctr_feature_interaction/jobformer_skill_aware_job_recommendation_with_semantic_enhanced_transformer/">JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用局部-全局注意力 Transformer 聚合邻居 JD 补全语义，并以技能分布引导两阶段召回+CTR排序，Recall@20 较 NRMS 提升 2.31%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Prompt-Tuning&quot;,&quot;冷启动&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/personalized_prompt_for_sequential_recommendation/">Personalized Prompt for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户画像生成个性化软前缀 Prompt 替代全参微调，配合 Prompt 导向对比学习，在三大数据集的 Few-shot/Zero-shot 冷启动上稳定超越 Fine-tuning。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平与可信推荐&quot;,&quot;端到端联邦训练&quot;,&quot;联邦推荐&quot;,&quot;个性化联邦学习 PFL&quot;,&quot;双层优化&quot;]"><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/personalized_recommendation_models_in_federated_settings_a_survey/">Personalized Recommendation Models in Federated Settings: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首篇以“个性化模型”为核心的 FedRecSys 综述，给出双层优化定义并系统梳理四类挑战与解法。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;ID-centric Pre-training&quot;,&quot;Cross-domain ID Matcher&quot;,&quot;行为-文本对比对齐&quot;]"><strong><a href="../../venues/TOIS/2025/cold_start_rec/id_centric_pre_training_for_recommendation/">ID-centric Pre-training for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以 ID 嵌入为中心的多域预训练范式 IDP，用行为感知跨域 ID 匹配器 CDIM 将预训练 ID 嵌入迁移至新域物品，冷/热启动均显著超越 UniSRec 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;个性化预训练&quot;,&quot;专家发现&quot;,&quot;投票导向任务&quot;]"><strong><a href="../../venues/TOIS/2025/llm_rec/pept_expert_finding_meets_personalized_pre_training/">PEPT: Expert Finding Meets Personalized Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将专家历史问答、投票分与专家ID统一进BERT预训练，用问题级MLM与投票预测双任务学习个性化兴趣与专长，六域CQA全库排序MRR全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与工业架构&quot;,&quot;离线评估与模型验证&quot;,&quot;邻域切片评估&quot;,&quot;子群公平性&quot;,&quot;Welch t 检验&quot;]"><strong><a href="../../venues/TORS/2024/debias_fairness_industrial/group_validation_in_recommender_systems_framework_for_multi_layer_performan/">Group Validation in Recommender Systems: Framework for Multi-layer Performance Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 PCC-KNN 将评分数据切成可重叠邻域，以损失差+Welch 单侧 t 检验筛出关键邻域，暴露被全局均值掩盖的小用户群失效。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;联邦学习&quot;,&quot;神经网络压缩&quot;,&quot;隐私保护&quot;]"><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/a_privacy_preserving_system_for_movie_recommendations_using_federated_learn/">A Privacy Preserving System for Movie Recommendations Using Federated Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端到端联邦电影推荐系统，提出 FedQ 队列式联邦训练缓解非独立同分布与小样本问题，结合 NNC 压缩降低通信开销，在 MovieLens 25M 上验证 16 万客户端规模。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;冷启动&quot;,&quot;综述&quot;,&quot;数据稀缺&quot;,&quot;知识迁移&quot;,&quot;自监督学习&quot;]"><strong><a href="../../venues/TORS/2025/cold_start_rec/data_scarcity_in_recommendation_systems_a_survey/">Data Scarcity in Recommendation Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理数据稀缺下五类缓解范式（增广/自监督/迁移/广度学习/知识图谱），并给出挑战与未来方向。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话建模&quot;,&quot;删失Weibull损失&quot;,&quot;多类型动作序列&quot;]"><strong><a href="../../venues/TORS/2025/sequential_rec/recommending_target_actions_outside_sessions_in_the_data_poor_insurance_dom/">Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向保险数据稀缺场景的跨会话 RNN 推荐，预测会话外购买目标，Full-Ranking 下 HR@3 较 SKNN_EB 提升 3.11%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;预训练音频表征&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/TORS/2026/multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;LLM 行为模拟&quot;,&quot;耦合漏斗过滤&quot;,&quot;严格冷启动&quot;]"><strong><a href="../../venues/WSDM/2025/cold_start_rec/large_language_model_simulator_for_cold_start_recommendation/">Large Language Model Simulator for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 模拟冷物品的用户交互行为，并以耦合漏斗（FAISS 过滤 + LLM 精炼）把十亿级候选压到数百，离线冷召回 NDCG 提升 37%+，线上冷启 GMV +16.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;双曲嵌入&quot;,&quot;域解耦&quot;]"><strong><a href="../../venues/WSDM/2025/cross_domain_multi_scenario/review_based_hyperbolic_cross_domain_recommendation/">Review-Based Hyperbolic Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论文本嵌入双曲空间做跨域推荐，用度数归一化与尺度对齐避免域对齐引发的层级坍塌，NDCG@10 较 SER 提升约 10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性 item-item 模型&quot;,&quot;时间间隔加权&quot;,&quot;趋势感知归一化&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/temporal_linear_item_item_model_for_sequential_recommendation/">Temporal Linear Item-Item Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在闭式解线性 item-item 模型上引入时间间隔加权与趋势感知归一化，Full-Ranking 下平均 NDCG@10 提升 9.07%，训练快 6.9–57 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;冷启动迁移&quot;,&quot;广告定向&quot;]"><strong><a href="../../venues/WSDM/2026/multi_task_reranking/cold_starting_podcast_ads_and_promotions_with_multi_task_learning_on_spotif/">Cold-Starting Podcast Ads and Promotions with Multi-Task Learning on Spotify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Spotify 用单一 5 任务 MTL 模型统一播客广告与推广定向，靠非对称损失掩码与源平衡采样实现跨渠道迁移，线上 eCPS 降 22%、i2s 升 18–24%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;对比学习&quot;,&quot;物品关系挖掘&quot;]"><strong><a href="../../venues/WSDM/2026/cold_start_rec/epiccbr_item_relation_enhanced_dual_scenario_contrastive_learning_for_cold_/">EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过物品对关系增强与流行度感知双场景对比学习，在冷启动捆绑推荐上实现最高 387% 的 Recall@20 提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;课程学习&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/WWW/2024/cold_start_rec/cold_start_bundle_recommendation_via_popularity_based_coalescence_and_curri/">Cold-start Bundle Recommendation via Popularity-based Coalescence and Curriculum Heating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CoHeat 以热度自适应加权融合「用户-捆绑」与「用户-物品」双图视图，并借课程升温对齐两视图，冷启动 bundle 的 nDCG@20 最高提升 193%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（打分校正）&quot;,&quot;Personal Popularity&quot;,&quot;Counterfactual Inference&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/debiasing_recommendation_with_personal_popularity/">Debiasing Recommendation with Personal Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出用户级个人流行度 PP 与反事实框架 PPAC，用代理变量解耦 PP/GP 直接效应，在等频干预测试集上全排序 Recall@50 最高提升 46.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;动态图预训练&quot;,&quot;提示学习&quot;,&quot;相对时间编码&quot;]"><strong><a href="../../venues/WWW/2024/graph_rec/graphpro_graph_pre_training_and_prompt_learning_for_recommendation/">GraphPro: Graph Pre-training and Prompt Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>时序提示注入图卷积 + 结构提示图与门控微调，动态快照评测下 Recall@20 领先 2.55%–9.77%，线上 CTR +1.53%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;训练期数据增强&quot;,&quot;LLM4Rec&quot;,&quot;成对比较提示&quot;,&quot;辅助 BPR 损失&quot;]"><strong><a href="../../venues/WWW/2024/cold_start_rec/large_language_models_as_data_augmenters_for_cold_start_item_recommendation/">Large Language Models as Data Augmenters for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 成对比较推理生成冷启动物品的合成偏好信号，经辅助 BPR 损失注入 ID-based 推荐模型；Amazon 全库排序下冷启动物品 R@50 提升近 21 倍且几乎不损 warm-start。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;有限库存商品推荐&quot;,&quot;序列建模&quot;]"><strong><a href="../../venues/WWW/2024/ctr_feature_interaction/metasplit_meta_split_network_for_limited_stock_product_recommendation/">MetaSplit: Meta-Split Network for Limited-Stock Product Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按库存切分用户序列并引入元缩放/位移网络与辅助损失，解决 C2C 孤品冷启与注意力偏置，线上 CTR +3.56%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/曝光层在线探索&quot;,&quot;Neural Contextual Bandit&quot;,&quot;Exploration-Regret Theory&quot;,&quot;Collaborative Bandits&quot;]"><strong><a href="../../venues/WWW/2024/rl_bandit_rec/neural_contextual_bandits_for_personalized_recommendation/">Neural Contextual Bandits for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>WWW&#x27;24 教程：以探索缓解物品流行度马太效应，串联线性→神经→协作式神经上下文 Bandit 的算法、Regret 理论与开放问题</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;多样性推荐&quot;,&quot;用户-类别匹配&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/on_practical_diversified_recommendation_with_controllable_category_diversit/">On Practical Diversified Recommendation with Controllable Category Diversity Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多样性拆为 U-/N-diversity 并两阶段解耦，用 top-K 类别门控实现可控多样性，离线 N-diversity HR@5 较 SOTA +129%，线上转化率 +6.66%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质图神经网络&quot;,&quot;双塔召回&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/WWW/2024/graph_rec/personalized_audiobook_recommendations_at_spotify_through_graph_neural_netw/">Personalized Audiobook Recommendations at Spotify Through Graph Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 2T-HGNN：将用户从图里解耦、仅在有声书-播客共听异质图上跑 HGNN 学物品长程表征，再接双塔做实时召回；A/B 上有声书新起播率 +46.83%、播放率 +25.82%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;检索增强&quot;,&quot;跨样本特征交互&quot;,&quot;级联注意力&quot;]"><strong><a href="../../venues/WWW/2024/ctr_feature_interaction/rat_retrieval_augmented_transformer_for_click_through_rate_prediction/">RAT: Retrieval-Augmented Transformer for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 BM25 检索 K 个历史样本作为参考上下文，以级联 intra/cross attention 建模样本内与跨样本特征交互，AUC 平均较 xDeepFM 提升 1.13%，长尾用户提升更显著。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;冷启动推荐&quot;,&quot;端到端联邦推荐&quot;,&quot;联邦推荐&quot;,&quot;物品属性表征&quot;,&quot;表征对齐&quot;]"><strong><a href="../../venues/WWW/2024/cold_start_rec/when_federated_recommendation_meets_cold_start_problem_separating_item_attr/">When Federated Recommendation Meets Cold-Start Problem: Separating Item Attributes and User Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个联邦冷启动推荐框架 IFedRec：服务器端学习物品属性元网络、客户端保留交互物品嵌入，双向对齐两表征使冷物品仅凭属性即可推理。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;创作者侧推荐&quot;,&quot;用户可用性计算&quot;,&quot;双边市场公平性&quot;]"><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/creator_side_recommender_system_challenges_designs_and_applications/">Creator-Side Recommender System: Challenges, Designs, and Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出创作者侧推荐系统 DualRec，将用户侧召回/排序镜像为物品找用户，并用用户可用性计算 UAC 解决用户可达性问题，快手线上 DAC 提升 2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;意图解耦&quot;,&quot;标题-内容一致性&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/WWW/2025/cold_start_rec/crown_a_novel_approach_to_comprehending_users_preferences_for_accurate_pers/">CROWN: A Novel Approach to Comprehending Users&#x27; Preferences for Accurate Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别引导意图解耦、标题-内容一致性加权与 GNN 混合用户表征，统一解决新闻推荐的多意图、读后偏好差异与冷启动三大难题。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;分层提示&quot;,&quot;新闻推荐&quot;,&quot;数据增强&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/enhancing_news_recommendation_with_hierarchical_llm_prompting/">Enhancing News Recommendation with Hierarchical LLM Prompting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层提示让 LLM 生成更丰富的新闻标题与实体，作为模型无关的数据增强模块，在 MIND 上以 AUC 68.88 超越 GLoCIM 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Semi-Bandit&quot;,&quot;可学习偏置项&quot;,&quot;分层 UCB&quot;]"><strong><a href="../../venues/WWW/2025/rl_bandit_rec/interactive_visualization_recommendation_with_hier_sucb/">Interactive Visualization Recommendation with Hier-SUCB</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将可视化推荐建模为带可学习偏置项的分层上下文组合半臂老虎机，把遗憾界从 $O(\sqrt{T\ln^3(nm^2T\ln T)})$降至$O(\sqrt{T\ln^3(m^2T\ln T)})$，在 Plot.ly 上 80/160 轮超越离线 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回+精排&quot;,&quot;端到端联合训练&quot;,&quot;内容编码器&quot;,&quot;推理缓存&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/legommenders_a_comprehensive_content_based_recommendation_library_with_llm_/">Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个支持内容编码器与行为/交互模块端到端联合训练的推荐库，内置 LLM 编码与数据增强，覆盖 1000+ 模型组合与 15 个数据集，推理缓存带来最高 50x 加速。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;扩散生成推荐&quot;,&quot;捆绑推荐&quot;]"><strong><a href="../../venues/WWW/2025/cold_start_rec/personalized_diffusion_model_reshapes_cold_start_bundle_recommendation/">Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用个性化条件扩散在分布空间生成捆绑包，配合解耦自注意力与KL正则，冷启动捆绑推荐全库排序Recall@20最高提升1155%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回+重排&quot;,&quot;视觉语言模型微调&quot;,&quot;贴纸个性化检索&quot;,&quot;效用评估&quot;]"><strong><a href="../../venues/WWW/2025/multimodal_rec/persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;数据增强/冷启动&quot;,&quot;表格数据合成&quot;,&quot;特征归因重要性采样&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/samplellm_optimizing_tabular_data_synthesis_in_recommendations/">SampleLLM: Optimizing Tabular Data Synthesis in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>两阶段 LLM 表格数据合成框架：聚类采样示例 + 特征归因重要性采样对齐分布，冷启动场景线上 RPM +2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;数据压缩/训练加速&quot;,&quot;数据集浓缩&quot;,&quot;提示演化&quot;,&quot;聚类用户合成&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/tf_dcon_leveraging_large_language_models_llms_to_empower_training_free_data/">TF-DCon: Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 ChatGPT 前向式浓缩文本内容与聚类合成假用户，免迭代优化压缩 CBR 数据集 95% 仍保 97% 性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;VTuber观众行为测量&quot;,&quot;逆向推荐（向主播推荐观众）&quot;,&quot;粉丝社区文化&quot;]"><strong><a href="../../venues/WWW/2025/social_group_rec/virtual_stars_real_fans_understanding_the_vtuber_ecosystem/">Virtual Stars, Real Fans: Understanding the VTuber Ecosystem</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>基于Bilibili 270万场直播的VTuber核心粉丝（会员）行为测量研究，逆向构建FanRanker将潜在付费会员推荐给主播，Top-50命中85%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;精排&quot;,&quot;长尾学习&quot;,&quot;层级语义树&quot;,&quot;Gumbel-Softmax&quot;]"><strong><a href="../../venues/WWW/2026/poi_spatiotemporal_rec/adaptive_location_hierarchy_learning_for_long_tailed_mobility_prediction/">Adaptive Location Hierarchy Learning for Long-Tailed Mobility Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出架构无关插件 ALOHA：LLM+CoT 自动构建城市定制四级语义层级，配合 Gumbel 扰动与节点自适应权重的层级损失，在 6 城数据集上对 RNN/GNN/Transformer/Diffusion 四类骨干一致提升，M@1 最高 +16.59%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;LLM Reasoning&quot;,&quot;GRPO&quot;,&quot;Cold-Start&quot;]"><strong><a href="../../venues/WWW/2026/cold_start_rec/llm_reasoning_for_cold_start_item_recommendation/">LLM Reasoning for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 Netflix 冷启动重排任务上系统对比结构化推理、软自洽与 SFT/GRPO 混合微调，推理微调模型在 Discovery 指标上超越线上生产排序模型最高 8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测与系统设计&quot;,&quot;用户一致性度量&quot;,&quot;分层评测&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/quantifying_user_coherence_a_unified_framework_for_analyzing_recommender_sy/">Quantifying User Coherence: A Unified Framework for Analyzing Recommender Systems Across Domains</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 Mean Surprise 与 Mean Conditional Surprise 两个信息论度量刻画用户画像一致性，揭示复杂模型增益集中于高一致用户，并支撑分层评测与用户分群建模。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;ODE 免训练&quot;]"><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;长尾序列推荐&quot;]"><strong><a href="../../venues/WWW/2026/cold_start_rec/tail_aware_data_augmentation_for_long_tail_sequential_recommendation/">Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性物品相似度构建候选集，设计 T-Substitute/T-Insert 两个尾部感知算子并在表征层 Mixup，实现头部与尾部双赢的长尾序列推荐数据增强。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;跨场景统计特征张量&quot;,&quot;稳定性感知模型更新&quot;]"><strong><a href="../../venues/WWW/2026/cold_start_rec/trinity_a_scenario_aware_recommendation_framework_for_large_scale_cold_star/">Trinity: A Scenario-Aware Recommendation Framework for Large-Scale Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向十亿级产品迁移的“特征-架构-更新”三位一体冷启动框架，Copilot 新场景 AUC 0.564→0.726、COPC 0.13→0.95，线上时长 +5.61%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;冷启动基准&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/vk_lsvd_a_large_scale_industrial_dataset_for_short_video_recommendation/">VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>发布 407 亿交互、千万用户、两千万短视频的工业级开放数据集，配全局时间切分与冷启动竞赛基准。</p>
</dd>
</dl>
