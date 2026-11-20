---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2023 推荐系统论文解读

17 篇 AAAI (RecSys Track) 2023 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI ×9 · AAAI 2023 ×8 · CTR预估与特征交叉 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨服务对比预训练&quot;,&quot;用户表征缩放定律&quot;]"><strong><a href="self_supervised_denoising/scaling_law_for_recommendation_models_towards_general_purpose_user_represen/">Scaling Law for Recommendation Models: Towards General-purpose User Representations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以层级双 Transformer + 跨服务 CLIP 式对比学习预训练通用用户表征 CLUE，在文本化物品描述上对齐异构服务，首次验证推荐用户建模的幂律缩放律，特征级迁移在离线与线上 CTR 上全面超越任务专属模型。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列&quot;,&quot;胶囊动态路由&quot;,&quot;动态GRU&quot;]"><strong><a href="sequential_rec/dynamic_multi_behavior_sequence_modeling_for_next_item_recommendation/">Dynamic Multi-Behavior Sequence Modeling for Next Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用胶囊动态路由按用户相关性融合多行为序列，并下沉到动态 GRU 实现 item 级个性化，全库排序 H@10 最高提升 38.5%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;集合到序列 (Set-to-Sequence)&quot;,&quot;策略梯度&quot;,&quot;知识追踪辅助任务&quot;]"><strong><a href="sequential_rec/set_to_sequence_ranking_based_concept_aware_learning_path_recommendation/">Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将学习路径推荐形式化为集合到序列生成，用概念自注意力编码 + LSTM 指针解码 + 策略梯度与知识追踪辅助任务，在 ASSIST09/Junyi/工业集上经 DKT/CoKT 模拟器回测显著超越规则与 RL 基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;用户-物品交互图&quot;,&quot;BM25 检索增强&quot;,&quot;门控融合&quot;]"><strong><a href="agent_conversational_rec/cola_improving_conversational_recommender_systems_by_collaborative_augmenta/">COLA: Improving Conversational Recommender Systems by Collaborative Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用用户-物品交互图注入流行度、BM25 检索相似会话做门控增强，全库排序下 REDIAL R@50 相对提升 7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式端到端&quot;,&quot;检索增强生成&quot;,&quot;可解释推荐&quot;,&quot;事实性幻觉抑制&quot;]"><strong><a href="agent_conversational_rec/factual_and_informative_review_generation_for_explainable_recommendation/">Factual and Informative Review Generation for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>PRAG 用个性化 Retriever 生成隐式评论查询，检索历史评论作证据并抽取 tf-idf 关键词，驱动 QA 模型生成可溯源且多样的推荐解释，在 Yelp/TripAdvisor/Amazon 上忠实度最高提升 145%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;多模态推荐&quot;,&quot;应用层端到端&quot;,&quot;情感上下文&quot;,&quot;区块链存证&quot;]"><strong><a href="multimodal_rec/emotion_aware_music_recommendation_system_enhancing_user_experience_through/">Emotion-Aware Music Recommendation System: Enhancing User Experience Through Real-Time Emotional Context</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 CNN 歌词情感分类配合实时情绪输入驱动歌曲推荐，并用 Alchemy 区块链存证偏好与版权；仅有界面截图级定性验证，无任何量化评测与消融实验。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端打分（跨域知识迁移）&quot;,&quot;差分隐私数据发布&quot;,&quot;生成式对抗迁移&quot;,&quot;表征解耦&quot;]"><strong><a href="cross_domain_multi_scenario/ppgencdr_a_stable_and_robust_framework_for_privacy_preserving_cross_domain_/">PPGenCDR: A Stable and Robust Framework for Privacy-Preserving Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用条件 GAN 在源域做选择性 RDP 扰动并发布生成器，配合控制论稳定器与批内特征解耦，在 Leave-One-Out + 99 负采样协议下较 PriCDR 提升 6–32%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;低精度训练&quot;,&quot;嵌入表压缩&quot;,&quot;量化&quot;]"><strong><a href="ctr_feature_interaction/adaptive_low_precision_training_for_embeddings_in_click_through_rate_predic/">Adaptive Low-Precision Training for Embeddings in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 ALPT，用随机舍入与逐特征可学习步长训练低精度嵌入表，在 Criteo/Avazu 上实现 8-bit 无损压缩与 3.2× 训练内存压缩。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;大 batch 训练&quot;,&quot;自适应梯度裁剪&quot;,&quot;Embedding 频次不均衡&quot;]"><strong><a href="ctr_feature_interaction/cowclip_reducing_ctr_prediction_model_training_time_from_12_hours_to_10_min/">CowClip: Reducing CTR Prediction Model Training Time from 12 hours to 10 minutes on 1 GPU</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 ID 频次差异使大 batch 下 LR/正则缩放规则失效，提出按列自适应裁剪 CowClip，单卡 128× batch 无精度损失、76.8× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;决策上下文建模&quot;,&quot;上下文交互单元&quot;,&quot;自适应兴趣聚合&quot;]"><strong><a href="ctr_feature_interaction/decision_making_context_interaction_network_for_click_through_rate_predicti/">Decision-Making Context Interaction Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 DCIN 建模页面内显式比较与粗排候选隐式决策上下文，CIU 过滤无关项、AIAU 聚合目标兴趣；MeituanAds AUC+0.0042，线上 CTR+2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;双流MLP&quot;,&quot;双线性融合&quot;]"><strong><a href="ctr_feature_interaction/finalmlp_an_enhanced_two_stream_mlp_model_for_ctr_prediction/">FinalMLP: An Enhanced Two-Stream MLP Model for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双流纯 MLP 加流专属门控与多头双线性融合，四个基准 AUC 全面超两流模型，线上 CTR +1.6%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;离线策略学习&quot;,&quot;重要性采样&quot;,&quot;最大内积检索&quot;]"><strong><a href="rl_bandit_rec/fast_offline_policy_optimization_for_large_scale_recommendation/">Fast Offline Policy Optimization for Large Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用协方差梯度+自归一化重要性采样+训练期 MIPS 索引，将大规模目录离线策略梯度训练从 O(P) 降至 O(log P)，提速 5–25× 且策略质量不降。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端决策支持&quot;,&quot;因果推断&quot;,&quot;逆倾向加权&quot;]"><strong><a href="debias_fairness_industrial/evaluating_digital_agriculture_recommendations_with_causal_inference/">Evaluating Digital Agriculture Recommendations with Causal Inference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用因果图与后门准则评估棉花播种推荐对产量的影响，在171块田实测 ATE 提升 12%–17%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;多重稳健估计 (MR)&quot;,&quot;逆倾向加权 (IPS)&quot;,&quot;误差插补&quot;,&quot;MNAR 去偏&quot;]"><strong><a href="debias_fairness_industrial/multiple_robust_learning_for_recommendation/">Multiple Robust Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多个倾向/误差插补模型构成基向量并岭回归求解组合系数，把 DR 升级为多重稳健估计器，Coat/Yahoo 全库评测 MSE 与 nDCG@10 显著优于 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/排序&quot;,&quot;定向拓扑攻击&quot;,&quot;节点掩码&quot;,&quot;LightGCN&quot;]"><strong><a href="debias_fairness_industrial/revisiting_item_promotion_in_gnn_based_collaborative_filtering_a_masked_tar/">Revisiting Item Promotion in GNN-based Collaborative Filtering: A Masked Targeted Topological Attack Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次将GNN协同过滤物品推广重述为定向拓扑攻击，以节点掩码降噪和目标列梯度实现小预算高效攻击；Gowalla/Yelp2018全库top-50命中显著超越启发式基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估&quot;,&quot;Item Sampling&quot;,&quot;Top-K Metric Estimation&quot;,&quot;EM&quot;]"><strong><a href="debias_fairness_industrial/towards_reliable_item_sampling_for_recommendation_evaluation/">Towards Reliable Item Sampling for Recommendation Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出MSE最优item-sampling估计器与自适应采样EM，低样本下准确恢复Top-K指标。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦训练（服务器聚合）&quot;,&quot;无目标投毒攻击&quot;,&quot;对比学习均匀性正则&quot;,&quot;拜占庭鲁棒聚合&quot;]"><strong><a href="debias_fairness_industrial/untargeted_attack_against_federated_recommendation_systems_via_poisonous_it/">Untargeted Attack against Federated Recommendation Systems via Poisonous Item Embeddings and the Defense</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 ClusterAttack 用簇内方差最小化的恶意 item 梯度塌缩嵌入空间打乱排序，并给出基于对比学习均匀性正则的 UNION 防御与服务器端梯度过滤。</p>
</dd>
</dl>
</section>
</div>
