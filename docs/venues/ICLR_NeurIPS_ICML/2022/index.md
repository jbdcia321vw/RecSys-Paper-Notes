---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2022 推荐系统论文解读

17 篇 ICLR / NeurIPS / ICML (RecSys) 2022 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** NeurIPS ×9 · CTR预估与特征交叉 ×3 · NeurIPS 2022 ×3 · ICLR 2022 ×2 · ICML ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;arXiv 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;重要性重采样&quot;,&quot;负采样偏差校正&quot;,&quot;难负样本缓存&quot;]"><strong><a href="collaborative_filtering/cache_augmented_inbatch_importance_resampling_for_training_recommender_retr/">Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用查询相关的批内重要性重采样加难负样本缓存，将 inbatch 采样偏差压向真实 softmax，五数据集 NDCG@10 最高相对提升 17.1%</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/全库排序&quot;,&quot;NTK&quot;,&quot;数据蒸馏&quot;]"><strong><a href="collaborative_filtering/infinite_recommendation_networks_a_data_centric_approach/">Infinite Recommendation Networks: A Data-Centric Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 NTK 将推荐自编码器推到无限宽瓶颈，闭式 KRR 求解，并以 Gumbel 数据蒸馏合成 0.1% 用户摘要保持 96–105% 性能。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2022&quot;,&quot;协同过滤（按指定归档）&quot;,&quot;非推荐链路/神经形态推理&quot;,&quot;ANN-SNN Conversion&quot;,&quot;Quantization Clip-Floor-Shift&quot;]"><strong><a href="collaborative_filtering/optimal_ann_snn_conversion_for_high_accuracy_and_ultra_low_latency_spiking_/">Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>按指定归档；论文实为 ANN-SNN 转换，提出量化 clip-floor-shift 激活，证明期望转换误差为零，CIFAR-10 仅 2 步达 91.18%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;动态参数生成&quot;,&quot;低秩分解&quot;]"><strong><a href="ctr_feature_interaction/apg_adaptive_parameter_generation_network_for_click_through_rate_prediction/">APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以实例条件动态生成低秩参数矩阵，解耦共享/特定模式，AUC普遍提升且训练内存大降。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;重要性重加权&quot;,&quot;流式训练&quot;]"><strong><a href="ctr_feature_interaction/generalized_delayed_feedback_model_with_post_click_information_in_recommend/">Generalized Delayed Feedback Model with Post-Click Information in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将点击后行为统一为随机后验反馈，用条件熵与时间衰减重加权代理损失，流式训练下 CVR 相对提升至 Oracle 差距的 74.9%/79.4%</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;富媒体广告拍卖&quot;,&quot;单调分配规则&quot;,&quot;无政府代价 PoA&quot;]"><strong><a href="ctr_feature_interaction/simple_mechanisms_for_welfare_maximization_in_rich_advertising_auctions/">Simple Mechanisms for Welfare Maximization in Rich Advertising Auctions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以绝对性价比单调分配广告空间再随机混合最大价值创意，得 3-近似诚实富媒体广告拍卖，GSP 下 PoA≤6，实测达 VCG 的 0.95 且快 9 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;策略层/预算约束决策&quot;,&quot;Contextual Bandits with Knapsacks&quot;,&quot;Conversion Model&quot;]"><strong><a href="rl_bandit_rec/contextual_bandits_with_knapsacks_for_a_conversion_model/">Contextual Bandits with Knapsacks for a Conversion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用转化耦合的CBwK与Logistic-UCB+原始LP实现(OPT/B)√T遗憾并硬预算约束。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端长期策略&quot;,&quot;对抗Bandit&quot;,&quot;自适应偏好&quot;,&quot;多样性约束&quot;]"><strong><a href="rl_bandit_rec/diversified_recommendations_for_agents_with_adaptive_preferences/">Diversified Recommendations for Agents with Adaptive Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将自适应偏好菜单推荐建模为对抗 bandit，提出 EIRD 基准与局部可学习假设，在熵多样性约束下取得 $\tilde{O}(T^{3/4})$ 遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;策略优化/LTV&quot;,&quot;偏好动态建模&quot;,&quot;安全偏移信任域&quot;,&quot;非齐次HMM&quot;]"><strong><a href="rl_bandit_rec/estimating_and_penalizing_induced_preference_shifts_in_recommender_systems/">Estimating and Penalizing Induced Preference Shifts in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可学用户偏好动态模型预测新策略诱导的偏好偏移，并以“安全偏移”信任域惩罚 RL 推荐器，兼顾参与度与偏好安全。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/在线探索&quot;,&quot;Dueling Bandit&quot;,&quot;Ellipsoid Method&quot;]"><strong><a href="rl_bandit_rec/learning_from_a_learning_user_for_optimal_recommendations/">Learning from a Learning User for Optimal Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户建模为同步学习的(c,γ)-理性智能体，提出RAES噪声鲁棒主动椭球搜索，仅用成对反馈获得 O~(d^2 T^{1/2+γ}) 遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Bandit&quot;,&quot;多购买选择模型&quot;,&quot;UCB 乐观探索&quot;]"><strong><a href="rl_bandit_rec/product_ranking_for_revenue_maximization_with_multiple_purchases/">Product Ranking for Revenue Maximization with Multiple Purchases</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出随机注意力跨度与购买预算的多次购买级联选择模型，推导按 $\lambda r/(1-q+q(1-s)\lambda)$ 降序的闭式最优排序，并用 MPB-UCB 在线算法在合成/淘宝半合成数据上取得最低遗憾。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练底座（分布式嵌入表分片）&quot;,&quot;强化学习设备放置&quot;,&quot;可泛化成本网络&quot;,&quot;算子融合&quot;]"><strong><a href="debias_fairness_industrial/dreamshard_generalizable_embedding_table_placement_for_recommender_systems/">DreamShard: Generalizable Embedding Table Placement for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用成本网络估计 MDP 替代真实 GPU 执行，训练可泛化的 RL 策略完成嵌入表分片，在 DLRM/生产表上较最强基线最高提速 19%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/排序（全链路风险目标）&quot;,&quot;域传输 (Domain Transportation)&quot;,&quot;Wasserstein 约束&quot;,&quot;逆概率加权 (IPW)&quot;]"><strong><a href="debias_fairness_industrial/from_intervention_to_domain_transportation_a_novel_perspective_to_optimize_/">From Intervention to Domain Transportation: A Novel Perspective to Optimize Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将推荐视为域传输，用 Wasserstein 约束把训练域与部署域绑定，以 WGAN 式极小极大博弈求解，在 Leave-One-Out Full-Ranking 协议下显著提升推荐器。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/相关性预估&quot;,&quot;因果用户建模&quot;,&quot;不变学习&quot;]"><strong><a href="debias_fairness_industrial/in_the_eye_of_the_beholder_robust_prediction_with_causal_user_modeling/">In the Eye of the Beholder: Robust Prediction with Causal User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户建模为有界理性因果决策者，用因果/反因果与相信者/怀疑者类型选择MMD正则，提升分布漂移下相关性预测鲁棒性。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;角度 Margin 对比损失&quot;]"><strong><a href="debias_fairness_industrial/incorporating_bias_aware_margins_into_contrastive_loss_for_collaborative_fi/">Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用流行度偏置提取器为每个交互定制角度 margin 并注入 softmax 对比损失，在 8 个数据集的平衡/非平衡/时间切分评测上同时提升头尾性能 12.1%–58.4%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;模型选型 / AutoML（元学习）&quot;,&quot;数据集元特征（meta-features）&quot;,&quot;算法选择与超参预测&quot;]"><strong><a href="debias_fairness_industrial/on_the_generalizability_and_predictability_of_recommender_systems/">On the Generalizability and Predictability of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>85 个数据集、315 个指标的元学习选型研究，RecZilla 直接预测算法与超参数，%Diff 33.2 优于 cf4cf-meta 43.5，推理仅 6.7 秒。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;工业级基准数据集&quot;,&quot;全链路评测基准&quot;,&quot;跨场景推荐&quot;,&quot;多任务 CTR&quot;,&quot;真实负反馈&quot;]"><strong><a href="debias_fairness_industrial/tenrec_a_large_scale_multipurpose_benchmark_dataset_for_recommender_systems/">Tenrec: A Large-scale Multipurpose Benchmark Dataset for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>腾讯四场景 500 万用户/1.4 亿交互数据集，含真实负反馈与跨域重叠，在 10 类任务上实证经典基线差距远小于文献宣称。</p>
</dd>
</dl>
</section>
</div>
