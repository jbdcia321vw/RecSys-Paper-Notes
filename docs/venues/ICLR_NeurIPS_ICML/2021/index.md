---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2021 推荐系统论文解读

12 篇 ICLR / NeurIPS / ICML (RecSys) 2021 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** ICML ×6 · NeurIPS ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;神经正切核&quot;,&quot;隐式正则与泛化界&quot;,&quot;曝光偏差去偏&quot;]"><strong><a href="collaborative_filtering/rethinking_neural_vs_matrix_factorization_collaborative_filtering_the_theor/">Rethinking Neural vs. Matrix-Factorization Collaborative Filtering: the Theoretical Perspectives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 NTK、隐式偏差与转导/归纳泛化界严格重审 MCF 与 NCF 之争，指出表达力等价但优化路径与评估协议（曝光偏差）才是胜负关键。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;归纳式矩阵分解&quot;,&quot;注意力关系学习&quot;,&quot;冷启动&quot;]"><strong><a href="collaborative_filtering/towards_open_world_recommendation_an_inductive_model_based_collaborative_fi/">Towards Open-World Recommendation: An Inductive Model-based Collaborative Filtering Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 key 用户矩阵分解元隐变量为基座，用多头注意力关系图做归纳式消息传递，免特征实现少样本/零样本新用户即时表征，ML-1M 新用户 RMSE 降 4.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;序列与会话推荐&quot;,&quot;召回(Next-Item 预测)&quot;,&quot;时间点过程&quot;,&quot;条件强度函数&quot;,&quot;注意力调制&quot;]"><strong><a href="sequential_rec/learning_self_modulating_attention_in_continuous_time_space_with_applicatio/">Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用时间点过程的时变条件强度调制注意力权重并加连续时间正则，Strong Generalization 协议下 HR@10 最高 +2.8%、NDCG@10 最高 +4.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;可学习阈值剪枝&quot;,&quot;混合维度嵌入表&quot;,&quot;Lottery Ticket 重训&quot;]"><strong><a href="ctr_feature_interaction/learnable_embedding_sizes_for_recommender_systems/">Learnable Embedding Sizes for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用可学习软阈值对嵌入矩阵做列稀疏剪枝，一次训练自动得到混合维度嵌入表，参数降 97–99% 且 AUC 反升。</p>
</dd>
<dt data-paper-tags="[&quot;ICML 2021&quot;,&quot;CTR/CVR 预估与特征交互&quot;,&quot;精排后定价/计费&quot;,&quot;一价拍卖保留价&quot;,&quot;零阶随机梯度&quot;,&quot;方差缩减&quot;]"><strong><a href="ctr_feature_interaction/reserve_price_optimization_for_first_price_auctions_in_display_advertising/">Reserve Price Optimization for First Price Auctions in Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将一价拍卖收入分解为需求与竞价两项并分别做参数化拟合与分位数截断降方差，零阶梯度收敛率由 Õ(N^{-1/4}) 提升到 Õ(N^{-1/3})。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回+精排级联&quot;,&quot;两阶段推荐&quot;,&quot;Mixture-of-Experts&quot;,&quot;Bandit遗憾分解&quot;]"><strong><a href="multi_task_reranking/on_component_interactions_in_two_stage_recommender_systems/">On component interactions in two-stage recommender systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用遗憾分解揭示两阶段推荐器组件交互，提出MoE学习物品池分配，提升prec@5/recall@5。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;上下文线性 Bandit&quot;,&quot;切割平面与凸几何&quot;,&quot;列表推荐与低遗憾在线学习&quot;]"><strong><a href="rl_bandit_rec/contextual_recommendations_and_low_regret_cutting_plane_algorithms/">Contextual Recommendations and Low-Regret Cutting-Plane Algorithms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>仅在仅能观测“最优动作身份”的定性反馈下，将上下文推荐归约为低遗憾切割平面算法，给出 $O(d\log T)$、$\exp(O(d\log d))$及列表推荐$O(d^2\log d)$ 的遗憾界。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线探索/排序&quot;,&quot;Incentivized Bandit&quot;,&quot;Self-Reinforcing Preference&quot;,&quot;Pólya Urn&quot;]"><strong><a href="rl_bandit_rec/incentivized_bandit_learning_with_self_reinforcing_user_preferences/">Incentivized Bandit Learning with Self-Reinforcing User Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出激励式 MAB 与自增强偏好模型，以多色 Pólya 罐设计三阶段 ALnETC/UCB-List，证明 $O(\log T)$ 遗憾与支付，仿真验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与工业系统&quot;,&quot;重排&quot;,&quot;反事实排序&quot;,&quot;双边市场实验设计&quot;]"><strong><a href="debias_fairness_industrial/a_b_testing_for_recommender_systems_in_a_two_sided_marketplace/">A/B Testing for Recommender Systems in a Two-sided Marketplace</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>UniCoRn 以反事实排序统一实现生产者侧 A/B 实验，α 显式权衡精度与成本，线上 viewee 指标 +0.51%/+0.57%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;曝光偏差&quot;,&quot;逆倾向加权&quot;,&quot;反馈回路&quot;]"><strong><a href="debias_fairness_industrial/correcting_exposure_bias_for_link_recommendation/">Correcting Exposure Bias for Link Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以倾向分重加权构造链接推荐的真风险估计器并联合训练，半合成引文数据 Precision 67.24→83.28，真风险估计 RMSE 由 1.50 降至 0.03，并抑制反馈回路与 FOS 同质化。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;安全多方计算&quot;,&quot;稀疏矩阵乘法&quot;,&quot;同态加密与 PIR&quot;]"><strong><a href="debias_fairness_industrial/exploiting_data_sparsity_in_secure_cross_platform_social_recommendation/">Exploiting Data Sparsity in Secure Cross-Platform Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>S3Rec 用 HE+PIR 构造两种安全稀疏矩阵乘法协议，将跨平台社交推荐协同训练中的社交项通信量从 O(km²) 降到 O(km)/O(αkm)，RMSE 与明文 Soreg 完全一致而通信量降低约 423×。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平与工业架构&quot;,&quot;离线评测/策略审计&quot;,&quot;随机可达性&quot;,&quot;凸优化&quot;]"><strong><a href="debias_fairness_industrial/quantifying_availability_and_discovery_in_recommender_systems_via_stochasti/">Quantifying Availability and Discovery in Recommender Systems via Stochastic Reachability</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 softmax 随机可达性凸程序审计推荐系统，在 ML-1M/LastFM/MIND 上量化流行度偏差是否内生于打分模型本身。</p>
</dd>
</dl>
</section>
</div>
