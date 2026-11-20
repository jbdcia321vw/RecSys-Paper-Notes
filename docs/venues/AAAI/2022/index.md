---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2022 推荐系统论文解读

13 篇 AAAI (RecSys Track) 2022 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI 2022 ×11 · AAAI ×2 · Contextual Bandit ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排/在线增量更新&quot;,&quot;Meta-Learning&quot;,&quot;参数级学习率&quot;,&quot;在线学习&quot;]"><strong><a href="collaborative_filtering/meta_learning_for_online_update_of_recommender_systems/">Meta-Learning for Online Update of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MeLON 用元学习为每个「交互-参数」对生成二维自适应学习率，突破重要性重加权与元优化的单向灵活性；在三个在线数据集上 HR@5 最高提升 29.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;意图解耦&quot;,&quot;对比学习&quot;]"><strong><a href="graph_rec/multi_view_intent_disentangle_graph_networks_for_bundle_recommendation/">Multi-view Intent Disentangle Graph Networks for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用邻域路由在全局/局部双视图上解耦用户 K 个意图，并以意图级 InfoNCE 对齐，NetEase 上 Recall@20 相对最强基线提升 38.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;序列建模/记忆压缩&quot;,&quot;可微数据选择&quot;,&quot;双层优化&quot;,&quot;梯度近似&quot;]"><strong><a href="sequential_rec/dips_differentiable_policy_for_sketching_in_recommender_systems/">DiPS: Differentiable Policy for Sketching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以双层优化端到端学习可微草图（sketching）策略，并用队列式近似梯度替代全历史重算，全库排序下同等精度省 50% 草图槽位。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;轨迹式元学习&quot;,&quot;时序动态建模&quot;]"><strong><a href="sequential_rec/leaping_through_time_with_gradient_based_adaptation_for_recommendation/">Leaping Through Time with Gradient-based Adaptation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用轨迹式元学习 Leap 替代 RNN 建模时序动态：GTL 抓跨时段共性、OTL 抓短期趋势，时间切分+1:99 采样排序下 HR@1 最高相对提升 18.59%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;生成式推荐&quot;,&quot;重排/展示层&quot;,&quot;Pointer-Generator 文本生成&quot;,&quot;推荐理由个性化&quot;]"><strong><a href="generative_rec/intelligent_online_selling_point_extraction_for_e_commerce_recommendation/">Intelligent Online Selling Point Extraction for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>IOSPE 以“粗筛-BERT-Pointer 生成-递归精筛”三级流水线自动生成卖点，并按用户画像做个性化指派；JD 全量上线使 CTR +1.89%、停留时长 +2.03%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮对话决策与模板生成&quot;,&quot;规则式对话动作树&quot;,&quot;简化知识图谱推理&quot;,&quot;冷启动 CRS 配置化框架&quot;]"><strong><a href="agent_conversational_rec/force_a_framework_of_rule_based_conversational_recommender_system/">FORCE: A Framework of Rule-Based Conversational Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出可配置的规则式对话推荐框架 FORCE，以简化知识图谱推理 + 三类对话动作树替代深度模型，在电影与中文医疗两个冷启动场景下意图预测与 Recall 显著超越随机基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;元学习 MAML&quot;,&quot;动态矩阵分解&quot;,&quot;时间敏感冷启动&quot;]"><strong><a href="cold_start_rec/a_dynamic_meta_learning_model_for_time_sensitive_cold_start_recommendations/">A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户隐偏好可加分解为时间特定（MAML 少样本适配）与时间演化（嵌套 RNN）两因子，在 Netflix/Last.fm/ML-1M 分期预测上 RMSE 降 1.7–5.1%、NDCG 升 3.3–7.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排后处理&quot;,&quot;概率校准&quot;,&quot;IPS 去偏&quot;]"><strong><a href="ctr_feature_interaction/obtaining_calibrated_probabilities_with_personalized_ranking_models/">Obtaining Calibrated Probabilities with Personalized Ranking Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 Gaussian/Gamma 参数化校准与 IPS 无偏风险最小化，校准个性化排序分数，ECE 最高降 25.85%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Cross-Task Knowledge Distillation&quot;,&quot;Quadruplet Ranking&quot;,&quot;Calibrated Distillation&quot;]"><strong><a href="multi_task_reranking/cross_task_knowledge_distillation_in_multi_task_recommendation/">Cross-Task Knowledge Distillation in Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用四元组辅助排序任务捕获跨任务细粒度排序知识，经校准蒸馏与误差纠错同步训练，在 TikTok/微信朋友圈样本级 AUC/Multi-AUC 显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;隐空间探索&quot;,&quot;长期奖励&quot;]"><strong><a href="rl_bandit_rec/context_uncertainty_in_contextual_bandits_with_applications_to_recommender_/">Context Uncertainty in Contextual Bandits with Applications to Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 REN，在 RNN 隐空间用 DPP 多样性项 + 表征不确定性 UCB 项联合打分，兼顾相关性与探索，理论上保持率最优次线性 regret，在线滚动评估中长期奖励显著超越 GRU4Rec/TCN/HRNN。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线探索/推荐决策&quot;,&quot;Best Arm Identification&quot;,&quot;Revealed Preference&quot;,&quot;Explorative User&quot;]"><strong><a href="rl_bandit_rec/learning_the_optimal_recommendation_from_explorative_users/">Learning the Optimal Recommendation from Explorative Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建探索型用户隐偏好Bandit模型，提出两阶段BAIR算法，在合成BAI任务上将最优臂识别交互数从O(log(1/δ))提升至O(1/δ)并证明紧致。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线推荐策略&quot;,&quot;Contextual Bandit&quot;,&quot;用户流失建模&quot;]"><strong><a href="rl_bandit_rec/modeling_attrition_in_recommender_systems_with_departing_bandits/">Modeling Attrition in Recommender Systems with Departing Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出策略依赖交互时长的离场 bandit，用 POMDP 信念阈值规划与 UCB 回报学习实现 Õ(√T) regret。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;在线审计/策略评估&quot;,&quot;Envy-freeness&quot;,&quot;Pure Exploration Bandits&quot;,&quot;Conservative Exploration&quot;]"><strong><a href="debias_fairness_industrial/online_certification_of_preference_based_fairness_for_personalized_recommen/">Online certification of preference-based fairness for personalized recommender systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将推荐公平审计转化为带保守探索的纯探索老虎机，在线认证用户级无嫉妒性，并以41用户/75臂实现与总用户数无关的概率审计。</p>
</dd>
</dl>
</section>
</div>
