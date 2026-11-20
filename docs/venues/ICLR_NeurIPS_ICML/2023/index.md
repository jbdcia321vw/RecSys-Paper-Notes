---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2023 推荐系统论文解读

21 篇 ICLR / NeurIPS / ICML (RecSys) 2023 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** NeurIPS 2023 ×6 · NeurIPS ×5 · ICML ×4 · ICLR ×2 · ICLR 2023 ×2 · ICML 2023 ×2 · 去偏公平与工业架构 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排&quot;,&quot;对抗负采样&quot;,&quot;难例挖掘&quot;]"><strong><a href="self_supervised_denoising/empowering_collaborative_filtering_with_principled_adversarial_contrastive_/">Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 AdvInfoNCE：用可学习难度 δ_j 对抗重加权负样本，证明等价 KL-DRO，在四个 Top-K 基准上显著提升 OOD 泛化。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;SVD 图增强&quot;]"><strong><a href="self_supervised_denoising/lightgcl_simple_yet_effective_graph_contrastive_learning_for_recommendation/">LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以截断 SVD 重构交互图生成全局协同视图，并与主视图直接对比，在五个基准的全库排序中 Recall@20 最高提升 23%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Session-based Recommendation&quot;,&quot;Multilingual Multi-locale Dataset&quot;,&quot;Cross-locale Transfer&quot;]"><strong><a href="sequential_rec/amazon_m2_a_multilingual_multi_locale_shopping_session_dataset_for_recommen/">Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>发布首个多语言多地区电商会话数据集（141万商品/360万会话），定义下一商品推荐、跨域迁移推荐与商品标题生成三任务，并揭示流行度与复制基线仍压制深度模型。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;属性图模式&quot;,&quot;记忆增强Transformer&quot;]"><strong><a href="sequential_rec/enhancing_user_intent_capture_in_session_based_recommendation_with_attribut/">Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以频繁属性图模式作记忆增强会话 Transformer，全库 top-K 评测，公开与工业数据平均提升约4.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;引导扩散 (Guided Diffusion)&quot;,&quot;无负采样&quot;]"><strong><a href="generative_rec/generate_what_you_prefer_reshaping_sequential_recommendation_via_guided_dif/">Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DreamRec 将序列推荐重塑为学习生成范式，用 Transformer 引导的无分类器扩散生成 oracle 物品，弃用负采样，三数据集 HR@20 相对提升 7%–21%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;RQ-VAE&quot;,&quot;Semantic ID&quot;,&quot;Generative Retrieval&quot;]"><strong><a href="generative_rec/recommender_systems_with_generative_retrieval/">Recommender Systems with Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TIGER 用 RQ-VAE 构建层次 Semantic ID，并以 Transformer 自回归生成下一物品 ID；在 Amazon 三类 Full-Ranking 协议下显著超越 SASRec/S3-Rec，并展示冷启动与多样性收益。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;交互式查询/在线校验&quot;,&quot;不确定性最小化&quot;,&quot;期望确定性增益&quot;,&quot;在线决策树&quot;]"><strong><a href="agent_conversational_rec/lending_interaction_wings_to_recommender_systems_with_conversational_agents/">Lending Interaction Wings to Recommender Systems with Conversational Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CORE 提出「离线打分—在线校验」范式，用不确定性最小化驱动对话智能体闭式决定查物品或属性，即插即用于 9 种 RS、8 个数据集，冷启动 S@5 最高提升 400%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端策略决策&quot;,&quot;内在奖励学习&quot;,&quot;双层多目标优化&quot;]"><strong><a href="agent_conversational_rec/multi_objective_intrinsic_reward_learning_for_conversational_recommender_sy/">Multi-Objective Intrinsic Reward Learning for Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将内在奖励学习建模为双层多目标优化，用 HRS 与 RPM 元梯度更新奖励，在三个 CRS 基准上显著提升成功率并缩短轮数。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;观点动力学&quot;,&quot;冲突感知&quot;,&quot;链接推荐&quot;]"><strong><a href="social_group_rec/on_the_relationship_between_relevance_and_conflict_in_online_social_link_re/">On the Relationship Between Relevance and Conflict in Online Social Link Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用FJ观点动力学分析社交链接推荐中相关性与冲突减少的关系，提出冲突感知CA指标并在Reddit/Twitter评估13种算法。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;CTR预估与特征交叉&quot;,&quot;机制设计/拍卖定价&quot;,&quot;UCB 在线学习&quot;,&quot;负遗憾 (Negative Regret)&quot;,&quot;点击付费拍卖&quot;]"><strong><a href="ctr_feature_interaction/improved_online_learning_algorithms_for_ctr_prediction_in_ad_auctions/">Improved Online Learning Algorithms for CTR Prediction in Ad Auctions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 PPC 广告拍卖 CTR 未知下的收入最大化建模为在线学习，UCB 机制在静态估价正间隔下取得 -Ω(T) 负遗憾，对抗性最坏情况达紧致 Õ(√T) 界。</p>
</dd>
<dt data-paper-tags="[&quot;ICML 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Pairwise Ranking Loss&quot;,&quot;Welfare Maximization&quot;]"><strong><a href="ctr_feature_interaction/pairwise_ranking_losses_of_click_through_rates_prediction_for_welfare_maxim/">Pairwise Ranking Losses of Click-Through Rates Prediction for Welfare Maximization in Ad Auctions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出面向广告拍卖福利最大化的加权 pairwise ranking loss，并借助教师网络校准；Criteo/DeepFM 上 welfare 提升 1.47%，AUC 基本持平。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/会话级RL&quot;,&quot;用户模拟器&quot;,&quot;长期留存&quot;]"><strong><a href="rl_bandit_rec/kuaisim_a_comprehensive_simulator_for_recommender_systems/">KuaiSim: A Comprehensive Simulator for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出多行为、跨会话推荐模拟器 KuaiSim，统一请求级列表、整会话与留存优化，并在 KuaiRand/ML-1m 给出可复现基准。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线策略&quot;,&quot;Interaction-Grounded Learning&quot;,&quot;潜在奖励学习&quot;,&quot;个性化去偏&quot;]"><strong><a href="rl_bandit_rec/personalized_reward_learning_with_interaction_grounded_learning_igl/">Personalized Reward Learning with Interaction-Grounded Learning (IGL)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用逆运动学与3状态潜在奖励学习个性化隐式反馈，在仿真、Facebook新闻与生产图像推荐中提升公平性并抑制负反馈。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;残差策略学习&quot;,&quot;离线强化学习&quot;,&quot;条件变分自编码器&quot;]"><strong><a href="rl_bandit_rec/resact_reinforcing_long_term_engagement_in_sequential_recommendation_with_r/">ResAct: Reinforcing Long-term Engagement in Sequential Recommendation with Residual Actor</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 CVAE 重构在线策略并叠加残差 Actor 做近端策略改进，配双重信息论正则，NCIS 离线评估下 Session Length 提升 35%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推荐&quot;,&quot;倾向得分估计&quot;,&quot;IPS&quot;]"><strong><a href="debias_fairness_industrial/estimating_propensity_for_causality_based_recommendation_without_exposure_d/">Estimating Propensity for Causality-based Recommendation without Exposure Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>仅用交互数据，以流行度成对先验与 Beta 正则估计倾向/曝光，接入 IPS 因果推荐，在 DH/ML 上逼近真值。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平与工业架构&quot;,&quot;推荐策略分析&quot;,&quot;Price of Anarchy&quot;,&quot;Random Utility Model&quot;,&quot;内容创作者博弈&quot;]"><strong><a href="debias_fairness_industrial/how_bad_is_top_k_recommendation_under_competing_content_creators/">How Bad is Top-$K$ Recommendation under Competing Content Creators?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以价格无政府比 (PoA) 证明：内容创作者以用户参与度博弈时，top-$K$推荐的福利损失永远被$1+\frac{1}{(1+\beta)\log K}$ 小常数上界卡死且紧；用户曝光激励则可任意差。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/排序学习&quot;,&quot;战略激励&quot;,&quot;多样性正则&quot;]"><strong><a href="debias_fairness_industrial/performative_recommendation_diversifying_content_via_strategic_incentives/">Performative Recommendation: Diversifying Content via Strategic Incentives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将创作者战略响应嵌入可微排序，以多样性正则激励内容多样化；Yelp 动态重训练下同 NDCG 损失使多样性翻倍。</p>
</dd>
<dt data-paper-tags="[&quot;ICML 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;平台机制设计&quot;,&quot;潜在博弈&quot;,&quot;尾部积分奖励&quot;]"><strong><a href="debias_fairness_industrial/rethinking_incentives_in_recommender_systems_are_monotone_rewards_always_be/">Rethinking Incentives in Recommender Systems: Are Monotone Rewards Always Beneficial?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>证明功绩制单调奖励 M3 存在常数福利损失，提出尾部积分 BRM 使创作者竞争成为潜在博弈并可用 BRCM 参数化优化社会福利。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与工业架构&quot;,&quot;市场机制/平台生态&quot;,&quot;Supply-Side Equilibrium&quot;,&quot;Specialization&quot;,&quot;博弈论&quot;]"><strong><a href="debias_fairness_industrial/supply_side_equilibria_in_recommender_systems/">Supply-Side Equilibria in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 D 维双线性博弈与凸对偶刻画推荐系统供给侧专业化相变，给出单 genre 均衡的充要条件与 β* 阈值，并在 MovieLens-100K 验证阈值随嵌入维度下降。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;双重稳健估计 (DR)&quot;,&quot;靶向学习 (Targeted Learning)&quot;,&quot;逆倾向加权 (IPS)&quot;]"><strong><a href="debias_fairness_industrial/tdr_cl_targeted_doubly_robust_collaborative_learning_for_debiased_recommend/">TDR-CL: Targeted Doubly Robust Collaborative Learning for Debiased Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用靶向学习把 DR 的校正项经验均值归零，并将插补误差拆为参数化+非参数化协同更新，Coat/Music 上 AUC、NDCG 全面超越 DR 家族且方差更小。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/表征学习&quot;,&quot;纵向联邦学习&quot;,&quot;隐私保护图神经网络&quot;,&quot;随机投影&quot;]"><strong><a href="debias_fairness_industrial/vertical_federated_graph_neural_network_for_recommender_system/">Vertical Federated Graph Neural Network for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首个纵向联邦 GNN 推荐框架 VerFedGNN，以随机投影传输邻居聚合矩阵、三元量化扰动公共梯度，RMSE 逼近集中式且去匿名攻击 F1 从 0.90 降至 0.02。</p>
</dd>
</dl>
</section>
</div>
