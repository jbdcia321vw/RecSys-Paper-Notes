---
hide:
  - toc
---

# 🏛️ ICLR / NeurIPS / ICML (RecSys) 2024 推荐系统论文解读

22 篇 ICLR / NeurIPS / ICML (RecSys) 2024 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** NeurIPS 2024 ×8 · NeurIPS ×5 · ICLR 2024 ×4 · CTR预估与特征交叉 ×2 · ICML ×2 · 特征交互 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交互" aria-pressed="false">特征交互<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;LLE 流形学习&quot;,&quot;EASE 闭式解&quot;,&quot;嵌入正则化&quot;]"><strong><a href="collaborative_filtering/implicitslim_and_how_it_improves_embedding_based_collaborative_filtering/">ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 EASE 闭式解嵌入 LLE 流形学习框架，用 Woodbury 恒等式在 $L\times L$空间直接抽取嵌入，作为任意 CF 模型的初始化与正则，免去$I\times I$ 相似度矩阵存储。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数设计&quot;,&quot;假阴性鲁棒&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="collaborative_filtering/psl_rethinking_and_improving_softmax_loss_from_pairwise_perspective_for_rec/">PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 Softmax Loss 改写为成对形式并把 exp 换成 tanh/atan/ReLU 有界激活，兼具更紧 DCG 上界、可控样本权重与 DRO 化 BPR 三重性质，IID/OOD/噪声三场景 NDCG@20 稳定提升 1.1%–5.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/训练&quot;,&quot;CVaR&quot;,&quot;iALS&quot;,&quot;卷积平滑&quot;]"><strong><a href="collaborative_filtering/safe_collaborative_filtering/">Safe Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以卷积平滑CVaR嵌入iALS，在保持平均Recall的同时显著改善尾部用户排序质量。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/排序（序列下一项）&quot;,&quot;图正则&quot;,&quot;嵌入优化&quot;]"><strong><a href="graph_rec/graph_enhanced_optimizers_for_structure_aware_recommendation_embedding_evol/">Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出SEvo，将图正则注入嵌入更新而非GNN模块；Leave-One-Out全库排序下多backbone平均提升9%–23%，百万节点数据最高139%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表征增强&quot;,&quot;测试时增强&quot;,&quot;消息传递机理&quot;,&quot;轻量图协同过滤&quot;]"><strong><a href="graph_rec/how_does_message_passing_improve_collaborative_filtering/">How Does Message Passing Improve Collaborative Filtering?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示消息传递主要靠前向邻居表征而非梯度，提出仅推理时聚合的TAG-CF，全库排序下冷/低度用户最高提升39.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列推荐&quot;,&quot;精排/生成式端到端&quot;,&quot;Instance-wise LoRA&quot;,&quot;Mixture of Experts&quot;,&quot;负迁移&quot;]"><strong><a href="sequential_rec/customizing_language_models_with_instance_wise_lora_for_sequential_recommen/">Customizing Language Models with Instance-wise LoRA for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 LoRA 拆成 MoE 专家并由序列表征门控，实现实例级参数定制，HR@1 平均相对 LoRA 提升 11.4% 且可训练参数不增。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;端到端可学习聚类&quot;,&quot;意图对比学习&quot;]"><strong><a href="sequential_rec/end_to_end_learnable_clustering_for_intent_learning_in_recommendation/">End-to-end Learnable Clustering for Intent Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将意图聚类中心设为可学习神经元，端到端优化聚类与对比学习，全库排序下NDCG@5较ICLRec提升8.9%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/下一物品预测&quot;,&quot;双视图语义协同融合&quot;,&quot;检索增强自蒸馏&quot;]"><strong><a href="sequential_rec/llm_esr_large_language_models_enhancement_for_long_tailed_sequential_recomm/">LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用冻结的 LLM 物品/用户语义嵌入搭建语义-协同双视图与相似用户检索自蒸馏，在不增加 LLM 推理开销的前提下同时缓解长尾物品与长尾用户难题。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;LLM推荐&quot;,&quot;重排&quot;,&quot;DPO 偏好对齐&quot;,&quot;Softmax 负采样&quot;]"><strong><a href="llm_rec/on_softmax_direct_preference_optimization_for_recommendation/">On Softmax Direct Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将成对 DPO 推广为多负样本 softmax 排序损失 S-DPO，理论桥接负采样 softmax 并自带难负样本挖掘，三数据集 HR@1 较次优提升 11.1%–47.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;无监督聚类&quot;,&quot;自监督学习&quot;]"><strong><a href="social_group_rec/identify_then_recommend_towards_unsupervised_group_recommendation/">Identify Then Recommend: Towards Unsupervised Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>无监督识别动态群组并用双重自监督 pretext 推荐，免组数/组标签；两数据集 HR/NDCG 显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦图学习&quot;,&quot;负迁移抑制&quot;]"><strong><a href="cross_domain_multi_scenario/federated_graph_learning_for_cross_domain_recommendation/">Federated Graph Learning for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 FedGCDR，以 HVH 联邦图学习 + DP 知识迁移 + 虚拟用户图扩展过滤负迁移，在 Amazon 16 域采样排序上显著优于 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;Embedding Collapse&quot;,&quot;Multi-Embedding&quot;,&quot;特征交互&quot;]"><strong><a href="ctr_feature_interaction/on_the_embedding_collapse_when_scaling_up_recommendation_models/">On the Embedding Collapse when Scaling up Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示推荐模型嵌入坍缩现象，用交互-坍缩理论解释其双面效应，提出多嵌入集+专属交互模块的 ME 设计，在 Criteo/Avazu 实现稳定可扩展的 AUC 提升</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;堆叠因子分解机&quot;,&quot;稠密缩放律&quot;,&quot;特征交互&quot;]"><strong><a href="ctr_feature_interaction/wukong_towards_a_scaling_law_for_large_scale_recommendation/">Wukong: Towards a Scaling Law for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以堆叠 FM + 线性压缩块构建纯稠密可缩放交互架构，在 6 公开数据集与 146B 内部数据上确立跨两数量级复杂度的推荐缩放律。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/在线探索策略&quot;,&quot;Strategic Bandit&quot;,&quot;Incentive-Compatible Mechanism&quot;,&quot;Clickbait&quot;]"><strong><a href="rl_bandit_rec/bandits_meet_mechanism_design_to_combat_clickbait_in_online_recommendation/">Bandits Meet Mechanism Design to Combat Clickbait in Online Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>把 CTR 建模为自利臂的策略动作，用 UCB-S 的淘汰威胁激励真实点击率，任意均衡下取得 Õ(√KT) 强策略遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR_NeurIPS_ICML 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;平台策略层/流量分配机制&quot;,&quot;博弈论均衡分析&quot;,&quot;探索强度机制设计&quot;,&quot;双层优化&quot;]"><strong><a href="rl_bandit_rec/unveiling_user_satisfaction_and_creator_productivity_trade_offs_in_recommen/">Unveiling User Satisfaction and Creator Productivity Trade-Offs in Recommendation Platforms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 Cournot 创作竞争博弈 C4 刻画推荐探索强度 β 对创作者产量的反噬，证明 U-β 增、V-β 减的权衡，并用隐函数梯度双层优化求得个性化 β*，仿真福利提升 &gt;20%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练与模型维护（隐私卸载）&quot;,&quot;Machine Unlearning&quot;,&quot;Recommendation Fairness&quot;,&quot;Benchmark&quot;]"><strong><a href="debias_fairness_industrial/cure4rec_a_benchmark_for_recommendation_unlearning_with_deeper_influence/">CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个推荐卸载综合基准，以完整性、效用、效率、公平四维及 core/edge/random 三类卸载集系统评测 5 种方法。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/排序预测&quot;,&quot;因果去偏&quot;,&quot;倾向得分&quot;,&quot;RKHS核平衡&quot;]"><strong><a href="debias_fairness_industrial/debiased_collaborative_filtering_with_kernel_based_causal_balancing/">Debiased Collaborative Filtering with Kernel-Based Causal Balancing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用RKHS核自适应平衡倾向得分，提出KBIPS/KBDR，在COAT/MUSIC/PRODUCT提升AUC/NDCG/F1。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;差分隐私&quot;,&quot;加性个性化&quot;]"><strong><a href="debias_fairness_industrial/federated_recommendation_with_additive_personalization/">Federated Recommendation with Additive Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>FedRAP 以稀疏全局物品嵌入加本地个性化偏置实现联邦推荐，在六数据集 Leave-One-Out 采样排序下显著优于 FL 基线。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv (2024)&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/在线曝光分配&quot;,&quot;多边公平约束优化&quot;,&quot;低regret Bandit&quot;]"><strong><a href="debias_fairness_industrial/interpolating_item_and_user_fairness_in_multi_sided_recommendations/">Interpolating Item and User Fairness in Multi-Sided Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出FAIR多边公平约束与FORM低regret在线算法，在Amazon模拟中将公平推荐收入损失从约38%降至&lt;10%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（CF 表征训练）&quot;,&quot;对抗协同过滤&quot;,&quot;投毒攻击鲁棒性&quot;,&quot;个性化扰动&quot;]"><strong><a href="debias_fairness_industrial/understanding_and_improving_adversarial_collaborative_filtering_for_robust_/">Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论证明 ACF 降误差并据此按用户嵌入尺度分配扰动幅度，PamaCF 在 Gowalla/Yelp/MIND 上兼顾推荐与抗投毒。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;平台生态长期多样性&quot;,&quot;用户-创作者双重影响&quot;,&quot;极化动力学&quot;]"><strong><a href="debias_fairness_industrial/user_creator_feature_polarization_in_recommender_systems_with_dual_influenc/">User-Creator Feature Polarization in Recommender Systems with Dual Influence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>构建用户-创作者特征动力学，证明双重影响下推荐系统必然极化，并发现 top-k 截断可反向提升长期多样性。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;重排/推荐策略分配&quot;,&quot;双边公平&quot;,&quot;Price of Fairness&quot;,&quot;冷启动误估&quot;]"><strong><a href="debias_fairness_industrial/user_item_fairness_tradeoffs_in_recommendations/">User-Item Fairness Tradeoffs in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>把用户-物品双边公平建成带物品公平约束的极小极大优化，解析证明「多样性红利」与「冷启动误估放大」，并在 arXiv 原型系统上实测验证。</p>
</dd>
</dl>
</section>
</div>
