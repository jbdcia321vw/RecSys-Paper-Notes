---
hide:
  - toc
---

# 🏛️ SIGIR 2023 推荐系统论文解读

56 篇 SIGIR 2023 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×4 · 对比学习 ×4 · 端到端生成式 ×3 · 离线强化学习 ×2 · 课程学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">22</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Dense Retrieval&quot;,&quot;Personalization&quot;,&quot;Unified Information Access&quot;]"><strong><a href="collaborative_filtering/a_personalized_dense_retrieval_framework_for_unified_information_access/">A Personalized Dense Retrieval Framework for Unified Information Access</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出统一信息访问双塔框架UIA，用功能编码+用户历史注意力个性化，在Lowe’s/ESCI全库排序显著超越任务专用与联合训练基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;替代商品推荐&quot;,&quot;弱监督学习&quot;]"><strong><a href="collaborative_filtering/a_transformer_based_substitute_recommendation_model_incorporating_weakly_su/">A Transformer-Based Substitute Recommendation Model Incorporating Weakly Supervised Customer Behavior Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以商品标题对+XLMR交互匹配替代关系，PR对数回归与负采样去噪，11市场单模型A/B收入+19%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/全库排序&quot;,&quot;ODE 图滤波&quot;,&quot;Blurring-Sharpening&quot;]"><strong><a href="collaborative_filtering/blurring_sharpening_process_models_for_collaborative_filtering/">Blurring-Sharpening Process Models for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出免训练 ODE 模糊-锐化范式直接处理交互矩阵，在三个基准全库排序 Recall/NDCG 超 43 个基线，NDCG 最高提升 3.74%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Embedding 尺寸搜索&quot;,&quot;连续动作空间强化学习&quot;,&quot;随机游走探索&quot;]"><strong><a href="collaborative_filtering/continuous_input_embedding_size_search_for_recommender_systems/">Continuous Input Embedding Size Search For Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 TD3 在连续动作空间为每个用户/物品搜索任意粒度 Embedding 尺寸，并以随机游走探索提升收敛，80–95% 稀疏率下 R@20 全面超越离散 RL 搜索。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;点积模型嵌入维度&quot;,&quot;流行度偏差与物品公平性&quot;]"><strong><a href="collaborative_filtering/curse_of_low_dimensionality_in_recommender_systems/">Curse of \&quot;Low\&quot; Dimensionality in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>论证点积模型低维嵌入是可表示排序数指数级塌缩的根源，实证其加剧流行度偏差、损害多样性与公平性，并给出上界 $n^{2d}$ 与锥体刻画。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;意图解缠&quot;,&quot;图对比学习&quot;,&quot;自适应增广&quot;]"><strong><a href="collaborative_filtering/disentangled_contrastive_collaborative_filtering/">Disentangled Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 K 个全局意图原型解缠用户/物品表征，并用解缠相似度驱动参数化图掩码做自适应对比增广，全库排序下 Recall@20 相对最优基线提升 2.8%–6.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图结构重连 (Graph Rewiring)&quot;,&quot;双边 Top-K 去噪与增广&quot;,&quot;用户-用户/物品-物品相关性&quot;]"><strong><a href="graph_rec/graph_collaborative_signals_denoising_and_augmentation_for_recommendation/">Graph Collaborative Signals Denoising and Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>预训练后双边 Top-K 重连二部图并补入用户-用户/物品-物品对称相关块，Amazon 全库排序 NDCG@20 相对提升 6.9%–13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;逻辑查询嵌入 (LQE)&quot;,&quot;多任务知识共享&quot;]"><strong><a href="graph_rec/logicrec_recommendation_with_users_logical_requirements/">LogicRec: Recommendation with Users&#x27; Logical Requirements</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户逻辑需求形式化为 KG 结构化逻辑查询，用 LQE 联合推断缺失事实并以多任务多门专家共享 A_l/A_u 答案集，Amazon-book hit@20 相对提升约 22%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;端到端训练/召回&quot;,&quot;Sharpness-Aware Minimization&quot;,&quot;双层优化&quot;,&quot;隐函数微分&quot;,&quot;泛化与长尾&quot;]"><strong><a href="graph_rec/sharpness_aware_graph_collaborative_filtering/">Sharpness-Aware Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双层优化+隐函数超梯度把锐度感知训练引入GNN协同过滤，全库排序下LightGCN骨干Recall@20最高提升13.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;增量SVD&quot;,&quot;图信号处理低通滤波&quot;,&quot;可解释交互推荐&quot;]"><strong><a href="graph_rec/triple_structural_information_modelling_for_accurate_explainable_and_intera/">Triple Structural Information Modelling for Accurate, Explainable and Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>增量SVD低通滤波＋无参数注意力＋物品转移矩阵三源融合，全库排序 HR@10 最高相对提升 203.8%，并支持频率级可解释交互推荐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;方向性扩散噪声&quot;,&quot;线性图 Transformer&quot;]"><strong><a href="self_supervised_denoising/a_directional_diffusion_graph_transformer_for_recommendation/">A Directional Diffusion Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以各向异性方向高斯噪声替代等方向噪声做嵌入级扩散，并用条件线性 Transformer 反向去噪，全库排序 Recall@20 相对 SOTA 提升约 3–4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回（全库排序）&quot;,&quot;变分图重构&quot;,&quot;聚类感知对比学习&quot;]"><strong><a href="self_supervised_denoising/generative_contrastive_graph_learning_for_recommendation/">Generative-Contrastive Graph Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以变分图重构采样替代数据增广生成对比视图，联合节点/簇级对比，在三个数据集 Full-Ranking 下 NDCG@20 较 LightGCN 提升 8.61%–28.17%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;Graph Transformer&quot;,&quot;Masked Autoencoding&quot;,&quot;Rationale Discovery&quot;]"><strong><a href="self_supervised_denoising/graph_transformer_for_recommendation/">Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GFormer 用拓扑感知图 Transformer 参数化发现协同 rationale 子图，并以 rationale-aware 掩码自编码替代随机对比增广；全库排序下 Recall@20 较 SGL 提升 5.2%–7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为对齐&quot;,&quot;KL散度去噪&quot;,&quot;隐变量推断&quot;]"><strong><a href="self_supervised_denoising/improving_implicit_feedback_based_recommendation_through_multi_behavior_ali/">Improving Implicit Feedback-Based Recommendation through Multi-Behavior Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户真实偏好设为隐变量，用 KL 散度对齐点击/购买两条行为链并交替去噪，三数据集 Recall@20 相对最优基线提升 0.5%–6.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与排序表征学习&quot;,&quot;知识图谱剪枝&quot;,&quot;对比去噪&quot;,&quot;伯努利掩码&quot;]"><strong><a href="self_supervised_denoising/knowledge_refined_denoising_network_for_robust_recommendation/">Knowledge-refined Denoising Network for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KRDN 用伯努利掩码剪枝 KG 任务无关三元组，并以协同/知识信号分歧对比式剔除噪声交互；全库排序下 NDCG@20 较 SOTA 提升 4.6%–12.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;梯度冲突矫正&quot;]"><strong><a href="self_supervised_denoising/multi_behavior_self_supervised_learning_for_recommendation/">Multi-behavior Self-supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为感知GNN+双层自判别对比+梯度混合矫正，Leave-One-Out全库排序下Recall@10平均提升约19%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知序列推荐&quot;,&quot;高斯混合注意力&quot;,&quot;Attentive FISM&quot;]"><strong><a href="sequential_rec/attention_mixtures_for_time_aware_sequential_recommendation/">Attention Mixtures for Time-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯混合注意力量化物品与时间上下文两条注意力流的权重，并融合 Attentive FISM 长兴趣，1K 采样排序下 HR/NDCG@10 全面领先时序 Transformer。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;集成学习&quot;,&quot;对比知识蒸馏&quot;,&quot;掩码物品预测&quot;]"><strong><a href="sequential_rec/ensemble_modeling_with_contrastive_knowledge_distillation_for_sequential_re/">Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 N 个并行双向 Transformer 做序列编码器集成，并以 ICL/CCL 表征对比蒸馏 + logits 协同 KL 蒸馏打通网络间知识迁移；5-core + Leave-One-Out + 全库排序下 NDCG@10 相对提升 20.7%~33.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;频域注意力&quot;,&quot;自相关周期建模&quot;]"><strong><a href="sequential_rec/frequency_enhanced_hybrid_attention_network_for_sequential_recommendation/">Frequency Enhanced Hybrid Attention Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户行为序列迁移到频域，用频率斜坡采样改造自注意力并叠加自相关周期注意力，Leave-One-Out 全库排序下在四个基准上实现 3.6%–10.9% 的稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图掩码自编码器&quot;,&quot;自适应数据增广&quot;]"><strong><a href="sequential_rec/graph_masked_autoencoder_for_sequential_recommendation/">Graph Masked Autoencoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MAERec 用以任务自适应路径掩码驱动的图掩码自编码器替代手工对比增广，在留一法全库排序下 HR@20 相对最强基线提升最高 11.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;线性注意力&quot;,&quot;长序列建模&quot;,&quot;高效 Transformer&quot;]"><strong><a href="sequential_rec/linrec_linear_attention_mechanism_for_long_term_sequential_recommender_syst/">LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ELU + 行/列 L2 归一化替换 Softmax，令 QK^T V 可交换计算顺序，把长序列自注意力从 O(N²) 降到 O(N)，并在 ML-1M/Gowalla 全库排序上持平或超越 SASRec/BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;可学习模型增强&quot;,&quot;元学习&quot;]"><strong><a href="sequential_rec/meta_optimized_contrastive_learning_for_sequential_recommendation/">Meta-optimized Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习模型增强器与元学习优化对比学习，在 LOO+Full-Ranking 下 HR/NDCG 相对最强基线提升 2.5–8.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;App 推荐数据集&quot;,&quot;全库排序基准&quot;]"><strong><a href="sequential_rec/mobilerec_a_large_scale_dataset_for_mobile_apps_recommendation/">MobileRec: A Large-Scale Dataset for Mobile Apps Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>构建 19.3M 条 Google Play 用户-App 时序交互（10,173 App / 48 类 / 0.7M 用户），提供 uid、时间戳与 App 元数据，并以 Leave-One-Out + Full-Ranking 协议给出 9 个序列推荐基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排/训练阶段负采样&quot;,&quot;难负例挖掘&quot;,&quot;加权物品转移图&quot;,&quot;课程学习&quot;]"><strong><a href="sequential_rec/neighborhood_based_hard_negative_mining_for_sequential_recommendation/">Neighborhood-based Hard Negative Mining for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在全局加权物品转移图上以 Jaccard 邻域重叠度构造难负采样分布，并用课程学习调度假阴性阈值；在三个 Amazon 数据集 Leave-One-Out + 1000 候选采样排序下，相对最强负采样基线 NDCG@20 稳定提升 1.0–2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;兴趣解耦&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/when_search_meets_recommendation_learning_disentangled_search_representatio/">When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SESRec 用查询-物品对齐 + 三元组对比自监督把搜索/推荐行为解耦为相似与相异兴趣，在快手与 Amazon 上 NDCG@10 相对提升 7.05%/2.48%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回（端到端生成式全库打分）&quot;,&quot;扩散模型去噪&quot;,&quot;隐空间压缩扩散&quot;,&quot;时间感知重加权&quot;]"><strong><a href="generative_rec/diffusion_recommender_model/">Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次将扩散模型引入全物品交互生成式推荐，通过缩小噪声尺度并预测 x0 保留个性化信息，辅以物品聚类隐空间扩散与时间重加权，Full-Ranking 下 R@10 最高提升 10.67%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;Prompt Learning&quot;,&quot;Cloze-style Mask Prediction&quot;,&quot;Multi-Prompt Ensembling&quot;]"><strong><a href="llm_rec/prompt_learning_for_news_recommendation/">Prompt Learning for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次将新闻点击预测改写为完形填空式 MLM 任务，设计离散/连续/混合三类模板与多模板概率集成；MIND 上 1:4 采样排序评测，跨类型集成 AUC 69.64，较 BERT-NRMS 相对提升 1.5%–4.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;策略学习与交互决策&quot;,&quot;超图神经网络&quot;,&quot;跨视图对比学习&quot;]"><strong><a href="agent_conversational_rec/multi_view_hypergraph_contrastive_policy_learning_for_conversational_recomm/">Multi-view Hypergraph Contrastive Policy Learning for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用三视图动态超图+分层超图编码+跨视图对比策略学习建模对话推荐中的多源用户偏好，Yelp 上 SR@15 提升 18.28%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;可解释推荐&quot;,&quot;解释生成&quot;,&quot;数据集构建&quot;]"><strong><a href="agent_conversational_rec/towards_explainable_conversational_recommender_systems/">Towards Explainable Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 CRS 可解释性 10 维评估视角，构建含 2,058 条高质量解释的 E-ReDial 数据集，T5+知识接地生成 Explanations 时 F1 提升 21.8%、人工总体评分从 1.14 提升至 3.67。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉-语言预训练&quot;,&quot;知识蒸馏&quot;]"><strong><a href="multimodal_rec/enhancing_dynamic_image_advertising_with_vision_language_pre_training/">Enhancing Dynamic Image Advertising with Vision-Language Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 20B 图文预训练 + 相关性蒸馏统一广告图检索与相关性重排，百度线上 CPM/CTR 提升 1.04%/1.865%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;稳定学习&quot;,&quot;HSIC 去相关&quot;,&quot;样本重加权&quot;]"><strong><a href="multimodal_rec/mining_stable_preferences_adaptive_modality_decorrelation_for_multimedia_re/">Mining Stable Preferences: Adaptive Modality Decorrelation for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 HSIC 度量模态相关度并做样本重加权，把多模态推荐从虚假偏好拉到稳定偏好，四骨干四数据集 IID 平均涨 3–22%、OOD 涨 2–48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;多模态推荐&quot;,&quot;端到端生成式（解释生成）&quot;,&quot;多模态解释生成&quot;,&quot;对比学习硬负采样&quot;,&quot;DPP 多样性选图&quot;]"><strong><a href="multimodal_rec/personalized_showcases_generating_multi_modal_explanations_for_recommendati/">Personalized Showcases: Generating Multi-Modal Explanations for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出“个性化展示”新任务，用 DPP 选图 + GPT-2 多模态解码生成图文解释，以跨模态/个性化对比学习提升多样性与视觉对齐，Distinct-2 提升近 5 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态编码器&quot;,&quot;端到端训练&quot;]"><strong><a href="multimodal_rec/where_to_go_next_for_recommender_systems_id_vs_modality_based_recommender_m/">Where to Go Next for Recommender Systems? ID- vs. Modality-based Recommender Models Revisited</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以统一骨干/采样/损失公平对比 IDRec 与纯模态 MoRec，证明 SASRec+端到端微调下 MoRec 已在常规/暖物品场景追平或超越 IDRec，冷启动碾压但训练开销 &gt;100×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;重排&quot;,&quot;概念瓶颈&quot;,&quot;最优传输&quot;,&quot;可编辑用户画像&quot;]"><strong><a href="cold_start_rec/editable_user_profiles_for_controllable_text_recommendation/">Editable User Profiles for Controllable Text Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>LACE以检索式概念瓶颈构建人类可读的用户画像，用最优传输计算个性化概念值进行文本推荐，在冷启动与零样本设定下NDCG@20较NeuKNN提升5–29%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/排序&quot;,&quot;长尾用户-物品联合建模&quot;,&quot;双边分支互增强&quot;,&quot;课程学习&quot;]"><strong><a href="cold_start_rec/melt_mutual_enhancement_of_long_tailed_user_and_item_for_sequential_recomme/">MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SIGIR 2023 长尾序列推荐：双边分支互增强+课程学习，联合提升尾用户/尾物品，Music 总体 HR@10 +8.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/排序&quot;,&quot;不确定性伪交互生成&quot;,&quot;教师-学生一致性学习&quot;]"><strong><a href="cold_start_rec/uncertainty_aware_consistency_learning_for_cold_start_item_recommendation/">Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>UCC 用不确定性伪交互生成与教师-学生一致性学习对齐冷/热物品分布，10-core/7:1:2、K=20 下 Recall 最高提升 27.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;双曲几何表征&quot;,&quot;线性化图 Transformer&quot;]"><strong><a href="cross_domain_multi_scenario/hgformer_hyperbolic_graph_transformer_for_recommendation/">Hgformer: Hyperbolic Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>双曲流形上的无参图卷积 + 线性化双曲交叉注意力，兼顾长尾局部结构与全局信息，全库排序下 Amazon-Book R@10 相对提升 17.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;时空轨迹与 POI 本地生活推荐&quot;,&quot;精排&quot;,&quot;时空特征选择&quot;,&quot;动态特征门控&quot;]"><strong><a href="poi_spatiotemporal_rec/exploring_the_spatiotemporal_features_of_online_food_recommendation_service/">Exploring the Spatiotemporal Features of Online Food Recommendation Service</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统剖析外卖场景时空特征价值，指出 self-attention 不适用的根因，提出时间差特征与动态上下文自适应模块 DCAM，离线 AUC 与线上 CTR 双赢</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端侧推理/召回&quot;,&quot;去中心化协同学习&quot;,&quot;知识蒸馏互信息最大化&quot;]"><strong><a href="poi_spatiotemporal_rec/model_agnostic_decentralized_collaborative_learning_for_on_device_poi_recom/">Model-Agnostic Decentralized Collaborative Learning for On-Device POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MAC 允许端侧 POI 推荐模型结构异构，用参考数据软决策+互信息蒸馏做去中心化协同，采样邻居降噪，1/6 模型体积下 HR@10 超 DCLR 3.8%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习&quot;,&quot;嵌入掩码&quot;,&quot;CVR预估&quot;]"><strong><a href="ctr_feature_interaction/contrastive_learning_for_conversion_rate_prediction/">Contrastive Learning for Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用嵌入掩码替代特征掩码构造对比视图，配合假负样本剔除与转化正样本纳入，缓解CVR稀疏，工业/公开集AUC+0.0079/+0.0066</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;纵向联邦学习&quot;,&quot;条件扩散模型生成嵌入&quot;,&quot;标签推断攻击防御&quot;]"><strong><a href="ctr_feature_interaction/fedads_a_benchmark_for_privacy_preserving_cvr_estimation_with_vertical_fede/">FedAds: A Benchmark for Privacy-Preserving CVR Estimation with Vertical Federated Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首个基于阿里真实广告日志的纵向联邦 CVR 基准，含 1130 万样本数据集、扩散式未对齐样本合成 (Diffu-AT) 与梯度 mixup+投影隐私防御 (MixPro) 的系统评测。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;不变学习&quot;,&quot;表征解耦&quot;]"><strong><a href="ctr_feature_interaction/reformulating_ctr_prediction_learning_invariant_feature_interactions_for_re/">Reformulating CTR Prediction: Learning Invariant Feature Interactions for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>按时间切分环境、解耦环境不变/特异嵌入做不变学习，剔除不稳定特征交叉；ML-10M/Douban 上 AUC 相对 FM 提升 1.63%/0.18%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;排序集成&quot;,&quot;用户意图建模&quot;,&quot;误差-歧义分解&quot;]"><strong><a href="multi_task_reranking/intent_aware_ranking_ensemble_for_personalized_recommendation/">Intent-aware Ranking Ensemble for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 IntEL，用意图感知的物品级权重融合多目标异构列表，经误差-歧义分解理论保证，Tmall/LifeData 全目标 NDCG 显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;离线强化学习&quot;,&quot;马太效应&quot;,&quot;熵惩罚&quot;]"><strong><a href="rl_bandit_rec/alleviating_matthew_effect_of_offline_reinforcement_learning_in_interactive/">Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>指出离线 RL 的保守性惩罚会放大马太效应，提出熵惩罚项 DORL，在 KuaiRec/KuaiRand 交互环境中累计收益提升 79.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;离线强化学习&quot;,&quot;Decision Transformer&quot;,&quot;因果奖励估计&quot;]"><strong><a href="rl_bandit_rec/causal_decision_transformer_for_recommender_systems_via_offline_reinforceme/">Causal Decision Transformer for Recommender Systems via Offline Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以因果决策 Transformer 将离线推荐改写为轨迹序列建模，绕开奖励函数手工设计，在 6 个真实数据集与 VirtualTB 模拟器上取得召回/CTR 领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;状态增广&quot;,&quot;对比学习&quot;]"><strong><a href="rl_bandit_rec/contrastive_state_augmentations_for_reinforcement_learning_based_recommende/">Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用四种局部状态增广平滑离线 RL 的 Q 值面，并对增广视图施加对比损失，在三个数据集与模拟在线环境中稳定提升序列推荐收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;策略学习/端到端排序&quot;,&quot;真实工业数据集&quot;,&quot;离线策略评估&quot;,&quot;Slate 推荐&quot;]"><strong><a href="rl_bandit_rec/rl4rs_a_real_world_dataset_for_reinforcement_learning_based_recommender_sys/">RL4RS: A Real-World Dataset for Reinforcement Learning based Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>发布首个开源真实工业级 RL 推荐数据集（Slate/SeqSlate），配套四层评估框架与数据理解工具；PPO 线上 GMV 较 DIEN 提升 23.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;预训练与提示调优&quot;,&quot;后门攻击&quot;,&quot;序列推荐&quot;]"><strong><a href="debias_fairness_industrial/attacking_pre_trained_recommendation/">Attacking Pre-trained Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示预训练序列推荐模型提供者可植入后门，提出基础替换与提示增强攻击，在 Leave-One-Out 协议下将目标物品曝光率提升数百倍且保持隐蔽。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;连续时间点过程&quot;,&quot;个性化位置去偏&quot;]"><strong><a href="debias_fairness_industrial/coupa_an_industrial_recommender_system_for_online_to_offline_service_platfo/">COUPA: An Industrial Recommender System for Online to Offline Service Platforms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用连续时间点过程+个性化位置选择器联合建模 O2O 场景的时间周期性与位置偏置，Alipay 线上 CTR 提升 2.47%–5.63%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;生成式端到端&quot;,&quot;Diffusion Recommender&quot;,&quot;Popularity Bias&quot;,&quot;Optimal Transport&quot;]"><strong><a href="debias_fairness_industrial/fairdiff_mitigating_the_self_reinforcing_matthew_effect_in_diffusion_recomm/">FairDiff: Mitigating the Self-Reinforcing Matthew Effect in Diffusion Recommender Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示扩散推荐模型优化偏置+先验失配叠加的自强化马太效应，用推理期流行度条件引导与最优传输语义校准即插即用地同时提升精度与长尾公平。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;评测基准&quot;,&quot;统计显著性检验&quot;,&quot;Copula 模拟&quot;]"><strong><a href="debias_fairness_industrial/inference_at_scale_significance_testing_for_large_search_and_recommendation/">Inference at Scale: Significance Testing for Large Search and Recommendation Experiments</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Copula 模拟实证 5 种成对显著性检验：大样本下 Sign/Wilcoxon 一类错误率严重膨胀，功效全部饱和至 100%，应改用效应量判据。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排/候选重排&quot;,&quot;数据集与评测基准&quot;,&quot;个性化商品搜索&quot;,&quot;真实查询与冷启动&quot;]"><strong><a href="debias_fairness_industrial/jdsearch_a_personalized_product_search_dataset_with_real_queries_and_full_i/">JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>发布京东 JDsearch 数据集：真实 query、不裁剪冷尾商品、200 候选内精排评估，TEM 较 QEM MRR 提升 25.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;物品侧公平曝光&quot;,&quot;生存分析&quot;,&quot;时效分布建模&quot;]"><strong><a href="debias_fairness_industrial/measuring_item_global_residual_value_for_fair_recommendation/">Measuring Item Global Residual Value for Fair Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用生存分析建模物品级时效分布 GRV，以模型无关的加权融合实现新老物品公平曝光，MIND 上 NeuMF 的 HR@5 提升 131%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序/精排&quot;,&quot;流行度去偏&quot;,&quot;IPS 逆倾向得分估计&quot;]"><strong><a href="debias_fairness_industrial/popularity_debiasing_from_exposure_to_interaction_in_collaborative_filterin/">Popularity Debiasing from Exposure to Interaction in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 IPL 准则——物品交互量应正比于喜欢它的用户数，用 IPS 估计并构造 std 正则，去偏同时反涨推荐精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全链路离线评估&quot;,&quot;全局时间线&quot;,&quot;数据泄漏&quot;,&quot;评测协议&quot;]"><strong><a href="debias_fairness_industrial/take_a_fresh_look_at_recommender_systems_from_an_evaluation_standpoint/">Take a Fresh Look at Recommender Systems from an Evaluation Standpoint</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>指出随机/留一划分缺失全局时间线导致 Popularity 基线病态与未来数据泄漏，提出 Recent Popularity 与时间窗评测方案。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;投毒攻击&quot;,&quot;反事实解释&quot;,&quot;神经逻辑推理&quot;]"><strong><a href="debias_fairness_industrial/the_dark_side_of_explanations_poisoning_recommender_systems_with_counterfac/">The Dark Side of Explanations: Poisoning Recommender Systems with Counterfactual Examples</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次揭示反事实解释 API 的可被投毒风险：用 CF 训练 Horn 子句代理模型并反向优化生成虚假用户画像，Yelp/ML-100K 上 HR@10 最高相对提升 45%。</p>
</dd>
</dl>
</section>
</div>
