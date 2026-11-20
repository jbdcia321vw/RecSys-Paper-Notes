---
hide:
  - toc
---

# 🏛️ WSDM 2023 推荐系统论文解读

15 篇 WSDM 2023 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;解耦表征&quot;,&quot;推荐多样性&quot;]"><strong><a href="collaborative_filtering/disentangled_representation_for_diversified_recommendations/">Disentangled Representation for Diversified Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DCRS 将用户偏好解耦为类别相关/无关分量，用对抗判别器约束，在 Full-Ranking 协议下兼顾准确率与类别多样性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;子模函数邻居选择&quot;,&quot;长尾损失重加权&quot;]"><strong><a href="graph_rec/dgrec_graph_neural_network_for_recommendation_with_diversified_embedding_ge/">DGRec: Graph Neural Network for Recommendation with Diversified Embedding Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用子模邻居选择、层注意力与长尾损失重加权在嵌入生成阶段直接做多样化，Recall 近似持平 LightGCN 而类目覆盖度显著领先。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图对比学习&quot;,&quot;元网络&quot;]"><strong><a href="graph_rec/heterogeneous_graph_contrastive_learning_for_recommendation/">Heterogeneous Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出元网络个性化异构图对比学习，在 Ciao/Epinions/Yelp 的 1:99 采样排序上 HR@10 较最强基线提升 2.0%–3.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;知识图谱对比学习&quot;,&quot;InfoNCE&quot;]"><strong><a href="self_supervised_denoising/knowledge_enhancement_for_contrastive_multi_behavior_recommendation/">Knowledge Enhancement for Contrastive Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KMCLR 以多行为 InfoNCE + 知识图谱双视图对比 + 一致性调制子图采样去噪，三数据集 HR@10 相对最佳基线提升 1.4%–9.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多层注意力混合&quot;,&quot;归纳偏置剪枝&quot;,&quot;会话表征读出&quot;]"><strong><a href="sequential_rec/efficiently_leveraging_multi_level_user_intent_for_session_based_recommenda/">Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 SparseVD 稀疏化证据证明会话推荐中 GNN 传播冗余，提出多层意图查询生成 + Lp 池化混合的 Atten-Mixer 读出门控，三数据集 HR@20 全面提升且训练快 7 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Neural Processes&quot;,&quot;兴趣动态建模&quot;,&quot;少样本序列推荐&quot;]"><strong><a href="sequential_rec/idnp_interest_dynamics_modeling_using_generative_neural_processes_for_seque/">IDNP: Interest Dynamics Modeling using Generative Neural Processes for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用空洞卷积+自注意力编码多尺度短期兴趣，以神经过程把长期兴趣视为连续函数实例，在四数据集少样本设定下大幅超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端对话评测（仿真）&quot;,&quot;用户模拟器&quot;,&quot;议程驱动交互模型&quot;]"><strong><a href="agent_conversational_rec/usersimcrs_a_user_simulation_toolkit_for_evaluating_conversational_recommen/">UserSimCRS: A User Simulation Toolkit for Evaluating Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>模块化议程驱动用户模拟器工具箱，新增满意度预测、人格/情境建模与条件式自然语言生成，仅用 8 条标注对话即可黑盒评测 IAI MovieBot，但仅有定性案例无定量基准。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;变分推断&quot;,&quot;知识图谱补全&quot;]"><strong><a href="agent_conversational_rec/variational_reasoning_over_incomplete_knowledge_graphs_for_conversational_r/">Variational Reasoning over Incomplete Knowledge Graphs for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将对话特定子图建模为离散隐变量，用变分贝叶斯同时补全不完整 KG 并按上下文选知识；REDIAL 上 Recall@1 相对提升 11.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;对话策略学习&quot;,&quot;元强化学习&quot;,&quot;元探索策略&quot;,&quot;正负反馈状态编码&quot;]"><strong><a href="cold_start_rec/meta_policy_learning_for_cold_start_conversational_recommendation/">Meta Policy Learning for Cold-Start Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用元强化学习为 CRS 学一个可快速个性化的元策略，配合元探索策略、可局部适配的推荐器与 TransGate 状态编码，在新用户上 SR@10 提升 5.2%–16.0% 并显著缩短对话轮数。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习&quot;,&quot;特征表征学习&quot;,&quot;长尾低频特征&quot;]"><strong><a href="ctr_feature_interaction/cl4ctr_a_contrastive_learning_framework_for_ctr_prediction/">CL4CTR: A Contrastive Learning Framework for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用随机/特征/维度三种嵌入扰动构造正对，叠加同域对齐与跨域均匀性三项自监督信号正则化嵌入层，模型无关且零推理开销，四数据集 AUC/LogLoss 全面超越 DCN-V2。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;特权特征 (Privileged Features)&quot;,&quot;表征对齐/相似度损失&quot;]"><strong><a href="multi_task_reranking/slate_aware_ranking_for_recommendation/">Slate-Aware Ranking for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>训练期用特权 slate 编码器与用户编码器做表征对齐，推理期仅用用户编码器注入列表互影响，精排 AUC +0.43%、线上 PV +0.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;列表推荐 (Slate Recommendation)&quot;,&quot;变分自编码器潜动作空间&quot;,&quot;Soft Actor-Critic&quot;]"><strong><a href="rl_bandit_rec/generative_slate_recommendation_with_reinforcement_learning/">Generative Slate Recommendation with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 VAE 将整条列表编码为连续原型动作空间，让 SAC 在低维潜空间选动作再解码成 slate，在 6 个模拟环境中显著超越分解式 RL 基线并有效抑制用户厌倦。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练(联邦)&quot;,&quot;联邦推荐遗忘&quot;,&quot;更新校准回滚&quot;,&quot;半难负采样&quot;]"><strong><a href="debias_fairness_industrial/federated_unlearning_for_on_device_recommendation/">Federated Unlearning for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用日志回滚+方向校准实现联邦推荐用户遗忘，配合重要性更新筛选与混合半难负采样，7× 加速且超越全量重训</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/模型压缩&quot;,&quot;知识蒸馏&quot;,&quot;流行度去偏&quot;,&quot;分组排序学习&quot;]"><strong><a href="debias_fairness_industrial/unbiased_knowledge_distillation_for_recommendation/">Unbiased Knowledge Distillation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示知识蒸馏的流行度偏差，提出按流行度分组的无偏蒸馏 UnKD；在三个数据集上提升 Recall@10 并显著改善非热门物品。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;无分布不确定性量化&quot;,&quot;分组阈值策略&quot;,&quot;逆倾向加权置信界&quot;]"><strong><a href="debias_fairness_industrial/uncertainty_quantification_for_fairness_in_two_stage_recommender_systems/">Uncertainty Quantification for Fairness in Two-Stage Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用无分布有限样本置信界为两阶段推荐的召回层设计分组阈值策略，保证弱势群体候选集的相关物品数达标且候选规模近最优。</p>
</dd>
</dl>
</section>
</div>
