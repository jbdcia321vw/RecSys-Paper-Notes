---
hide:
  - toc
---

# 🏛️ KDD 2023 推荐系统论文解读

28 篇 KDD 2023 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/模型设计&quot;,&quot;联合搜索&quot;,&quot;AutoML&quot;]"><strong><a href="collaborative_filtering/efficient_and_joint_hyperparameter_and_architecture_search_for_collaborativ/">Efficient and Joint Hyperparameter and Architecture Search for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出CF超参与架构联合搜索：超参筛减+频率子图两阶段BORE/RF搜索，全库Recall@20最高提升13.02%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;LightGCN&quot;,&quot;多准则推荐&quot;,&quot;PairNorm&quot;]"><strong><a href="graph_rec/criteria_tell_you_more_than_ratings_criteria_preference_aware_light_graph_c/">Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多准则评分展开为准则-物品扩展二部图，用轻量图卷积 + 用户准则偏好/物品准则双嵌入建模，全库排序下 P@5 最高提升 141.2%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;有效电导&quot;,&quot;难负样本采样&quot;,&quot;数据子采样&quot;]"><strong><a href="graph_rec/graph_based_model_agnostic_data_subsampling_for_recommendation_systems/">Graph-Based Model-Agnostic Data Subsampling for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用用户-物品二部图的有效电导度量负样本难度并做边传播平滑，实现免 pilot 模型的数据子采样，KuaiRec/MIND 上 AUC 稳定超越均匀采样且可与模型法互补融合。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱合理化&quot;,&quot;掩码自编码器&quot;,&quot;跨视图对比学习&quot;]"><strong><a href="graph_rec/knowledge_graph_self_supervised_rationalization_for_recommendation/">Knowledge Graph Self-Supervised Rationalization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KGRec 用注意力合理化分数驱动掩码重建生成式任务与去噪跨视图对比学习，Full-Ranking 下 Recall@20 较 SOTA 提升 2.3%–4.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自适应视图生成&quot;,&quot;图边去噪&quot;,&quot;对比学习&quot;]"><strong><a href="self_supervised_denoising/adaptive_graph_contrastive_learning_for_recommendation/">Adaptive Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 VGAE 生成视图与可学习边去噪视图替代手工图增广，为 LightGCN 提供自适应对比信号，全库排序下 Recall@20 相对提升约 5–8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;双层优化&quot;,&quot;梯度匹配&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="self_supervised_denoising/efficient_bi_level_optimization_for_recommendation_denoising/">Efficient Bi-Level Optimization for Recommendation Denoising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将隐式反馈去噪建模为双层优化，用权重生成器+一步梯度匹配高效求解，LightGCN 上 Recall@20 最高提升 10.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;一致性训练&quot;,&quot;双向KL散度&quot;]"><strong><a href="sequential_rec/ct4rec_simple_yet_effective_consistency_training_for_sequential_recommendat/">CT4Rec: Simple yet Effective Consistency Training for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>仅加双向KL一致性正则，在四数据集采样排序上超CL4SRec且线上CTR提升2.717%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双层级三元图&quot;,&quot;时段分解与时间门控&quot;,&quot;外卖推荐&quot;]"><strong><a href="sequential_rec/modeling_dual_period_varying_preferences_for_takeaway_recommendation/">Modeling Dual Period-Varying Preferences for Takeaway Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将外卖用户-商家-菜品三元图按时段硬切分为双层级周期子图，配合用户感知与时间感知软门控融合，采样排序下 Hit@10 相对提升 3.6–7.3%，线上 GMV +0.70%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;无 ID 文本表征&quot;,&quot;两阶段微调&quot;,&quot;跨域零样本迁移&quot;]"><strong><a href="sequential_rec/text_is_all_you_need_learning_language_representations_for_sequential_recom/">Text Is All You Need: Learning Language Representations for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Recformer 弃用物品 ID，将商品展平为键值属性“句子”，用 Longformer 联合建模语言理解与序列转移；Amazon 六域 leave-one-out 全库排序 NDCG@10 平均提升 15.83%，零样本提升 39.78%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;实时行为序列建模&quot;,&quot;实时-批式混合排序&quot;]"><strong><a href="sequential_rec/transact_transformer_based_realtime_user_action_model_for_recommendation_at/">TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Pinterest 精排用 Transformer 编码最近 100 条实时行为并与 PinnerFormer 批式长期向量混合，离线 HIT@3/repin +9.4%、线上 repin +11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;生成式推荐&quot;,&quot;重排/端到端生成式&quot;,&quot;GFlowNet&quot;,&quot;列表级多样性探索&quot;]"><strong><a href="generative_rec/generative_flow_network_for_listwise_recommendation/">Generative Flow Network for Listwise Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GFN4Rec 用生成流网络做列表级推荐，以对数尺度奖励匹配损失保证生成概率与列表奖励成正比，在在线模拟器中兼顾高质量与多样性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端对话推荐&quot;,&quot;反事实数据增强&quot;,&quot;对话流语言模型&quot;,&quot;课程对抗训练&quot;]"><strong><a href="agent_conversational_rec/improving_conversational_recommendation_systems_via_counterfactual_data_sim/">Improving Conversational Recommendation Systems via Counterfactual Data Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出反事实对话模拟CFCRS，用课程对抗编辑用户表征并生成对话，缓解CRS数据稀缺，在ReDial/INSPIRED上稳定提升多种骨干。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;长尾偏差解耦&quot;,&quot;混合专家MoE&quot;,&quot;双侧分支网络&quot;]"><strong><a href="cold_start_rec/empowering_long_tail_item_recommendation_through_cross_decoupling_network_c/">Empowering Long-tail Item Recommendation through Cross Decoupling Network (CDN)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用物品侧记忆/泛化 MoE 与用户侧双分支采样解耦长尾双重偏差，γ-adapter 软化尾部迁移，HR@50 整体+8.4%、尾部+11.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多漏斗提名&quot;,&quot;实时序列建模&quot;]"><strong><a href="cold_start_rec/fresh_content_needs_more_attention_multi_funnel_fresh_content_recommendatio/">Fresh Content Needs More Attention: Multi-funnel Fresh Content Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>工业级专属新鲜内容推荐栈：泛化双塔+近实时序列模型多漏斗提名、Bandit 预打分与用户-语料共同分流 A/B，DUIC@1K 提升 0.65%、曝光覆盖提升 4%~7.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;负迁移抑制&quot;,&quot;样本级信息流加权&quot;,&quot;强化学习选择器&quot;]"><strong><a href="cross_domain_multi_scenario/a_collaborative_transfer_learning_framework_for_cross_domain_recommendation/">A Collaborative Transfer Learning Framework for Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用对称孪生双塔在线评估源域增益、以 REINFORCE 训练选择器对源样本逐条加权，配合域特异对比正则抑制负迁移，美团线上 CTR +4.37%、GMV +5.43%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;自监督预训练&quot;,&quot;特征损坏与恢复&quot;,&quot;噪声对比估计&quot;]"><strong><a href="ctr_feature_interaction/map_a_model_agnostic_pretraining_framework_for_click_through_rate_predictio/">MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向多域类别数据提出特征损坏-恢复预训练框架，派生 MFP 与 RFD 两种自监督任务，在 Avazu/Criteo 上为 DCNv2 等基座带来最高 1.22% AUC 提升，且预训练提速约 18×</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;序数回归树分解&quot;,&quot;不确定性方差约束&quot;,&quot;后门调整去偏&quot;]"><strong><a href="ctr_feature_interaction/tree_based_progressive_regression_model_for_watch_time_prediction_in_short_/">Tree based Progressive Regression Model for Watch-Time Prediction in Short-video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将观看时长序数分位离散化并用条件依赖二叉树逐层二分类，输出多项式分布求期望与方差，后门调整无缝嵌入，Kuaishou 线上时长 +0.25%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;标签纠正&quot;,&quot;反事实标注&quot;]"><strong><a href="ctr_feature_interaction/unbiased_delayed_feedback_label_correction_for_conversion_rate_prediction/">Unbiased Delayed Feedback Label Correction for Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用辅助 LC 模型对未转化样本做无偏标签纠正，配合反事实标注与嵌入交替训练，Criteo 上填补 Vanilla→Oracle 差距的 77.6% AUC。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Task-to-Task Fusion&quot;,&quot;门控混合专家 (MoE)&quot;]"><strong><a href="multi_task_reranking/adatt_adaptive_task_to_task_fusion_network_for_multitask_learning_in_recomm/">AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用任务对级门控融合+原生专家残差构建多层 AdaTT，在工业短视频 70B 样本与 UCI Census 上稳定超越 PLE/MMoE。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;训练稳定性&quot;,&quot;自适应梯度裁剪&quot;,&quot;Adagrad 优化器&quot;,&quot;多任务共享底层&quot;]"><strong><a href="multi_task_reranking/improving_training_stability_for_multitask_ranking_models_in_recommender_sy/">Improving Training Stability for Multitask Ranking Models in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对多任务精排 loss 发散，提出基于 L∞ 更新范数与双阈值自适应裁剪的 Clippy，Large+DCN 上 AUC 72.37 且可吃 2× 学习率</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;在线原始-对偶&quot;,&quot;曝光约束&quot;,&quot;跨通道建模&quot;]"><strong><a href="multi_task_reranking/multi_channel_integrated_recommendation_with_exposure_constraints/">Multi-channel Integrated Recommendation with Exposure Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多通道信息流推荐建模为带曝光上下界的在线线性规划，用对偶曝光分配与 PCR/CAR 上下文重排联合求解，淘宝 A/B 点击提升 3.00%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;延迟奖励贝叶斯滤波&quot;,&quot;长期留存探索&quot;]"><strong><a href="rl_bandit_rec/impatient_bandits_optimizing_recommendations_for_the_long_term_without_dela/">Impatient Bandits: Optimizing Recommendations for the Long-Term Without Delay</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用贝叶斯滤波+元学习将逐日揭示的播客活跃轨迹推断为60天粘性，实现无需等待奖励的Thompson采样内容冷启动探索，显著优于延迟反馈与短期代理基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;近端因果推断&quot;,&quot;潜在混淆因子识别&quot;]"><strong><a href="debias_fairness_industrial/debiasing_recommendation_by_learning_identifiable_latent_confounders/">Debiasing Recommendation by Learning Identifiable Latent Confounders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用代理变量做近端因果推断，学习可识别的潜在混淆因子，为反事实反馈提供可识别性保证；三数据集全库排序 NDCG@5 最高提升 3.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评测与基准&quot;,&quot;切片指标 (Slice-based Metrics)&quot;,&quot;行为测试 (Behavioral Tests)&quot;,&quot;双向公平性&quot;]"><strong><a href="debias_fairness_industrial/evalrs_2023_well_rounded_recommender_systems_for_real_world_deployments/">EvalRS 2023. Well-Rounded Recommender Systems For Real-World Deployments</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出以切片指标、行为测试与双向公平性为核心的多维推荐评测范式，并发布 WASABI 增强音乐数据集与开源评测工具。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;逆倾向得分 (IPS)&quot;,&quot;离线策略评估 (OPE)&quot;]"><strong><a href="debias_fairness_industrial/off_policy_evaluation_of_ranking_policies_under_diverse_user_behavior/">Off-Policy Evaluation of Ranking Policies under Diverse User Behavior</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 AIPS：以动作-奖励交互矩阵刻画上下文相关的异构用户行为，实现任意行为分布下无偏且方差最优的排序策略离线评估，并用树模型直接优化行为模型以最小化 MSE。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;路径特定反事实公平&quot;,&quot;潜中介分析&quot;,&quot;弱监督变分推断&quot;]"><strong><a href="debias_fairness_industrial/path_specific_counterfactual_fairness_for_recommender_systems/">Path-Specific Counterfactual Fairness for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双潜中介分解敏感特征的公平/不公平因果路径，以最小改动原则构造零路径特定偏差的公平世界，PSF-VAE 弱监督变分推断在保持多样性的同时显著降低不公平命中率。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/评分预测&quot;,&quot;Shilling Attack&quot;,&quot;Fake Review Generation&quot;,&quot;Reinforcement Learning&quot;,&quot;Adversarial Training&quot;]"><strong><a href="debias_fairness_industrial/shilling_black_box_review_based_recommender_systems_through_fake_review_gen/">Shilling Black-box Review-based Recommender Systems through Fake Review Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首个生成式假评论攻击 RBRS，RL+三类奖励与方面生成，黑盒显著提升预测漂移并验证对抗训练防御。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;投毒攻击&quot;,&quot;新闻推荐&quot;]"><strong><a href="debias_fairness_industrial/ua_fedrec_untargeted_attack_on_federated_news_recommendation/">UA-FedRec: Untargeted Attack on Federated News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出联邦新闻推荐无目标投毒攻击，联合新闻相似性、用户模型与样本量三重扰动，1% 恶意客户端即可显著降效并绕过防御。</p>
</dd>
</dl>
</section>
</div>
