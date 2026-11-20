---
hide:
  - toc
---

# 📖 ACM TOIS 2023 推荐系统论文解读

29 篇 ACM TOIS 2023 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×26 · TOIS 2023 ×2 · 去偏公平与工业架构 ×2 · 因果推断去偏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断去偏" aria-pressed="false">因果推断去偏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;高斯偏好分布&quot;,&quot;协方差偏好关联&quot;,&quot;可解释属性画像&quot;]"><strong><a href="collaborative_filtering/dual_preference_distribution_learning_for_item_recommendation/">Dual Preference Distribution Learning for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布对用户\&quot;物品级一般偏好+属性级细粒度偏好\&quot;双分布建模，协方差刻画偏好关联并顺带产出属性画像解释；Leave-One-Out + 1:100 采样排序下 NDCG@10 相对提升 1.7%–13.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;MIPS 最大内积检索&quot;,&quot;近邻图路由&quot;,&quot;强化学习与奖励塑形&quot;]"><strong><a href="collaborative_filtering/reinforcement_routing_on_proximity_graph_for_efficient_recommendation/">Reinforcement Routing on Proximity Graph for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Reinforce+自评基线训练 GCN 智能体在近邻图上路由，并以势函数奖励塑形注入少量 ground truth；固定 IPC 预算下全库召回全面超越 ip-NSW/IPDG/Mobius。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图预训练&quot;,&quot;局部子图元学习&quot;,&quot;FiLM 图级适配&quot;]"><strong><a href="graph_rec/an_adaptive_graph_pre_training_framework_for_localized_collaborative_filter/">An Adaptive Graph Pre-training Framework for Localized Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用局部子图替代用户/物品嵌入做可迁移的元 GNN，并由图属性驱动 adaptor 生成定制 GCN，稀疏目标图上 HR@5 最高提升 81.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;异质信息网络&quot;,&quot;课程式预训练&quot;,&quot;子图对比学习&quot;]"><strong><a href="graph_rec/curriculum_pre_training_heterogeneous_subgraph_transformer_for_top_n_recomm/">Curriculum Pre-Training Heterogeneous Subgraph Transformer for Top-N Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>交互特定异质子图 + 复合槽位嵌入 Transformer + 由易到难课程式预训练，Leave-One-Out/1000 负采样下 HR@20 较最强基线提升 9.9%–25.0%</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/可解释排序&quot;,&quot;时间感知知识图谱&quot;,&quot;强化学习路径推理&quot;]"><strong><a href="graph_rec/time_aware_path_reasoning_on_knowledge_graph_for_recommendation/">Time-aware Path Reasoning on Knowledge Graph for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>时间戳聚类构建TCKG，个性化时间奖励引导RL路径推理，Amazon@10最高较ADAC提升14.71% NDCG。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回与推荐评估&quot;,&quot;下一购物篮推荐&quot;,&quot;重复-探索分解&quot;,&quot;可复现评估&quot;]"><strong><a href="sequential_rec/a_next_basket_recommendation_reality_check/">A Next Basket Recommendation Reality Check</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>复现三类NBR方法，提出重复/探索指标与GP-TopFreq基线，三数据集全物品Top-K显示深度模型无一致优势。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理/模型压缩&quot;,&quot;组合编码&quot;,&quot;知识蒸馏&quot;]"><strong><a href="sequential_rec/efficient_on_device_session_based_recommendation/">Efficient On-Device Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以组合编码压缩物品嵌入并用双向自监督蒸馏保容，30×压缩下实现6–8×端侧推理加速且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;偏好编辑自监督&quot;,&quot;多偏好表征&quot;]"><strong><a href="sequential_rec/improving_transformer_based_sequential_recommenders_through_preference_edit/">Improving Transformer-based Sequential Recommenders through Preference Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 K 个特殊 token 表征多偏好，并以“偏好分离-重组”自监督构造跨序列信号，Beauty/ML-100k 上 Recall@20 提升显著</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回+排序(全链路综述)&quot;,&quot;新闻文本表征&quot;,&quot;点击噪声与责任化推荐&quot;]"><strong><a href="sequential_rec/personalized_news_recommendation_methods_and_challenges/">Personalized News Recommendation: Methods and Challenges</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以核心问题重构新闻推荐综述框架，梳理新闻/用户建模、排序、训练与评测技术，指出冷启动、点击噪声与离线-在线鸿沟三大挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时间感知图卷积&quot;,&quot;高阶连通性&quot;,&quot;自注意力聚合&quot;]"><strong><a href="sequential_rec/position_enhanced_and_time_aware_graph_convolutional_network_for_sequential/">Position-enhanced and Time-aware Graph Convolutional Network for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在用户-物品二部图上定义位置增强/时间感知图卷积与自注意力聚合器，同步建模用户与物品的时序动态并堆叠多层捕获高阶连通性，Movielens/Amazon R@10 较 DGCF 提升约 9-11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;多任务统一 Seq2Seq&quot;,&quot;Prompt-based Learning&quot;,&quot;多目标对话推荐&quot;]"><strong><a href="agent_conversational_rec/a_unified_multi_task_learning_framework_for_multi_goal_conversational_recom/">A Unified Multi-task Learning Framework for Multi-goal Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>UniMIND 用 Seq2Seq + Prompt 将 MG-CRS 的目标规划、主题预测、物品推荐与回复生成四任务统一，两基准全任务 SOTA，主题预测 Hit@1 提升 66%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式端到端（解释理由生成）&quot;,&quot;Prompt Learning&quot;,&quot;连续提示&quot;,&quot;可解释推荐&quot;]"><strong><a href="agent_conversational_rec/personalized_prompt_learning_for_explainable_recommendation/">Personalized Prompt Learning for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户/物品 ID 视为离散或连续提示注入 GPT-2，配合顺序调优与推荐正则，在三个可解释推荐数据集上显著提升解释文本质量与特征解释性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态语义实体图&quot;,&quot;情感加权图卷积&quot;]"><strong><a href="multimodal_rec/megcf_multimodal_entity_graph_collaborative_filtering_for_personalized_reco/">MEGCF: Multimodal Entity Graph Collaborative Filtering for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用语义实体替代完整深度特征并构建情感加权对称线性GCN，缓解MFE与UIM失配；留一法+99负采样下NDCG@10最高提升8.87%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/表征预训练&quot;,&quot;自监督图预训练&quot;,&quot;图对比学习&quot;]"><strong><a href="cold_start_rec/a_multi_strategy_based_pre_training_method_for_cold_start_recommendation/">A Multi-Strategy based Pre-Training Method for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MPT 在 PT-GNN 上并联 Transformer 路径编码器与对比学习借口任务，元学习模拟冷启动，外延 Recall@20 相对提升 12%–20%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;乘积专家(PoE)&quot;,&quot;归纳式图自编码器&quot;]"><strong><a href="cold_start_rec/multi_auxiliary_augmented_collaborative_variational_auto_encoder_for_tag_re/">Multi-Auxiliary Augmented Collaborative Variational Auto-encoder for Tag Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多项式VAE、内容VAE与归纳式VGAE经PoE融合及MSE互约束耦合协同与多源辅助信息，在ml-20m/citeulike全标签库排序下Recall@20提升55%–99%</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排/CTR预估&quot;,&quot;元学习&quot;,&quot;残差偏好&quot;]"><strong><a href="cold_start_rec/resus_warm_up_cold_users_via_meta_learning_residual_user_preferences_in_ctr/">RESUS: Warm-Up Cold Users via Meta-Learning Residual User Preferences in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将冷用户CTR预测解耦为全局基偏好与用户残差偏好，用共享预测器+NN/岭回归基学习器，在三个公开集上显著提升冷用户AUC并降低推理开销。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;综述&quot;,&quot;双层分类法&quot;,&quot;重叠用户对齐&quot;,&quot;知识迁移&quot;]"><strong><a href="cross_domain_multi_scenario/a_survey_on_cross_domain_recommendation_taxonomies_methods_and_future_direc/">A Survey on Cross-domain Recommendation: Taxonomies, Methods, and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出两级分类法（9 种用户/物品重叠场景 × 4 类推荐任务），按场景系统梳理 CDR 方法族、数据集与研究空白。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;异构图表征解耦&quot;,&quot;多目标跨域迁移&quot;]"><strong><a href="cross_domain_multi_scenario/heterogeneous_graph_based_framework_with_disentangled_representations_learn/">Heterogeneous Graph-based Framework with Disentangled Representations Learning for Multi-target Cross Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以单一异构全局图建模多域用户/物品，用域特异与域共享双通道图卷积解耦表征，多域同时提升 HR@10 最高约 10%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端侧个性化排序&quot;,&quot;去中心化协同学习&quot;,&quot;自监督POI表征&quot;,&quot;差分隐私&quot;]"><strong><a href="poi_spatiotemporal_rec/decentralized_collaborative_learning_framework_for_next_poi_recommendation/">Decentralized Collaborative Learning Framework for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>端侧去中心化 DCLR：地理/类别自监督预训练 + 双邻居 MIM 融合，缓解稀疏并逼近集中式 STAN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;AutoML&quot;,&quot;搜索空间&quot;,&quot;深度推荐系统&quot;]"><strong><a href="ctr_feature_interaction/automl_for_deep_recommender_systems_a_survey/">AutoML for Deep Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统梳理深度推荐AutoML的五类搜索空间与策略，给出统一分类、复杂度对比与工业落地启示。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估协议&quot;,&quot;全局时间线&quot;,&quot;数据泄漏&quot;,&quot;非采样排序&quot;]"><strong><a href="debias_fairness_industrial/a_critical_study_on_data_leakage_in_recommender_system_offline_evaluation/">A Critical Study on Data Leakage in Recommender System Offline Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TOIS 2023｜去偏公平/工业评测｜系统揭示 leave-last-one-out 忽略全局时间线造成未来交互泄漏，量化四模型四数据集上精度与排序失稳，并提出 timeline 增量评测方案。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全链路(数据/排序/重排)&quot;,&quot;双边公平&quot;,&quot;摊销公平&quot;,&quot;校准公平&quot;]"><strong><a href="debias_fairness_industrial/a_survey_on_the_fairness_of_recommender_systems/">A Survey on the Fairness of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首篇系统综述：以“过程/结果×群体/个体×七概念”定义体系、三视角分类与数据-排序-重排方法学，统一 60+ 篇公平推荐文献并指明双边联合公平等空白。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果后门调整&quot;,&quot;混合专家 MoE&quot;]"><strong><a href="debias_fairness_industrial/addressing_confounding_feature_issue_for_causal_recommendation/">Addressing Confounding Feature Issue for Causal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用后门调整 + MoE 专家架构重估 P(Y|U,do(X))，消除视频时长等混杂特征对交互的虚假直接影响，在 Kwai/Wechat 上相对最优基线平均提升 26.9%/5.0%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;倾向性得分 IPS&quot;,&quot;反馈回路&quot;]"><strong><a href="debias_fairness_industrial/bias_and_debias_in_recommender_system_a_survey_and_future_directions/">Bias and Debias in Recommender System: A Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以反馈回路为骨架统一 7 类推荐偏差的因果定义，给出 IPS/插补/因果图等去偏方法分类学与开放挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/模型训练&quot;,&quot;随机数据集&quot;,&quot;上界最小化&quot;,&quot;系统诱导偏差&quot;]"><strong><a href="debias_fairness_industrial/bounding_system_induced_biases_in_recommender_systems_with_a_randomized_dat/">Bounding System-Induced Biases in Recommender Systems with A Randomized Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TOIS 2023 去偏推荐：以随机策略完整反馈定义理想损失，推导三角不等式与可分性泛化上界，提出 DUB 直接优化可解上界项；在 Yahoo! R3 与 Product 上 AUC 稳定超越强基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;混淆变量平衡&quot;,&quot;对抗学习&quot;]"><strong><a href="debias_fairness_industrial/debiased_recommendation_with_user_feature_balancing/">Debiased Recommendation with User Feature Balancing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户特征投影+IPM分布平衡替代IPS实现因果去偏，配合裁剪/采样/对抗三种平衡项策略与潜混淆建模，在6个数据集上NDCG@10最高提升6.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;学习哈希&quot;,&quot;离散优化&quot;]"><strong><a href="debias_fairness_industrial/lightfr_lightweight_federated_recommendation_with_privacy_preserving_matrix/">LightFR: Lightweight Federated Recommendation with Privacy-preserving Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用学习哈希把联邦 MF 的用户/物品表征压成二值码，配联邦离散坐标下降与梯度聚合，精度追平联邦 SOTA，存储与推理开销降约一个量级。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信隐私&quot;,&quot;召回&quot;,&quot;隐私推断攻击&quot;,&quot;曝光数据保护&quot;]"><strong><a href="debias_fairness_industrial/on_the_user_behavior_leakage_from_recommender_system_exposure/">On the User Behavior Leakage from Recommender System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示系统曝光序列可反推用户历史点击的隐私漏洞，用编码器-解码器黑盒攻击在 Zhihu/MIND 上实现最高 87.72% Recall@10，并提出两阶段曝光替换保护机制。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;工业架构与评测基准&quot;,&quot;隐私计算&quot;,&quot;数据披露机制&quot;,&quot;多智能体模拟&quot;]"><strong><a href="debias_fairness_industrial/studying_the_impact_of_data_disclosure_mechanism_in_recommender_systems_via/">Studying the Impact of Data Disclosure Mechanism in Recommender Systems via Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出隐私感知推荐框架，以隐私计算建模用户细粒度数据披露决策，并用多智能体 bandit 模拟证明细粒度机制优于“全有或全无”。</p>
</dd>
</dl>
</section>
</div>
