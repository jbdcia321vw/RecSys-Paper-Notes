---
hide:
  - toc
---

# 🏛️ KDD 2022 推荐系统论文解读

27 篇 KDD 2022 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 超图神经网络 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图神经网络" aria-pressed="false">超图神经网络<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;洛伦兹模型&quot;,&quot;间隔排序损失&quot;,&quot;信息负采样&quot;]"><strong><a href="collaborative_filtering/hicf_hyperbolic_informative_collaborative_filtering/">HICF: Hyperbolic Informative Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>拆解双曲与欧氏 CF 在头/尾物品上的行为差异，用几何感知自适应 margin 与双曲信息负采样改造间隔排序损失，全库指标最高提升 12.92%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;二值量化&quot;,&quot;图协同过滤&quot;]"><strong><a href="collaborative_filtering/learning_binarized_graph_representations_with_multi_faceted_quantization_re/">Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>BiGeaR 以逐层二值化、Top-R 推理蒸馏和 Dirac 梯度估计，在5个基准上接近全精度 LightGCN 且约8倍加速。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;对齐-均匀性&quot;,&quot;免负采样损失&quot;]"><strong><a href="collaborative_filtering/towards_representation_alignment_and_uniformity_in_collaborative_filtering/">Towards Representation Alignment and Uniformity in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>从超球面均匀性与对齐性视角重构 CF 损失，提出无需负采样的 DirectAU，简单 MF 编码器在三个数据集全库排序下超 SOTA 最高 14.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图 Transformer&quot;,&quot;自监督去噪&quot;]"><strong><a href="graph_rec/self_supervised_hypergraph_transformer_for_recommender_systems/">Self-Supervised Hypergraph Transformer for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 K 个可学习隐超边 + 线性注意力 Transformer 建模全局协同关系，并以元网络标注边可靠度做局部-全局自增广，全库排序 Recall@20 最高提升 44%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;捆绑包推荐&quot;,&quot;LightGCN&quot;]"><strong><a href="self_supervised_denoising/crosscbr_cross_view_contrastive_learning_for_bundle_recommendation/">CrossCBR: Cross-view Contrastive Learning for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>KDD 2022][自监督对比学习与交互去噪] 以 LightGCN 分别学习捆绑包视图与物品视图，用跨视图 InfoNCE 对齐双视图并增强自判别性，三个数据集 Full-Ranking 下 NDCG@20 最高相对提升 29.6%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;超图神经网络&quot;,&quot;低秩自注意力&quot;]"><strong><a href="sequential_rec/multi_behavior_hypergraph_enhanced_transformer_for_sequential_recommendatio/">Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>多尺度低秩 Transformer 捕获行为感知短期转移，超图网络建模跨行为长程依赖，Leave-One-Out+1:100 采样排序下 HR@5 最高提升 48.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双塔 Transformer&quot;,&quot;因果掩码序列建模&quot;]"><strong><a href="sequential_rec/nxtpost_user_to_post_recommendations_in_facebook_groups/">NxtPost: User to Post Recommendations in Facebook Groups</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用预训练帖子向量替代词表嵌入的双塔 Transformer 因果序列召回，短/长期双损失在 Facebook Groups 上线，离线 Hits@1 绝对提升 49%、线上 +0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;个性化注意力&quot;,&quot;购物篮推荐&quot;]"><strong><a href="sequential_rec/parsrec_explainable_personalized_attention_fused_recurrent_sequential_recom/">PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用显式用户嵌入作注意力 Query 融合 RNN，建模当前购物篮部分行为；Nielsen 采样排序 NDCG@5 较 BERT4Rec 提升 27.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;通用序列表征学习&quot;,&quot;跨域迁移预训练&quot;]"><strong><a href="sequential_rec/towards_universal_sequence_representation_learning_for_recommender_systems/">Towards Universal Sequence Representation Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>UniSRec 用商品文本 + 参数化白化 + MoE 适配器学习 ID-agnostic 的通用物品/序列表征，多域对比预训练后仅微调少量参数即可跨域跨平台迁移，Pantry Recall@10 提升 38.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;预训练语言模型加速&quot;,&quot;训练框架优化&quot;]"><strong><a href="llm_rec/training_large_scale_news_recommenders_with_pretrained_language_models_in_t/">Training Large-Scale News Recommenders with Pretrained Language Models in the Loop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SpeedyFeed 用集中式编码、缓存复用、BusLM 与自回归用户建模，把 PLM 新闻推荐训练加速 128.7×，AUC 73.74 全面超越 NRMS。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;提示学习 (Prompt Learning)&quot;,&quot;知识图谱语义融合&quot;,&quot;回复模板一致性&quot;]"><strong><a href="agent_conversational_rec/towards_unified_conversational_recommender_systems_via_knowledge_enhanced_p/">Towards Unified Conversational Recommender Systems via Knowledge-Enhanced Prompt Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>冻结 DialoGPT，用知识增强软提示统一推荐与回复生成，并以回复模板回流推荐，ReDial R@10 相对提升 26.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态表征学习&quot;,&quot;多任务学习&quot;]"><strong><a href="multimodal_rec/itemsage_learning_product_embeddings_for_shopping_recommendations_at_pinter/">ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用单层Transformer聚合PinSAGE图像与哈希文本特征，以logQ校正分离式负采样与多任务学习产出单一商品嵌入，A/B测试GMV最高+6.85%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/候选生成&quot;,&quot;对比学习&quot;,&quot;异质图偏好网络&quot;,&quot;冷启动迁移&quot;]"><strong><a href="cross_domain_multi_scenario/contrastive_cross_domain_recommendation_in_matching/">Contrastive Cross-domain Recommendation in Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向召回阶段的跨域推荐：构建六类异质节点偏好图+邻居相似度损失，并用域内子图对比与用户/类目/邻居三种域间对比实现知识迁移，微信看一看线上 CTR +14.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排（出价预估）&quot;,&quot;删失数据建模&quot;,&quot;邻域似然损失&quot;,&quot;特征交叉&quot;]"><strong><a href="ctr_feature_interaction/arbitrary_distribution_modeling_with_censorship_in_real_time_bidding_advert/">Arbitrary Distribution Modeling with Censorship in Real-Time Bidding Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出邻域似然损失 NLL + ADM 框架，在无分布假设下建模 RTB 中标价删失分布，MAE 较 DLF 降 29%，推理仅 3ms。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;超图神经网络&quot;,&quot;互信息最大化&quot;,&quot;任意阶特征交互&quot;]"><strong><a href="ctr_feature_interaction/detecting_arbitrary_order_beneficial_feature_interactions_for_recommender_s/">Detecting Arbitrary Order Beneficial Feature Interactions for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HIRS 用超图超边预测直接生成任意阶有益特征交互，以 s-Infomax/Infomin 三性质约束 + L0 稀疏正则引导生成，ML-1M R@10 相对最优基线提升 5.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;情感计算&quot;,&quot;双塔多标签&quot;,&quot;弱监督标注&quot;]"><strong><a href="multi_task_reranking/affective_signals_in_a_social_media_recommender_system/">Affective Signals in a Social Media Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建14类情感taxonomy与CARE弱标注管线，用双塔多标签模型产出32维情感embedding注入Facebook排序打分模块，离线归一化AUC损失降&gt;8%，20M用户在线A/B违规内容曝光降&gt;0.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;探索-利用权衡 (E&amp;E)&quot;,&quot;对抗梯度扰动&quot;,&quot;反馈回路去偏&quot;]"><strong><a href="rl_bandit_rec/adversarial_gradient_driven_exploration_for_deep_click_through_rate_predict/">Adversarial Gradient Driven Exploration for Deep Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>把探索价值从“不确定性估计”重构为“对抗扰动引发的模型更新收益”，并配合动态门控在 Yahoo! R6B 与阿里展示广告线上 A/B 取得 CTR +6.4% 的收益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端侧决策（请求插入）&quot;,&quot;因果 uplift&quot;,&quot;资源约束动态规划&quot;]"><strong><a href="rl_bandit_rec/intelligent_request_strategy_design_in_recommender_system/">Intelligent Request Strategy Design in Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出端侧自适应请求插入框架 AdaRequest，用 CUBE 建模实时意图、CREST 估计因果 uplift、DRP 在资源约束下动态规划，离线 Qini AUUC 显著优于基线，淘宝在线 GMV 提升超过 3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Batch RL&quot;,&quot;Multi-Task Fusion&quot;]"><strong><a href="rl_bandit_rec/multi_task_fusion_via_reinforcement_learning_for_long_term_user_satisfactio/">Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将多任务融合建模为会话内 MDP，用 Batch RL 学习个性化融合权重，并以保守离线估计与线上 A/B 验证长期满意度。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;工业架构&quot;,&quot;训练系统/分布式训练&quot;,&quot;嵌入表分片&quot;,&quot;强化学习&quot;,&quot;成本建模&quot;]"><strong><a href="debias_fairness_industrial/autoshard_automated_embedding_table_sharding_for_recommender_systems/">AutoShard: Automated Embedding Table Sharding for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用神经成本模型与深度强化学习自动分片嵌入表，在 MetaSyn/MetaProd 上平衡跨 GPU 负载并实现最高约 1.7× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;冷启动用户建模&quot;,&quot;元学习&quot;,&quot;对抗公平&quot;,&quot;多任务对抗&quot;]"><strong><a href="debias_fairness_industrial/comprehensive_fair_meta_learned_recommender_system/">Comprehensive Fair Meta-learned Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首次将个体/反事实/组公平统一为多任务对抗并嵌入元学习冷启动，在三个真实数据集上显著降AUC/CF/GF且总体不损MAE。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;隐私攻击评测（黑盒 top-N 输出）&quot;,&quot;成员推断攻击&quot;,&quot;解耦表征（Gaussian/vMF）&quot;,&quot;样本重加权与交替训练&quot;]"><strong><a href="debias_fairness_industrial/debiasing_learning_for_membership_inference_attacks_against_recommender_sys/">Debiasing Learning for Membership Inference Attacks Against Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Gaussian+vMF 双先验 VAE 解耦差异向量、以 truth-level score 交替重加权，同时消除影子/目标推荐器落差与隐状态估计偏差，黑盒 AUC 由 &lt;0.7 升至 0.98+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序 (Next-Item 全库打分)&quot;,&quot;曝光偏差&quot;,&quot;逆倾向打分 (IPS/ITPS)&quot;,&quot;BERT4Rec / Cloze Task&quot;]"><strong><a href="debias_fairness_industrial/debiasing_the_cloze_task_in_sequential_recommendation_with_bidirectional_tr/">Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>证明 IPS 无法去偏序列推荐的 Cloze 任务，提出引入 Choice 变量与时变曝光倾向的 ITPS 无偏估计器，在半合成与真实数据上稳定优于 BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长混杂偏差&quot;,&quot;后门调整&quot;,&quot;观看时长分位数回归&quot;]"><strong><a href="debias_fairness_industrial/deconfounding_duration_bias_in_watch_time_prediction_for_video_recommendati/">Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用因果图指出视频时长是同时影响曝光与观看时长的混杂因子，提出分时长组的分位数回归 D2Q/Res-D2Q，在快手 13 亿样本与线上 A/B 中显著提升观看时长。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端云协同调度（会话内机制选择）&quot;,&quot;Uplift 建模&quot;,&quot;反事实样本构造&quot;,&quot;标签平滑预算约束&quot;]"><strong><a href="debias_fairness_industrial/device_cloud_collaborative_recommendation_via_meta_controller/">Device-Cloud Collaborative Recommendation via Meta Controller</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用因果 uplift 构造反事实样本，训练端侧 Meta Controller 在云会话推荐/端侧重排/云端刷新三种机制间动态调度，兼顾兴趣突变与算力预算。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练（联邦训练阶段攻击）&quot;,&quot;联邦推荐安全&quot;,&quot;硬负采样投毒&quot;]"><strong><a href="debias_fairness_industrial/fedattack_effective_and_covert_poisoning_attack_on_federated_recommendation/">FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用全局最难正负样本污染联邦推荐的本地训练数据，1% 恶意客户端即可使 HR@5 降 8%，并绕过鲁棒聚合与恶意梯度检测。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;知识图谱增强攻击&quot;,&quot;分层强化学习策略&quot;,&quot;黑盒投毒/逃逸攻击&quot;]"><strong><a href="debias_fairness_industrial/knowledge_enhanced_black_box_attacks_for_recommendations/">Knowledge-enhanced Black-box Attacks for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用物品知识图谱约束动作空间与状态表征，配合分层策略网络与 PPO 在纯黑盒下生成高质量虚假用户画像，ML-1M 上 HR@20 相对 PoisonRec 提升 10.2%。</p>
</dd>
</dl>
</section>
</div>
