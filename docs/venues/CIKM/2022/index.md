---
hide:
  - toc
---

# 🏛️ CIKM 2022 推荐系统论文解读

48 篇 CIKM 2022 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×3 · 对比学习 ×3 · 去偏公平与可信推荐 ×2 · 曝光偏差 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;Gumbel 分布&quot;,&quot;变分编码器&quot;,&quot;评分长尾失衡&quot;]"><strong><a href="collaborative_filtering/a_gumbel_based_rating_prediction_framework_for_imbalanced_recommendation/">A Gumbel-based Rating Prediction Framework for Imbalanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 Gumbel 变分编码器把评分频率映射到非正态空间，多尺度卷积融合评论交互特征并加残差个性化，5 数据集上 MAE 最高相对降 57.82%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;三元组重要性加权&quot;,&quot;双层优化&quot;,&quot;端到端聚类&quot;]"><strong><a href="collaborative_filtering/adapting_triplet_importance_of_implicit_feedback_for_personalized_recommend/">Adapting Triplet Importance of Implicit Feedback for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以双层优化学习 (u,i,j) 三元组重要性权重，结合单/多兴趣状态设计与端到端物品聚类，Recall@20 较最优基线提升 3–21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Recall@N 直接优化&quot;,&quot;损失函数空间统一&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="collaborative_filtering/croloss_towards_a_customizable_loss_for_retrieval_models_in_recommender_sys/">CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 Recall@N 重写为可微的成对比较核函数加权形式，用幂律权重 $w_\alpha$ 定制检索规模 N 并派生 Lambda 双核梯度法；Amazon/Taobao 召回指标相对交叉熵提升 5–12%，线上 A/B 收入 +4.75%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;跨通道对比学习&quot;,&quot;多路召回融合&quot;]"><strong><a href="collaborative_filtering/mic_model_agnostic_integrated_cross_channel_recommenders/">MIC: Model-agnostic Integrated Cross-channel Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨通道对比学习把 U2I/U2U/I2I 三路召回统一进单模型，作为即插即用插件同时提升召回精度与多样性，并在腾讯新闻线上生效。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;原型对比学习&quot;]"><strong><a href="collaborative_filtering/prototypical_contrastive_learning_and_adaptive_interest_selection_for_candi/">Prototypical Contrastive Learning and Adaptive Interest Selection for Candidate Generation in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微掩码按用户活跃度自适应选择兴趣向量数目，并以聚类原型对比学习缓解负采样类冲突，1:100 采样排序下 HR@10 相对提升 5–9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表征学习&quot;,&quot;强化学习元路径选择&quot;,&quot;异质信息网络&quot;,&quot;元路径推荐&quot;]"><strong><a href="graph_rec/automatic_meta_path_discovery_for_effective_graph_based_recommendation/">Automatic Meta-Path Discovery for Effective Graph-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出RMS强化学习元路径自动选择与HRec注意力聚合推荐器；在Yelp/Douban/TCL的Leave-One-Out、1正+499负采样排序下，HR@3较最佳基线最高提升12.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;上下文漂移解耦&quot;,&quot;随机游走图嵌入&quot;,&quot;多头交互注意力&quot;]"><strong><a href="graph_rec/hysage_a_hybrid_static_and_adaptive_graph_embedding_network_for_context_dri/">HySAGE: A Hybrid Static and Adaptive Graph Embedding Network for Context-Drifting Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>解耦静态交互图嵌入与漂移上下文自适应嵌入，注意力交互融合，全库未交互排序下 HR@10 最高提升 28.4%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多层交互对比学习&quot;,&quot;知识图谱推荐&quot;,&quot;非局部图&quot;]"><strong><a href="graph_rec/improving_knowledge_aware_recommendation_with_multi_level_interactive_contr/">Improving Knowledge-aware Recommendation with Multi-level Interactive Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 KGIC：以 GNN 层为视图做图内 CF-KG 与图间局部-非局部两层交互对比，三数据集 AUC 相对提升 0.67%–3.35%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;子图级图神经网络&quot;,&quot;捆绑包推荐&quot;,&quot;迁移推荐&quot;]"><strong><a href="graph_rec/suger_a_subgraph_based_graph_convolutional_network_method_for_bundle_recomm/">SUGER: A Subgraph-based Graph Convolutional Network Method for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以用户-捆绑包为中心抽取异质 k 跳子图，用两级 RGCN 图级编码器打分，并删除中心边消除标签泄漏；Recall@20 在 Youshu/Netease 上相对 BGCN 提升 39%/90%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低秩分解&quot;,&quot;图卷积简化&quot;]"><strong><a href="graph_rec/svd_gcn_a_simplified_graph_convolution_paradigm_for_recommendation/">SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用截断 SVD 替代 GCN 邻域聚合，仅取 K 个奇异向量并重归一化，全库排序精度提升且百倍加速。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM Computing Surveys&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与表征学习&quot;,&quot;对比学习&quot;,&quot;重建学习&quot;,&quot;对抗学习&quot;]"><strong><a href="self_supervised_denoising/a_comprehensive_survey_on_self_supervised_learning_for_recommendation/">A Comprehensive Survey on Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统综述 170+ 篇 SSL4Rec 工作，提出对比/重建/对抗三范式与九场景视图中心分类，汇总评估协议并维护开源库 SSLRec。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;免数据增强图对比学习&quot;,&quot;自适应层融合&quot;]"><strong><a href="self_supervised_denoising/adaptive_fusion_self_supervised_learning_for_recommendation/">Adaptive Fusion Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>AFGCL以免数据增强的传播层自适应融合与交互锚点对比目标，提升稀疏推荐精度并降低训练成本。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;个性化兴趣持续性&quot;,&quot;时序频次分箱&quot;,&quot;稀疏历史补充&quot;]"><strong><a href="sequential_rec/beyond_learning_from_next_item_sequential_recommendation_via_personalized_i/">Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用「训练期末段消费预测」任务刻画个性化兴趣持续性(PIS)，并以内在/外在双路稀疏历史补充方案与原型度量偏好学习联合训练，11 数据集全库采样排序全面超越 10 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/contrastvae_contrastive_variational_autoencoder_for_sequential_recommendati/">ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双视角ContrastELBO耦合双分支VAE与潜空间对比增强，缓解后验坍缩，Full-Ranking下R@40最高提升19.34%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;双网络解耦&quot;,&quot;训练-推理鸿沟&quot;,&quot;双向KL知识迁移&quot;]"><strong><a href="sequential_rec/disentangling_past_future_modeling_in_sequential_recommendation_via_dual_ne/">Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双网络分离过去/未来建模，并以双向KL迁移互补知识，缓解训练-推理鸿沟；4数据集采样排序上HR@1平均提升12.03%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;连续时间动态图&quot;]"><strong><a href="sequential_rec/evolutionary_preference_learning_via_graph_nested_gru_ode_for_session_based/">Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将会话建模为连续时间动态图，以GNG-ODE与t-Alignment学习偏好连续演化，全库排序下MRR@20最高提升6.05%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;解释引导数据增强&quot;,&quot;对比学习视图选择&quot;]"><strong><a href="sequential_rec/explanation_guided_contrastive_learning_for_sequential_recommendation/">Explanation Guided Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用解释方法（Saliency/Occlusion）估计序列物品重要性，指导正负视图增强与自监督+监督对比学习；四数据集 Full-Ranking 下 NDCG@5 较 CL4SRec 提升 14%–22%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（端到端 Next-Item 预测）&quot;,&quot;隐式会话上下文&quot;,&quot;图聚类与上下文预测&quot;]"><strong><a href="sequential_rec/implicit_session_contexts_for_next_item_recommendations/">Implicit Session Contexts for Next-Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>会话-物品图 GraphSAGE 嵌入 + K-means 聚类生成隐式会话上下文，用双 Bi-LSTM 上下文预测器作先验特征，Foursquare MRR 相对最优基线 +36.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;会话内上下文&quot;,&quot;曝光偏置建模&quot;,&quot;多任务学习&quot;]"><strong><a href="sequential_rec/intra_session_context_aware_feed_recommendation_in_live_systems/">Intra-session Context-aware Feed Recommendation in Live Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将信息流会话建模为点击/滑动的马尔可夫事件链，以会话内曝光序列编码上下文，多任务联合学习并用 GPT 式贪心序列生成优化总曝光与总点击；线上人均曝光 +1.16%、转化 +1.56%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;多视图图神经网络&quot;,&quot;自监督信号&quot;]"><strong><a href="sequential_rec/multi_level_contrastive_learning_framework_for_sequential_recommendation/">Multi-level Contrastive Learning Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MCLSR 构建序列/用户-物品/用户-用户/物品-物品四视图，在兴趣层与特征层做跨视图对比学习以补足稀疏监督信号，四数据集 Full-Ranking 下 Recall@20 平均相对提升约 7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;层级自注意力&quot;,&quot;时间门控融合&quot;]"><strong><a href="sequential_rec/time_lag_aware_sequential_recommendation/">Time Lag Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用层级自注意力分离全局稳定/局部波动的用户偏好，并以时间差驱动的维度级门向量融合长短期表征，Amazon Book 上 Hit@20 相对提升 55%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮对话决策与Top-K排序&quot;,&quot;决策树随机森林&quot;,&quot;因子分解树&quot;,&quot;负反馈更新&quot;]"><strong><a href="agent_conversational_rec/rethinking_conversational_recommendations_is_decision_tree_all_you_need/">Rethinking Conversational Recommendations: Is Decision Tree All You Need?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用用户-物品交互树/随机森林替代RL，按属性提问并排序候选，SR@10最高提升41.5%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;域解耦表征&quot;,&quot;跨域对比学习&quot;]"><strong><a href="social_group_rec/disentangled_contrastive_learning_for_social_recommendation/">Disentangled Contrastive Learning for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户表征按协同域/社交域显式解耦，并用跨域对比学习迁移社交知识，Dianping/Ciao 上 NDCG 相对 SGL 提升 5.3%/6.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;图特征迁移&quot;,&quot;多模态语义链接&quot;]"><strong><a href="cold_start_rec/gift_graph_guided_feature_transfer_for_cold_start_video_click_through_rate_/">GIFT: Graph-guIded Feature Transfer for Cold-Start Video Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建物理+多模态语义异构链接，用 metapath 引导的 target attention 把热视频的 ID 表征与统计特征迁移给新视频，淘宝线上 CTR 提升 6.82%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;多域双塔检索&quot;,&quot;领域自适应&quot;]"><strong><a href="cross_domain_multi_scenario/adaptive_domain_interest_network_for_multi_domain_recommendation/">Adaptive Domain Interest Network for Multi-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>ADI 用共享/领域专属双路网络 + DSBN + SE 域注意力 + 跨域自训练伪标签，在阿里展示广告三场景召回上全场景 Recall 提升，线上 RPM +1.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;对比学习&quot;,&quot;互信息最大化&quot;]"><strong><a href="cross_domain_multi_scenario/contrastive_cross_domain_sequential_recommendation/">Contrastive Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>联合建模序列内/间物品关系，双目标学单域/跨域表征，并以对比 Infomax 互信息约束跨域迁移；修正时间穿越划分后 HR@10 相对 SOTA 提升 14%+</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;树形元结构 TMS&quot;,&quot;Tree2vec&quot;,&quot;异构图嵌入&quot;]"><strong><a href="cross_domain_multi_scenario/gres_graphical_cross_domain_recommendation_for_supply_chain_platform/">GReS: Graphical Cross-domain Recommendation for Supply Chain Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为餐饮供应链平台设计「菜品-类目-食材」三层树形元结构与 Tree2vec(GCN+BERT) 编码，配合异构图 Node2vec 与元素级注意力融合，HR@10 相对最强基线提升 7.35%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;场景自适应门控 (ML-SAT)&quot;,&quot;场景监督对比预训练&quot;]"><strong><a href="cross_domain_multi_scenario/scenario_adaptive_and_self_supervised_model_for_multi_scenario_personalized/">Scenario-Adaptive and Self-Supervised Model for Multi-Scenario Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SASS 用场景自适应门控做细粒度全局→单场景信息迁移，并以场景监督对比预训练挖掘无标签全空间样本，淘宝短视频 AWT +8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构图神经网络&quot;,&quot;双查询注意力融合&quot;,&quot;距离协方差&quot;]"><strong><a href="ctr_feature_interaction/aminergnn_heterogeneous_graph_neural_network_for_paper_click_through_rate_p/">AMinerGNN: Heterogeneous Graph Neural Network for Paper Click-through Rate Prediction with Fusion Query</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建 user-paper-keyword 异构图统一表征，用距离协方差注意力动态融合 userID 与 keyword 双查询，AMiner_KRS 上 AUC 0.899（较 MGDSPR +3.2%），线上 UV-CTR 提升 9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;图检索中心节点&quot;,&quot;多场景短期意图&quot;,&quot;个性化门控融合&quot;]"><strong><a href="ctr_feature_interaction/graph_based_long_term_and_short_term_interest_model_for_click_through_rate_/">Graph Based Long-Term And Short-Term Interest Model for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用图检索结构替代长行为截断/硬类目过滤，结合多场景 GRU 短期意图与个性化门控融合，离线 AUC 达 0.9830，线上 CTR +4.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长短期兴趣解耦&quot;,&quot;自监督对比学习&quot;]"><strong><a href="ctr_feature_interaction/hierarchically_fusing_long_and_short_term_user_interests_for_click_through_/">Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HIFN 以查询/目标/因果三路编码短期兴趣、注意力聚合长期兴趣、门控自适应融合并用自监督代理对比解耦长短期，Fliggy 线上 CTR +5.13%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;早期特征交互&quot;,&quot;对比学习正则&quot;]"><strong><a href="ctr_feature_interaction/inttower_the_next_generation_of_two_tower_model_for_pre_ranking_system/">IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Light-SE 特征选择、FE-Block 早期细粒度交互与对比交互正则，在保持双塔解耦部署的前提下逼近单塔精排效果。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Hawkes Process&quot;,&quot;Hierarchical RNN&quot;,&quot;Entire Space Multi-Task&quot;]"><strong><a href="ctr_feature_interaction/multi_scale_user_behavior_network_for_entire_space_multi_task_learning/">Multi-Scale User Behavior Network for Entire Space Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层循环网络 HEROES 用 Hawkes 过程与门控跨层通道，在 observation→click→conversion 全空间多尺度建模用户行为，CVR AUC 最高相对提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;召回/精排嵌入层&quot;,&quot;嵌入表剪枝&quot;,&quot;神经架构搜索&quot;]"><strong><a href="ctr_feature_interaction/optembed_learning_optimal_embedding_table_for_click_through_rate_prediction/">OptEmbed: Learning Optimal Embedding Table for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可学习场级阈值剪冗余嵌入 + 均匀采样超网与进化搜索定场级维度，在 Criteo/Avazu/KDD12 上省 39%–97% 参数且 AUC 最高 +0.15%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;LSH哈希采样&quot;,&quot;长序列用户兴趣建模&quot;,&quot;端到端训练&quot;]"><strong><a href="ctr_feature_interaction/sampling_is_all_you_need_on_modeling_long_term_user_behaviors_for_ctr_predi/">Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用多轮 SimHash 碰撞概率替代 target attention，直接 gather 同签名行为构造长序列用户兴趣，AUC 媲美全注意力且训练快 5–11×，美团线上升 +2.98% CTR。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长用户行为序列建模&quot;,&quot;稀疏注意力记忆网络&quot;,&quot;实时推理效率&quot;]"><strong><a href="ctr_feature_interaction/sparse_attentive_memory_network_for_click_through_rate_prediction_with_long/">Sparse Attentive Memory Network for Click-through Rate Prediction with Long Sequences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以目标物品为 Query、千长行为序列为知识库，用目标+记忆双查询稀疏注意力与 O(1) 次迭代记忆更新，实现 O(L) 复杂度长序列 CTR 建模，线上 A/B 提升 CTR 7.30%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习视觉编码&quot;,&quot;样本选择偏差去偏&quot;]"><strong><a href="ctr_feature_interaction/visual_encoding_and_debiasing_for_ctr_prediction/">Visual Encoding and Debiasing for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用自监督+点击监督两阶段对比学习微调视觉编码器，并在 CTR 塔内加对比去偏网络与门控融合，淘宝十亿级数据 AUC +6.8%、线上 CTR +4~5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;端侧推理&quot;,&quot;自适应束搜索&quot;,&quot;实时特征交叉&quot;]"><strong><a href="multi_task_reranking/real_time_short_video_recommendation_on_mobile_devices/">Real-time Short Video Recommendation on Mobile Devices</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>端侧轻量排序模型融合实时反馈交叉特征，并以自适应束搜索做上下文感知重排，线上有效播放/点赞/关注分别 +1.28%/+8.22%/+13.6%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;校准推荐&quot;,&quot;置信度加权&quot;]"><strong><a href="multi_task_reranking/towards_confidence_aware_calibrated_recommendation/">Towards Confidence-aware Calibrated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出置信度感知校准重排CCL，用用户画像大小加权MILP平衡相关性、校准与多样性，MovieLens上nDCG与覆盖率提升</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;逆倾向加权 IPS&quot;,&quot;曝光偏差&quot;,&quot;邻居聚合&quot;]"><strong><a href="debias_fairness_industrial/debiasing_neighbor_aggregation_for_graph_neural_network_in_recommender_syst/">Debiasing Neighbor Aggregation for Graph Neural Network in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>把逆倾向权重下推进 GNN 邻居聚合，用加权随机游走拉普拉斯归一化，推理期免训练重聚合即可在 Alexa 上提升 NDCG@5 14.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;逆倾向得分 IPS&quot;,&quot;序列推荐&quot;,&quot;双视角去偏&quot;]"><strong><a href="debias_fairness_industrial/dually_enhanced_propensity_score_estimation_in_sequential_recommendation/">Dually Enhanced Propensity Score Estimation in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出用户/物品双视角时序 IPS 与双 Transformer，理论无偏并显著提升序列推荐。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;回声室缓解&quot;]"><strong><a href="debias_fairness_industrial/dynamic_causal_collaborative_filtering/">Dynamic Causal Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用含环因果图与马尔可夫吸收态分析刻画推荐回声室，提出后门调整+反事实历史的 ∂CCF，内容多样性变化平均降低 7.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据基准与离线评估&quot;,&quot;Exposure Bias&quot;,&quot;Missing-at-Random&quot;,&quot;多任务反馈信号&quot;]"><strong><a href="debias_fairness_industrial/kuairand_an_unbiased_sequential_recommendation_dataset_with_randomly_expose/">KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>快手无感插入百万级随机曝光视频，构建首个带丰富特征与12类反馈的大规模无偏序列推荐数据集，支撑 IPS 全库评估。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估基准&quot;,&quot;全观测数据&quot;,&quot;曝光偏差&quot;]"><strong><a href="debias_fairness_industrial/kuairec_a_fully_observed_dataset_and_insights_for_evaluating_recommender_sy/">KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>发布快手全观测数据集KuaiRec，揭示数据密度与曝光偏差会改变CRS离线评测排序，矩阵补全仅部分缓解。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回(端到端联邦训练)&quot;,&quot;联邦学习&quot;,&quot;用户聚类&quot;,&quot;图神经网络&quot;]"><strong><a href="debias_fairness_industrial/personalized_federated_recommendation_via_joint_representation_learning_use/">Personalized Federated Recommendation via Joint Representation Learning, User Clustering, and Model Adaptation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 PerFedRec，用联邦 GNN 联合学习表征、用户聚类与三级模型自适应，实现个性化联邦推荐并降低通信开销，HR@10 较 FedAvg 平均提升 29.47%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排（候选集内排序）&quot;,&quot;对话推荐系统&quot;,&quot;流行度偏差&quot;,&quot;双策略学习&quot;]"><strong><a href="debias_fairness_industrial/quantifying_and_mitigating_popularity_bias_in_conversational_recommender_sy/">Quantifying and Mitigating Popularity Bias in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统量化对话推荐中曝光/成功率/对话效用三重流行度偏差，提出 PAL+CSM+DPL 三阶段模型无关去偏框架，在 Yelp/Lastfm 上平均降偏 42.5% 且成功率提升 12.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序评测/鲁棒性分析&quot;,&quot;Rank List Sensitivity&quot;,&quot;交互级扰动&quot;,&quot;级联效应&quot;]"><strong><a href="debias_fairness_industrial/rank_list_sensitivity_of_recommender_systems_to_interaction_perturbations/">Rank List Sensitivity of Recommender Systems to Interaction Perturbations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出排序列表敏感度 RLS 指标与 CASPER 级联扰动法，量化单条交互扰动致全库排序剧变，并揭示低准确率用户更脆弱</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;用户侧推荐 (User-side RecSys)&quot;,&quot;度量恢复 (Metric Recovery)&quot;,&quot;公平性公理化&quot;]"><strong><a href="debias_fairness_industrial/towards_principled_user_side_recommender_systems/">Towards Principled User-side Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 k-NN 度量恢复理论证明用户侧推荐可行，并提出满足一致性/可靠性/局部性三公理的 Consul 算法，通信开销降低约 8–19 倍且性能近 Oracle。</p>
</dd>
</dl>
</section>
</div>
