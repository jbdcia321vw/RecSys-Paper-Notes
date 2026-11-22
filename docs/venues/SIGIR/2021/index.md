---
hide:
  - toc
---

# 🏛️ SIGIR 2021 推荐系统论文解读

40 篇 SIGIR 2021 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×2 · 元学习 ×2 · 冷启动 ×2 · 图对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图对比学习" aria-pressed="false">图对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;无负采样自举学习&quot;,&quot;动量目标编码器&quot;,&quot;邻居随机增广&quot;]"><strong><a href="collaborative_filtering/bootstrapping_user_and_item_representations_for_one_class_collaborative_fil/">Bootstrapping User and Item Representations for One-Class Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>BUIR 用在线/目标双编码器互预测替代负采样, 配合邻居随机丢弃增广, 在极稀疏隐式反馈上 Full-Ranking P@10 相对最优基线提升 22.8%–58.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;跨批次负采样&quot;,&quot;双塔模型&quot;,&quot;内存队列&quot;]"><strong><a href="collaborative_filtering/cross_batch_negative_sampling_for_training_two_tower_recommenders/">Cross-Batch Negative Sampling for Training Two-Tower Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>利用编码器 embedding stability，以 FIFO memory bank 复用历史 batch 的 item 向量作为跨批次负样本，在几乎零额外编码开销下把负样本量从 128 扩至 2560，NDCG@50 最高提升 12.19%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;属性图匹配&quot;,&quot;内外交互解耦&quot;,&quot;图神经网络&quot;]"><strong><a href="collaborative_filtering/neural_graph_matching_based_collaborative_filtering/">Neural Graph Matching based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将属性交互解耦为内交互与交叉交互，用双属性图匹配结构分别做特征学习与偏好匹配，AUC/NDCG@10 最高提升 4.3%/16.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;混合推荐&quot;,&quot;神经表征学习&quot;,&quot;自编码器正则化&quot;]"><strong><a href="collaborative_filtering/neural_representations_in_hybrid_recommender_systems_prediction_versus_regu/">Neural Representations in Hybrid Recommender Systems: Prediction versus Regularization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>论证混合推荐中神经表征应直接用于打分而非充作正则项，删去 MF 表征与解码器后 RMSE 降 3.8% 且训练更快、内存更省。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;二跳路径聚合&quot;,&quot;多兴趣建模&quot;]"><strong><a href="collaborative_filtering/path_based_deep_network_for_candidate_item_matching_in_recommenders/">Path-based Deep Network for Candidate Item Matching in Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 TrigNet×SimNet 对用户-物品二跳路径逐条打分并聚合召回，兼顾个性化与兴趣多样性，淘宝线上 PCTR +18.04%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样分布&quot;,&quot;参数密度估计&quot;,&quot;范数裁剪&quot;]"><strong><a href="collaborative_filtering/scalable_personalised_item_ranking_through_parametric_density_estimation/">Scalable Personalised Item Ranking through Parametric Density Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以指数族参数密度估计推导点式排序风险与最优负采样分布（排序器自采样、免额外生成器），配合范数裁剪稳定训练；Full-Ranking 下 nDCG@20 较 DREGN-CF 最高 +18%，收敛快 1–2 个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;排序优化&quot;,&quot;集合排序&quot;,&quot;自适应负采样&quot;]"><strong><a href="collaborative_filtering/set2setrank_collaborative_set_to_set_ranking_for_implicit_feedback_based_re/">Set2setRank: Collaborative Set to Set Ranking for Implicit Feedback based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出集合到集合两级排序损失，结合自适应掩码负采样，在Full-Ranking下提升隐式反馈Top-N推荐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;元学习&quot;,&quot;行为依赖迁移&quot;]"><strong><a href="graph_rec/graph_meta_network_for_multi_behavior_recommendation/">Graph Meta Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元学习为每个用户动态生成行为异质变换与预测层参数，联合建模多行为图传播与跨类型依赖；Leave-One-Out + 1:99 采样排序下，Taobao HR@10 相对 NGCF_M 提升 31.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;Knowledge Graph&quot;,&quot;Co-Attention&quot;,&quot;Interactive Matching&quot;]"><strong><a href="graph_rec/personalized_news_recommendation_with_knowledge_aware_interactive_matching/">Personalized News Recommendation with Knowledge-aware Interactive Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>KIM 用知识协同编码器与语义协同编码器实现候选新闻与用户兴趣的双向交互匹配，在 MIND/Feeds 上 nDCG@5 相对提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱嵌入&quot;,&quot;度量学习&quot;,&quot;共现关系建模&quot;]"><strong><a href="graph_rec/ugrec_modeling_directed_and_undirected_relations_for_recommendation/">UGRec: Modeling Directed and Undirected Relations for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将知识图谱有向关系与物品共现无向关系统一建模：有向关系用 TransD 关系空间投影+头尾关系感知注意力，无向关系用关系专属超平面投影最小化距离，Amazon 三数据集 Leave-One-Out 全库排序下 HR@20 相对 SOTA 提升 3.7%–5.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;长尾去偏&quot;]"><strong><a href="self_supervised_denoising/self_supervised_graph_learning_for_recommendation/">Self-supervised Graph Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在 LightGCN 上叠加节点自判别对比任务，用节点/边丢弃与随机游走构造多视图，全库排序下 Recall@20 最高提升 9.13%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;假阴性去偏&quot;,&quot;拓扑相似度&quot;]"><strong><a href="self_supervised_denoising/topology_aware_debiased_self_supervised_graph_learning_for_recommendation/">Topology-aware Debiased Self-supervised Graph Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用交互共现拓扑筛选假阴性并转为辅助正样本，在三个数据集 Full-Ranking 下 Recall@20 最高相对提升约 3.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;反事实序列合成&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/causerec_counterfactual_user_sequence_synthesis_for_sequential_recommendati/">CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>通过反事实概念替换合成用户序列并做对比学习，使用户表征对噪声不敏感、对关键兴趣更信任；Amazon Books NDCG@50 相对最强基线提升 22.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/候选重排&quot;,&quot;Transformer 序列编码&quot;,&quot;评论级细粒度匹配&quot;,&quot;注意力可解释个性化&quot;]"><strong><a href="sequential_rec/learning_a_fine_grained_review_based_transformer_model_for_personalized_pro/">Learning a Fine-Grained Review-based Transformer Model for Personalized Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将查询、用户评论与物品评论拼成单序列交给 Transformer 做评论级细粒度匹配，Amazon 三品类 BM25 候选重排 MRR 最高相对提升 60%，并天然支持注意力可解释与动态个性化。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;兴趣图构建&quot;,&quot;动态图池化&quot;,&quot;AUGRU&quot;]"><strong><a href="sequential_rec/sequential_recommendation_with_graph_neural_networks/">Sequential Recommendation with Graph Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将松散物品序列用度量学习重构为兴趣图，经簇/查询双感知图卷积与动态图池化提炼激活核心兴趣，长序列场景 AUC 提升约 4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;代理选择 (Proxy Selection)&quot;,&quot;温度 Softmax 硬选择&quot;,&quot;超平面投影度量学习&quot;]"><strong><a href="sequential_rec/unsupervised_proxy_selection_for_session_based_recommender_systems/">Unsupervised Proxy Selection for Session-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用可学习的 K 个代理在无监督下为匿名会话选择&quot;用户长期兴趣&quot;，结合短期兴趣与超平面投影距离，全库排序 R@20 相对提升最高 45.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Pre-trained Language Model&quot;,&quot;Attention Pooling&quot;,&quot;多语言新闻推荐&quot;]"><strong><a href="llm_rec/empowering_news_recommendation_with_pre_trained_language_models/">Empowering News Recommendation with Pre-trained Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 PLM 替换新闻编码器并以 Attention 池化 token 隐状态、仅微调顶层两层，MIND 上 AUC 68.18→70.64，多语言 7 市场 AUC 61.07→62.83，线上点击 +8.53%/+10.68%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;在线推荐/候选臂选择&quot;,&quot;Contextual Bandit&quot;,&quot;Relative Feedback&quot;]"><strong><a href="agent_conversational_rec/comparison_based_conversational_recommender_system_with_relative_bandit_fee/">Comparison-based Conversational Recommender System with Relative Bandit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 RelativeConUCB，用成对 key-term 相对反馈与差异岭回归更新偏好；真实数据漂移场景累积遗憾较 ConUCB 降低 44.75%–49.96%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;澄清问题生成&quot;,&quot;知识图谱关系注意力&quot;,&quot;TransH 偏好翻译&quot;]"><strong><a href="agent_conversational_rec/learning_to_ask_appropriate_questions_in_conversational_recommendation/">Learning to Ask Appropriate Questions in Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>KBQG 用 KG 关系注意力引导澄清问题生成，简化动作空间后在两数据集上 SR 相对 CRM 提升 7.2%–12.9% 且显著缩短对话轮次</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;Soft Attribute Critiquing&quot;,&quot;Debiased Crowdsourcing&quot;,&quot;Weighted Rank Correlation&quot;]"><strong><a href="agent_conversational_rec/on_interpretation_and_measurement_of_soft_attributes_for_recommendation/">On Interpretation and Measurement of Soft Attributes for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将软属性批评表示为相对锚点物品的个性化偏序，用去偏众包与加权 Gamma 度量，并以无/弱/全监督三档打分器实现属性排序。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;一致性邻居采样&quot;,&quot;关系注意力&quot;]"><strong><a href="social_group_rec/consisrec_enhancing_gnn_for_social_recommendation_via_consistent_neighbor_a/">ConsisRec: Enhancing GNN for Social Recommendation via Consistent Neighbor Aggregation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出一致性查询+邻居采样+关系注意力，抑制社交图中上下文级与关系级不一致噪声，在 Ciao/Epinions 评分预测上 RMSE 相对降 2.79%/1.74%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;好友圈分组&quot;,&quot;个性化搜索&quot;]"><strong><a href="social_group_rec/group_based_personalized_search_by_integrating_search_behaviour_and_friend_/">Group based Personalized Search by Integrating Search Behaviour and Friend Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>FNPS 用“关系圈+行为圈”双通道好友分组与掩码交叉注意力构建群体画像，在微信搜索日志重排上 MAP 较原始排序 +3.22%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;图注意力网络&quot;,&quot;元学习嵌入生成&quot;]"><strong><a href="cold_start_rec/learning_graph_meta_embeddings_for_cold_start_ads_in_click_through_rate_pre/">Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用广告属性图 + GAT + MAML 式元学习为新广告 ID 生成初始嵌入，在三个真实数据集上对五种 CTR 主干模型实现冷启动与预热双阶段 AUC 提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/精排通用模块&quot;,&quot;元学习&quot;,&quot;冷 ID 嵌入热身&quot;,&quot;特征变换去噪&quot;]"><strong><a href="cold_start_rec/learning_to_warm_up_cold_item_embeddings_for_cold_start_recommendation_with/">Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元缩放/平移网络将冷物品 ID 嵌入变换到热特征空间并以交互用户均值去噪，三点数据集八基线 + 六基础模型上 AUC 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;异质图卷积&quot;]"><strong><a href="cold_start_rec/privileged_graph_distillation_for_cold_start_recommendation/">Privileged Graph Distillation for Cold Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将属性作为节点构建异质图教师模型，用二阶间接链接构造无 CF 边的学生图，通过特权图蒸馏实现新用户/新物品/双冷启动统一推荐，Yelp/XING/Amazon 全库排序 HR/NDCG 平均最高提升 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 MAML&quot;,&quot;翻译式转移模式&quot;]"><strong><a href="cold_start_rec/sequential_recommendation_for_cold_start_users_with_meta_transitional_learn/">Sequential Recommendation for Cold-start Users with Meta Transitional Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将冷启动序列推荐重构为少样本任务，用翻译式转移编码器 + MAML 双层优化实现仅 K=2~4 次交互即可快速适配，时间戳冷启动划分下 Hit@1/MRR 显著超越 SASRec/BERT4Rec/MetaCF。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;Meta Learning&quot;,&quot;MAML&quot;,&quot;EMCDR&quot;,&quot;冷启动&quot;]"><strong><a href="cross_domain_multi_scenario/transfer_meta_framework_for_cross_domain_recommendation_to_cold_start_users/">Transfer-Meta Framework for Cross-domain Recommendation to Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用全量预训练替代重叠用户重训，并以 MAML 式任务导向元网络隐式迁移源域用户表征，6 个跨域冷启动任务 AUC 最高提升 4.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;时空轨迹与 POI 本地生活推荐&quot;,&quot;召回&quot;,&quot;叙事驱动推荐&quot;,&quot;众包分级相关性评测&quot;,&quot;冷启动&quot;]"><strong><a href="poi_spatiotemporal_rec/pointrec_a_test_collection_for_narrative_driven_point_of_interest_recommend/">POINTREC: A Test Collection for Narrative-driven Point of Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 695K POI + 112 条真实自然语言叙事需求的冷启动 POI 测试集，配众包 4 级相关性池化标注与三种结构化查询基线，NDCG@5 最优 0.6784。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回/相关性粗排&quot;,&quot;行为图增强&quot;,&quot;知识蒸馏长尾&quot;]"><strong><a href="ctr_feature_interaction/adsgnn_behavior_graph_augmented_relevance_modeling_in_sponsored_search/">AdsGNN: Behavior-Graph Augmented Relevance Modeling in Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以点击二部图作互补信号，在节点/边/Token 三级融合语义与行为图，配合领域预训练与长尾蒸馏，ROC-AUC 0.881、线上 RPM +1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;位置偏置&quot;,&quot;位置感知兴趣聚合&quot;,&quot;训练推理一致性&quot;]"><strong><a href="ctr_feature_interaction/deep_position_wise_interaction_network_for_ctr_prediction/">Deep Position-wise Interaction Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>DPIN 将候选物品与展示位置解耦为 item 塔与 position 塔并行打分，训练与推理一致地用真实位置建模，并提出 PAUC 指标；线上 A/B 提升 CTR 2.25%、RPM 2.15%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;双重稳健估计&quot;,&quot;选择偏差去偏&quot;,&quot;反事实学习&quot;]"><strong><a href="ctr_feature_interaction/enhanced_doubly_robust_learning_for_debiasing_post_click_conversion_rate_es/">Enhanced Doubly Robust Learning for Debiasing Post-click Conversion Rate Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>从 DR 估计量的偏差-方差分解出发，把误差插补目标改为最小化方差并引入双学习伪标签机制，CVR 去偏在 Coat/Yahoo!R3 全排序评测上稳定超越 IPS 与 DR-JL。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自注意力特征交叉&quot;,&quot;离散选择模型&quot;,&quot;统一框架&quot;]"><strong><a href="ctr_feature_interaction/looking_at_ctr_prediction_again_is_attention_all_you_need/">Looking at CTR Prediction Again: Is Attention All You Need?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用离散选择模型重构 CTR 目标，提出 ST∘AL∘FI∘EL 统一框架与自注意力二阶算子 SAM 族，Criteo/Avazu 上 SAM2E 取最优 AUC 0.8115/0.7891。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR预估与特征交叉&quot;,&quot;训练系统/离线训练加速&quot;,&quot;超大 Embedding 表&quot;,&quot;分布式模型并行&quot;,&quot;缓存与流水线重叠&quot;]"><strong><a href="ctr_feature_interaction/scalefreectr_mixcache_based_distributed_training_system_for_ctr_models_with/">ScaleFreeCTR: MixCache-based Distributed Training System for CTR Models with Huge Embedding Table</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 VSI 批内去重、GPU HBM 混合缓存 MixCache 与三阶段流水线消除 Host↔GPU 参数搬运延迟，Criteo-TB 上训练吞吐达 PS 的 1.6×/1.3×、HugeCTR 的 5.6×/6.9×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Metric Optimization&quot;,&quot;RBP Loss&quot;]"><strong><a href="multi_task_reranking/new_insights_into_metric_optimization_for_ranking_based_recommendation/">New Insights into Metric Optimization for Ranking-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>系统比较 RR/AP/nDCG/RBP 作为 pairwise 与 listwise 优化目标，提出 RBP 损失，揭示优化指标≠评估指标更优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;元学习双层优化&quot;,&quot;风险差异统一框架&quot;,&quot;逆倾向加权&quot;]"><strong><a href="debias_fairness_industrial/autodebias_learning_to_debias_for_recommendation/">AutoDebias: Learning to Debias for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将各类数据偏置统一为风险差异，用小型无偏数据以元学习双层优化自动学习重加权/插补参数，Yahoo!R3 NDCG@5 提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;因果去偏&quot;,&quot;流行度偏差&quot;,&quot;do-calculus&quot;]"><strong><a href="debias_fairness_industrial/causal_intervention_for_leveraging_popularity_bias_in_recommendation/">Causal Intervention for Leveraging Popularity Bias in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将物品流行度识别为曝光-交互之间的混杂因子，用 do-calculus 解耦训练并在推理期反向注入期望流行度，Kwai 上 Recall@20 相对 BPRMF 提升约 165%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;反事实推断&quot;,&quot;标题党去偏&quot;]"><strong><a href="debias_fairness_industrial/clicks_can_be_cheating_counterfactual_recommendation_for_mitigating_clickba/">Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在因果图中补入曝光→打分直连边，反事实扣除曝光的 NDE，仅用点击数据即把 N@10 提升 7.5%–11.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排后处理（解释生成层）&quot;,&quot;反事实解释&quot;,&quot;影响函数 FIA&quot;,&quot;可解释推荐&quot;]"><strong><a href="debias_fairness_industrial/counterfactual_explanations_for_neural_recommenders/">Counterfactual Explanations for Neural Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>ACCENT 将影响函数从单点扩展到「推荐项-替代项」点对，迭代填平分数差生成最小反事实解释集，重训验证下解释集规模比 FIA 缩小约 1.5–1.7 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排 (解释排序)&quot;,&quot;近重复检测 LSH/MinHash&quot;,&quot;Learning-to-Rank 基准数据集&quot;]"><strong><a href="debias_fairness_industrial/extra_explanation_ranking_datasets_for_explainable_recommendation/">EXTRA: Explanation Ranking Datasets for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 MinHash-LSH 将近重复评论句折叠为全局解释库，构造三个(user,item,explanation)三元组基准，把可解释推荐评测统一为解释全库排序。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边公平&quot;,&quot;曝光公平&quot;,&quot;后处理启发式&quot;]"><strong><a href="debias_fairness_industrial/tfrom_a_two_sided_fairness_aware_recommendation_model_for_both_customers_an/">TFROM: A Two-sided Fairness-Aware Recommendation Model for Both Customers and Providers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出双边公平后处理重排模型 TFROM：位置敏感曝光 + 质量升序贪心填位，离线/在线两版在三数据集上以 &lt;10% 质量损失大幅降低提供者曝光方差。</p>
</dd>
</dl>
</section>
</div>
