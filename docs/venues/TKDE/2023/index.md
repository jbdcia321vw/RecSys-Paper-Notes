---
hide:
  - toc
---

# 📖 IEEE TKDE 2023 推荐系统论文解读

28 篇 IEEE TKDE 2023 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDE ×25 · TKDE 2023 ×3 · 知识图谱推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱推荐" aria-pressed="false">知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/粗排&quot;,&quot;神经推荐综述&quot;,&quot;图神经网络&quot;,&quot;辅助信息融合&quot;]"><strong><a href="collaborative_filtering/a_survey_on_accuracy_oriented_neural_recommendation_from_collaborative_filt/">A Survey on Accuracy-oriented Neural Recommendation: From Collaborative Filtering to Information-rich Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以数据使用为轴重构神经推荐综述：统一框架下划分协同过滤、内容增强与时空序列三类，梳理表征/交互建模与辅助信息融合演进及开放问题</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲表示学习&quot;,&quot;语义邻居聚合&quot;,&quot;陀螺向量空间&quot;]"><strong><a href="collaborative_filtering/hyperbolic_neural_collaborative_recommender/">Hyperbolic Neural Collaborative Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用热核加权关系图构造用户/物品语义邻居，并在 Poincaré 球内以陀螺向量算子做深度聚合，CTR AUC 较最强基线最高 +4.15%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;曝光置信度建模&quot;,&quot;随机游走信息量采样&quot;]"><strong><a href="collaborative_filtering/samwalker_recommendation_with_informative_sampling_strategy/">SamWalker++: recommendation with informative sampling strategy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用伪社交网络图扩散核自适应分配隐式反馈置信度，并沿网络随机游走采样正比于置信度的信息量实例，稀疏数据 Pre@5 提升 67.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;变分信息瓶颈&quot;,&quot;用户特征去噪&quot;]"><strong><a href="collaborative_filtering/variational_bandwidth_auto_encoder_for_hybrid_recommender_systems/">Variational Bandwidth Auto-encoder for Hybrid Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户相关的信道带宽动态调节辅助特征信息的流入量，同时缓解评分稀疏不确定性与特征噪声过拟合，三数据集全库排序 Recall@20 最高提升 7.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;反事实去偏&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="graph_rec/causal_inference_for_knowledge_graph_based_recommendation/">Causal Inference for Knowledge Graph based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用因果图拆解 KG 推荐中结构混淆与属性捷径偏差，以 do 干预+反事实推断实现去偏属性偏好建模，Full-Ranking 下 Recall@20 提升 3.79%~6.66%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱注意力&quot;,&quot;偏置随机游走&quot;,&quot;门控上下文融合&quot;]"><strong><a href="graph_rec/contextualized_graph_attention_network_for_recommendation_with_item_knowled/">Contextualized Graph Attention Network for Recommendation with Item Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户特定图注意力聚合 KG 局部邻居、偏置随机游走+GRU 抽取非局部高阶上下文、物品特定注意力建模历史交互；FM/ML/BC 6:2:2 划分全库排序下 HR@20 较最强基线提升 8.3%/1.0%/5.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;哈希学习 (Learning to Hash)&quot;,&quot;汉明空间图卷积&quot;,&quot;高阶相似度传播&quot;]"><strong><a href="graph_rec/hs_gcn_hamming_spatial_graph_convolutional_networks_for_recommendation/">HS-GCN: Hamming Spatial Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在汉明空间直接做图卷积传播，显式建模一阶与高阶汉明相似度，7 个数据集全库排序 HR@50 平均超最强哈希基线 26.2%、逼近实值 GNN 且训练/推理提速 2.4×/4.9×。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;评论图构建&quot;,&quot;个性化图池化&quot;,&quot;类型感知图注意力&quot;]"><strong><a href="graph_rec/learning_hierarchical_review_graph_representations_for_recommendation/">Learning Hierarchical Review Graph Representations for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为每个用户/物品单独构建带词序边类型的评论图，用类型感知图注意力与个性化图池化学习层级语义表征，6 个 Amazon/Yelp 数据集上 MSE 较最强基线 DAML 平均再降 1.14%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对抗变分推断&quot;,&quot;多峰后验嵌入&quot;]"><strong><a href="self_supervised_denoising/dual_adversarial_variational_embedding_for_robust_recommendation/">Dual Adversarial Variational Embedding for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双侧 VAE + 对抗式隐式后验正则实现个性化降噪与多峰偏好建模，Leave-One-Out + 1:99 采样排序下 NDCG@5 提升约 1.3%–8.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;分块记忆网络&quot;,&quot;外部记忆&quot;]"><strong><a href="sequential_rec/cmnrec_sequential_recommendations_with_chunk_accelerated_memory_network/">CmnRec: Sequential Recommendations with Chunk-accelerated Memory Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以认知分块压缩邻近隐状态，仅在 chunk 时刻读写外部记忆，训练最高约 6.34×、推理 10.07× 加速且 MRR@5 不降反升。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;解耦表征学习&quot;,&quot;会话图神经网络&quot;]"><strong><a href="sequential_rec/disentangled_graph_neural_networks_for_session_based_recommendation/">Disentangled Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将物品嵌入解耦为 K 个因子并用因子级相似度 GGNN 传播、距离相关解耦，按因子注意力聚合会话表征，三数据集 P@20 最高 +16.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态图神经网络&quot;,&quot;跨序列协同信号&quot;]"><strong><a href="sequential_rec/dynamic_graph_neural_networks_for_sequential_recommendation/">Dynamic Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将全部用户序列编织成带时间戳与双向顺序的五元组动态图，用相对顺序感知注意力+短期注意力做子图消息传播，把 next-item 预测转为动态图链路预测，Beauty 上 NDCG@10 相对提升 10.56%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;多行为序列推荐&quot;,&quot;时间图Transformer&quot;]"><strong><a href="sequential_rec/multi_behavior_sequential_recommendation_with_temporal_graph_transformer/">Multi-Behavior Sequential Recommendation with Temporal Graph Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为感知Transformer与时间图联合建模多行为短长期依赖，HR@10较MBGCN提升22.5%/12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回/Next-Item排序&quot;,&quot;共享账户&quot;,&quot;跨域序列推荐&quot;,&quot;角色拆分&quot;]"><strong><a href="sequential_rec/parallel_split_join_networks_for_shared_account_cross_domain_sequential_rec/">Parallel Split-Join Networks for Shared-account Cross-domain Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TKDE 2023 [序列与会话推荐] 提出 PSJNet，以角色拆分与跨域联结解决共享账户跨域序列推荐，全库排序下 MRR/Recall 显著超越 GRU4REC/HGRU4REC</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列推荐&quot;,&quot;排序&quot;,&quot;强化学习路径探索&quot;,&quot;可解释推荐&quot;]"><strong><a href="sequential_rec/reinforcement_learning_based_path_exploration_for_sequential_explainable_re/">Reinforcement Learning based Path Exploration for Sequential Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TMER-RL 用强化学习探索连续物品间元路径，并以双注意力融合时序与路径解释，在 4 个数据集 500 负采样排序上显著优于 TMER。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;会话推荐&quot;,&quot;解耦图神经网络&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/transition_information_enhanced_disentangled_graph_neural_networks_for_sess/">Transition Information Enhanced Disentangled Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>位置感知全局图+解耦因子GNN，双通道对比学习建模会话转移，Tmall P@20提升11.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;图结构学习&quot;,&quot;对比多模态融合&quot;]"><strong><a href="multimodal_rec/latent_structure_mining_with_contrastive_modality_fusion_for_multimedia_rec/">Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TKDE 2023 多模态推荐：模态感知图学习+对比融合，Amazon 全排序 R@20 提升 9.2%–24.1%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE 2023&quot;,&quot;社交与群组推荐&quot;,&quot;精排（评分预测 + Top-N 排序）&quot;,&quot;去中心化图&quot;,&quot;偏置向量化&quot;,&quot;社交关系加权&quot;]"><strong><a href="social_group_rec/gdsrec_graph_based_decentralized_collaborative_filtering_for_social_recomme/">GDSRec: Graph-Based Decentralized Collaborative Filtering for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>把评分偏置建成向量并构造去中心化图，配合偏好相似度加权的可微社交强度；Ciao/Epinions 上 MAE 相对 DiffNet++ 降低 0.9%–3.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;意图解耦&quot;,&quot;特征自监督&quot;]"><strong><a href="cold_start_rec/intent_disentanglement_and_feature_self_supervision_for_novel_recommendatio/">Intent Disentanglement and Feature Self-supervision for Novel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户意图解耦为从众（热度）与偏好两因子并配以协同-内容特征自监督，统一优化精度/新颖性/覆盖度，F1@5 相对提升 32%+</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 MAML&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="cold_start_rec/metakg_meta_learning_on_knowledge_graph_for_cold_start_recommendation/">MetaKG: Meta-learning on Knowledge Graph for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>双元学习器（协同感知局部更新 + 知识感知全局更新）配自适应任务调度，在 UC/IC/UIC 三类冷启动场景全库排序下平均提升 62%–75%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;异构图嵌入&quot;,&quot;元素级注意力&quot;,&quot;个性化训练&quot;]"><strong><a href="cross_domain_multi_scenario/a_unified_framework_for_cross_domain_and_cross_system_recommendations/">A Unified Framework for Cross-Domain and Cross-System Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出GA统一框架，以异构图嵌入、元素级注意力与个性化训练同时提升DTCDR/MTCDR/CDR+CSR，1:99采样排序下平均超越SOTA 9.04%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;正交度量映射&quot;,&quot;对偶学习&quot;]"><strong><a href="cross_domain_multi_scenario/dual_metric_learning_for_effective_and_efficient_cross_domain_recommendatio/">Dual Metric Learning for Effective and Efficient Cross-Domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用正交隐空间映射+双向对偶迭代做跨域迁移，仅 8 个重叠用户即可超越用全量重叠用户的 SOTA，Book-Movie RMSE 降至 0.2184。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排/序列打分&quot;,&quot;共享账号用户识别&quot;,&quot;层次强化学习域过滤&quot;]"><strong><a href="cross_domain_multi_scenario/reinforcement_learning_enhanced_shared_account_cross_domain_sequential_reco/">Reinforcement Learning-enhanced Shared-account Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 RL-ISN：用注意力潜在用户聚类建模共享账号、用层次强化学习域过滤器剔除跨域噪声，HVIDEO/HAMAZON 上 HR@10 最高提升 29.1%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排/召回&quot;,&quot;跨域迁移&quot;,&quot;自适应微调&quot;,&quot;用户级策略网络&quot;]"><strong><a href="cross_domain_multi_scenario/user_specific_adaptive_fine_tuning_for_cross_domain_recommendations/">User-specific Adaptive Fine-tuning for Cross-domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出用户级自适应微调 UAF，以策略网络逐用户选择冻结/微调层，在 Tencent 冷启动数据上稳定超越全局微调与 PeterRec。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端生成式&quot;,&quot;替代行程生成&quot;,&quot;MCMC采样&quot;]"><strong><a href="poi_spatiotemporal_rec/deepalttrip_top_k_alternative_itineraries_for_trip_recommendation/">DeepAltTrip: Top-k Alternative Itineraries for Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>双图自编码器+双向LSTM学习历史路线，无需显式流行度/多样性建模即可生成k条既热门又互不相似的替代行程，组合指标最高相对提升29.24%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;可微架构搜索&quot;,&quot;嵌入维度剪枝&quot;]"><strong><a href="ctr_feature_interaction/aim_automatic_interaction_machine_for_click_through_rate_prediction/">AIM: Automatic Interaction Machine for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一门控搜索自动挑选特征交叉、交叉函数与嵌入维度，AUC领先且线上CTR提升4.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;临时兴趣建模&quot;,&quot;非典型交互&quot;]"><strong><a href="rl_bandit_rec/learning_from_atypical_behavior_temporary_interest_aware_recommendation_bas/">Learning from Atypical Behavior: Temporary Interest Aware Recommendation Based on Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双智能体 Actor-Critic 将&quot;非典型交互&quot;从噪声重构为临时兴趣，无监督分类器教推荐器平衡长期偏好与短期兴趣，全库排序下 HR@20 提升最高 15.19%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/评分预测&quot;,&quot;本地差分隐私&quot;,&quot;矩阵分解&quot;,&quot;高斯混合模型&quot;,&quot;有界Laplace&quot;]"><strong><a href="debias_fairness_industrial/private_and_utility_enhanced_recommendations_with_local_differential_privac/">Private and Utility Enhanced Recommendations with Local Differential Privacy and Gaussian Mixture Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出BLP本地扰动与MoG-MF噪声建模，在LDP下用10折CV、RMSE/F-score验证精度提升与单轮通信优势。</p>
</dd>
</dl>
</section>
</div>
