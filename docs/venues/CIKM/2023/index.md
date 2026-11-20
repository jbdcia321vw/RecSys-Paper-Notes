---
hide:
  - toc
---

# 🏛️ CIKM 2023 推荐系统论文解读

59 篇 CIKM 2023 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CTR预估与特征交叉 ×5 · CIKM 2022 ×3 · Contextual Bandit ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">24</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负样本采样&quot;,&quot;Mixup 数据增广&quot;]"><strong><a href="collaborative_filtering/dimension_independent_mixup_for_hard_negative_sample_in_collaborative_filte/">Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>从采样区域视角重塑隐式反馈负采样，提出首个 Area-wise 方法 DINS（边界定义+逐维独立 Mixup+多层池化），在 Full-Ranking 协议下 MF backbone 上 Recall@20 最高提升 36.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数统一分析&quot;,&quot;对齐与均匀性&quot;,&quot;流行度去偏边缘&quot;]"><strong><a href="collaborative_filtering/toward_a_better_understanding_of_loss_functions_for_collaborative_filtering/">Toward a Better Understanding of Loss Functions for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用统一数学框架把 7 种 CF 损失还原为 SSM 的极限特例，并提出边缘感知对齐+加权均匀性损失 MAWU，MF/LightGCN 全库排序下稳定超越 DirectAU 与 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Triplet Loss&quot;,&quot;矩阵分解&quot;,&quot;公平性评测&quot;]"><strong><a href="collaborative_filtering/triplet_losses_based_matrix_factorization_for_robust_recommendations/">Triplet Losses-based Matrix Factorization for Robust Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多重 triplet 损失 + 稀有实体加权矩阵分解，在 LFM-1b 多面公平评测下 MRED 类指标较基线改善 55–89%，总分第 3。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;对齐与一致性&quot;,&quot;高阶邻域聚合&quot;,&quot;轻量图卷积&quot;]"><strong><a href="graph_rec/graph_based_alignment_and_uniformity_for_recommendation/">Graph-based Alignment and Uniformity for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以高阶邻域聚合压缩稠密表示并对齐用户/物品，层间池化缓解稀疏，Office R@20 较 DirectAU 提升 16.64%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;拓扑接近度 Topological Closeness&quot;,&quot;GNN 表达性理论&quot;,&quot;免学习推荐算法&quot;]"><strong><a href="graph_rec/how_expressive_are_graph_neural_networks_in_recommendation/">How Expressive are Graph Neural Networks in Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出三层 GNN 表达性度量框架，用链路级「拓扑接近度」替代图同构度量，并给出免学习 GTE 算法与六数据集全库排序验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;随机播放&quot;,&quot;会话推荐&quot;,&quot;转移增广&quot;]"><strong><a href="self_supervised_denoising/muse_music_recommender_system_with_shuffle_play_recommendation_enhancement/">MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对音乐随机播放会话唯一转移率过高的问题，提出转移频次引导的会话增广与细粒度匹配，在 MSSD 全库排序上稳定超越 12 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;超图推荐&quot;,&quot;一致性感知增强&quot;,&quot;购物篮推荐&quot;]"><strong><a href="self_supervised_denoising/robust_basket_recommendation_via_noise_tolerated_graph_contrastive_learning/">Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>BNCL 以双行为图/超图与一致性感知对比学习抑制购物篮噪声，全库排序下 R@60 最高 21.10。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;局部-全局混合&quot;,&quot;Squeeze-Excitation Attention&quot;]"><strong><a href="sequential_rec/adamct_adaptive_mixture_of_cnn_transformer_for_sequential_recommendation/">AdaMCT: Adaptive Mixture of CNN-Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将局部卷积滤波注入 Transformer，并用层感知、用户自适应的混合系数与 Sigmoid 挤压-激励注意力统一建模长短期兴趣；在三个 Amazon 基准的 1:100 采样排序下 Recall@10 相对最优基线提升 4.1%–8.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态融合顺序&quot;,&quot;异质图注意力&quot;,&quot;非侵入式融合&quot;]"><strong><a href="sequential_rec/adaptive_multi_modalities_fusion_in_sequential_recommendation_systems/">Adaptive Multi-Modalities Fusion in Sequential Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用模态图 + 双注意力异步门控，让每个节点自适应选择早期/晚期融合顺序，Amazon 六域 HR@5 平均 +8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回（序列召回）&quot;,&quot;自注意力校准&quot;,&quot;对抗扰动去噪&quot;]"><strong><a href="sequential_rec/attention_calibration_for_transformer_based_sequential_recommendation/">Attention Calibration for Transformer-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐中注意力权重错配现象，提出空间校准器与对抗校准器直接修正注意力分布，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 8.65%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自编码器协同过滤&quot;,&quot;增量推理&quot;,&quot;物品转移矩阵&quot;]"><strong><a href="sequential_rec/autoseqrec_autoencoder_for_efficient_sequential_recommendation/">AutoSeqRec: Autoencoder for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用单个共享编码器同时重建用户-物品矩阵与转移矩阵行/列，增量推理只更新输入矩阵不更新参数，ML-100K 上 Recall@10 相对提升 35%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;数据增强(召回前处理)&quot;,&quot;Diffusion Model&quot;,&quot;SU-Net&quot;,&quot;长尾用户&quot;]"><strong><a href="sequential_rec/diffusion_augmentation_for_sequential_recommendation/">Diffusion Augmentation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用扩散模型一次性生成高质量伪前序物品序列做数据增强，配合 SU-Net 与两种引导策略，在三个数据集三种骨干上稳定提升 HR/NDCG，尤其缓解长尾用户。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长期行为序列&quot;,&quot;两阶段相关检索&quot;,&quot;参与度门控注意力&quot;]"><strong><a href="sequential_rec/query_dominant_user_interest_network_for_large_scale_search_ranking/">Query-dominant User Interest Network for Large-Scale Search Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>两阶段相关行为检索+参与度融合注意力，Amazon四集全指标平均提升21.19%，线上CTR +7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;文本匹配双塔&quot;,&quot;流行度去偏&quot;,&quot;注意力稀疏&quot;]"><strong><a href="sequential_rec/text_matching_improves_sequential_recommendation_by_reducing_popularity_bia/">Text Matching Improves Sequential Recommendation by Reducing Popularity Biases</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TASTE 用模板将物品与交互历史文本化，经 T5 双塔文本匹配做下一物品推荐，配合会话切分注意力稀疏建模长历史，Full-Ranking 下平均提升约 18%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理与更新&quot;,&quot;组合码&quot;,&quot;更新压缩&quot;]"><strong><a href="sequential_rec/towards_communication_efficient_model_updating_for_on_device_session_based_/">Towards Communication-Efficient Model Updating for On-Device Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出组合码压缩物品嵌入，并用栈/队列与MMD自适应压缩端侧更新，在Gowalla/Lastfm上以60倍更小更新逼近云端重训精度。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;重复消费时间间隔&quot;,&quot;序列匹配&quot;]"><strong><a href="sequential_rec/tsrec_enhancing_repeat_aware_recommendation_from_a_temporal_sequential_pers/">TSRec: Enhancing Repeat-Aware Recommendation from a Temporal-Sequential Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>从时间间隔与序列匹配双视角建模重复消费，三大模块在三个数据集的时间切分采样排序协议下 HR@5 最高相对提升 5.47%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式端到端&quot;,&quot;零样本提示&quot;,&quot;重复物品捷径&quot;,&quot;内容知识与协同知识解耦&quot;]"><strong><a href="agent_conversational_rec/large_language_models_as_zero_shot_conversational_recommenders/">Large Language Models as Zero-Shot Conversational Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用极简提示把 GPT-3.5/4、Vicuna、BAIZE 直接当零样本对话推荐器，揭露 CRS 评测中的「重复物品捷径」，并构建 63 万级 Reddit-Movie 真实数据集</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;多粒度注意力&quot;,&quot;群组偏好聚合&quot;,&quot;组图卷积&quot;]"><strong><a href="social_group_rec/multi_granularity_attention_model_for_group_recommendation/">Multi-Granularity Attention Model for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用子集/群组/超集三级粒度注意力（K-Means 子集聚类 + 组图 GCN + 自注意力融合）挖掘稀疏用户隐偏好，离线 HR/NDCG 全面领先，线上 CTR +1.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;LLM 域不变文本特征&quot;,&quot;域自适应多任务学习&quot;,&quot;Aspect Gating Fusion&quot;]"><strong><a href="cold_start_rec/an_unified_search_and_recommendation_foundation_model_for_cold_start_scenar/">An Unified Search and Recommendation Foundation Model for Cold-Start Scenario</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 LLM 域不变文本特征 + 域门控融合 + 域自适应多任务预训练统一建模搜索与推荐，冷启新场景微调后离线 AUC 提升 2.2–4.0 个百分点、线上 PVCTR +17.54%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排（评分预测）&quot;,&quot;元学习&quot;,&quot;自适应损失&quot;]"><strong><a href="cold_start_rec/meta_learning_with_adaptive_weighted_loss_for_imbalanced_cold_start_recomme/">Meta-Learning with Adaptive Weighted Loss for Imbalanced Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 MELO，用序列状态编码器生成用户级自适应损失权重修正 MAML 内循环，在四个评分数据集上显著降低 RMSE/MAE。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;商品知识图谱预训练&quot;,&quot;零样本物品表征&quot;,&quot;任务导向适配层&quot;]"><strong><a href="cold_start_rec/zero_shot_item_based_recommendation_via_multi_task_product_knowledge_graph_/">Zero-shot Item-based Recommendation via Multi-task Product Knowledge Graph Pre-Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在商品知识图谱上多任务预训练 SGC 编码器并用 ToA 适配层微调，为无交互物品推断归纳式表征；Xmarket 18 国全库排序 NDCG@20 相对提升 18%–83%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Adapter&quot;,&quot;Hyper-Network&quot;,&quot;多领域 CTR&quot;]"><strong><a href="cross_domain_multi_scenario/hamur_hyper_adapter_for_multi_domain_recommendation/">HAMUR: Hyper Adapter for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用即插即用领域适配器配合共享超网络实例级动态生成适配器权重，端到端训练，MovieLens/Ali-CCP 上 AUC 全面超越 STAR/APG/MMOE。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;嵌入解耦&quot;,&quot;随机游走域对齐&quot;]"><strong><a href="cross_domain_multi_scenario/multi_domain_recommendation_with_embedding_disentangling_and_domain_alignme/">Multi-domain Recommendation with Embedding Disentangling and Domain Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>EDDA 在模型与嵌入双层解耦域间/域内知识，并用随机游走停靠向量对齐跨域相似节点，采样排序下 AUC 最高 +7.6%、Recall +41.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多视图时空子图&quot;,&quot;实时兴趣门控融合&quot;]"><strong><a href="poi_spatiotemporal_rec/stgin_spatial_temporal_graph_interaction_network_for_large_scale_poi_recomm/">STGIN: Spatial-Temporal Graph Interaction Network for Large-scale POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>四视图时空子图+GAT聚合、实时序列门控融合，工业数据 HitRate@200 达 25.87%，线上 CTR +1.1%/RPM +6.3%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;时间提示 (Temporal Prompt)&quot;,&quot;移位窗口地理编码&quot;]"><strong><a href="poi_spatiotemporal_rec/timestamps_as_prompts_for_geography_aware_location_recommendation/">Timestamps as Prompts for Geography-Aware Location Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以时间戳为解码 Prompt 显式建模待预测时刻，并用移位窗口缓解网格硬边界，Five Dataset Full 采样排序全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (13)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;增量更新&quot;,&quot;特征先验&quot;,&quot;贝叶斯模型先验&quot;,&quot;长尾物品&quot;]"><strong><a href="ctr_feature_interaction/an_incremental_update_framework_for_online_recommenders_with_data_driven_pr/">An Incremental Update Framework for Online Recommenders with Data-Driven Prior</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用特征级 CTR 先验 (FP) 与贝叶斯输出级模型先验 (MP) 约束滑窗增量训练，流式切分评估下 Criteo/CIKM2019 AUC 超 ASMG，线上 CTR +1.99%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;展示上下文&quot;,&quot;位置偏置&quot;,&quot;用户行为建模&quot;]"><strong><a href="ctr_feature_interaction/deep_context_interest_network_for_click_through_rate_prediction/">Deep Context Interest Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以点击级展示上下文与位置注意力建模用户兴趣，31天工业数据AUC 0.66475，线上CTR/RPM各+1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;属性感知激活&quot;,&quot;动态参数生成&quot;,&quot;相似性与多样性偏好&quot;]"><strong><a href="ctr_feature_interaction/dpan_dynamic_preference_based_and_attribute_aware_network_for_relevant_reco/">DPAN: Dynamic Preference-based and Attribute-aware Network for Relevant Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>属性级双激活+双维压缩重表达分离相似/多样表征，条件动态参数生成融合，AUC 0.6098（RelaImpr +10.69%），线上 CTR +7.62%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Delayed Feedback&quot;,&quot;Entire Space Modeling&quot;,&quot;多任务辅助学习&quot;]"><strong><a href="ctr_feature_interaction/entire_space_cascade_delayed_feedback_modeling_for_effective_conversion_rat/">Entire Space Cascade Delayed Feedback Modeling for Effective Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 ECVR 分解为 CVR×(1−RFR)，用全空间 CVRFR 任务加级联延迟辅助塔一并解决 DS/SSB/CDF，线上 ECVR +5.21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩特征交互&quot;,&quot;SENet位级注意力&quot;,&quot;模型压缩&quot;]"><strong><a href="ctr_feature_interaction/fibinet_reducing_model_size_by_low_rank_feature_interaction_layer_for_ctr_p/">FiBiNet++: Reducing Model Size by Low Rank Feature Interaction Layer for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用内积双线性+低秩交互层与分组位级SENet+重构FiBiNet，非嵌入参数压缩12–16倍且AUC全面反超。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;实例级嵌入维度搜索&quot;,&quot;伯努利门与极化正则&quot;]"><strong><a href="ctr_feature_interaction/ihas_instance_wise_hierarchical_architecture_search_for_deep_learning_recom/">iHAS: Instance-wise Hierarchical Architecture Search for Deep Learning Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用带极化正则的伯努利门为每个样本搜出字段级嵌入维度，再聚类并重训簇专属 DLRM；Avazu AUC 0.7815（+0.0038），推理最快。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多域学习&quot;,&quot;动态加权损失&quot;]"><strong><a href="ctr_feature_interaction/masked_multi_domain_network_multi_type_and_multi_scenario_conversion_rate_p/">Masked Multi-Domain Network: Multi-Type and Multi-Scenario Conversion Rate Prediction with a Single Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用参数共享组合、自动掩码与动态加权损失，在单模型单数据集中实现多转化类型×多展示场景 CVR 预测，AUC 与线上 CVR/CPM 均显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排/出价&quot;,&quot;Bid Shading&quot;,&quot;Multi-task Learning&quot;]"><strong><a href="ctr_feature_interaction/mebs_multi_task_end_to_end_bid_shading_for_multi_slot_display_advertising/">MEBS: Multi-task End-to-end Bid Shading for Multi-slot Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向多广告位展示广告，用多任务端到端出价折减最大化广告主盈余，离线 Surplus 与线上 GMV/ROI 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多哈希码本&quot;,&quot;交叉特征记忆&quot;]"><strong><a href="ctr_feature_interaction/memonet_memorizing_all_cross_features_representations_efficiently_via_multi/">MemoNet: Memorizing All Cross Features&#x27; Representations Efficiently via Multi-Hash Codebook Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出多哈希码本HCNet与MemoNet，以独立记忆存储全部交叉特征，在三大CTR数据集和线上A/B均显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;自适应特征选择&quot;,&quot;混合专家(MoE)&quot;,&quot;重要性分数退火&quot;]"><strong><a href="ctr_feature_interaction/mvfs_multi_view_feature_selection_for_recommender_system/">MvFS: Multi-view Feature Selection for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多子网+门控的 MoE 式控制器为每条样本自适应选择特征域，并以逐域 tanh 退火替代跨域重加权，在 Avazu/Criteo 四个 CTR 骨干上 AUC 全面超越 AdaFS。</p>
</dd>
</dl>
<p><a href="ctr_feature_interaction/">查看全部13篇「CTR/CVR 预估与特征交叉」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;粗排&quot;,&quot;排序一致性对齐&quot;,&quot;分块采样&quot;,&quot;ECPM 排序学习&quot;]"><strong><a href="multi_task_reranking/copr_consistency_oriented_pre_ranking_for_online_advertising/">COPR: Consistency-Oriented Pre-Ranking for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将粗排↔精排的分数对齐松弛为 ECPM 排序对齐，用分块采样与 ΔNDCG 加权 pairwise 损失优化一致性，淘宝线上 +12.3% CTR、+5.6% RPM。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;任务特定底层表征&quot;,&quot;超网络条件Transformer&quot;,&quot;SENet式特征重标定&quot;]"><strong><a href="multi_task_reranking/deep_task_specific_bottom_representation_network_for_multi_task_recommendat/">Deep Task-specific Bottom Representation Network for Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CIKM 2023 多任务推荐论文，提出 DTRN 以超网络条件 Transformer 与 SENet 式任务特定底层表征缓解负迁移；在电商与 Tenrec 样本级 AUC/LogLoss 上全面优于 MMoE/PLE/AITM，在线进入率 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;稀疏MoE路由&quot;,&quot;多模态特征融合&quot;]"><strong><a href="multi_task_reranking/efficient_multi_task_learning_via_generalist_recommender/">Efficient Multi-Task Learning via Generalist Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GRec 用多模态 Wide&amp;Deep + 并行 Transformer + 任务-句子级稀疏 MoE 路由，在电信场景离线 AP 与线上 A/B 全面超越 MMoE 与单任务模型。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;投影解耦&quot;,&quot;多专家网络&quot;]"><strong><a href="multi_task_reranking/parallel_knowledge_enhancement_based_framework_for_multi_behavior_recommend/">Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以并行流知识投影增强级联多行为 GNN，并用投影解耦多专家消除 MTL 梯度冲突，三数据集 HR@10 最高提升 95.16%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排/模型选择&quot;,&quot;帕累托前沿&quot;,&quot;多目标优化&quot;,&quot;校准推荐&quot;]"><strong><a href="multi_task_reranking/post_hoc_selection_of_pareto_optimal_solutions_in_search_and_recommendation/">Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 PDU 从帕累托前沿事后挑出唯一可部署解：用样本级 utopia 距离替代聚合点距离，并支持用户级校准。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;嵌入维度搜索&quot;,&quot;流式推荐&quot;]"><strong><a href="rl_bandit_rec/dynamic_embedding_size_search_with_minimum_regret_for_streaming_recommender/">Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将流式推荐中的动态嵌入维度搜索建模为非平稳 LinUCB 赌博机，用频率+兴趣多样性指标做上下文，取得亚线性遗憾与更低内存/时延。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/在线探索&quot;,&quot;Epistemic Neural Network&quot;,&quot;Thompson Sampling&quot;,&quot;Contextual Bandit&quot;]"><strong><a href="rl_bandit_rec/scalable_neural_contextual_bandit_for_recommender_systems/">Scalable Neural Contextual Bandit for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 ENR：用 EpiNet 式认知不确定性头与 Thompson 采样替代神经老虎机的高代价矩阵求逆，在 MIND/KuaiRec 在线评测中 CTR +9%、评分 +6%，交互节省 29–56%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练与在线推理服务&quot;,&quot;LSH 动态稀疏&quot;,&quot;CPU 推理加速&quot;]"><strong><a href="debias_fairness_industrial/bolt_an_automated_deep_learning_framework_for_training_and_deploying_large_/">BOLT: An Automated Deep Learning Framework for Training and Deploying Large-Scale Search and Recommendation Models on Commodity CPU Hardware</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 LSH 动态稀疏层在商用 CPU 上训练十亿级搜索推荐模型，辅以稀疏超参自动调优与稀疏推理，精度持平 GPU 基线而延迟与碳排降一个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平与工业可信&quot;,&quot;精排&quot;,&quot;双稳健估计&quot;,&quot;选择偏差&quot;,&quot;不确定性门控&quot;]"><strong><a href="debias_fairness_industrial/cdr_conservative_doubly_robust_learning_for_debiased_recommendation/">CDR: Conservative Doubly Robust Learning for Debiased Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 DR 去偏中占比&gt;35% 的“有毒插补”，用 MC-Dropout 估计插补均值/方差做保守门控，即插即用，KuaiRand NDCG@5 提升 6.27%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平与工业系统&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;对比学习&quot;,&quot;反事实采样&quot;]"><strong><a href="debias_fairness_industrial/contrastive_counterfactual_learning_for_causality_aware_interpretable_recom/">Contrastive Counterfactual Learning for Causality-aware Interpretable Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用因果图定位曝光混杂，提出CCL以反事实正采样做对比学习去偏，在Coat/Yahoo! R3无偏测试集全库排序上超越IPS/CVIB。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排（推理期图增广）&quot;,&quot;反事实图增广&quot;,&quot;人口统计学均等&quot;,&quot;二部图扰动&quot;]"><strong><a href="debias_fairness_industrial/counterfactual_graph_augmentation_for_consumer_unfairness_mitigation_in_rec/">Counterfactual Graph Augmentation for Consumer Unfairness Mitigation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 B 维可微二值边选择向量对二部图做反事实增广，以 ApproxNDCG 组间差损失只抬升劣势群体效用，让“解释不公平”直接变成“缓解不公平”。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（用户表征学习）&quot;,&quot;本地差分隐私 LDP&quot;,&quot;去中心化图神经网络&quot;,&quot;超图兴趣凝聚&quot;]"><strong><a href="debias_fairness_industrial/decentralized_graph_neural_network_for_privacy_preserving_recommendation/">Decentralized Graph Neural Network for Privacy-Preserving Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DGREC 用「用户内物品超图兴趣凝聚 + 邻域 1-bit 梯度编码共享」，构建首个去中心化 GNN 推荐框架，满足 (1.5,ε)-RDP 且解码无偏，全库排序 recall@20 较最佳隐私基线提升 7.6%–25.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端评测&quot;,&quot;行为测试&quot;,&quot;切片公平性&quot;,&quot;推荐系统评估&quot;]"><strong><a href="debias_fairness_industrial/evalrs_a_rounded_evaluation_of_recommender_systems/">EvalRS: a Rounded Evaluation of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出EvalRS挑战，以LFM音乐数据和RecList切片/行为测试补充MRR/HR，采用留一法、25%用户×4次随机评审与bootstrap校验，推动全方位评测。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;工业架构与可信系统&quot;,&quot;训练系统/端到端&quot;,&quot;分布式混合并行&quot;,&quot;Meta-IO 数据流水线&quot;,&quot;冷启动元学习&quot;]"><strong><a href="debias_fairness_industrial/g_meta_distributed_meta_learning_in_gpu_clusters_for_large_scale_recommende/">G-Meta: Distributed Meta Learning in GPU Clusters for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向 MAML 类元学习 DLRM 的 GPU 混合并行训练系统，以 AlltoAll/AllReduce 分工与 Meta-IO 流水线把端到端吞吐提升 22%+、模型交付缩短 4 倍，线上 CVR +6.48%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022 (EvalRS DataChallenge)&quot;,&quot;去偏公平&quot;,&quot;召回 + 列表级配额重排&quot;,&quot;Item-based VAE&quot;,&quot;流行度偏差 MRED&quot;]"><strong><a href="debias_fairness_industrial/item_based_variational_auto_encoder_for_fair_music_recommendation/">Item-based Variational Auto-encoder for Fair Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 VAE-CF 转置为物品视角并按流行度分组训练、加组间重建公平正则, 配合配额式列表拼装与 BPRMF 兜底, 用 MRED 与行为测试换取长尾曝光公平。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平与工业架构&quot;,&quot;数据与评测基准&quot;,&quot;搜索推荐联合建模&quot;,&quot;多行为序列&quot;]"><strong><a href="debias_fairness_industrial/kuaisar_a_unified_search_and_recommendation_dataset/">KuaiSAR: A Unified Search And Recommendation Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首个快手真实搜索-推荐统一行为数据集，含跨服务跳转与搜索入口标签，2.6万用户1966万交互</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/排序（物品侧特征编辑）&quot;,&quot;Algorithmic Recourse&quot;,&quot;Counterfactual Explanation&quot;,&quot;多利益相关方可解释性&quot;]"><strong><a href="debias_fairness_industrial/recrec_algorithmic_recourse_for_recommender_systems/">RecRec: Algorithmic Recourse for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次形式化内容提供方视角的推荐补救，用特征空间梯度下降+L1/IHT 稀疏化把物品推进目标人群 top-10，三数据集成功率 80–100%、RBO&gt;99.4%。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部14篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
