---
hide:
  - toc
---

# 🏛️ KDD 2024 推荐系统论文解读

45 篇 KDD 2024 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 端到端生成式 ×5 · CTR预估与特征交叉 ×2 · 双层优化 ×2 · 对比学习 ×2 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR预估与特征交叉" aria-pressed="false">CTR/CVR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排/召回模型压缩&quot;,&quot;持续学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="collaborative_filtering/continual_collaborative_distillation_for_recommender_system/">Continual Collaborative Distillation for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 CCD，在非平稳数据流中协同更新教师与学生：双代理回放防遗忘、新实体初始化与教师吸收学生侧知识；Yelp 学生 H-mean 较最佳基线提升 10.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;三部图元路径&quot;,&quot;对比学习&quot;]"><strong><a href="graph_rec/consistency_and_discrepancy_based_contrastive_tripartite_graph_learning_for/">Consistency and Discrepancy-Based Contrastive Tripartite Graph Learning for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可达/不可达元路径导出一致性与差异性系数，经 GCN 极限理论免训练预计算，配合 CD 对比损失解决三部图推荐极端冷启动。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱图滤波&quot;,&quot;广义图归一化&quot;,&quot;免训练协同过滤&quot;]"><strong><a href="graph_rec/how_powerful_is_graph_filtering_for_recommendation/">How Powerful is Graph Filtering for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以谱锐度视角提出广义图归一化 G²N 与个性化图滤波 IGF，免训练闭式 SGFCF 在四个密度差异数据集上 nDCG@10 最高提升 14.3%，训练耗时降低三个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;偏序推荐图&quot;,&quot;多行为协同过滤&quot;,&quot;偏序 BPR 采样&quot;]"><strong><a href="graph_rec/multi_behavior_collaborative_filtering_with_partial_order_graph_convolution/">Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以分级偏序秩函数融合多行为图为单张加权 POG，POGCN 图卷积 + 偏序 BPR 采样，全库排序下三基准 Mean NDCG 提升 19.67%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;知识图谱推荐&quot;,&quot;健康行为干预&quot;]"><strong><a href="graph_rec/nudgerank_digital_algorithmic_nudging_for_personalized_health/">NudgeRank: Digital Algorithmic Nudging for Personalized Health</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可扩展异构知识图谱+注意力图卷积实现百万级人群个性化健康提醒，离线Precision@3提升45%，线上日步数+6.17%、MVPA+7.61%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Box Embedding&quot;,&quot;标签感知推荐&quot;]"><strong><a href="graph_rec/when_box_meets_graph_neural_network_in_tag_aware_recommendation/">When Box Meets Graph Neural Network in Tag-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户/物品/标签统一建模为超矩形盒，用交并逻辑算子替代 GNN 均值聚合，并以 Gumbel 平滑盒体积作为匹配分，全库排序下全面超越 LFGCF。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;全链路训练/学习范式&quot;,&quot;交互去噪&quot;,&quot;隐式反馈&quot;]"><strong><a href="self_supervised_denoising/double_correction_framework_for_denoising_recommendation/">Double Correction Framework for Denoising Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024交互去噪推荐针对单次损失判定噪声的不稳定性，提出DCF双重校正框架：确认损失+硬样本搜索与渐进式标签翻转，显著提升推荐精度。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;决策边界感知扰动&quot;,&quot;对抗图对比学习&quot;]"><strong><a href="self_supervised_denoising/towards_robust_recommendation_via_decision_boundary_aware_graph_contrastive/">Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>决策边界感知扰动、关系感知对抗图对比与间隔最大化；五数据集全库排序最高相对提升 14.14%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图卷积等价&quot;,&quot;简单对比协同过滤&quot;]"><strong><a href="self_supervised_denoising/unifying_graph_convolution_and_contrastive_learning_in_collaborative_filter/">Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论证明对比损失等价双图卷积，提出无GCN的SCCF，在4数据集全库排序最高提升6.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;数据侧（训练集重构）&quot;,&quot;数据集再生&quot;,&quot;双层优化&quot;,&quot;序列模式挖掘&quot;]"><strong><a href="sequential_rec/dataset_regeneration_for_sequential_recommendation/">Dataset Regeneration for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出数据中心的 DR4SR/DR4SR+ 框架，用多样性促进的 Transformer 再生器把原始序列再生为高质量转移模式数据集，配合双层优化个性化打分，在 4 个数据集 5 类骨干上 Full-Ranking NDCG@20 最高提升 21.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;序列推荐&quot;,&quot;端侧重排推理&quot;,&quot;端云协同个性化&quot;,&quot;子网掩码剪枝&quot;,&quot;超网络&quot;]"><strong><a href="sequential_rec/diet_customized_slimming_for_incompatible_networks_in_sequential_recommenda/">DIET: Customized Slimming for Incompatible Networks in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>云端按实时序列用 GRU+超网络生成二进制子网 mask 并做 filter 级校正下发边缘，传输参数降至 3.1%、FLOPs 减半，NDCG 最高 +22.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式&quot;,&quot;Gen-RecSys 分类学&quot;,&quot;LLM4Rec 与 RAG&quot;]"><strong><a href="generative_rec/a_review_of_modern_recommender_systems_using_generative_models_gen_recsys/">A Review of Modern Recommender Systems Using Generative Models (Gen-RecSys)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按数据模态（交互/文本/多模态）×生成范式（AE/AR/GAN/Diffusion/LLM）构建 Gen-RecSys 二维分类学，并系统梳理评估协议与社会危害框架。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;双流生成架构&quot;,&quot;分级 K-Means 语义ID&quot;,&quot;行为-语义协同&quot;]"><strong><a href="generative_rec/eager_two_stream_generative_recommender_with_behavior_semantic_collaboratio/">EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>共享编码器加双解码器并行生成行为码与语义码，配全局对比与语义引导迁移，Beauty Recall@5 相对 TIGER 提升 31.49%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;大语言模型推荐 (LLM4Rec)&quot;,&quot;生成式端到端&quot;,&quot;多侧面物品标识&quot;,&quot;受限解码&quot;,&quot;生成接地&quot;]"><strong><a href="llm_rec/bridging_items_and_language_a_transition_paradigm_for_large_language_model_/">Bridging Items and Language: A Transition Paradigm for Large Language Model-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>多侧面标识(ID+标题+属性)+FM-index 位置无关受限生成+聚合接地，全库排序下 Yelp R@5 提升 33.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;模型可解释性&quot;,&quot;代理模型对齐&quot;,&quot;多模态嵌入对齐&quot;]"><strong><a href="llm_rec/recexplainer_aligning_large_language_models_for_explaining_recommendation_m/">RecExplainer: Aligning Large Language Models for Explaining Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用行为/意图/混合三种对齐把 LLM 训练成黑盒双塔推荐模型的替身解释器，并在三数据集上验证保真度与解释质量</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多模态推荐&quot;,&quot;端到端生成式&quot;,&quot;多模态大模型适配&quot;,&quot;AIGC 个性化生成&quot;]"><strong><a href="multimodal_rec/multimodal_pretraining_adaptation_and_generation_for_recommendation_a_surve/">Multimodal Pretraining, Adaptation, and Generation for Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以预训练-适配-生成三轴重构多模态推荐文献地图，剖析语义空间与行为空间的错配矛盾，并指出统一评测与推荐基础模型的空白。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;社交与群组推荐&quot;,&quot;召回/表征学习&quot;,&quot;信息瓶颈&quot;,&quot;社交图去噪&quot;]"><strong><a href="social_group_rec/graph_bottlenecked_social_recommendation/">Graph Bottlenecked Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用信息瓶颈原理学习去噪社交图：偏好引导的可微边丢弃 + HSIC 瓶颈正则，Full-Ranking 下 NDCG@20 最高相对提升 17.06%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态对比学习&quot;,&quot;关系感知预训练&quot;]"><strong><a href="cold_start_rec/larp_language_audio_relational_pre_training_for_cold_start_playlist_continu/">LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出语言-音频三阶段对比预训练 LARP，在无重叠冷启动歌单续写的全库排序中显著超越多模态基线。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;MAML 元学习&quot;,&quot;Meta Embedding&quot;,&quot;工业级个性化&quot;]"><strong><a href="cold_start_rec/limaml_personalization_of_deep_recommender_models_via_meta_learning/">LiMAML: Personalization of Deep Recommender Models via Meta Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将推荐网络解耦为 Meta/Global 双块，仅对 Meta Block 做 MAML 元学习并离线蒸馏成定长 meta embedding，实现百亿参数模型的少样本个性化在线服务，线上 CTR 提升 2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;超网络生成特征图&quot;,&quot;元学习暖启动&quot;]"><strong><a href="cold_start_rec/warming_up_cold_start_ctr_prediction_by_learning_item_specific_feature_inte/">Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用超网络为每个新物品生成专属特征交互图，配合可证任意阶交互的定制 GNN 与元学习暖启动，在 MovieLens/Taobao 冷启动与三阶段暖启动 AUC 上全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;双意图建模&quot;,&quot;图传播与嵌入翻译&quot;]"><strong><a href="cross_domain_multi_scenario/unified_dual_intent_translation_for_joint_modeling_of_search_and_recommenda/">Unified Dual-Intent Translation for Joint Modeling of Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用搜索查询监督推荐侧需求意图生成，并以双意图翻译传播统一建模搜索与推荐，离线 Hit@5 约 +4.8%，线上 GMV +1.46%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;端到端生成式&quot;,&quot;扩散模型&quot;,&quot;云边端协同&quot;]"><strong><a href="poi_spatiotemporal_rec/diffusion_based_cloud_edge_device_collaborative_learning_for_next_poi_recom/">Diffusion-Based Cloud-Edge-Device Collaborative Learning for Next POI Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024 [时空轨迹与POI推荐] 提出云-边-端三层扩散协同框架DCPR，云端学类别运动模式、边端注入区域POI嵌入与时空偏置、端侧patch微调，配合DDIM式采样加速，在Foursquare/Weeplace采样排序上取得SOTA并兼顾端侧效率。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨域终身序列&quot;,&quot;对比学习&quot;]"><strong><a href="ctr_feature_interaction/cross_domain_lifelong_sequential_modeling_for_online_click_through_rate_pre/">Cross-Domain LifeLong Sequential Modeling for Online Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>跨域终身序列CTR；CRP对比学习对齐跨域物品表征，LAP三级注意力金字塔；工业A/B CTR +2.93%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排后校准/线上推理&quot;,&quot;Post-hoc Calibration&quot;,&quot;Basis Function Ensemble&quot;,&quot;Multi-Field Calibration&quot;]"><strong><a href="ctr_feature_interaction/deep_ensemble_shape_calibration_multi_field_post_hoc_calibration_in_online_/">Deep Ensemble Shape Calibration: Multi-Field Post-hoc Calibration in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多字段后置校准解耦为基函数形状校准与全字段数值校准，免分箱提升稀疏字段数据利用率，工业线上 CVR +2.5%、GMV +4.0%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征选择&quot;,&quot;AUKC&quot;,&quot;基准评测&quot;]"><strong><a href="ctr_feature_interaction/erase_benchmarking_feature_selection_methods_for_deep_recommender_systems/">ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一评测 11 种特征选择方法×4 个 DRS backbone，提出 AUKC 稳定性指标，工业在线裁剪 30% 特征降 20% 延迟且效果不降。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;广告位分配/拍卖机制&quot;,&quot;自动机制设计&quot;,&quot;联合拍卖&quot;,&quot;JRegNet&quot;]"><strong><a href="ctr_feature_interaction/joint_auction_in_the_online_advertising_market/">Joint Auction in the Online Advertising Market</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出门店-品牌联合拍卖与JRegNet，在合成/美团日志上以近DSIC/IR约束提升收入15.9%–19.8%，推理1–3ms。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排后广告拍卖/计费&quot;,&quot;OCPC&quot;,&quot;转化报告操纵&quot;,&quot;拍卖博弈&quot;]"><strong><a href="ctr_feature_interaction/optimized_cost_per_click_in_online_advertising_a_theoretical_analysis/">Optimized Cost Per Click in Online Advertising: A Theoretical Analysis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用两广告主扩展式博弈比较 OCPC/CPC/CPA，揭示站外转化操纵与外部选项下的定价优势。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;排序损失&quot;,&quot;负样本梯度消失&quot;]"><strong><a href="ctr_feature_interaction/understanding_the_ranking_loss_for_recommendation_with_sparse_user_feedback/">Understanding the Ranking Loss for Recommendation with Sparse User Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从负样本梯度消失视角解释辅助排序损失，Criteo_x1全样本AUC/BCE评测与腾讯线上GMV提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩分解&quot;,&quot;模型压缩&quot;]"><strong><a href="ctr_feature_interaction/unified_low_rank_compression_framework_for_click_through_rate_prediction/">Unified Low-rank Compression Framework for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出AFM统一低秩压缩框架，对MLP输出特征与Embedding表做PCA低秩近似；在Criteo/Avazu/XYZ上以3–5×压缩实现更高AUC与1.35–2.7×吞吐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;对比上下文编码&quot;,&quot;多样性推荐&quot;]"><strong><a href="multi_task_reranking/contextual_distillation_model_for_diversified_recommendation/">Contextual Distillation Model for Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用兴趣感知 MMR 作教师、Gumbel-Top-k 对比上下文编码器作学生，把 O(NK²) 的贪心多样性重排蒸馏成 O(N log K) 的 point-wise 打分，全链路可用。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;梯度冲突消解&quot;,&quot;多目标优化&quot;]"><strong><a href="multi_task_reranking/gradcraft_elevating_multi_task_recommendations_through_holistic_gradient_cr/">GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出“幅度柔性对齐 + 全局方向去冲突”的两阶段梯度整形，在 Wechat/Kuaishou 上 AV-G 分别提升 1.06%/3.79%，线上 VV +0.95%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;非自回归生成&quot;,&quot;对比解码&quot;]"><strong><a href="multi_task_reranking/non_autoregressive_generative_models_for_reranking_recommendation/">Non-autoregressive Generative Models for Reranking Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>非自回归生成器一次解码全部位置，配合匹配模型、序列级 unlikelihood 训练与对比解码；Avito AUC 0.7234、快手 Recall@6 74.86%，推理约 5 倍提速。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;LTV 预测&quot;,&quot;跨域迁移&quot;,&quot;负迁移抑制&quot;]"><strong><a href="rl_bandit_rec/adsnet_cross_domain_ltv_prediction_with_an_adaptive_siamese_network_in_adve/">ADSNet: Cross-Domain LTV Prediction with an Adaptive Siamese Network in Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用伪孪生双网逐样本度量外部数据增益，门控拒收负迁移样本以做跨域 LTV 预测；工业数据全量 AUC/GINI 评测，GINI 提升约 2%，线上 LTV +3.47%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;奖励分解&quot;,&quot;列表重排&quot;]"><strong><a href="rl_bandit_rec/future_impact_decomposition_in_request_level_recommendations/">Future Impact Decomposition in Request-level Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在请求级列表 MDP 下将 A2C 的 TD 目标与优势函数按 item 分解重加权，KuaiRand 上奖励/深度较 HAC 提升 26.7%/20.9%，并完成亿级短视频平台 A/B 验证。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Policy Gradient 大动作空间&quot;,&quot;LLM-as-Reward / Novelty 优化&quot;]"><strong><a href="rl_bandit_rec/optimizing_novelty_of_top_k_recommendations_using_large_language_models_and/">Optimizing Novelty of Top-k Recommendations using Large Language Models and Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 top-k 新颖性优化重构为 ⟨query,item⟩ 二值动作策略梯度，用 GPT-3.5 作奖励模型，把样本复杂度从 $A^2$降到$A$，工业搜索引擎 Novelty@200 提升 36%、A/B 点击率 +0.26%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/多任务&quot;,&quot;表征学习&quot;,&quot;维度坍缩&quot;,&quot;兴趣解缠&quot;]"><strong><a href="debias_fairness_industrial/ads_recommendation_in_a_collapsed_and_entangled_world/">Ads Recommendation in a Collapsed and Entangled World</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>腾讯广告系统从特征编码、维度坍缩与任务兴趣纠缠三方面重构表征学习，并以在线 A/B 验证多嵌入、解缠与训练策略的 GMV 收益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端生成式&quot;,&quot;对抗攻击&quot;,&quot;LLM Agent&quot;,&quot;Prompt Tuning&quot;]"><strong><a href="debias_fairness_industrial/cheatagent_attacking_llm_empowered_recommender_systems_via_llm_agent/">CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用掩码定位插入位 + LLM 智能体生成对抗扰动并以前缀提示自反思优化，在黑盒下显著击穿 P5/TALLRec 推荐系统。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差 (Duration Bias)&quot;,&quot;反事实观看时长 (CWT)&quot;]"><strong><a href="debias_fairness_industrial/counteracting_duration_bias_in_video_recommendation_via_counterfactual_watc/">Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024｜去偏与工业推荐：以“反事实观看时长(CWT)截断”重构时长偏差，用经济学边际收益推导变换函数与截断似然，KuaiRand/WeChat/Product 上 MAE 降约 25%、AUC 逼近 Oracle，线上 MWT +2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;双稳健估计&quot;,&quot;噪声反馈&quot;]"><strong><a href="debias_fairness_industrial/debiased_recommendation_with_noisy_feedback/">Debiased Recommendation with Noisy Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次联合建模 MNAR 与结果测量误差 (OME)，提出 OME-EIB/IPS/DR 估计量、弱可分误差参数辨识与交替去噪训练，在 Coat/Music/KuaiRec 上 AUC 相对提升约 1.5–2.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;用户关系图&quot;,&quot;个性化物品嵌入&quot;]"><strong><a href="debias_fairness_industrial/gpfedrec_graph_guided_personalization_for_federated_recommendation/">GPFedRec: Graph-guided Personalization for Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>服务端用客户端物品嵌入构建用户关系图引导联邦聚合，Leave-One-Out 全库排序下 HR@10 最高相对提升 3.38%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/排序策略层&quot;,&quot;用户偏好动态&quot;,&quot;危害缓解&quot;,&quot;隐函数定理梯度&quot;]"><strong><a href="debias_fairness_industrial/harm_mitigation_in_recommender_systems_under_user_preference_dynamics/">Harm Mitigation in Recommender Systems under User Preference Dynamics</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 MNL 选择+吸引力偏好动态建模“推荐→兴趣漂移→站外有害消费”路径，在平稳点用隐函数定理求梯度优化 CTR-危害权衡，CTR 与危害同时优于基线，最高 77%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;互惠推荐&quot;,&quot;因果推断&quot;]"><strong><a href="debias_fairness_industrial/revisiting_reciprocal_recommender_systems_metrics_formulation_and_method/">Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出覆盖/稳定性/双边均衡五项互惠推荐新指标，并以双边因果 treatment + 空位重排求解；全库排序下 CRecall 最高提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段数据投毒&quot;,&quot;对比学习推荐&quot;,&quot;谱分析&quot;,&quot;双层优化&quot;]"><strong><a href="debias_fairness_industrial/unveiling_vulnerabilities_of_contrastive_recommender_systems_to_poisoning_a/">Unveiling Vulnerabilities of Contrastive Recommender Systems to Poisoning Attacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从谱分析揭示 InfoNCE 的全局离散效应正是 CL 推荐易受定向投毒的根源，提出双目标 CLeaR 攻击，Hit-Ratio@50 平均提升 1.5–3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（冷启动内容匹配）/供给生态&quot;,&quot;创作者竞争博弈&quot;,&quot;用户福利优化&quot;,&quot;乘法权重更新&quot;]"><strong><a href="debias_fairness_industrial/user_welfare_optimization_in_recommender_systems_with_competing_content_cre/">User Welfare Optimization in Recommender Systems with Competing Content Creators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户满意度动态重加权 UIR/SMT/HMT 引导创作者竞争，三周线上 A/B 使 LTR +1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回 (Next-POI 全库打分)&quot;,&quot;Membership Inference Attack (LiRA)&quot;,&quot;Training Data Extraction&quot;,&quot;DP-SGD / Selective DP&quot;]"><strong><a href="debias_fairness_industrial/where_have_you_been_a_study_of_privacy_risk_for_point_of_interest_recommend/">Where Have You Been? A Study of Privacy Risk for Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个面向 POI 推荐模型的位置/轨迹级隐私攻击套件（抽取+成员推断），揭示全库排序模型可被黑盒查询大幅记忆训练签到，并证明现有 DP 防御效用崩塌。</p>
</dd>
</dl>
</section>
</div>
