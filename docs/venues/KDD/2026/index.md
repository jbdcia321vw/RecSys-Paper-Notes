---
hide:
  - toc
---

# 🏛️ KDD 2026 推荐系统论文解读

104 篇 KDD 2026 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 生成式推荐 ×25 · 大语言模型推荐 ×8 · 端到端生成式 ×8 · 去偏公平与可信推荐 ×7 · 语义ID ×6 · 多任务学习 ×5 · CTR预估与特征交叉 ×4 · GRPO 强化学习 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">33</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">25</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">23</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习" aria-pressed="false">GRPO 强化学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Scaling Law" aria-pressed="false">Scaling Law<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sparse MoE" aria-pressed="false">Sparse MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Uplift Modeling" aria-pressed="false">Uplift Modeling<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型合并" aria-pressed="false">模型合并<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义 ID" aria-pressed="false">生成式推荐与离散语义 ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式端到端" aria-pressed="false">生成式端到端<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式重排" aria-pressed="false">生成式重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏注意力" aria-pressed="false">稀疏注意力<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束解码" aria-pressed="false">约束解码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;语义因子路由&quot;,&quot;对齐与均匀性&quot;]"><strong><a href="collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲图神经网络&quot;,&quot;层次聚类与探索利用&quot;,&quot;语义-协同对齐&quot;]"><strong><a href="collaborative_filtering/breaking_information_cocoons_a_hyperbolic_framework_for_balancing_explorati/">Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>洛伦兹双曲空间对齐 LLM 语义与协同信号，并用 Dasgupta 代价驱动的无超参层次聚类实现用户可调探索-利用权衡，效用最高 +5.49%、多样性 +11.39%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔稠密检索&quot;,&quot;Matryoshka 表征学习&quot;,&quot;多语言多垂类&quot;]"><strong><a href="collaborative_filtering/scaling_multilingual_semantic_search_in_uber_eats_delivery/">Scaling Multilingual Semantic Search in Uber Eats Delivery</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于 gte-Qwen2 双塔统一多语言多垂类语义召回，InfoNCE+Triplet NCE 两阶段训练与 MRL 多维度服务，Store R@20 相对 Legacy 提升 88%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;频率信号缩放&quot;,&quot;空间翻转&quot;]"><strong><a href="graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;特征频域谱滤波&quot;,&quot;可靠性感知融合&quot;]"><strong><a href="self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (11)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="sequential_rec/bending_the_scaling_law_curve_in_large_scale_recommendation_systems/">Bending the Scaling Law Curve in Large-Scale Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>ULTRA-HSTU以物品-动作合并、半局部稀疏注意力与注意力截断实现5×训练/21×推理扩展效率，工业A/B消费指标提升4.11%–8.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;隐扩散模型&quot;,&quot;多行为序列&quot;,&quot;旋转位置编码&quot;]"><strong><a href="sequential_rec/from_agnostic_to_specific_latent_preference_diffusion_for_multi_behavior_se/">From Agnostic to Specific: Latent Preference Diffusion for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 FatsMB：以 BaRoPE 增强的多行为自编码器构建统一偏好隐空间，再用隐扩散模型把「行为无关」偏好定向迁移到「行为特定」偏好并解码生成，全库排序下 Recall@10 最高提升 20.56%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;指数幂时间编码&quot;,&quot;对角稀疏位置注意力&quot;]"><strong><a href="sequential_rec/fuxi_efficient_sequential_recommendation_with_exponential_power_temporal_en/">FuXi-$γ$: Efficient Sequential Recommendation with Exponential-Power Temporal Encoder and Diagonal-Sparse Positional Mechanism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用艾宾浩斯遗忘曲线启发的指数幂时间编码器替代分桶查表，配合 Toeplitz 对角稀疏位置剪枝，在 Full-Ranking 下精度 SOTA 且训练/推理加速最高 4.74×/6.18×。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;访客旅程建模&quot;,&quot;工业搜索排序&quot;]"><strong><a href="sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;语义捷径&quot;,&quot;交互去噪&quot;]"><strong><a href="sequential_rec/lisrec_modeling_user_preferences_with_learned_item_shortcuts_for_sequential/">LISRec: Modeling User Preferences with Learned Item Shortcuts for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>任务无关语义编码构建个性化物品图，取最大连通分量作语义捷径去噪；Amazon/Yelp 全库排序较 TASTE 最高 +14.5% R@20。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Scaling Laws&quot;,&quot;Multi-head Latent Attention&quot;,&quot;多阶段架构&quot;]"><strong><a href="sequential_rec/llatte_scaling_laws_for_multi_stage_sequence_modeling_in_large_scale_ads_re/">LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模块化 LLaTTE 与异步两阶段架构，揭示语义特征可弯曲缩放曲线，线上 NE 降 0.25%、转化提升 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;统一参数化&quot;,&quot;用户-物品解耦&quot;]"><strong><a href="sequential_rec/mixformer_co_scaling_up_dense_and_sequence_in_industrial_recommenders/">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一参数空间统一序列建模与稠密特征交叉，配合用户-物品解耦实现请求级复用，AUC 提升 1.28% 且 FLOPs 降 36%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Prior-Fitted Network&quot;,&quot;免更新推理&quot;,&quot;合成先验&quot;]"><strong><a href="sequential_rec/one_sequential_recommendation_model_pretrained_from_synthetic_priors_predic/">One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 hDCSBM 合成先验预训练 PFN，以支持集条件化实现免更新推理的序列推荐，五基准平均超最优基线 7.53%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;生成式推荐&quot;,&quot;大模型缩放定律&quot;,&quot;双塔离线推理&quot;]"><strong><a href="sequential_rec/scaling_recommender_transformers_to_one_billion_parameters/">Scaling Recommender Transformers to One Billion Parameters</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>ARGUS 将自回归预训练解耦为反馈预测与下一物品预测双任务，把推荐 Transformer 编码器扩到 10 亿参数，线上听歌时长 +2.26%、点赞率 +6.37%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;双流交叉注意力&quot;,&quot;模型-基础设施协同设计&quot;,&quot;非自回归并行解码&quot;]"><strong><a href="sequential_rec/transx_scaling_transformer_based_recommendation_via_behavioral_and_serving_/">TransX: Scaling Transformer-based Recommendation via Behavioral and Serving Stream Crossings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>把推荐重构为“行为流×服务流”交叉的序列到序列动作转导，配合近线摊销编码与KV缓存，离线AUC 0.862、线上CTR +6.0%而在线算力仅基线约20%</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部11篇「序列与会话推荐」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (23)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;生成式推荐&quot;,&quot;训练-推理对齐&quot;,&quot;Beam Search&quot;,&quot;前缀级排序优化&quot;]"><strong><a href="generative_rec/apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;图结构解码&quot;,&quot;似然陷阱&quot;]"><strong><a href="generative_rec/breaking_the_likelihood_trap_consistent_generative_recommendation_with_grap/">Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用有向无环图解码器扩展非自回归生成空间，并以 Gumbel-Softmax 打通评估器梯度，破解似然陷阱，快手线上 Views +0.78%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回&quot;,&quot;时间感知多目标预测&quot;,&quot;条件引导稀疏注意力&quot;]"><strong><a href="generative_rec/climber_pilot_a_non_myopic_generative_recommendation_model_towards_better_i/">Climber-Pilot: A Non-Myopic Generative Recommendation Model Towards Better Instruction-Following</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以时间感知多目标预测 TAMIP 缓解单步推理短视，用条件引导稀疏注意力 CGSA 实现指令遵循召回，网易云线上 Like Rate +4.24%、延迟仅 +0.32ms</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端推荐&quot;,&quot;协同记忆库&quot;,&quot;潜在上下文压缩&quot;,&quot;目标感知检索&quot;]"><strong><a href="generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;归一化流&quot;,&quot;连续生成式回归&quot;]"><strong><a href="generative_rec/flowtime_towards_continuous_generative_watch_time_prediction_via_flow_based/">FlowTime: Towards Continuous Generative Watch Time Prediction via Flow-based Personalized Priors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出连续生成式回归新范式，用一步VAE+归一化流个性化先验建模多峰观看时长分布，KuaiRec MAE降0.676%、线上播放时长+1.044%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID构建&quot;,&quot;生成式检索&quot;,&quot;工业数据集&quot;]"><strong><a href="generative_rec/forge_forming_semantic_identifiers_for_generative_retrieval_in_industrial_d/">FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统化拆解 SID 构建全流程（多模态特征/量化编码/冲突后处理），提出免训练的 Embedding HitRate 与 Gini 系数代理指标，淘宝线上交易量 +0.35%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;语义ID量化&quot;,&quot;RSPO 列表级强化学习&quot;,&quot;动态Beam服务&quot;]"><strong><a href="generative_rec/generative_recommendation_for_large_scale_advertising/">Generative Recommendation for Large-Scale Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>快手广告 GR4AD：UA-SID 语义ID + LazyAR 惰性自回归解码 + VSL/RSPO 在线价值对齐与动态 Beam 服务，线上 A/B 广告收入 +4.28%、吞吐 +117%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID量化&quot;,&quot;分层位置编码&quot;]"><strong><a href="generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;Token级信用分配&quot;,&quot;残差策略优化&quot;,&quot;语义ID&quot;]"><strong><a href="generative_rec/hierarchical_residual_policy_optimization_for_generative_recommendations/">Hierarchical Residual Policy Optimization for Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将物品级反馈分解为 SID 前缀残差信用，用 RRPO 做保守 token 级策略优化，KuaiSim 会话回报 10.53 vs 8.14，线上 IAA Target Cost 最高 +3.49%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式训练&quot;,&quot;动态哈希嵌入表&quot;,&quot;分布式训练系统&quot;]"><strong><a href="generative_rec/mtgenrec_an_efficient_distributed_training_system_for_generative_recommenda/">MTGenRec: An Efficient Distributed Training System for Generative Recommendation Models in Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>KDD 2026 工业系统论文：以动态哈希嵌入表、自动合表、两阶段ID去重与动态序列批处理，将GRM训练吞吐提升1.6–2.4×，线上PV_CTR提升1.31%。</p>
</dd>
</dl>
<p><a href="generative_rec/">查看全部23篇「生成式推荐与离散语义 ID」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;LLM推荐&quot;,&quot;生成式端到端&quot;,&quot;List-level SFT&quot;,&quot;IL-GRPO&quot;,&quot;查询推荐&quot;]"><strong><a href="llm_rec/aigq_an_end_to_end_hybrid_generative_architecture_for_e_commerce_query_reco/">AIGQ: An End-to-End Hybrid Generative Architecture for E-commerce Query Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个淘宝首页 HintQ 端到端生成式查询推荐框架，通过列表级 IL-SFT 与 IL-GRPO 双层奖励及离线-近线混合部署，线上 GMV +10.68%、UCTR +7.42%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;LLM推荐&quot;,&quot;生成式推荐&quot;,&quot;方面解耦&quot;,&quot;多重图消息传递&quot;]"><strong><a href="llm_rec/copersona_collaborative_persona_graphs_for_robust_llm_personalization/">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用方面级多重人物图+双分支推理补全稀疏用户画像，Amazon六域生成式评测全面超越DEP/DPL。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语义画像生成&quot;,&quot;知识图谱传播&quot;]"><strong><a href="llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026 TSMO&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;隐式意图生成&quot;,&quot;教师学生蒸馏&quot;]"><strong><a href="llm_rec/improving_item_discoverability_in_e_commerce_search_via_related_intent_gene/">Improving Item Discoverability in e-Commerce Search via Related Intent Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用闭源LLM+蒸馏SLM生成替代/互补/主题三类隐式意图以扩展电商搜索召回，会话购买预测与LLM评审双轨评估，覆盖流量60%→80%、成本仅30%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;隐空间推理&quot;,&quot;直觉先验&quot;,&quot;双注意力编码器&quot;]"><strong><a href="llm_rec/intuition_guided_latent_reasoning_for_llm_based_recommendation/">Intuition-Guided Latent Reasoning for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 top-K 候选集构建“推荐直觉”嵌入锚定 LLM 隐空间推理起点，Amazon 三数据集 Recall@5 平均提升约 6%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;指令微调数据集&quot;,&quot;LoRA 参数高效微调&quot;]"><strong><a href="llm_rec/itdr_an_instruction_tuning_dataset_for_enhancing_large_language_models_in_r/">ITDR: An Instruction Tuning Dataset for Enhancing Large Language Models in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建覆盖 2 根任务/7 子任务、13 个公开数据集的 19.5 万条指令数据集 ITDR，使 3B–9B 开源 LLM 的推荐综合指标反超闭源大模型。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;多轮 Grounding&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="llm_rec/mgfrec_towards_reinforced_reasoning_recommendation_with_multiple_groundings/">MGFRec: Towards Reinforced Reasoning Recommendation with Multiple Groundings and Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将单步 grounding 扩展为多轮物品空间接地并引入用户 Agent 反馈，用 GRPO 训练 LLM 推理推荐，Full-Ranking 下 H@5 最高提升约 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;KG-RAG&quot;,&quot;MoE&quot;,&quot;多智能体强化学习&quot;]"><strong><a href="llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;画像外推&quot;,&quot;不变性学习&quot;]"><strong><a href="llm_rec/proex_a_unified_framework_leveraging_large_language_model_with_profile_extr/">ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用四步 CoT 为每个用户/物品生成多面画像并做环境不变性融合，在三个数据集上对 6 个基模型平均提升 5%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;基准审计&quot;,&quot;推理期干预&quot;]"><strong><a href="llm_rec/whose_name_comes_up_ii_benchmarking_and_intervention_based_auditing_of_llm_/">Whose Name Comes Up? II: Benchmarking and Intervention-Based Auditing of LLM-Based Scholar Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 LLMScholarBench，联合审计 22 个 LLM 在 5 类学者推荐任务、3 类推理期干预下的技术质量与社会代表性权衡。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;Agent Memory&quot;,&quot;Hierarchical Retrieval&quot;,&quot;RAG&quot;]"><strong><a href="agent_conversational_rec/hierarchical_long_term_semantic_memory_for_linkedin_s_hiring_agent/">Hierarchical Long-Term Semantic Memory for LinkedIn&#x27;s Hiring Agent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>HLTM 用 schema 对齐的层级记忆树 + 多视图表示 + 无损增量索引，为 LinkedIn 招聘智能体实现低延迟、隐私隔离的长期个性化记忆，正确率提升 5%+、检索 F1 提升 10%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排/在线策略实时调整&quot;,&quot;LLM Agent 工具编排&quot;,&quot;仿真增强知识蒸馏&quot;]"><strong><a href="agent_conversational_rec/interactive_recommendation_agent_with_active_user_commands/">Interactive Recommendation Agent with Active User Commands</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出可下达自然语言命令的交互式推荐流 IRF 与 Parser-Planner 双 Agent 框架 RecBot，经仿真蒸馏落地，三月在线 A/B 提升 GMV 1.40%、NFF 降 0.71%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回+重排（对话推荐）&quot;,&quot;Variational EM&quot;,&quot;Latent Intent&quot;]"><strong><a href="agent_conversational_rec/latentcrs_a_variational_em_framework_for_bridging_semantics_and_behavior_in/">LatentCRS: A Variational EM Framework for Bridging Semantics and Behavior in LLM-based Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以离散潜在意图为桥，用变分EM对齐LLM语义与行为序列；三数据集单/多轮全库排序，Recall@5较CRAG最高提升73.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;个性化安全对齐&quot;,&quot;GDPO 多奖励解耦&quot;]"><strong><a href="agent_conversational_rec/safecrs_personalized_safety_alignment_for_llm_based_conversational_recommen/">SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SafeRec 个性化安全基准与 Safe-SFT+Safe-GDPO 两阶段框架，在 SafeMovie 上将安全违规率相对降低 96.5% 且保持竞争力推荐质量。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Agentic RAG&quot;,&quot;意图结构化&quot;,&quot;科学数据集推荐&quot;]"><strong><a href="agent_conversational_rec/sciencedb_ai_an_llm_driven_agentic_recommender_system_for_large_scale_scien/">ScienceDB AI: An LLM-Driven Agentic Recommender System for Large-Scale Scientific Data Sharing Services</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向千万级科学数据集共享平台，用意图感知器+结构化记忆压缩+可信两阶段RAG构建单智能体对话式推荐，离线指标较最强基线提升约50%，线上CTR较关键词检索提升200%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;对话交互式推荐（端到端生成）&quot;,&quot;Thought Pattern Distillation&quot;,&quot;多智能体分层规划&quot;]"><strong><a href="agent_conversational_rec/thought_augmented_planning_for_llm_powered_interactive_recommender_agent/">Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TAIRA 用思维模式蒸馏与分层规划增强 LLM 多智能体交互推荐，在 7 类场景 3 难度模拟查询上 SR 相对提升 9.7%–15.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多模态推荐与内容理解&quot;,&quot;召回&quot;,&quot;个人相册检索&quot;,&quot;多源意图推理&quot;]"><strong><a href="multimodal_rec/photobench_beyond_visual_matching_towards_personalized_intent_driven_photo_/">PhotoBench: Beyond Visual Matching Towards Personalized Intent-Driven Photo Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个真实个人相册多源意图检索基准，以视觉-时空-身份-事件画像合成查询，Full-Set/F1 评测揭示模态鸿沟与源融合悖论。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;Hybrid-Query Connector&quot;,&quot;可靠性门控&quot;]"><strong><a href="multimodal_rec/text_guided_visual_representation_learning_for_robust_multimodal_e_commerce/">Text-Guided Visual Representation Learning for Robust Multimodal E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以结构化元数据初始化语义查询、双流解耦视觉证据并用可靠性双门控校准，在 134K 全库 I2I 检索上 H@100 相对提升 6.04%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;统一嵌入与重排&quot;,&quot;GRPO 强化学习&quot;,&quot;Matryoshka 表征&quot;]"><strong><a href="multimodal_rec/uninote_a_unified_embedding_model_for_multimodal_representation_and_ranking/">UniNote: A Unified Embedding Model for Multimodal Representation and Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>小红书工业级 I2I 统一嵌入模型：对比 SFT + GRPO 相对重排两阶段训练，10 类检索任务全 SOTA，MRL 支持弹性维度部署。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;虚拟交互增强&quot;,&quot;模态相似度&quot;]"><strong><a href="multimodal_rec/vi_mmrec_similarity_aware_training_cost_free_virtual_user_item_interactions/">VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态相似度构造虚拟交互矩阵并以统计权重融合，零训练开销即插即用地缓解多模态推荐数据稀疏，Full-Ranking 下 Recall@10 最高提升 22.25%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;伪标签纠正&quot;,&quot;多模态内容相似&quot;]"><strong><a href="cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;冷启动&quot;,&quot;召回/精排/重排全链路&quot;,&quot;内容冷启动&quot;,&quot;探索与去偏&quot;,&quot;工业级度量框架&quot;]"><strong><a href="cold_start_rec/pinequalizer_full_funnel_content_exploration_and_debiasing_system_at_pinter/">PinEqualizer: Full Funnel Content Exploration and Debiasing System at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向工业级搜索/推荐全漏斗的新内容冷启动系统，以去偏替代纯探索，配合三层可扩展度量框架，线上新鲜内容曝光提升350%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端序列推荐&quot;,&quot;扩散模型&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/align_for_fusion_harmonizing_triple_preferences_via_dual_oriented_diffusion/">Align-for-Fusion: Harmonizing Triple Preferences via Dual-oriented Diffusion for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出align-for-fusion：检索混合域行为噪声并对源/目标双域扩散，四设置Last-item排序HR@10最高+8.22%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;原子意图缓存&quot;,&quot;目标感知意图检索&quot;]"><strong><a href="cross_domain_multi_scenario/atomic_intent_reasoning_bringing_llm_semantics_to_industrial_cross_domain_r/">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 意图推理离线化为原子行为-意图对缓存，在线经意图树构建与目标感知检索实现约 400× 加速，快手 A/B 测试 GMV +3.446%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Tokenization&quot;,&quot;多任务学习&quot;]"><strong><a href="cross_domain_multi_scenario/mdl_a_unified_multi_distribution_learner_in_large_scale_industrial_recommen/">MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将场景/任务Token化并与特征逐层交叉注意，在抖音搜索0.5B模型上QAUC相对提升0.23–0.77%，线上LT30 +0.0626%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;模型合并&quot;,&quot;任务向量&quot;]"><strong><a href="cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多域多目标MDMO&quot;,&quot;组合合并与知识蒸馏&quot;]"><strong><a href="cross_domain_multi_scenario/meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;LLM Persona 建模&quot;,&quot;Doppelganger 跨域迁移&quot;]"><strong><a href="cross_domain_multi_scenario/multi_tap_multi_criteria_target_adaptive_persona_modeling_for_cross_domain_/">Multi-TAP: Multi-criteria Target Adaptive Persona Modeling for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 将用户偏好拆成五类语义 Persona 表征域内异质性，再用 doppelganger 交叉注意力做目标域自适应的选择性迁移；Amazon 时间切分全库排序下 HR@5 最高提升 36.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;模型合并&quot;,&quot;锐度感知优化&quot;,&quot;LoRA 重参数化&quot;]"><strong><a href="cross_domain_multi_scenario/sharpness_aware_model_merging_with_salience_recovery_for_llm_based_cross_do/">Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用锐度感知平坦极小值对齐 + 重尾重参数化恢复显著偏好信号，破解 LLM 跨域序列推荐模型合并中的参数冲突与多域饱和瓶颈。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;冷启动POI预测&quot;,&quot;因果推断&quot;,&quot;时空图神经网络&quot;]"><strong><a href="poi_spatiotemporal_rec/causalpoi_spatio_temporal_graph_based_causal_modeling_for_cold_start_poi_ch/">CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用功能交互图+结构对齐的干预/对照图做冷启动POI签到量预测与因果提升估计，四区域RMSE最高降57.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端序列预测&quot;,&quot;Magnetic Laplacian&quot;,&quot;Complex-valued Mamba&quot;]"><strong><a href="poi_spatiotemporal_rec/mag_mamba_modeling_coupled_spatiotemporal_asymmetry_for_poi_recommendation/">Mag-Mamba: Modeling Coupled spatiotemporal Asymmetry for POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>时变方向场编码为磁相位并注入复值Mamba衰减-旋转递推，NYC/TKY/CA的NDCG@1最高提升14.71%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;生成式推荐&quot;,&quot;LLM推理&quot;,&quot;优势校准&quot;,&quot;语义ID&quot;]"><strong><a href="poi_spatiotemporal_rec/think2go_generative_next_poi_recommendation_with_llm_reasoning/">Think2Go: Generative Next POI Recommendation with LLM Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一SFT与RL推理的生成式Next-POI框架，用KDE时空认知不确定性与难度感知奖励差校准优势函数，Acc@1较GNPR提升5.8–7.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Attention Sink&quot;,&quot;语言模型CTR&quot;,&quot;行为序列建模&quot;]"><strong><a href="ctr_feature_interaction/ctr_sink_attention_sink_for_language_models_in_click_through_rate_predictio/">CTR-Sink: Attention Sink for Language Models in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在文本化行为序列中插入携带时间距离的 [SINK] 锚点并配合两阶段训练与 sink 间注意力增强，缓解语义碎片化，AUC 稳定提升 0.2–0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动机制设计 (AMD)&quot;,&quot;联合广告拍卖&quot;,&quot;可微排序确定性分配&quot;]"><strong><a href="ctr_feature_interaction/deterministic_allocation_and_anonymous_joint_advertising_in_e_commerce_plat/">Deterministic-Allocation and Anonymous Joint Advertising in E-commerce Platforms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用无位置编码 Transformer + 可微排序算子生成匿名、确定性分配的联合广告拍卖机制，离线与线上 A/B 收入显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;数值特征嵌入&quot;,&quot;流式学习&quot;]"><strong><a href="ctr_feature_interaction/distribution_aware_end_to_end_embedding_for_streaming_numerical_features_in/">Distribution-Aware End-to-End Embedding for Streaming Numerical Features in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DAES用跳采样蓄水池估计流式分位数，并在分位空间做场感知调制，实现数值特征端到端嵌入，AUC稳定超越PLE/DAE。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;统一序列建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="ctr_feature_interaction/est_towards_efficient_scaling_laws_in_click_through_rate_prediction_via_uni/">EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LCA+CSA 把特征与行为装进单序列实现全统一建模，剥离冗余交互，在淘宝展示广告取得 0.99% AUC 与 3.27% RPM 提升</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;有效秩&quot;,&quot;表征塌缩&quot;,&quot;Token Mixing&quot;]"><strong><a href="ctr_feature_interaction/expand_more_shrink_less_shaping_effective_rank_dynamics_for_dense_scaling_i/">Expand More, Shrink Less: Shaping Effective-Rank Dynamics for Dense Scaling in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>从有效秩视角诊断 RankMixer 的阻尼振荡塌缩，提出参数化全混合与 GLU-PFFN 实现“多扩张少收缩”，Criteo/Avazu 上 AUC 提升超 0.001。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;显式特征交叉&quot;,&quot;指数交叉网络&quot;,&quot;低开销聚合&quot;]"><strong><a href="ctr_feature_interaction/fcn_fusing_exponential_and_linear_cross_network_for_click_through_rate_pred/">FCN: Fusing Exponential and Linear Cross Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 LCN+ECN 纯显式交叉网络，以半秩 LCA 与 Tri-BCE 在六数据集样本级 AUC/Logloss 上无需 DNN 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Field-Aware Transformer&quot;,&quot;Basis-Composed Hypernetwork&quot;,&quot;Scaling Law&quot;]"><strong><a href="ctr_feature_interaction/from_scaling_to_structured_expressivity_rethinking_transformers_for_ctr_pre/">From Scaling to Structured Expressivity: Rethinking Transformers for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用域感知参数重构 Transformer 块，将复杂度从词表规模 n 转向域数 F，配合基组合超网络实现可预测幂律缩放，离线 AUC 最高 +4.38%，线上 CTR +2.33%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;GD-RTB 混合竞价&quot;,&quot;对偶变量引导&quot;,&quot;列表级生成评估&quot;]"><strong><a href="ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;缩放定律&quot;,&quot;高效注意力&quot;,&quot;模型-效率协同设计&quot;]"><strong><a href="ctr_feature_interaction/kunlun_establishing_scaling_laws_for_massive_scale_recommendation_systems_t/">Kunlun: Establishing Scaling Laws for Massive-Scale Recommendation Systems through Unified Architecture Design</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Kunlun 以 GDPA/HSP/CompSkip 协同设计将 MFU 17%→37%，在 Meta Ads 70B+ 样本上实现 2× 缩放效率与 0.79% NE 增益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多归因学习&quot;,&quot;多任务学习&quot;,&quot;MoE&quot;]"><strong><a href="ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
</dl>
<p><a href="ctr_feature_interaction/">查看全部14篇「CTR/CVR 预估与特征交叉」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;稠密监督蒸馏&quot;,&quot;累积回归&quot;]"><strong><a href="multi_task_reranking/degre_dense_supervised_generative_reranking_for_recommendation/">DeGRe: Dense-supervised Generative Reranking for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>离线束搜索挖掘前瞻序列+累积回归稠密监督蒸馏至轻量生成器，单次贪心解码逼近全局最优，线上GMV+3.75%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;宏图专家&quot;,&quot;多任务图合并&quot;]"><strong><a href="multi_task_reranking/macro_graph_of_experts_for_billion_scale_multi_task_recommendation/">Macro Graph of Experts for Billion-Scale Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以MTMG合并多任务宏图，结合宏图专家与级联读出；Taobao平均AUC提升13.42%，线上GMV较MMoE提升16.46%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Transformer-Native 架构&quot;,&quot;梯度截断与任务解耦&quot;]"><strong><a href="multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;策略个性化&quot;,&quot;Uplift建模&quot;,&quot;因果推断&quot;,&quot;双边市场&quot;]"><strong><a href="multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Sparse MoE&quot;,&quot;Progressive Expert Routing&quot;,&quot;Load Balancing&quot;]"><strong><a href="multi_task_reranking/smes_towards_scalable_multi_task_recommendation_via_expert_sparsity/">SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>KDD 2026 多任务精排：渐进式共享/私有专家稀疏路由与全局负载均衡，工业 A/B 观看时长 +0.31%、延迟降 50%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;激励广告&quot;,&quot;约束策略优化&quot;,&quot;拉格朗日对偶&quot;]"><strong><a href="rl_bandit_rec/generative_optimization_for_incentivized_advertising_with_global_level_cons/">Generative Optimization for Incentivized Advertising with Global Level Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>把激励金分配重构为 λ 条件自回归生成，用 SCPO 单一策略跨 ROI 约束泛化；IA 数据 REV +18.5%、RVR 相对降 18.4%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Uplift Modeling&quot;,&quot;多处理因果推断&quot;,&quot;多目标权衡&quot;]"><strong><a href="rl_bandit_rec/heterogeneous_multi_treatment_uplift_modeling_for_trade_off_optimization_in/">Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用异构多处理增益建模+在线动态权重，个性化权衡短视频消费与商业化指标，快手全量部署</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;分层强化学习&quot;,&quot;自动出价&quot;,&quot;LLM Agent&quot;]"><strong><a href="rl_bandit_rec/hoba_hierarchical_on_policy_bidding_agents_for_adaptive_online_advertising/">HOBA: Hierarchical On-Policy Bidding Agents for Adaptive Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三层时间尺度解耦出价：LLM 小时级推理超参、SARSA 分钟级因果去偏选专家、专家池逐次拍卖执行，AuctionNet 与线上 A/B 双验证。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/干预选择&quot;,&quot;Contextual Bandit&quot;,&quot;影响力概率估计&quot;,&quot;帕累托权衡&quot;]"><strong><a href="rl_bandit_rec/learning_peer_influence_probabilities_with_linear_contextual_bandits/">Learning Peer Influence Probabilities with Linear Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将同伴影响力概率学习建模为双目标线性上下文 Bandit，刻画 Regret–RMSE 帕累托前沿并提出 InfluenceCB 以单参数 β 导航权衡。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;不确定性量化&quot;,&quot;UCB 探索&quot;,&quot;风险规避降权&quot;]"><strong><a href="rl_bandit_rec/uncertainty_calibrated_recommendations_for_low_active_users/">Uncertainty-Aware Adaptive Recommendation across User Lifecycle</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 EPE Critic 与经验贝叶斯先验方差量化逐样本认知不确定性，对低活跃用户风险规避降权、高活跃用户 UCB 探索加成，14 天线上 A/B 验证留存与多样性双收益。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;Bradley-Terry&quot;,&quot;算法排名一致性&quot;,&quot;数据集分类学&quot;]"><strong><a href="debias_fairness_industrial/bradley_terry_rankings_for_recommender_systems_across_dataset_taxonomies/">Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Bradley-Terry 概率排序替代指标均值聚合，在 14 算法×89 数据集上给出数据集感知的推荐算法排名与免运行预测。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Uplift Modeling&quot;,&quot;Cannibalization&quot;,&quot;SUTVA&quot;]"><strong><a href="debias_fairness_industrial/canniuplift_a_holistic_framework_for_mitigating_seller_and_incentive_cannib/">CanniUplift: A Holistic Framework for Mitigating Seller and Incentive Cannibalization in E-commerce Uplift Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用平台级GMV一致性约束与核销路径分解，缓解多卖家电商uplift建模中的卖家级与激励级蚕食，线上ΔGMV相对提升4.08%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;黎曼流形聚合&quot;,&quot;联邦推荐&quot;]"><strong><a href="debias_fairness_industrial/canonfedrec_a_canonical_geometric_framework_for_personalized_federated_reco/">CanonFedRec: A Canonical Geometric Framework for Personalized Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示联邦推荐中聚合物品嵌入崩塌的几何成因，用黎曼投影聚合+散度弹性对齐在单位超球面上重建全局拓扑，显存降低最多 40×。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果提升建模 (Uplift Modeling)&quot;,&quot;离线回放 (Offline Replay)&quot;,&quot;双稳健估计 (Doubly Robust)&quot;]"><strong><a href="debias_fairness_industrial/deep_learning_causal_retrieval_optimization_for_efficient_e_commerce_distri/">Deep-learning Causal Retrieval Optimization for Efficient e-commerce Distribution in Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将购物候选生成器触发建模为因果决策，用 DR 多任务提升学习+随机 Holdout+线性回放，线上触发率降 85% 而购物会话持平。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测&quot;,&quot;LLM-as-a-Judge&quot;,&quot;语义代理&quot;,&quot;MNAR 去偏&quot;]"><strong><a href="debias_fairness_industrial/llm_as_a_judge_for_reliable_and_explainable_offline_evaluation_in_top_k_rec/">LLM-as-a-Judge for Reliable and Explainable Offline Evaluation in Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 从用户文本行为构建语义偏好代理替代 holdout ID 匹配，以 reasoning-then-scoring 生成可解释 Top-K 评测，模型排序一致性较常规测试提升 40–150%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;创作者竞争博弈&quot;,&quot;偏差-方差权衡&quot;,&quot;正则化策略&quot;]"><strong><a href="debias_fairness_industrial/lower_bias_higher_welfare_how_creator_competition_reshapes_bias_variance_tr/">Lower Bias, Higher Welfare: How Creator Competition Reshapes Bias-Variance Tradeoff in Recommendation Platforms?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>博弈论建模创作者竞争如何重塑平台正则化选择，理论与实验一致表明策略环境下应弱正则（低偏差）以最大化用户福利。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;流行度去偏&quot;,&quot;嵌入几何校正&quot;]"><strong><a href="debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排（Feed 排序）&quot;,&quot;用户-算法反馈回路&quot;,&quot;人类被试在线实验&quot;,&quot;智能体仿真&quot;]"><strong><a href="debias_fairness_industrial/rewarding_engagement_and_personalization_in_popularity_based_rankings_ampli/">Rewarding Engagement and Personalization in Popularity-Based Rankings Amplifies Extremism and Polarization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 5 参数动态流行度排序模型 + 1534 名被试在线实验，因果证明参与度奖励与个性化叠加会把消费推向极端与极化（极端同立场点击 +20%）。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;谱坍缩&quot;,&quot;流行度偏差&quot;,&quot;序列推荐&quot;]"><strong><a href="debias_fairness_industrial/the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
</dl>
</section>
</div>
