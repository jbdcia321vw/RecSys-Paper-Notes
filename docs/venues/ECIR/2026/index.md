---
hide:
  - toc
---

# 🏛️ ECIR 2026 推荐系统论文解读

11 篇 ECIR 2026 论文解读，覆盖 7 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 去偏公平与可信推荐 ×2 · 大语言模型推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图划分相似度&quot;,&quot;可复现性评测&quot;,&quot;长尾分析&quot;]"><strong><a href="collaborative_filtering/a_reproducible_and_fair_evaluation_of_partition_aware_collaborative_filteri/">A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在统一用户留出划分与 TPE 调参下复现 FPSR/FPSR+，发现 BISM 在 Amazon-CDs 反超，FPSR 家族在长尾显著占优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;LLM 提示去噪&quot;,&quot;后训练用户画像清洗&quot;]"><strong><a href="self_supervised_denoising/post_training_denoising_of_user_profiles_with_llms_in_collaborative_filteri/">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 零样本/少样本提示在推理前清洗用户画像，无需重训 CF 模型即获最高 13% NDCG/HR 提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;层次化语义ID&quot;,&quot;平衡树构建&quot;,&quot;约束k-means加速&quot;]"><strong><a href="generative_rec/efficient_optimization_of_hierarchical_identifiers_for_generative_recommend/">Efficient Optimization of Hierarchical Identifiers for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>复现SEATER并指出其平衡k-ary树构建是训练瓶颈，提出贪心/混合聚类将建树时间降至2%/5-8%且检索质量持平甚至更优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户画像蒸馏&quot;,&quot;序列推荐&quot;]"><strong><a href="llm_rec/pre_trained_llms_meet_sequential_recommenders_efficient_user_centric_knowle/">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预训练 LLM 生成文本用户画像并蒸馏进 SASRec/BERT4Rec 隐层，两阶段训练实现零推理开销的语义增强，Recall@10 最高提升 23.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;自然语言用户画像&quot;,&quot;学术文献推荐基准&quot;]"><strong><a href="llm_rec/scinup_natural_language_user_interest_profiles_for_scientific_literature_re/">SciNUP: Natural Language User Interest Profiles for Scientific Literature Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建千名作者的自然语言兴趣画像学术推荐基准，系统对比稀疏/稠密/LLM重排，RRF 融合达 NDCG@10 0.4481。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM Reranking&quot;,&quot;外部推荐器适配&quot;]"><strong><a href="agent_conversational_rec/improving_conversational_recommendation_with_contextual_adaptation_of_exter/">Improving Conversational Recommendation with Contextual Adaptation of External Recommenders and LLM-based Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用轻量实体级序列推荐器产出候选、LLM 做上下文适配重排，ReDial/INSPIRED 上 H@5 平均提升 54%/25%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;数据合成/仿真训练&quot;,&quot;Reference-Free 仿真&quot;,&quot;角色掩码损失&quot;,&quot;双智能体交互&quot;]"><strong><a href="agent_conversational_rec/interplay_training_independent_simulators_for_reference_free_conversational/">Interplay: Training Independent Simulators for Reference-Free Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>训练独立用户/推荐双模拟器，以目标属性替代目标物品消除 oracle 泄漏，8B 模型 Recall@1 超 32B 基线 15 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;用户模拟&quot;,&quot;LLM-as-a-Judge&quot;]"><strong><a href="agent_conversational_rec/usersimcrs_v2_simulation_based_evaluation_for_conversational_recommender_sy/">UserSimCRS v2: Simulation-Based Evaluation for Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>升级对话推荐用户模拟评测工具包，引入 LLM 模拟器、统一数据格式与 LLM-as-a-Judge 指标，案例揭示模拟器间系统排序显著分歧。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态剔除&quot;,&quot;实证复现研究&quot;]"><strong><a href="multimodal_rec/are_multimodal_embeddings_truly_beneficial_for_recommendation_a_deep_dive_i/">Are Multimodal Embeddings Truly Beneficial for Recommendation? A Deep Dive into Whole vs. Individual Modalities</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态剔除（常量/噪声替换）在14个模型上做受控实证，发现文本主导、图像近乎无效，简单融合基线增益源于架构伪影。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测指标&quot;,&quot;个体公平&quot;,&quot;用户相似度&quot;,&quot;有效性差异&quot;]"><strong><a href="debias_fairness_industrial/measuring_individual_user_fairness_with_user_similarity_and_effectiveness_d/">Measuring Individual User Fairness with User Similarity and Effectiveness Disparity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 PUF 指标，用用户相似度加权有效性差异，首次同时刻画个体公平的两大要素，4 数据集 7 推荐器验证其敏感性与计算效率。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;模糊逻辑&quot;,&quot;可解释规则学习&quot;]"><strong><a href="debias_fairness_industrial/modeling_behavioral_patterns_in_news_recommendations_using_fuzzy_neural_net/">Modeling Behavioral Patterns in News Recommendations Using Fuzzy Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用多层模糊神经网络端到端学习可读逻辑规则预测新闻点击，阈值可调控制规则复杂度，EB-NeRD 上 nDCG@10 相对 NRMS 提升约 21.9%。</p>
</dd>
</dl>
</section>
</div>
