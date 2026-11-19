---
hide:
  - toc
---

# 🏛️ RecSys 2024 推荐系统论文解读

49 篇 RecSys 2024 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 去偏公平与可信推荐 ×2 · 可解释推荐 ×2 · 特征交叉 ×2 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;条件检索 Conditional Retrieval&quot;,&quot;双塔 + In-batch 负采样&quot;,&quot;特征交叉&quot;]"><strong><a href="collaborative_filtering/bootstrapping_conditional_retrieval_for_user_to_item_recommendations/">Bootstrapping Conditional Retrieval for User-to-Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品侧 topic 元数据为普通 user-item 互动样本合成条件，在用户塔内做 user×condition 早期特征交叉，无需条件专属训练数据即可支持条件检索；Pinterest 全量上线带来 +0.26% WAU，服务成本仅为过滤式双塔的 1/4。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;长尾查询&quot;,&quot;LLM 数据增强&quot;]"><strong><a href="collaborative_filtering/embedding_based_retrieval_for_long_tail_search_queries_in_ecommerce/">Embedding based retrieval for long tail search queries in ecommerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Best Buy 用 LLM 合成查询、领域 RoBERTa 预训练、q2q→q2p 微调与权重融合优化双塔长尾召回，线上转化 +3%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;BPR&quot;,&quot;可复现性研究&quot;,&quot;负采样&quot;]"><strong><a href="collaborative_filtering/revisiting_bpr_a_replicability_study_of_a_common_recommender_system_baselin/">Revisiting BPR: A Replicability Study of a Common Recommender System Baseline</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统复现 BPR 并解耦四大实现特征，证明调优后的 BPR 在全局时间切分下可追平甚至超越 EASE/Mult-VAE。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式召回&quot;,&quot;生成式 LLM 微调&quot;,&quot;对比对齐&quot;,&quot;BM25 文本检索&quot;]"><strong><a href="sequential_rec/calrec_contrastive_alignment_of_generative_llms_for_sequential_recommendati/">CALRec: Contrastive Alignment of Generative LLMs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>两阶段微调生成式 LLM，双塔对比对齐 + 文本生成损失，Amazon 5-core 全库排序 Recall@1 提升 37.3%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/全库打分&quot;,&quot;乘积量化&quot;,&quot;子ID推理&quot;,&quot;PQTopK&quot;]"><strong><a href="sequential_rec/efficient_inference_of_sub_item_id_based_sequential_recommendation_models_w/">Efficient Inference of Sub-Item Id-based Sequential Recommendation Models with Millions of Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用PQTopK重排子ID打分循环，将RecJPQ增强SASRec在百万物品Gowalla的CPU推理加速4.5×，并验证千万级目录可行。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;个性化流行度&quot;,&quot;重复消费&quot;]"><strong><a href="sequential_rec/enhancing_sequential_music_recommendation_with_personalized_popularity_awar/">Enhancing Sequential Music Recommendation with Personalized Popularity Awareness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户级流行度先验映射为 logit 偏移注入 Transformer 序列模型，NDCG@100 相对提升 25.2%–69.8%。</p>
</dd>
<dt data-paper-tags="[&quot;Knowledge-Based Systems (KBS)&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;多面向解耦表征&quot;,&quot;门控融合&quot;]"><strong><a href="sequential_rec/modeling_multi_aspect_preferences_and_intents_for_multi_behavioral_sequenti/">Modeling Multi-aspect Preferences and Intents for Multi-behavioral Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>多面向投影+行为增强LSTM+精炼注意力，联合建模偏好/意图并按面向门控融合，采样排序下较最强基线显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;注意力权重细化&quot;,&quot;高阶依赖建模&quot;]"><strong><a href="sequential_rec/pay_attention_to_attention_for_sequential_recommendation/">Pay Attention to Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AWRSR，把自注意力权重矩阵本身当作表征再做一次注意力细化，以极低开销捕获高阶 item 转移；在 Beauty/ML-1M/Anime 上对 SASRec 与 STOSA 普遍带来 2%–14% 相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可扩展交叉熵&quot;,&quot;MIPS 难负样本&quot;]"><strong><a href="sequential_rec/scalable_cross_entropy_loss_for_sequential_recommendations_with_large_item_/">Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以随机桶中心做 top-k 分桶，把全库 softmax 近似为桶内小 logits 矩阵，并用 Mix 投影缓解桶塌缩；全库排序下显存最多降 100 倍而指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;预训练/召回表征&quot;,&quot;滑窗数据增广&quot;,&quot;推荐基础模型&quot;]"><strong><a href="sequential_rec/sliding_window_training_utilizing_historical_recommender_systems_data_for_f/">Sliding Window Training -- Utilizing Historical Recommender Systems Data for Foundation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出滑窗式训练采样器，在不增大 FM 输入窗口的前提下把用户多年历史灌入预训练；混合最近窗+滑窗后 MRR 提升 8.8%、item 侧 Recall 提升 14.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;PLM 参数冗余&quot;,&quot;行为微调初始化&quot;,&quot;层间注意力分层&quot;]"><strong><a href="sequential_rec/the_elephant_in_the_room_rethinking_the_usage_of_pre_trained_language_model/">The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 PLM 行为序列建模的参数冗余，改用行为微调 PLM 仅做物品嵌入初始化，配合轻量 ID 序列模型大幅提升且零额外推理开销</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;重听行为建模&quot;,&quot;ACT-R 认知架构&quot;]"><strong><a href="sequential_rec/transformers_meet_act_r_repeat_aware_and_sequential_listening_session_recom/">Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 ACT-R 认知激活权重改造 Transformer 会话序列建模，兼顾重听与探索，Last.fm/Deezer 全库 Top-10 排序 NDCG 提升 32.5%/9.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;精排&quot;,&quot;Semantic ID&quot;,&quot;RQ-VAE&quot;,&quot;SentencePiece 子词哈希&quot;,&quot;冷启动泛化&quot;]"><strong><a href="generative_rec/better_generalization_with_semantic_ids_a_case_study_in_ranking_for_recomme/">Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 RQ-VAE 把冻结内容向量压成 8 层语义 ID，再以 SPM 子词哈希替代 YouTube 随机哈希视频 ID，冷启与长尾切片 CTR 提升且整体质量不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;生成式检索 Generative Retrieval&quot;,&quot;多任务联合训练&quot;,&quot;原子 ID 表征正则化&quot;]"><strong><a href="generative_rec/bridging_search_and_recommendation_in_generative_retrieval_does_one_task_he/">Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用单个 T5 联合训练搜索与推荐的生成式检索，以仿真+真实数据论证流行度与隐表征双重正则化机制，全库 R@30 平均 +16%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;创意生成（精排后呈现层）&quot;,&quot;Latent Diffusion / ControlNet&quot;,&quot;Contextual Bandit (LinUCB)&quot;]"><strong><a href="generative_rec/dynamic_product_image_generation_and_recommendation_at_scale_for_personaliz/">Dynamic Product Image Generation and Recommendation at Scale for Personalized E-commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 ControlNet 边缘约束扩散生成商品背景并回贴原商品，再用 LinUCB 选 prompt，线上 A/B 中 CTR 相对提升约 15%–40%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;序列推荐&quot;,&quot;排序蒸馏&quot;]"><strong><a href="llm_rec/distillation_matters_empowering_sequential_recommenders_to_match_the_perfor/">Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 BIGRec 教师蒸馏至轻量序列模型：置信/一致性加权排序蒸馏 + projector-offset 嵌入蒸馏，平均提升 47.97%，推理降至秒级。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;贝叶斯优化&quot;,&quot;自然语言偏好引导&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="agent_conversational_rec/bayesian_optimization_with_llm_based_acquisition_functions_for_natural_lang/">Bayesian Optimization with LLM-Based Acquisition Functions for Natural Language Preference Elicitation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将自然语言偏好引导建模为贝叶斯优化，用 NLI 蕴含分维护 Beta 效用后验，并以 TS/UCB 指导 LLM 生成查询；10 轮对话后 Yelp MRR@10 达 0.27，超最优单体 LLM 基线 58.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;说服性解释&quot;,&quot;可持续推荐&quot;]"><strong><a href="agent_conversational_rec/less_is_more_towards_sustainability_aware_persuasive_explanations_in_recomm/">Less is More: Towards Sustainability-Aware Persuasive Explanations in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出“可持续性感知说服性解释”，基于 Cialdini 六原则在书/健康食品/汽车三域开展 158 人被试内排序研究，发现权威原则说服力与有效性最高。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;预训练音频表征&quot;,&quot;迁移学习&quot;,&quot;内容-协同混合推荐&quot;]"><strong><a href="multimodal_rec/comparative_analysis_of_pretrained_audio_representations_in_music_recommend/">Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统评测 6 种预训练音频后端（MusiCNN/MERT/EncodecMAE 等）在 KNN、浅层网络与 BERT4Rec 三种推荐器上的冻结嵌入效果，揭示 MIR 榜单与音乐推荐性能严重错位。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;单分支权重共享&quot;,&quot;多模态对齐&quot;,&quot;缺失模态鲁棒性&quot;]"><strong><a href="cold_start_rec/a_multimodal_single_branch_embedding_network_for_recommendation_in_cold_sta/">A Multimodal Single-Branch Embedding Network for Recommendation in Cold-Start and Missing Modality Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用权重共享的单分支网络统一编码交互与多模态内容，Item/User-SiBraR 在 Full-Ranking 协议下物品冷启动 nDCG@10 最高相对提升 70%，并显著缓解缺失模态与模态鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元数据对齐正则&quot;,&quot;闭式 EASE 变体&quot;,&quot;语义特征嵌入&quot;]"><strong><a href="cold_start_rec/marec_metadata_alignment_for_cold_start_recommendation/">MARec: Metadata Alignment for cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用一条 item-item 相似度对齐正则把元数据嵌入注入 EASE/VAE 骨干，冷启 hr@10 提升 8.4%–53.8%，训练快两个数量级且热启不掉点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;Prompt Tuning&quot;,&quot;对比学习去偏&quot;]"><strong><a href="cold_start_rec/prompt_tuning_for_item_cold_start_recommendation/">Prompt Tuning for Item Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以高价值正反馈（pinnacle feedback）替代内容描述作为软提示，配合物品级个性化提示网络与双提示增强损失，在 4 个数据集 leave-one-out 采样排序上显著超越 SOTA，并落地亿级短视频平台 A/B 实验。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;ADMM 全局变量一致性&quot;,&quot;隐式反馈 ALS&quot;,&quot;用户因子共享&quot;]"><strong><a href="cross_domain_multi_scenario/cross_domain_latent_factors_sharing_via_implicit_matrix_factorization/">Cross-Domain Latent Factors Sharing via Implicit Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 ADMM 全局变量一致性把隐式 ALS 矩阵分解扩展为跨域版本，仅交换扰动用户因子即可共享隐因子，热启动 NDCG@10 最高相对提升 55.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;场景关系图 SRG&quot;,&quot;动态冷启动&quot;,&quot;自适应专家混合 AEM&quot;]"><strong><a href="cross_domain_multi_scenario/scene_wise_adaptive_network_for_dynamic_cold_start_scenes_optimization_in_c/">Scene-wise Adaptive Network for Dynamic Cold-start Scenes Optimization in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用场景关系图+用户感知相似度注意力为动态冷启动场景迁移先验，并以可微门控 MoE 自适应分配专家，线上 CTR +5.64%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR预估与特征交叉&quot;,&quot;重排&quot;,&quot;后处理概率校准&quot;,&quot;Top-N 排序偏差&quot;]"><strong><a href="ctr_feature_interaction/calibrating_the_predictions_for_top_n_recommendations/">Calibrating the Predictions for Top-N Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 Top-N 列表内存在排名相关的过/低估偏差，提出秩分组+秩折扣加权的通用校准优化方法与 ECE@N/RDECE@N 指标，KuaiRec 上 ECE@20 最高降 89%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;实时场景语义理解&quot;,&quot;跨模态对比对齐&quot;,&quot;LLM 离线推理&quot;]"><strong><a href="ctr_feature_interaction/larr_large_language_model_aided_real_time_scene_recommendation_with_semanti/">LARR: Large Language Model Aided Real-time Scene Recommendation with Semantic Understanding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>三阶段将 LLM 转为 POI/用户文本嵌入器，用双向编码器聚合实时场景语义并与协同信号对齐，离线化推理使美团外卖 CTR AUC +0.58%，线上 CTR +2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;FwFM&quot;,&quot;低秩分解&quot;,&quot;低延迟推理&quot;]"><strong><a href="ctr_feature_interaction/low_rank_field_weighted_factorization_machines_for_low_latency_item_recomme/">Low Rank Field-Weighted Factorization Machines for Low Latency Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以对角+低秩重参数化 FwFM 字段交互矩阵，上下文缓存使逐物品推断降至 $O(\rho|I|k)$，同参数下优于剪枝并降低线上延迟。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;拆分学习&quot;,&quot;标签差分隐私&quot;,&quot;LoRA&quot;]"><strong><a href="ctr_feature_interaction/privacy_preserving_conversion_modeling_in_data_clean_room/">Privacy Preserving Conversion Modeling in Data Clean Room</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>数据洁净室内以批量聚合梯度+LoRA 适配器+标签 DP 去偏实现隐私合规的 CVR 联合训练，ROC-AUC 仅降 0.2% 而通信开销降 4 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;服务时在线学习&quot;,&quot;代理评估器&quot;,&quot;电商推荐&quot;]"><strong><a href="multi_task_reranking/do_not_wait_learning_re_ranking_model_without_user_feedback_at_serving_time/">Do Not Wait: Learning Re-Ranking Model Without User Feedback At Serving Time in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用代理评估器在服务时刻为每个请求在线生成瞬态参数修正，无需等待延迟反馈即提升重排效果，离线 NDCG@10 +0.65%、线上购买数 +2.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;统一上下文建模&quot;,&quot;特征交叉&quot;,&quot;个性化搜索&quot;]"><strong><a href="multi_task_reranking/joint_modeling_of_search_and_recommendations_via_an_unified_contextual_reco/">Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Netflix 以 {user, query, country, source entity, task} 为共享上下文训练单一深度排序器，插补缺失上下文并做特征交叉，统一服务搜索与推荐，个性化后离线指标 +7%/+10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;跨内容类型排序&quot;,&quot;曝光预算与多样性&quot;,&quot;线上 A/B 测试&quot;]"><strong><a href="multi_task_reranking/ranking_across_different_content_types_the_robust_beauty_of_multinomial_ble/">Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式混合(MB)在打分后按内容类型采样组列表，实现可解释、可重训稳定的跨内容类型曝光预算，Amazon Music 线上 A/B 播客时长 +18.82% 且整体互动 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Hybrid Targets&quot;,&quot;梯度调节&quot;,&quot;标签嵌入&quot;]"><strong><a href="multi_task_reranking/touch_the_core_exploring_task_dependence_among_hybrid_targets_for_recommend/">Touch the Core: Exploring Task Dependence Among Hybrid Targets for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次提出混合目标（离散转化+连续核心值）多任务学习，用标签嵌入与信息融合显式传递依赖，并设计梯度方向/量级调节策略，离线三数据集与线上 A/B 均显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文组合赌博机 (CCB)&quot;,&quot;离线策略评估 (OPE)&quot;,&quot;奖励分解&quot;]"><strong><a href="rl_bandit_rec/effective_off_policy_evaluation_and_learning_in_contextual_combinatorial_ba/">Effective Off-Policy Evaluation and Learning in Contextual Combinatorial Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>因子化动作空间 + 主/残差效应分解的 OPCB 估计器，仅对主动作重要性采样、残差回归，KuaiRec 上 MSE 较 IPS 降 92%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/融合权重决策&quot;,&quot;Off-Policy Learning&quot;,&quot;Counterfactual Risk Minimisation&quot;,&quot;Effective Sample Size&quot;,&quot;Multi-Objective Scalarisation&quot;]"><strong><a href="rl_bandit_rec/multi_objective_recommendation_via_multivariate_policy_learning/">Multi-Objective Recommendation via Multivariate Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多目标融合权重视为连续动作，用 CRM/SNIPS 学习策略，提出 ESS 校正提升小样本置信区间覆盖率，线上 A/B 显著提升留存与时长。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;排序策略学习（离线评估）&quot;,&quot;上下文赌博机&quot;,&quot;逆倾向得分&quot;,&quot;基线校正&quot;]"><strong><a href="rl_bandit_rec/optimal_baseline_corrections_for_off_policy_contextual_bandits/">Optimal Baseline Corrections for Off-Policy Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 IPS/SNIPS/BanditNet/DR 统一为 β-IPS 基线校正，闭式求方差最优 β，在 OBP 与 ZOZOTOWN 上降低 OPE 误差并加速 OPL。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线列表探索&quot;,&quot;Engagement-Utility 错位&quot;,&quot;流行度偏差&quot;]"><strong><a href="rl_bandit_rec/the_fault_in_our_recommendations_on_the_perils_of_optimizing_the_measurable/">The Fault in Our Recommendations: On the Perils of Optimizing the Measurable</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用重复选择模型揭示 engagement 与 utility 错位，提出 PEAR 后验探索；δ=0.99 时 utility +53.4% 仅损 engagement 0.12%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;离线评估/策略学习&quot;,&quot;逆倾向得分 IPS&quot;,&quot;反事实方差缩减&quot;]"><strong><a href="rl_bandit_rec/text_rm_ope_off_policy_estimation_with_pairs_of_policies/">Δ-OPE: Off-Policy Estimation with Pairs of Policies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 off-policy 评估从单策略估值改为成对策略差值估计，用加性控制变量 Δβ-IPS 显著降方差，仿真与线上 A/B 双验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;多任务学习&quot;,&quot;教师偏差泄漏&quot;]"><strong><a href="debias_fairness_industrial/bridging_the_gap_unpacking_the_hidden_challenges_in_knowledge_distillation_/">Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>针对推荐蒸馏中的数据漂移、教师选型与多学生标签共享三大工业挑战，提出持续更新教师 + 分离 logits 的辅助蒸馏框架，线上互动率最高提升 1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;post-filter&quot;,&quot;multi-sided fairness&quot;,&quot;POI recommendation&quot;]"><strong><a href="debias_fairness_industrial/capri_fair_integration_of_multi_sided_fairness_in_contextual_poi_recommenda/">CAPRI-FAIR: Integration of Multi-sided Fairness in Contextual POI Recommendation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>免训练后置重排同时注入POI提供者与消费者公平，在Yelp/Gowalla上以Precision与GCE评测权衡长尾曝光和不活跃用户精度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/候选集排序&quot;,&quot;新闻推荐数据集&quot;,&quot;Beyond-Accuracy 评估&quot;,&quot;性别导向偏置&quot;]"><strong><a href="debias_fairness_industrial/eb_nerd_a_large_scale_dataset_for_news_recommendation/">EB-NeRD: A Large-Scale Dataset for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>发布百万用户丹麦新闻推荐数据集与 RecSys&#x27;24 Challenge 基准，通过打乱位置的曝光日志揭示流行度偏置与性别导向差异。</p>
</dd>
<dt data-paper-tags="[&quot;AI Magazine&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/前端解释层&quot;,&quot;可解释推荐&quot;,&quot;音乐推荐系统&quot;,&quot;知识图谱多跳解释&quot;]"><strong><a href="debias_fairness_industrial/explainability_in_music_recommender_systems/">Explainability in Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>立场综述：把 RS 解释目标与 XAI 目标对齐，梳理特征/样例/知识图谱三类解释及用户研究与离线指标双轨评估，并剖析工业 MRS 的‘简洁 vs 复杂’矛盾。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平与可信隐私&quot;,&quot;解释层/决策支持&quot;,&quot;多利益相关方推荐&quot;,&quot;职业知识图谱&quot;,&quot;可解释推荐&quot;]"><strong><a href="debias_fairness_industrial/explainable_multi_stakeholder_job_recommender_systems/">Explainable Multi-Stakeholder Job Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>招聘推荐三方利益相关方解释偏好研究 + 职业知识图谱注意力 GNN（OKRA），发现用户无法区分真实/随机解释，主张转向决策支持式解释。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;后处理(图数据增广)&quot;,&quot;消费者公平性 (Consumer Fairness)&quot;,&quot;图协同过滤增广&quot;,&quot;采样策略&quot;]"><strong><a href="debias_fairness_industrial/fair_augmentation_for_graph_collaborative_filtering/">Fair Augmentation for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>复现并形式化“公平图增广”后处理范式，推导 3 类新采样策略，在 5 数据集×16 模型上揭示大图高效用模型有效、公平知识不可迁移。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边匹配市场&quot;,&quot;嫉妒自由&quot;,&quot;Nash Social Welfare&quot;]"><strong><a href="debias_fairness_industrial/fair_reciprocal_recommendation_in_matching_markets/">Fair Reciprocal Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用NSW与Frank-Wolfe交替优化双边互惠推荐，在保持匹配数同时近零嫉妒，合成/真实约会数据验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;上下文 Bandit&quot;,&quot;贝叶斯决策论&quot;,&quot;变分推断&quot;]"><strong><a href="debias_fairness_industrial/position_paper_why_the_shooting_in_the_dark_method_dominates_recommender_sy/">Position Paper: Why the Shooting in the Dark Method Dominates Recommender Systems Practice; A Call to Abandon Anti-Utopian Thinking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>论证工业界「离线代理指标盲选 + A/B 验证」缺乏理论依据，主张用定制似然与变分贝叶斯直接瞄准奖励优化。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;线上评测与实验平台&quot;,&quot;A/B 测试&quot;,&quot;代理指标与统计功效&quot;,&quot;多重假设检验校正&quot;]"><strong><a href="debias_fairness_industrial/powerful_a_b_testing_metrics_and_where_to_find_them/">Powerful A/B-Testing Metrics and Where to Find Them</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用数百个历史 A/B 实验构建指标校验管线，量化 I/II/III 类错误与统计功效；联合 DAU/Engagers/TimeSpent 在 95% 置信下相对降低 II 类错误 35%，或把所需样本量缩小 3.5 倍。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;计算社会选择&quot;,&quot;异构公平性&quot;,&quot;公平感知重排&quot;]"><strong><a href="debias_fairness_industrial/social_choice_for_heterogeneous_fairness_in_recommendation/">Social Choice for Heterogeneous Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用计算社会选择的多智能体架构 SCRUF-D 将三类异构提供方公平性定义分别交由独立 Agent，在分配—聚合两阶段中实现多目标公平重排，且几乎不损 nDCG。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部12篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
