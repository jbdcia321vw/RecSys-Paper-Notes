---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2024 推荐系统论文解读

18 篇 ACL / EMNLP (RecSys & CRS) 2024 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** ACL ×5 · 大语言模型推荐 ×5 · 端到端生成式 ×5 · ACL 2024 ×3 · 生成式端到端 ×3 · EMNLP ×2 · EMNLP 2024 ×2 · LLM4Rec ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式端到端" aria-pressed="false">生成式端到端<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指令微调" aria-pressed="false">指令微调<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;协同过滤与双塔召回&quot;,&quot;排序&quot;,&quot;模块化新闻推荐&quot;,&quot;监督对比学习&quot;,&quot;方面定制&quot;]"><strong><a href="collaborative_filtering/train_once_use_flexibly_a_modular_framework_for_multi_aspect_neural_news_re/">Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用监督对比学习训练内容/方面专用新闻编码器，推理时线性组合相似度实现按需个性化与多样化，MIND上nDCG@10相对提升2.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL_EMNLP&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;重排&quot;,&quot;Denoising Attention&quot;,&quot;Query-aware User Modeling&quot;]"><strong><a href="self_supervised_denoising/denoising_attention_for_query_aware_user_modeling_in_personalized_search/">Denoising Attention for Query-aware User Modeling in Personalized Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ACL/EMNLP 2024 [自监督对比学习与交互去噪] 以有界对齐+阈值移位+ReLU 门控与 max-归一化重构查询感知用户建模，Top-1000 重排协议下 MAP 较最佳注意力基线提升 14–17%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;生成式端到端&quot;,&quot;LLM 参数高效微调 (LoRA/QLoRA)&quot;,&quot;Mixup 混合检索&quot;,&quot;服饰电商冷启动&quot;]"><strong><a href="sequential_rec/sequential_llm_framework_for_fashion_recommendation/">Sequential LLM Framework for Fashion Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以推荐专用 Prompt + 搜索记忆 + LoRA 微调的 Falcon-7B 生成候选，再用 ID/标题双通道 Mixup 检索排序；Leave-One-Out 全库排序下 Clothing 类目 Recall@10 较 CORE 提升 30.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端&quot;,&quot;LLM in-context optimizer&quot;,&quot;Explore-Exploit&quot;]"><strong><a href="generative_rec/generative_explore_exploit_training_free_optimization_of_generative_recomme/">Generative Explore-Exploit: Training-free Optimization of Generative Recommender Systems using LLM Optimizers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ACL 2024 主会精准笔记：免微调、用CTR反馈+LLM in-context优化的生成式推荐，探索-利用双语料生成让CTR随迭代持续上升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NAACL 2024 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;指令微调&quot;,&quot;掩码物品建模&quot;,&quot;贝叶斯个性化排序&quot;]"><strong><a href="llm_rec/aligning_large_language_models_with_recommendation_knowledge/">Aligning Large Language Models with Recommendation Knowledge</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 MIM/MLM/BPR 经典算子改写为自然语言辅助任务微调 LLM，配合去 user-ID 的精简推荐提示，在 Amazon 三域全库检索上超越 TIGER 达 19%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;高阶交互&quot;,&quot;全词嵌入&quot;]"><strong><a href="llm_rec/enhancing_high_order_interaction_awareness_in_llm_based_recommender_model/">Enhancing High-order Interaction Awareness in LLM-based Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用LightGCN随机特征传播生成图感知全词嵌入并注入T5，配合免训练重排；Leave-One-Out下直接与序列推荐均显著领先。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;召回/打分&quot;,&quot;提示工程&quot;,&quot;文本增强&quot;]"><strong><a href="llm_rec/llm_rec_personalized_recommendation_via_prompting_large_language_models/">LLM-Rec: Personalized Recommendation via Prompting Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用四类提示让LLM增强物品文本，拼接原始描述嵌入并以轻量MLP打分；在MovieLens-1M/Recipe的1:1000采样排序上NDCG@10相对MLP提升8.54%/21.72%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;理由蒸馏&quot;,&quot;提示蒸馏&quot;]"><strong><a href="llm_rec/rdrec_rationale_distillation_for_llm_based_recommendation/">RDRec: Rationale Distillation for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Llama-2-7b + CoT 从评论蒸馏「用户偏好/物品属性」双句理由，作为第四个生成任务注入 T5-small 的 POD 提示蒸馏框架；Leave-One-Out 生成式全库排序下 Top-N 提升 12–42%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;领域持续预训练&quot;,&quot;指令微调&quot;,&quot;文本化物品表征&quot;]"><strong><a href="llm_rec/recgpt_generative_pre_training_for_text_based_recommendation/">RecGPT: Generative Pre-training for Text-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个领域自适应全量预训练的 7B 推荐 LLM（20.5B token 推荐语料持续预训练 + 10 万指令微调），以语义相似度对齐解码结果，在 4 个基准上刷新评分预测与序列推荐 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;LLM4Rec&quot;,&quot;后验解释生成&quot;,&quot;可解释推荐&quot;,&quot;混合专家适配器&quot;,&quot;复现研究&quot;]"><strong><a href="llm_rec/reproducibility_study_of_xrec_large_language_models_for_explainable_recomme/">Reproducibility Study of \&quot;XRec: Large Language Models for Explainable Recommendation\&quot;</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 Llama-3 复现 XRec 并做 MoE 反事实消融，证实协同注入提分，但指出其并非全面超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;LLM推荐&quot;,&quot;评分预测/精排&quot;,&quot;自然语言用户画像&quot;,&quot;可审视推荐&quot;]"><strong><a href="llm_rec/transparent_and_scrutable_recommendations_using_natural_language_user_profi/">Transparent and Scrutable Recommendations Using Natural Language User Profiles</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>LLM从评论生成可编辑NL画像，仅以画像微调GPT-2评分预测；warm-start两数据集接近MF/PETER+并验证画像编辑可改推荐。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2024 Findings&quot;,&quot;大语言模型推荐 (LLM4Rec)&quot;,&quot;解释生成/推荐后处理&quot;,&quot;协同指令微调&quot;,&quot;Mixture-of-Experts Adapter&quot;]"><strong><a href="llm_rec/xrec_large_language_models_for_explainable_recommendation/">XRec: Large Language Models for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出模型无关 XRec，用 LightGCN 协同分词器与 MoE 适配器把协同信号注入 LLM 各层，在三数据集语义解释指标上显著优于 ID-based 基线且 USR≈1。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;可信解释&quot;,&quot;自反思精炼&quot;,&quot;LLM4Rec&quot;]"><strong><a href="agent_conversational_rec/beyond_persuasion_towards_conversational_recommender_system_with_credible_e/">Beyond Persuasion: Towards Conversational Recommender System with Credible Explanations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>免训练两阶段 PC-CRS：可信感知说服策略引导生成 + 自反思迭代精炼，模拟器评估下可信度 +8.17%、说服力 +5.07%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端（对话-推荐联合建模）&quot;,&quot;多通道超图&quot;,&quot;多兴趣自监督学习&quot;,&quot;马太效应缓解&quot;]"><strong><a href="agent_conversational_rec/mitigating_matthew_effect_multi_hypergraph_boosted_multi_interest_self_supe/">Mitigating Matthew Effect: Multi-Hypergraph Boosted Multi-Interest Self-Supervised Learning for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以「物品/实体/词 × group/joint/purchase」九路 motif 超图学多层级多兴趣并经 InfoNCE 对齐，REDIAL 上 R@10 提升 11.5% 且覆盖率最高、推荐流行度最低。</p>
</dd>
<dt data-paper-tags="[&quot;NAACL 2024 (Findings)&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;Self-Inspiring Planning&quot;,&quot;LLM Agent + Tool Use&quot;]"><strong><a href="agent_conversational_rec/recmind_large_language_model_powered_agent_for_recommendation/">RecMind: Large Language Model Powered Agent For Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出免训练 LLM 推荐智能体 RecMind，以 Self-Inspiring 规划保留全部历史推理分支，外挂个性化记忆与 SQL/Search/摘要工具；Amazon/Yelp 上 MAE 0.6892 超越全训练 P5，直接推荐 HR@5 较最强提示基线 +24.7%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2024 Industry&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;召回&quot;,&quot;Fusion-in-Decoder&quot;,&quot;Context Glancing&quot;]"><strong><a href="ctr_feature_interaction/improving_retrieval_in_sponsored_search_by_leveraging_query_context_signals/">Improving Retrieval in Sponsored Search by Leveraging Query Context Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用Web结果与GPT-4 Query Profile增强查询，FiD融合dense/NAR检索，P@100提升19.9%，在线收入+1.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Pointwise-Pairwise&quot;,&quot;RTL Rank Aggregation&quot;,&quot;Learning-to-Rank&quot;]"><strong><a href="multi_task_reranking/efficient_pointwise_pairwise_learning_to_rank_for_news_recommendation/">Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GLIMPSE 以单次多任务生成式微调同时习得点式相关性与成对偏好，再用 O(K) 右到左相邻交换精排，在 MIND/Adressa 全库排序上超越 UniTRec 等 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;可信隐私与工业架构&quot;,&quot;排序/重排&quot;,&quot;文本攻击&quot;,&quot;隐蔽性评估&quot;]"><strong><a href="debias_fairness_industrial/stealthy_attack_on_large_language_model_based_recommendation/">Stealthy Attack on Large Language Model based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示LLM推荐对测试期标题文本攻击的脆弱性，黑盒改写可使目标曝光最高提升520%，且整体NDCG与语义几乎不变。</p>
</dd>
</dl>
</section>
</div>
