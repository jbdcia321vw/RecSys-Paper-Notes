---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2021 推荐系统论文解读

20 篇 ACL / EMNLP (RecSys & CRS) 2021 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** EMNLP 2021 ×6 · 端到端生成式 ×5 · EMNLP ×3 · ACL ×2 · ACL 2021 ×2 · 新闻推荐 ×2 · 联邦学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/粗排" aria-pressed="false">召回/粗排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP-Findings&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;Aspect-Sentiment Pair&quot;,&quot;Review-Based Rating Prediction&quot;]"><strong><a href="collaborative_filtering/recommend_for_a_reason_unlocking_the_power_of_unsupervised_aspect_sentiment/">Recommend for a Reason: Unlocking the Power of Unsupervised Aspect-Sentiment Co-Extraction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>无监督抽取（方面,情感）对并用双通道注意力-属性估计器做评分预测，7 个 Amazon 数据集 MSE 相对降 1.3%–4.1%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;抽取式摘要&quot;,&quot;BERT&quot;,&quot;K-Means&quot;]"><strong><a href="collaborative_filtering/unsupervised_extractive_summarization_based_representations_for_accurate_an/">Unsupervised Extractive Summarization-Based Representations for Accurate and Explainable Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 BERT 句向量+K-Means 抽取评论摘要，统一用户/物品表征与解释；Amazon 5-core 评分预测平均 RMSE 0.9210，较最佳基线降 0.52%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NAACL&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱路径推理&quot;,&quot;神经逻辑规则&quot;,&quot;解释忠实性&quot;]"><strong><a href="graph_rec/faithfully_explainable_recommendation_via_neural_logic_reasoning/">Faithfully Explainable Recommendation via Neural Logic Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用马尔可夫逻辑规则+EM 迭代联合训练 KG 编码器与规则权重，并以个性化规则分引导路径推理，兼顾推荐精度与解释忠实度（JS 散度、用户调研双验证）。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021 Findings&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/粗排&quot;,&quot;新闻推荐&quot;,&quot;跨选择门控&quot;,&quot;层次聚类图&quot;]"><strong><a href="graph_rec/neural_news_recommendation_with_collaborative_news_encoding_and_structural_/">Neural News Recommendation with Collaborative News Encoding and Structural User Encoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 CNE+SUE，用标题-内容跨选择/交叉注意力和层次聚类图编码用户历史；在 200K-MIND 的 AUC 69.55，较最佳基线 +0.92。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化主题注意力&quot;,&quot;神经话题模型&quot;,&quot;用户-标签互动预测&quot;]"><strong><a href="sequential_rec/howyoutagtweets_learning_user_hashtagging_preferences_via_personalized_topi/">#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 BERTweet+神经话题模型联合训练并对话题标签上下文做用户查询式个性化主题注意力，预测用户未来是否会使用某话题标签；全局时间切分、全库排序下 MAP 0.3114 超越 LSTM-ATT 12.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;层级兴趣建模&quot;,&quot;多粒度兴趣匹配&quot;]"><strong><a href="sequential_rec/hierec_hierarchical_user_interest_modeling_for_personalized_news_recommenda/">HieRec: Hierarchical User Interest Modeling for Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用「子话题—话题—用户」三级兴趣树替代单一用户向量，并按候选新闻所属话题/子话题做多粒度分层匹配，MIND 上 AUC 相对提升 3.1%、nDCG@5 提升 7.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;无序集合生成&quot;,&quot;BERT [MASK] 生成&quot;,&quot;多模态早融合&quot;]"><strong><a href="generative_rec/leveraging_order_free_tag_relations_for_context_aware_recommendation/">Leveraging Order-Free Tag Relations for Context-Aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>SOG 以 BERT 编码器 + [MASK] 迭代回填实现无序标签集合生成，配合 1-to-M KL 训练与贪心解码，在 Instagram / Stack Overflow 全标签库排序下 P@1 提升 24.1% / 8.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮对话式推荐（端到端生成）&quot;,&quot;树状知识图谱推理&quot;,&quot;对话动作（Dialog Act）序列化&quot;]"><strong><a href="agent_conversational_rec/cr_walker_tree_structured_graph_reasoning_and_dialog_acts_for_conversationa/">CR-Walker: Tree-Structured Graph Reasoning and Dialog Acts for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 walker cell 在 KG 上做两跳树状推理，并把推理树前序线性化为对话动作来约束 GPT-2 生成；ReDial 覆盖率 14.7、知识 F1 47.7，人工评测四项全面超越人类参考答案。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;知识图谱多跳推理&quot;,&quot;强化学习策略搜索&quot;,&quot;路径片段融合&quot;]"><strong><a href="agent_conversational_rec/crfr_improving_conversational_recommender_systems_via_flexible_fragments_re/">CRFR: Improving Conversational Recommender Systems via Flexible Fragments Reasoning on Knowledge Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用强化学习在物品/概念双知识图谱上做显式多跳推理，以柔性路径片段而非完整路径增强推荐与回复生成，ReDial R@10 相对提升 10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;系统与工具链(端到端)&quot;,&quot;模块化解耦框架&quot;,&quot;三子任务联合建模&quot;,&quot;统一评测协议&quot;]"><strong><a href="agent_conversational_rec/crslab_an_open_source_toolkit_for_building_conversational_recommender_syste/">CRSLab: An Open-Source Toolkit for Building Conversational Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个开源 CRS 工具链：将推荐/对话/策略三子任务统一形式化，提供四级解耦数据流、6 数据集、18 模型与全栈自动评测接口。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;双语平行语料&quot;,&quot;跨语言对话推荐&quot;]"><strong><a href="agent_conversational_rec/durecdial_2_0_a_bilingual_parallel_corpus_for_conversational_recommendation/">DuRecDial 2.0: A Bilingual Parallel Corpus for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首个中英平行对话推荐语料，定义单语/多语/跨语五任务，验证英文数据可反向增益中文推荐。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回+重排&quot;,&quot;批评到偏好改写 (Critique Interpretation)&quot;,&quot;少样本提示与观点检索&quot;]"><strong><a href="agent_conversational_rec/it_doesn_t_look_good_for_a_date_transforming_critiques_into_preferences_for/">It doesn&#x27;t look good for a date: Transforming Critiques into Preferences for Conversational Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GPT-3 few-shot 把自由批评改写为正向偏好，再用 USE 检索 / BERT 重排评论论据；检索 Precision@1 提升 124%，重排 nDCG@1 提升 19–59%。</p>
</dd>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;多子目标规划&quot;,&quot;知识图谱&quot;,&quot;知识增强生成&quot;]"><strong><a href="agent_conversational_rec/kers_a_knowledge_enhanced_framework_for_recommendation_dialog_systems_with_/">KERS: A Knowledge-Enhanced Framework for Recommendation Dialog Systems with Multiple Subgoals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以子目标规划筛选知识，并用顺序注意力、知识门控与词典增强抑制噪声，DuRecDial 上 F1 与知识 F1 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;神经模板槽位填充&quot;,&quot;充分多注意力物品选择&quot;,&quot;新物品零样本&quot;]"><strong><a href="agent_conversational_rec/learning_neural_templates_for_recommender_dialogue_system/">Learning Neural Templates for Recommender Dialogue System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>解耦对话生成与物品推荐，用 [ITEM] 槽位模板 + 富信息注意力选择器填充，ReDial 上 ReR@1 与 Dist-4 大幅刷新 SOTA 并支持新物品零样本推荐。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;可解释推荐&quot;,&quot;个性化文本生成&quot;,&quot;多任务学习&quot;]"><strong><a href="agent_conversational_rec/personalized_transformer_for_explainable_recommendation/">Personalized Transformer for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用“上下文预测”辅助任务把用户/物品 ID 映射到解释词空间，使 2 层无预训练 Transformer 兼顾个性化解释生成与评分预测，BLEU-4 超微调 BERT 与 NETE 约 24–28%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL Findings 2021&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式（对话推荐+回复生成）&quot;,&quot;情感感知评论检索&quot;,&quot;评论注意力解码器&quot;]"><strong><a href="agent_conversational_rec/revcore_review_augmented_conversational_recommendation/">RevCore: Review-augmented Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用情感一致评论检索增强REDIAL对话推荐画像与回复生成；R@1 6.1、Dist-3 0.558。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;时变流行度预测&quot;,&quot;流行度去偏&quot;,&quot;冷启动用户&quot;]"><strong><a href="cold_start_rec/pp_rec_news_recommendation_with_personalized_user_interest_and_time_aware_n/">PP-Rec: News Recommendation with Personalized User Interest and Time-aware News Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将时变新闻流行度分数与个性化匹配分数门控融合，用内容+时效+实时CTR统一预测流行度并对用户行为做流行度去偏，MSN/Feeds 上 MRR 最高提升 22.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;Findings of EMNLP 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回/粗排&quot;,&quot;三塔相关性建模&quot;,&quot;竞价图&quot;,&quot;解耦图编码&quot;]"><strong><a href="ctr_feature_interaction/leveraging_bidding_graphs_for_advertiser_aware_relevance_modeling_in_sponso/">Leveraging Bidding Graphs for Advertiser-Aware Relevance Modeling in Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将竞价图引入 query-keyword-advertiser 三塔相关性建模，ROC-AUC 0.801 较 TextGNN 提升 0.4 个点。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/点击预估&quot;,&quot;联邦学习&quot;,&quot;安全聚合&quot;,&quot;新闻推荐&quot;]"><strong><a href="debias_fairness_industrial/efficient_fedrec_efficient_federated_learning_framework_for_privacy_preserv/">Efficient-FedRec: Efficient Federated Learning Framework for Privacy-Preserving News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>拆分服务器大新闻模型与端侧轻量用户模型，仅传新闻表示和安全聚合梯度，端侧通信降至2.18M且Adressa AUC达79.08。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP Findings&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回+精排两阶段&quot;,&quot;联邦学习&quot;,&quot;本地差分隐私&quot;]"><strong><a href="debias_fairness_industrial/uni_fedrec_a_unified_privacy_preserving_news_recommendation_framework_for_m/">Uni-FedRec: A Unified Privacy-Preserving News Recommendation Framework for Model Training and Online Serving</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>联邦多兴趣召回+客户端本地精排：兴趣分解-聚合+LDP 噪声，MIND R@100 +27.7%，隐私泄漏率 17.0%→0.55%。</p>
</dd>
</dl>
</section>
</div>
