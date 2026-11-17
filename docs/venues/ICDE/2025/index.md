---
hide:
  - toc
---

# 🏛️ ICDE 2025 推荐系统论文解读

7 篇 ICDE 2025 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 大语言模型推荐 ×3

<div class="conf-index">
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl>
<dt><strong><a href="graph_rec/path_based_summary_explanations_for_graph_recommenders_extended_version/">Path-based summary explanations for graph recommenders (extended version)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次提出路径解释的摘要化：用 Steiner Tree 与 Prize-Collecting Steiner Tree 将多条用户-物品解释路径压缩为连通子图，在可理解性、多样性、隐私与可扩展性上全面优于逐条路径基线。</p>
</dd>
</dl>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl>
<dt><strong><a href="llm_rec/darec_a_disentangled_alignment_framework_for_large_language_model_and_recom/">DaRec: A Disentangled Alignment Framework for Large Language Model and Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从信息论证明零间隙对齐次优，提出解耦共享/特有表征+全局局部结构对齐的即插即用框架，全库排序下 Recall@20 最高提升 5.33%。</p>
</dd>
<dt><strong><a href="llm_rec/delrec_distilling_sequential_pattern_to_enhance_llms_based_sequential_recom/">DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用软提示两阶段蒸馏传统序列模型的时间感知与推荐模式，再微调 LLM 做序列推荐，四数据集 HR/NDCG 全面超越 SOTA。</p>
</dd>
<dt><strong><a href="llm_rec/making_large_language_models_better_knowledge_miners_for_online_marketing_w/">Making Large Language Models Better Knowledge Miners for Online Marketing with Progressive Prompting Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用渐进式提示增强把 LLM 变成可控的营销知识图谱挖掘器，并以教师蒸馏出可部署的 LightPAIR，受众定向覆盖提升 15%–122%。</p>
</dd>
</dl>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl>
<dt><strong><a href="cold_start_rec/all_in_one_heterogeneous_interaction_modeling_for_cold_start_rating_predict/">All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HIRE 用纯 MHSA 在采样的预测上下文上端到端学习用户/物品/属性三层异构交互，三种冷启动场景下 Precision/NDCG/MAP 平均超 12 个基线 0.21/0.29/0.22。</p>
</dd>
</dl>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl>
<dt><strong><a href="cross_domain_multi_scenario/personalized_multi_interest_modeling_for_cross_domain_recommendation_to_col/">Personalized Multi-Interest Modeling for Cross-Domain Recommendation to Cold-Start Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用归一化流增强神经过程，将单峰高斯先验转为多峰分布以建模用户多兴趣，并引入偏好池捕获跨用户共同偏好，五场景 MAE 平均降约 45%。</p>
</dd>
</dl>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl>
<dt><strong><a href="ctr_feature_interaction/graphex_a_graph_based_extraction_method_for_advertiser_keyphrase_recommenda/">GraphEx: A Graph-based Extraction Method for Advertiser Keyphrase Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用二分图从商品标题抽取关键词排列，绕开点击数据偏差，CPU 毫秒级推理，线上广告收入提升 8.3%。</p>
</dd>
</dl>
</div>
