---
hide:
  - toc
---

# 🏛️ WSDM 2024 推荐系统论文解读

24 篇 WSDM 2024 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐与内容理解" aria-pressed="false">多模态推荐与内容理解<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;属性模拟&quot;,&quot;物品共现矩阵&quot;,&quot;多兴趣建模&quot;]"><strong><a href="collaborative_filtering/attribute_simulation_for_item_embedding_enhancement_in_multi_interest_recom/">Attribute Simulation for Item Embedding Enhancement in Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品共现矩阵近似物品-属性关联矩阵，提出即插即用的 SimEmb 模块替代 ID Embedding 做多兴趣召回，4 个数据集全库排序 Recall@20 平均相对提升 25.59%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Embedding 尺寸搜索&quot;,&quot;神经架构搜索&quot;,&quot;内存预算约束&quot;]"><strong><a href="collaborative_filtering/budgeted_embedding_table_for_recommender_systems/">Budgeted Embedding Table For Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>BET 用「预算受限的表级动作采样」取代逐实例尺寸搜索，并以 DeepSets 集合表征代理评估动作适应度，在 80–95% 三档稀疏率下稳定超越 CIESS 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;维度坍缩&quot;,&quot;率失真编码长度&quot;,&quot;非对比学习&quot;]"><strong><a href="collaborative_filtering/towards_mitigating_dimensional_collapse_of_representations_in_collaborative/">Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以率失真编码长度替代 InfoNCE，消除维度坍缩，免增广免负采样，ML10M Recall@10 提升 14.21%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱扩散&quot;,&quot;对比学习去噪&quot;]"><strong><a href="graph_rec/diffkg_knowledge_graph_diffusion_model_for_recommendation/">DiffKG: Knowledge Graph Diffusion Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KG 扩散生成任务相关子图，并以协同卷积与对比增广去噪，在三个数据集 Full-Ranking 下 Recall@20 最高提升 43.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱传播&quot;,&quot;多样性-精度权衡&quot;,&quot;对齐与均匀性&quot;]"><strong><a href="graph_rec/knowledge_graph_context_enhanced_diversified_recommendation/">Knowledge Graph Context-Enhanced Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KG-Diverse，以实体/关系覆盖度为细粒度多样性度量，用关系感知 KG 传播、多样性感知用户聚合与条件对齐-均匀性正则，在三个数据集上大幅提升覆盖度而精度损失有限。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/预训练&quot;,&quot;任务超图&quot;,&quot;过渡注意力&quot;]"><strong><a href="graph_rec/unified_pretraining_for_recommendation_via_task_hypergraphs/">Unified Pretraining for Recommendation via Task Hypergraphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多类预训练任务统一为任务超图超边预测，过渡注意力判别迁移至推荐，全库排序 Recall@10 最高提升 107%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序评测&quot;,&quot;模块化基准平台&quot;,&quot;自监督数据增广&quot;]"><strong><a href="self_supervised_denoising/sslrec_a_self_supervised_learning_framework_for_recommendation/">SSLRec: A Self-Supervised Learning Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出统一模块化 SSL 推荐基准，覆盖五大场景约30个模型，并以全库排序协议复现评测。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图对比学习&quot;,&quot;跨子序列监督&quot;]"><strong><a href="sequential_rec/intent_contrastive_learning_with_cross_subsequences_for_sequential_recommen/">Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用同一目标物品的跨序列子序列构造粗/细粒度意图对比监督，全库排序 NDCG@20 最高提升 44.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;线性递归单元&quot;,&quot;递归并行化&quot;,&quot;增量推理&quot;]"><strong><a href="sequential_rec/linear_recurrent_units_for_sequential_recommendation/">Linear Recurrent Units for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用线性递归单元替代自注意力，递归并行训练与O(1)增量推理，全库排序平均提升4.39%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;机器遗忘&quot;,&quot;会话分片聚类&quot;,&quot;隐私保护&quot;]"><strong><a href="sequential_rec/on_the_effectiveness_of_unlearning_in_session_based_recommendation/">On the Effectiveness of Unlearning in Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 SRU：相似会话聚片 + 会话-质心双源注意力聚合 + 协同/邻居/随机额外删除，并定义 HIT@K 遗忘有效性度量，遗忘重训提速 3.7× 且精度反超 SISA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;LLM 内容编码器&quot;,&quot;提示数据增强&quot;,&quot;部分冻结与 LoRA&quot;]"><strong><a href="llm_rec/once_boosting_content_based_recommendation_with_both_open_and_closed_source/">ONCE: Boosting Content-based Recommendation with Both Open- and Closed-source Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用开源 LLaMA 微调做内容编码器 (DIRE)、闭源 GPT-3.5 提示做数据增强 (GENRE)，MIND/Goodreads 采样排序下 AUC 最高 +11.1%，N@5 最高 +24.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;生成模型推荐&quot;,&quot;GRE-Score&quot;,&quot;提示词-模型检索&quot;]"><strong><a href="multimodal_rec/gemrec_towards_generative_model_recommendation/">GEMRec: Towards Generative Model Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 Prompt-Model 检索 + 生成模型排序两阶段框架，开源 GEMRec-18K（90 提示词×200 模型×18K 图像），用 GRE-Score 融合 CLIP 相关性、成对多样性与流行度完成候选预排序。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排&quot;,&quot;逆对偶损失&quot;,&quot;逆梯度元学习&quot;]"><strong><a href="multimodal_rec/inverse_learning_with_extremely_sparse_feedback_for_recommendation/">Inverse Learning with Extremely Sparse Feedback for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用逆对偶损失对无标签样本做双侧自动标注，再用元学习逆梯度仲裁更新方向，Micro Video/ML1M 上 AUC 最高提升 9.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排（训练标签生成层）&quot;,&quot;双层优化与元学习&quot;,&quot;时长去偏&quot;,&quot;可微 Top-k&quot;]"><strong><a href="multimodal_rec/labelcraft_empowering_short_video_recommendations_with_automated_label_craf/">LabelCraft: Empowering Short Video Recommendations with Automated Label Crafting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将短视频标签生成建模为双层优化，用 MAML 式元学习以平台目标（观看时长/互动/留存）反传梯度训练可微标注模型，Kuaishou/Wechat 上 NWTG@10 较最优基线最高 +4.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;代理嵌入&quot;,&quot;参数高效&quot;]"><strong><a href="cold_start_rec/proxy_based_item_representation_for_attribute_and_context_aware_recommendat/">Proxy-based Item Representation for Attribute and Context-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用代理嵌入加权和替代物品查表，面向属性/上下文序列推荐，Leave-One-Out+100负采样下最高约17%提升且仅用10%参数。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回-冷启动映射&quot;,&quot;神经过程 (Neural Process)&quot;,&quot;元学习个性化映射&quot;]"><strong><a href="cross_domain_multi_scenario/cdrnp_cross_domain_recommendation_to_cold_start_users_via_neural_process/">CDRNP: Cross-Domain Recommendation to Cold-Start Users via Neural Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以神经过程将全部用户的跨域映射建模为同一随机过程，捕捉重叠用户与冷启动用户的偏好相关性，三组 Amazon 场景 MAE 相对 SOTA 最高降 50.67%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;组原型注意力&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/mixed_attention_network_for_cross_domain_sequential_recommendation/">Mixed Attention Network for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用局部/全局双通道编码 + 物品相似度/序列融合/组原型三重注意力，在无重叠用户假设下做双目标跨域序列推荐，稀疏域 AUC 最高 +8.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;模体(Motif)采样与超图编码&quot;,&quot;Prompt Tuning&quot;,&quot;对比预训练&quot;]"><strong><a href="cross_domain_multi_scenario/motif_based_prompt_learning_for_universal_cross_domain_recommendation/">Motif-Based Prompt Learning for Universal Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MOP 用 butterfly/triangle/random-walk 三类模体构造跨域共享嵌入，并以「预训练+Prompt Tuning」把 SSL 与推荐统一为模体相似度学习，在四类 CDR 场景上取得稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;原型对比学习&quot;,&quot;实体图预训练&quot;]"><strong><a href="cross_domain_multi_scenario/peace_prototype_learning_augmented_transferable_framework_for_cross_domain_/">PEACE: Prototype lEarning Augmented transferable framework for Cross-domain rEcommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以实体图为桥梁、原型学习增强的多兴趣预训练框架，实现目标域零样本与轻量微调迁移，线上 CTR 最高 +35.63%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特权特征蒸馏&quot;,&quot;列表级蒸馏损失&quot;,&quot;概率校准&quot;]"><strong><a href="ctr_feature_interaction/calibration_compatible_listwise_distillation_of_privileged_features_for_ctr/">Calibration-compatible Listwise Distillation of Privileged Features for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将特权特征蒸馏从 pointwise 升级为 listwise，提出校准兼容(calibration-compatible)的 CLID 损失，在 Web30K/Istella-S/阿里线上广告数据上同时提升 NDCG@10 与 GAUC 且不损 LogLoss/ECE。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Exposure Bias&quot;,&quot;SNIPS&quot;]"><strong><a href="debias_fairness_industrial/debiasing_sequential_recommenders_through_distributionally_robust_optimizat/">Debiasing Sequential Recommenders through Distributionally Robust Optimization over System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以系统曝光模拟器构造 DRO 名义分布，叠加高曝光惩罚与 SNIPS 评估器，SNIPS 全库排序下 Recall@20 最高 +25.9%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段防护&quot;,&quot;模型抽取攻击&quot;,&quot;可微交换矩阵&quot;]"><strong><a href="debias_fairness_industrial/defense_against_model_extraction_attacks_on_recommender_systems/">Defense Against Model Extraction Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个推荐模型抽取防御 GRO：以可微交换矩阵反传 top-k 梯度，用学生模型模拟攻击者并最大化其损失，在全库排序下压低代理模型效用且保持目标模型效用。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;稀疏训练&quot;,&quot;模型压缩&quot;,&quot;彩票假设&quot;]"><strong><a href="debias_fairness_industrial/dynamic_sparse_learning_a_novel_paradigm_for_efficient_recommendation/">Dynamic Sparse Learning: A Novel Paradigm for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从零训练固定预算稀疏嵌入表，周期性剪枝/再生权重，训练与推理 MACs 同降 30%–75% 且性能持平。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回/生成式表征&quot;,&quot;因果去偏&quot;,&quot;预训练跨域推荐&quot;]"><strong><a href="debias_fairness_industrial/pre_trained_recommender_systems_a_causal_debiasing_perspective/">Pre-trained Recommender Systems: A Causal Debiasing Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以层次贝叶斯因果去偏框架 PreRec 分离域内流行度偏差与跨域隐式偏差，实现跨市场/跨平台零样本与小样本推荐，Recall@K% 最高提升 87.7%。</p>
</dd>
</dl>
</section>
</div>
