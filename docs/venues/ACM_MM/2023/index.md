---
hide:
  - toc
---

# 🏛️ ACM MM 2023 推荐系统论文解读

11 篇 ACM MM 2023 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×5 · 属性遗忘 (Attribute Unlearning) ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性遗忘 (Attribute Unlearning)" aria-pressed="false">属性遗忘 (Attribute Unlearning)<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;知识图谱提示&quot;,&quot;知识树掩码&quot;]"><strong><a href="sequential_rec/knowledge_prompt_tuning_for_sequential_recommendation/">Knowledge Prompt-tuning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将知识图谱经关系模板转成知识提示，并用知识树掩码去噪，与 T5 掩码个性化提示联合调优；Leave-One-Out + Full-Ranking 下 books NDCG@5 较 P5 提升 40.65%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态特征融合&quot;,&quot;在线知识蒸馏&quot;,&quot;去偏 In-batch 损失&quot;]"><strong><a href="sequential_rec/online_distillation_enhanced_multi_modal_transformer_for_sequential_recomme/">Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ID-aware 多模态 Transformer 与在线蒸馏解耦 ID/模态特征的优化冲突，四数据集 Recall@10 相对最优基线提升 7.4%–13.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布表征&quot;]"><strong><a href="sequential_rec/personalized_behavior_aware_transformer_for_multi_behavior_sequential_recom/">Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布刻画物品/行为/位置并借 Wasserstein 距离重构自注意力，把个性化多行为模式与行为-时间协同注入 Transformer；Leave-One-Out + HR/NDCG 上 Taobao HR@5 相对最优基线提升 6.37%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图结构冻结&quot;,&quot;度数敏感边剪枝&quot;]"><strong><a href="multimodal_rec/a_tale_of_two_graphs_freezing_and_denoising_graph_structures_for_multimodal/">A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>冻结多模态物品-物品图并对用户-物品图做度数敏感剪枝去噪，全排序协议下相对 LATTICE 平均提升 19.07%，内存最高降 6×。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;数据与特征工程&quot;,&quot;多模态特征抽取&quot;,&quot;可复现框架&quot;]"><strong><a href="multimodal_rec/ducho_a_unified_framework_for_the_extraction_of_multimodal_features_in_reco/">Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一 TensorFlow/PyTorch/Transformers 三后端的多模态特征抽取框架，以 YAML 配置解耦模态、来源与抽取层，提供 Docker 化可复现管线。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导模态去噪&quot;,&quot;多视图图卷积&quot;,&quot;行为感知融合&quot;]"><strong><a href="multimodal_rec/multi_view_graph_convolutional_network_for_multimedia_recommendation/">Multi-View Graph Convolutional Network for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为引导净化模态噪声 + U-I/I-I 双视图编码 + 行为感知自适应融合与互信息自监督，Amazon 三集全排序 Recall@20 最高提升 22.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;不变风险最小化&quot;,&quot;帕累托多目标优化&quot;]"><strong><a href="multimodal_rec/pareto_invariant_representation_learning_for_multimedia_recommendation/">Pareto Invariant Representation Learning for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 IID-OOD 权衡建模为多目标优化，梯度自适应加权 ERM/IRM 学不变掩码，三数据集 IID 与 OOD 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;语义引导特征提取&quot;,&quot;模态互补融合&quot;]"><strong><a href="multimodal_rec/semantic_guided_feature_distillation_for_multimodal_recommendation/">Semantic-Guided Feature Distillation for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用语义标签监督的 DNN 教师 + 浅层学生知识蒸馏替代联合训练的特征提取器，缓解稀疏下的提取失效，三类骨干最高提升 8.56%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排/创意上线前预评估&quot;,&quot;多模态CTR预测&quot;,&quot;注意力热图归因&quot;,&quot;LLM广告内容分析&quot;]"><strong><a href="ctr_feature_interaction/against_opacity_explainable_ai_and_large_language_models_for_effective_digi/">Against Opacity: Explainable AI and Large Language Models for Effective Digital Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 ViT 替换 ABN 构建 SoWide-v2 多模态三档 CTR 分类器，并用注意力热图与 GPT-3.5 流水线 SODA 为广告主侧提供可解释、可执行的创意优化洞察。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;后处理/模型修正&quot;,&quot;属性遗忘 (Attribute Unlearning)&quot;,&quot;属性推断攻击 (AIA)&quot;,&quot;MMD 分布对齐&quot;]"><strong><a href="debias_fairness_industrial/making_users_indistinguishable_attribute_wise_unlearning_in_recommender_sys/">Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出训练后属性遗忘 PoT-AU，用可分性损失+正则损失仅微调用户嵌入，D2D-MMD 变体把攻击 AUC 压至随机水平且推荐指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（训练后嵌入级遗忘后处理）&quot;,&quot;属性遗忘 (Attribute Unlearning)&quot;,&quot;用户嵌入分布对齐&quot;,&quot;可复现性&quot;]"><strong><a href="debias_fairness_industrial/reproducibility_companion_paper_making_users_indistinguishable_attribute_wi/">Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>复现属性遗忘 U2U-R/D2D-R 两法：在 NMF/LightGCN 上验证 ML-100K/1M/LFM-2B 的推荐保真度、抗攻击性与百倍级效率优势。</p>
</dd>
</dl>
</section>
</div>
