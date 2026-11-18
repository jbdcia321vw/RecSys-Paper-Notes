---
hide:
  - toc
---

# 📖 ACM TORS 2025 推荐系统论文解读

17 篇 ACM TORS 2025 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TORS ×17 · 去偏公平与可信推荐 ×7 · 可解释推荐 ×2

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两步接地" aria-pressed="false">两步接地<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主题模型" aria-pressed="false">主题模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全库排序" aria-pressed="false">全库排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全链路（召回/排序/解释）" aria-pressed="false">全链路（召回/排序/解释）<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="删失Weibull损失" aria-pressed="false">删失Weibull损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态重排" aria-pressed="false">动态重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单品公平" aria-pressed="false">单品公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排+重排" aria-pressed="false">召回+精排+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/重排/端到端" aria-pressed="false">召回/重排/端到端<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可信推荐 TRS" aria-pressed="false">可信推荐 TRS<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积溢出影响" aria-pressed="false">图卷积溢出影响<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线用户研究" aria-pressed="false">在线用户研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基准复现与超越精度评测" aria-pressed="false">基准复现与超越精度评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体公平" aria-pressed="false">多智能体公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态Schema形式化" aria-pressed="false">多模态Schema形式化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多类型动作序列" aria-pressed="false">多类型动作序列<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入坍缩" aria-pressed="false">嵌入坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序数逻辑回归" aria-pressed="false">序数逻辑回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="度量归一化" aria-pressed="false">度量归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响函数" aria-pressed="false">影响函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知推荐" aria-pressed="false">意图感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐遗忘" aria-pressed="false">推荐遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据投毒攻击" aria-pressed="false">数据投毒攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据稀缺" aria-pressed="false">数据稀缺<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光公平" aria-pressed="false">曝光公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光感知推荐" aria-pressed="false">曝光感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有效秩" aria-pressed="false">有效秩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="权重共享NAS" aria-pressed="false">权重共享NAS<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型更新/遗忘" aria-pressed="false">模型更新/遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力机制" aria-pressed="false">注意力机制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户簇定向攻击" aria-pressed="false">用户簇定向攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识迁移" aria-pressed="false">知识迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵分解" aria-pressed="false">矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社会选择" aria-pressed="false">社会选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经网络压缩" aria-pressed="false">神经网络压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="系统性文献综述" aria-pressed="false">系统性文献综述<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统计信息注入" aria-pressed="false">统计信息注入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述" aria-pressed="false">综述<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述与分类体系" aria-pressed="false">综述与分类体系<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督学习" aria-pressed="false">自监督学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测与度量" aria-pressed="false">评测与度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语气感知解释" aria-pressed="false">语气感知解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱均衡" aria-pressed="false">谱均衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超网路径采样" aria-pressed="false">超网路径采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨会话建模" aria-pressed="false">跨会话建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视角权衡" aria-pressed="false">跨视角权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软硬件协同设计" aria-pressed="false">软硬件协同设计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护" aria-pressed="false">隐私保护<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高阶关联" aria-pressed="false">高阶关联<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入坍缩&quot;,&quot;谱均衡&quot;,&quot;有效秩&quot;]"><strong><a href="collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;对比学习&quot;,&quot;高阶关联&quot;]"><strong><a href="collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回+重排&quot;,&quot;意图感知推荐&quot;,&quot;综述与分类体系&quot;]"><strong><a href="sequential_rec/a_survey_on_intent_aware_recommender_systems/">A Survey on Intent-aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理意图感知推荐(IARS)的三大技术路线与评估实践，指出离线精度导向、可复现性不足与显式意图建模缺失三大空白。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话建模&quot;,&quot;删失Weibull损失&quot;,&quot;多类型动作序列&quot;]"><strong><a href="sequential_rec/recommending_target_actions_outside_sessions_in_the_data_poor_insurance_dom/">Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向保险数据稀缺场景的跨会话 RNN 推荐，预测会话外购买目标，Full-Ranking 下 HR@3 较 SKNN_EB 提升 3.11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;两步接地&quot;,&quot;全库排序&quot;,&quot;统计信息注入&quot;]"><strong><a href="llm_rec/a_bi_step_grounding_paradigm_for_large_language_models_in_recommendation_sy/">A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 BIGRec 两步接地范式：先指令微调 LLM 生成物品语义 token，再以 L2 距离对齐真实物品空间，全库排序下 1024 样本即超越全量训练基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;可解释推荐&quot;,&quot;语气感知解释&quot;,&quot;在线用户研究&quot;,&quot;序数逻辑回归&quot;]"><strong><a href="agent_conversational_rec/impact_of_tone_aware_explanations_in_recommender_systems/">Impact of Tone-Aware Explanations in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成六种语气解释做在线成对用户研究，序数逻辑回归揭示语气对 10 项感知指标、领域与人格年龄的差异化影响。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;精排&quot;,&quot;可解释推荐&quot;,&quot;主题模型&quot;,&quot;注意力机制&quot;]"><strong><a href="agent_conversational_rec/topic_centric_explanations_for_news_recommendation/">Topic-Centric Explanations for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双层注意力主题模型 BATM 同时做新闻推荐与主题级解释，MIND 全库排序 AUC 提升约 0.4-0.5 个点，并用 NPMI/W2V 量化主题可解释性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态Schema形式化&quot;,&quot;基准复现与超越精度评测&quot;]"><strong><a href="multimodal_rec/formalizing_multimedia_recommendation_through_multimodal_deep_learning/">Formalizing Multimedia Recommendation through Multimodal Deep Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次将多媒体推荐对齐多模态深度学习五步流水线，提出 Which/How/When 三问形式化 Schema，并在 Elliot 上统一复现 10 个模型、兼顾精度与超越精度指标。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;冷启动&quot;,&quot;综述&quot;,&quot;数据稀缺&quot;,&quot;知识迁移&quot;,&quot;自监督学习&quot;]"><strong><a href="cold_start_rec/data_scarcity_in_recommendation_systems_a_survey/">Data Scarcity in Recommendation Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理数据稀缺下五类缓解范式（增广/自监督/迁移/广度学习/知识图谱），并给出挑战与未来方向。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;权重共享NAS&quot;,&quot;超网路径采样&quot;,&quot;软硬件协同设计&quot;]"><strong><a href="ctr_feature_interaction/towards_automated_model_design_on_recommender_systems/">Towards Automated Model Design on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用权重共享超网 NAS 在稠密连接+异构算子+维度+量化空间上自动搜索 CTR 模型，Criteo/KDD 降 0.001~0.003 LogLoss，FLOPs 降 8.5×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;联邦学习&quot;,&quot;神经网络压缩&quot;,&quot;隐私保护&quot;]"><strong><a href="debias_fairness_industrial/a_privacy_preserving_system_for_movie_recommendations_using_federated_learn/">A Privacy Preserving System for Movie Recommendations Using Federated Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端到端联邦电影推荐系统，提出 FedQ 队列式联邦训练缓解非独立同分布与小样本问题，结合 NNC 压缩降低通信开销，在 MovieLens 25M 上验证 16 万客户端规模。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;全链路（召回/排序/解释）&quot;,&quot;可信推荐 TRS&quot;,&quot;跨视角权衡&quot;]"><strong><a href="debias_fairness_industrial/a_survey_on_trustworthy_recommender_systems/">A Survey on Trustworthy Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理可解释、公平、隐私、鲁棒、可控五大可信推荐维度及其交叉关系，提出跨视角关系矩阵与评估局限反思。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;社会选择&quot;,&quot;多智能体公平&quot;,&quot;动态重排&quot;]"><strong><a href="debias_fairness_industrial/dynamic_fairness_aware_recommendation_through_multi_agent_social_choice/">Dynamic Fairness-aware Recommendation through Multi-agent Social Choice</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多利益相关方公平推荐建模为「分配+聚合」两阶段社会选择问题，SCRUF-D 框架动态调度多个公平性智能体，在真实与合成数据上实现多维权衡。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;数据投毒攻击&quot;,&quot;矩阵分解&quot;,&quot;用户簇定向攻击&quot;]"><strong><a href="debias_fairness_industrial/evaluating_impact_of_user_cluster_targeted_attacks_in_matrix_factorisation_/">Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统分析 MF 推荐中面向用户簇的投毒攻击，揭示 V 矩阵是攻击传播主通道，簇内真实评分量决定脆弱性。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测与度量&quot;,&quot;单品公平&quot;,&quot;曝光公平&quot;,&quot;度量归一化&quot;]"><strong><a href="debias_fairness_industrial/evaluation_measures_of_individual_item_fairness_for_recommender_systems_a_c/">Evaluation Measures of Individual Item Fairness for Recommender Systems: A Critical Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统批判 8 种单品曝光公平度量，修正 5 类理论缺陷并给出归一化方案与选用指南。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/重排/端到端&quot;,&quot;曝光感知推荐&quot;,&quot;系统性文献综述&quot;,&quot;曝光偏差&quot;]"><strong><a href="debias_fairness_industrial/impression_aware_recommender_systems/">Impression-Aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首篇曝光感知推荐系统(IARS)系统性综述，提出统一理论框架与三维分类体系，厘清模型、数据集与评估协议。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型更新/遗忘&quot;,&quot;影响函数&quot;,&quot;推荐遗忘&quot;,&quot;图卷积溢出影响&quot;]"><strong><a href="debias_fairness_industrial/recommendation_unlearning_via_influence_function/">Recommendation Unlearning via Influence Function</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用影响函数一步更新模型实现推荐遗忘，联合直接损失影响与图卷积溢出影响，剪枝加速后较重训提速 250 倍以上。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
