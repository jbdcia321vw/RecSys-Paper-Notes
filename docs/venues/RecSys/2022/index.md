---
hide:
  - toc
---

# 🏛️ RecSys 2022 推荐系统论文解读

27 篇 RecSys 2022 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 去偏公平与可信推荐 ×3 · 多模态推荐 ×2 · 重复消费建模 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复消费建模" aria-pressed="false">重复消费建模<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;互惠推荐 RRS&quot;,&quot;可转移效用匹配 (TU Matching)&quot;,&quot;IPFP 均衡求解&quot;]"><strong><a href="collaborative_filtering/matching_theory_based_recommender_systems_in_online_dating/">Matching Theory-based Recommender Systems in Online Dating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Choo-Siow 可转移效用匹配替代传统融合函数，把互惠偏好与用户承接容量统一到闭式均衡解，并在 tapple 千万级用户平台落地。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;iALS&quot;,&quot;频率缩放正则化&quot;,&quot;超参调优&quot;]"><strong><a href="collaborative_filtering/revisiting_the_performance_of_ials_on_item_recommendation_benchmarks/">Revisiting the Performance of iALS on Item Recommendation Benchmarks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>复核 iALS 在四个经典基准上的性能，用频率缩放正则、参考尺度 λ* 与超大嵌入维度这套调参配方，把被广泛认为落后的 iALS 拉回到 SOTA 水平。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回（KGNN 训练加速）&quot;,&quot;激活量化&quot;,&quot;随机舍入&quot;,&quot;显存压缩&quot;]"><strong><a href="graph_rec/tinykg_memory_efficient_training_framework_for_knowledge_graph_neural_recom/">TinyKG: Memory-Efficient Training Framework for Knowledge Graph Neural Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用均匀量化+随机舍入把 KGNN 前向激活压缩为 INT2 存显存，反向反量化算梯度，激活内存降 7× 而 Recall@20 仅降约 1.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (10)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;可复现性研究&quot;,&quot;BERT4Rec&quot;,&quot;掩码自监督训练&quot;]"><strong><a href="sequential_rec/a_systematic_review_and_replicability_study_of_bert4rec_for_sequential_reco/">A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复盘 134 次 BERT4Rec vs SASRec 对比，指出原始实现默认配置严重欠拟合（需 30× 训练时长），并给出 Hugging Face 高效复现与 ALBERT4Rec 改进。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;重排/序列生成&quot;,&quot;音乐 Tour 生成&quot;,&quot;用户中心评估&quot;]"><strong><a href="sequential_rec/a_user_centered_investigation_of_personal_music_tours/">A User-Centered Investigation of Personal Music Tours</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首次用半结构化访谈评估 Greedy 与 Optimal 两种音乐 Tour 生成算法，发现 Greedy 因 segue 多样性更受偏好，揭示多样性、编排与熟悉度为关键属性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排/排序&quot;,&quot;会话内自适应&quot;,&quot;实时搜索推荐&quot;]"><strong><a href="sequential_rec/augmenting_netflix_search_with_in_session_adapted_recommendations/">Augmenting Netflix Search with In-Session Adapted Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向Netflix搜索预查询画布，融合当前会话实时行为与长期偏好，离线排序指标相对生产模型提升6%，并讨论实时不缓存等工程约束。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可微二值掩码&quot;,&quot;Jacobian 正则&quot;,&quot;ARM 无偏梯度估计&quot;]"><strong><a href="sequential_rec/denoising_self_attentive_sequential_recommendation/">Denoising Self-attentive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微二值掩码剪枝自注意力的噪声项并叠加 Jacobian 谱正则，Leave-One-Out + 101 候选采样排序下比各 Transformer 骨干相对提升 5.05%–19.55%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/曝光策略&quot;,&quot;mere exposure effect&quot;,&quot;probit 回归&quot;,&quot;重复消费建模&quot;]"><strong><a href="sequential_rec/discovery_dynamics_leveraging_repeated_exposure_for_user_and_music_characte/">Discovery Dynamics: Leveraging Repeated Exposure for User and Music Characterization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 9M 条曝光日志拟合 probit 二次模型，验证新曲曝光—收听呈倒 U 型（峰值约 12 次），并以峰值位置刻画曲目熟悉度与用户年龄差异。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;训练目标设计&quot;,&quot;列表损失 λRank&quot;]"><strong><a href="sequential_rec/effective_and_efficient_training_for_sequential_recommendation_using_recenc/">Effective and Efficient Training for Sequential Recommendation using Recency Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出近期性序列采样 RSS 训练目标，替代序列延续与掩码任务，配合 λRank 列表损失，1 小时训练即可超越 16 小时 BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话行为建模&quot;,&quot;物品稀缺域推荐&quot;]"><strong><a href="sequential_rec/learning_recommendations_from_user_actions_in_the_item_poor_insurance_domai/">Learning Recommendations from User Actions in the Item-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首个保险领域跨会话推荐模型：以 3 元组网站行为序列为输入、GRU 编码多次会话，预测会话之外发生的购买行为，Full-Ranking HR@3 较 SKNN_EB 提升 3.04%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;重复消费建模&quot;,&quot;指针网络&quot;]"><strong><a href="sequential_rec/modeling_user_repeat_consumption_behavior_for_online_novel_recommendation/">Modeling User Repeat Consumption Behavior for Online Novel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>NovelNet 用 7 维细粒度交互属性编码每次阅读、BiGRU 建模序列、指针网络+pointwise 损失统一推荐已读与新小说，MRR@1 达 47.02%（+4.65%）。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Transformer&quot;,&quot;长短期兴趣建模&quot;]"><strong><a href="sequential_rec/rethinking_personalized_ranking_at_pinterest_an_end_to_end_approach/">Rethinking Personalized Ranking at Pinterest: An End-to-End Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>PinnerFormer 用稠密全动作损失编码长期兴趣，实时序列捕捉短期意图，线上 Repin +12.5%、Ads CTR +14.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态偏好建模&quot;,&quot;反馈回路仿真&quot;]"><strong><a href="sequential_rec/towards_psychologically_grounded_dynamic_preference_models/">Towards Psychologically-Grounded Dynamic Preference Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出心理学接地方法论，将单纯曝光、操作性条件作用、享乐适应形式化为偏好向量增量算子，仿真揭示振荡动态与参与度/多样性指标失效</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式端到端&quot;,&quot;多智能体强化学习&quot;,&quot;束推荐&quot;,&quot;完形预训练&quot;]"><strong><a href="agent_conversational_rec/bundle_mcr_towards_conversational_bundle_recommendation/">Bundle MCR: Towards Conversational Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 Bundle MCR 新场景与 Bunt 多智能体架构，将束推荐从一次性生成扩展为多轮交互，离线完形预训练+在线RL微调，Steam F1达0.822。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;对话交互层(端到端体验)&quot;,&quot;Rasa 任务型对话管理&quot;,&quot;Furhat 社交机器人多模态&quot;,&quot;多轮偏好启发&quot;]"><strong><a href="agent_conversational_rec/dagfinn_a_conversational_conference_assistant/">DAGFiNN: A Conversational Conference Assistant</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>基于 Rasa 的开源多域多模态会议助手，可在网页聊天与 Furhat 社交机器人间同步对话，ECIR&#x27;22 现场实测以 3–6 轮短对话完成 POI 与会程推荐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;音频内容分析&quot;,&quot;情绪条件推荐&quot;,&quot;工业级部署&quot;]"><strong><a href="multimodal_rec/flow_moods_recommending_music_by_moods_on_deezer/">Flow Moods: Recommending Music by Moods on Deezer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 Flow Moods 工业系统：将六类情绪作为显式上下文，融合专业策展标注、musicnn 音频嵌入与随机森林情绪分类器，对协同过滤检索结果做情绪硬过滤与兜底，已在千万级用户上线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;可复用自注意力&quot;,&quot;异构实体序列建模&quot;,&quot;穿搭生成&quot;]"><strong><a href="multimodal_rec/reusable_self_attention_recommender_systems_in_fashion_industry_application/">Reusable Self-Attention Recommender Systems in Fashion Industry Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>Zalando 用单一可复用 Transformer 统一穿搭排序、会话推荐与穿搭生成，靠异构实体/交互共享嵌入+时间衰减会话+目标掩码，线上 A/B 留存最高 +30%、会话互动 +130%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;增量学习&quot;,&quot;知识蒸馏&quot;,&quot;Warm-Start&quot;]"><strong><a href="ctr_feature_interaction/an_incremental_learning_framework_for_large_scale_ctr_prediction/">An Incremental Learning framework for Large-scale CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>Taboola 用 Warm-Start 初始化并从旧教师蒸馏，仅在数小时新样本上微调，训练提速 12.5×，线上 A/B 的 RPM 提升 0.53%–0.85%，新内容 CTR 显著上升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线探索与排序策略&quot;,&quot;差分隐私&quot;,&quot;线性Contextual Bandit&quot;]"><strong><a href="rl_bandit_rec/dynamic_global_sensitivity_for_differentially_private_contextual_bandits/">Dynamic Global Sensitivity for Differentially Private Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用模型收敛的动态全局敏感度+树机制降低隐私LinUCB噪声，将附加Regret降至$\tilde O(\log T\sqrt{T}/\epsilon)$，在合成与LastFM上优于Private LinUCB。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;MC Dropout 不确定性估计&quot;,&quot;实时竞价出价探索&quot;,&quot;反馈回路破环&quot;]"><strong><a href="rl_bandit_rec/exploration_with_model_uncertainty_at_extreme_scale_in_real_time_bidding/">Exploration with Model Uncertainty at Extreme Scale in Real-Time Bidding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 MC Dropout 的请求级预测不确定性驱动 RTB 出价探索，线上 A/B 相对提升 Revenue 5.0%、CTR 9.6%、离线 AUC +0.24%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;在线推理服务(端到端)&quot;,&quot;GPU 嵌入缓存&quot;,&quot;层次化参数服务器&quot;]"><strong><a href="debias_fairness_industrial/a_gpu_specialized_inference_parameter_server_for_large_scale_deep_recommend/">A GPU-specialized Inference Parameter Server for Large-Scale Deep Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HugeCTR HPS 用 GPU 嵌入缓存 + CPU/SSD 三级层次存储，把 TB 级嵌入表推理下沉到 GPU，端到端延迟相对 CPU 基线加速 5–62×。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;召回/输入序列层&quot;,&quot;对抗攻击与鲁棒性&quot;,&quot;Dirichlet邻域采样&quot;]"><strong><a href="debias_fairness_industrial/defending_substitution_based_profile_pollution_attacks_on_sequential_recomm/">Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首个替换式画像污染攻击+防御框架，用梯度引导攻击、Dirichlet邻域采样与对抗训练，NDCG@10波动降34.4%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;上下文融合&quot;,&quot;组公平性&quot;,&quot;时间偏差&quot;]"><strong><a href="debias_fairness_industrial/exploring_the_impact_of_temporal_bias_in_point_of_interest_recommendation/">Exploring the Impact of Temporal Bias in Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出以签到时段（工作/休闲）划分用户组，量化 GeoSoCa/LORE 等上下文感知 POI 模型的时间不公平，并用和式/加权和融合替代乘积融合以降低 ΔnDCG。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/离线评估&quot;,&quot;Commonality 共性度量&quot;,&quot;文化公民权 Cultural Citizenship&quot;,&quot;公共媒体 PSM&quot;]"><strong><a href="debias_fairness_industrial/measuring_commonality_in_recommendation_of_cultural_content_recommender_sys/">Measuring Commonality in Recommendation of Cultural Content: Recommender Systems to Enhance Cultural Citizenship</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出群体级共性度量 commonality，以 RBP 折损类别召回与联合概率刻画全体用户的共同文化曝光；MovieLens-1M 上 20 个系统排序显示其与 NDCG、α-NDCG 显著负相关，与公平性指标不相关。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;工业架构与系统&quot;,&quot;精排训练与在线推理服务&quot;,&quot;Hybrid Sparse Embedding&quot;,&quot;Hierarchical Parameter Server&quot;]"><strong><a href="debias_fairness_industrial/merlin_hugectr_gpu_accelerated_recommender_system_training_and_inference/">Merlin HugeCTR: GPU-accelerated Recommender System Training and Inference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向 TB 级 CTR 模型的 GPU 训练/推理一体化框架：混合稀疏并行、嵌入训练缓存 ETC 与三级分层参数服务器 HPS，MLPerf DLRM 训练最高 24.6×、在线推理 5~62× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;规范性多样性&quot;,&quot;秩感知散度&quot;]"><strong><a href="debias_fairness_industrial/radio_rank_aware_divergence_metrics_to_measure_normative_diversity_in_news_/">RADio: Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将新闻规范性多样性统一为秩感知 f-散度（JS）框架，在 MIND 上对 6 种推荐器做五维 DART 度量与敏感性分析。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;去偏公平与可信推荐&quot;,&quot;全链路设计治理（非排序阶段）&quot;,&quot;算法厌恶&quot;,&quot;人本推荐&quot;,&quot;用户主体性&quot;]"><strong><a href="debias_fairness_industrial/recommender_systems_and_algorithmic_hate/">Recommender Systems and Algorithmic Hate</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以多平台案例研究归纳“算法厌恶”的四类成因（无知、伤害、目标错位、主体性缺失）并提出人本设计、透明化与可配置三条缓解议程。</p>
</dd>
</dl>
</section>
</div>
