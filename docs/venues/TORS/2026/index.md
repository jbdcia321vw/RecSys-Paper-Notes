---
hide:
  - toc
---

# 📖 ACM TORS 2026 推荐系统论文解读

26 篇 ACM TORS 2026 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TORS ×20 · TORS 2026 ×6 · 大语言模型推荐 ×4 · 去偏公平与可信推荐 ×3 · 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;Cellular Sheaf&quot;,&quot;Sheaf Diffusion&quot;,&quot;BPR&quot;]"><strong><a href="graph_rec/sheaf4rec_sheaf_neural_networks_for_graph_based_recommender_systems/">Sheaf4Rec: Sheaf Neural Networks for Graph-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用细胞层 (Cellular Sheaf) 将节点/边表示为向量空间+限制映射,以 Sheaf 拉普拉斯扩散替代单向量 GCN,NDCG@10 最高相对提升 11.29%,且推理耗时反降 2.5%–37%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;精排打分&quot;,&quot;Mixture-of-Experts&quot;,&quot;长短期偏好融合&quot;]"><strong><a href="sequential_rec/compositions_of_variant_experts_for_integrating_short_term_and_long_term_pr/">Compositions of Variant Experts for Integrating Short-Term and Long-Term Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 BPR/GRU4Rec 异构专家 + 可学习门控动态融合用户长短期偏好，全库排序下 MRR 最高提升 38.03%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回与排序&quot;,&quot;LLM4Rec&quot;,&quot;LLM Embeddings&quot;,&quot;Sequential Recommendation&quot;]"><strong><a href="sequential_rec/improving_sequential_recommendations_with_llms/">Improving Sequential Recommendations with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统比较 LLM 嵌入初始化、提示微调与混合路由；Beauty/DH 全库排序 NDCG@20 最高 +45%/+9%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;捆绑生成&quot;,&quot;强化学习策略&quot;,&quot;LLM 意图分解&quot;]"><strong><a href="sequential_rec/text2bundle_towards_personalized_query_based_bundle_generation/">Text2Bundle: Towards Personalized Query-based Bundle Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出查询驱动捆绑生成任务 QBG，用 LLM 意图分解 + 统一状态编码 + DQN 逐步选品，集合级 P/R/F1 上精确率较 BYOB† 提升约 26–32%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;可复现性研究&quot;,&quot;扩散模型推荐&quot;,&quot;弱基线问题&quot;]"><strong><a href="generative_rec/diffusion_recommender_models_and_the_illusion_of_progress_a_concerning_stud/">Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>复现 9 个 SIGIR&#x27;23-24 扩散推荐模型，仅 25% 结果可复现，调优简单基线全面反超，并揭示 DDPM 与 top-n 推荐任务的概念错配。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排/重排&quot;,&quot;开放世界知识注入&quot;,&quot;因子分解提示&quot;,&quot;HEIN&quot;]"><strong><a href="llm_rec/efficient_and_deployable_knowledge_infusion_for_open_world_recommendations_/">Efficient and Deployable Knowledge Infusion for Open-World Recommendations via Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>REKI用因子分解提示离线抽取用户/物品开放世界知识，经HEIN压缩为32维增广向量注入任意CRM；CTR/重排AUC提升约1%–1.5%，华为A/B提升7%/1.99%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;弱用户识别&quot;,&quot;子群体鲁棒性&quot;,&quot;混合任务分配&quot;]"><strong><a href="llm_rec/efficient_and_responsible_adaptation_of_large_language_models_for_robust_to/">Efficient and Responsible Adaptation of Large Language Models for Robust Top-k Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以「稀疏度+实例级 AUC」双阈值筛出弱用户，仅对其做 LLM 上下文重排，弱用户数降约 85–99% 且开销仅为全量调用的 1/3。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;终身序列建模&quot;,&quot;LoRA 参数高效微调&quot;,&quot;软提示增强&quot;]"><strong><a href="llm_rec/full_stack_optimized_large_language_models_for_lifelong_sequential_behavior/">Full-Stack Optimized Large Language Models for Lifelong Sequential Behavior Comprehension in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 LLM 长行为序列不理解问题，从数据(SUBR)、提示(SPA)、参数(CFLoRA)三层全栈优化，CTR 全库 AUC 相对提升 0.4–0.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;音乐推荐评估&quot;,&quot;LLM-as-Judge&quot;,&quot;提示工程&quot;]"><strong><a href="llm_rec/music_recommendation_with_large_language_models_challenges_opportunities_an/">Music Recommendation with Large Language Models: Challenges, Opportunities, and Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统梳理 LLM 驱动音乐推荐在用户建模、物品建模与自然语言请求三方面的变革，并提出 NLP 启发的评估分类体系与 MRS 成功/风险维度框架。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;多智能体协作&quot;,&quot;流行度去偏&quot;]"><strong><a href="agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;预训练音频表征&quot;,&quot;冷启动&quot;]"><strong><a href="multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/冷启动表征迁移&quot;,&quot;耦合自编码器&quot;,&quot;潜因子非线性映射&quot;,&quot;两阶段联合优化&quot;]"><strong><a href="cross_domain_multi_scenario/a_cross_domain_recommender_system_using_deep_coupled_autoencoders/">A cross-domain recommender system using deep coupled autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 CACDR/LFACDR 两种耦合自编码器跨域框架，以“源编码器+MLP映射+目标解码器”端到端联合优化替代分阶段训练，在 MovieLens/Netflix 冷启动物品 RMSE 上平均降低 6.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;全链路(数据/算法/评估)&quot;,&quot;立场与综述论文&quot;,&quot;多利益相关方推荐&quot;,&quot;评估协议批判&quot;]"><strong><a href="poi_spatiotemporal_rec/point_of_interest_recommendation_pitfalls_and_viable_solutions/">Point of Interest Recommendation: Pitfalls and Viable Solutions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统梳理POI推荐在数据集、算法与评估三维度的20项陷阱，并提出六方向研究议程与陷阱-方案映射表</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨模态对比对齐&quot;,&quot;PLM 知识蒸馏&quot;]"><strong><a href="ctr_feature_interaction/ctrl_connect_collaborative_and_language_model_for_ctr_prediction/">CTRL: Connect Collaborative and Language Model for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将表格与提示文本视为双模态，用双向 InfoNCE + MaxSim 子空间细粒度对齐把 PLM 世界知识蒸馏进轻量协同模型，在线只部署协同塔，AUC 最高 +4.45% 且零额外推理延迟。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;LLM零样本重排&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="multi_task_reranking/enhancing_recommendation_diversity_by_re_ranking_with_large_language_models/">Enhancing Recommendation Diversity by Re-ranking with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 8 种零样本提示让 LLM 对 MF 候选列表做多样性重排，Anime/Goodreads 上优于随机重排但逊于贪心方法，延迟高 3 个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;碳足迹感知推荐&quot;,&quot;绿色度重排&quot;]"><strong><a href="multi_task_reranking/towards_carbon_footprint_aware_recommender_systems_for_greener_item_recomme/">Towards Carbon Footprint-Aware Recommender Systems for Greener Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建首个含碳足迹的食谱数据集 RecipeEmission，基准测试 9 种传统算法发现精度优化忽视绿色度，提出 α 加权重排以极小精度损失换取最高 +89.1% 绿色度提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线列表推荐/重排与探索&quot;,&quot;Contextual Bandit&quot;,&quot;Thompson Sampling&quot;,&quot;Reward-Informed Segmentation&quot;]"><strong><a href="rl_bandit_rec/a_reward_informed_semi_personalized_bandit_approach_for_enhancing_accuracy_/">A Reward-Informed Semi-Personalized Bandit Approach for Enhancing Accuracy and Serendipity in Online Slate Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>按物品学习奖励监督决策树划分用户段，并在叶级运行 Thompson Sampling 生成列表，兼顾 regret 与 serendipity。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;双边匹配推荐&quot;,&quot;嫉妒自由&quot;,&quot;Nash社会福利&quot;,&quot;Sinkhorn&quot;]"><strong><a href="debias_fairness_industrial/balancing_fairness_and_high_match_rates_in_reciprocal_recommender_systems_a/">Balancing Fairness and High Match Rates in Reciprocal Recommender Systems: A Nash Social Welfare Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双边匹配推荐建模为推荐机会分配，用 Nash 社会福利交替优化实现近无嫉妒的公平推荐，并以 Sinkhorn 加速至大规模场景。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/多步重训练&quot;,&quot;反馈回路&quot;,&quot;因果推断&quot;]"><strong><a href="debias_fairness_industrial/breaking_feedback_loops_in_recommender_systems_with_causal_inference/">Breaking Feedback Loops in Recommender Systems with Causal Inference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用因果干预分布与CAFL重加权打破多步推荐反馈回路，仿真MSE较Pan降4.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Calibrated Recommendation&quot;,&quot;KL-Divergence&quot;,&quot;Post-processing Reranking&quot;]"><strong><a href="debias_fairness_industrial/calibrated_recommendations_survey_and_future_directions/">Calibrated Recommendations: Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统综述 53 篇校准推荐工作，统一拆解「基线排序→候选→分布构造→散度度量→权衡重排」七步框架，并量化揭示评测协议短板（88% 仅离线、2% 线上、27.1% 未报告划分）。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/首页Feed排序&quot;,&quot;受控个性化&quot;,&quot;新闻推荐A/B测试&quot;]"><strong><a href="debias_fairness_industrial/controlled_personalization_in_legacy_media_online_services_a_case_study_in_/">Controlled Personalization in Legacy Media Online Services: A Case Study in News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>传统媒体首页以20%权重融合矩阵分解个性化分，34天5.8万订阅者A/B显示CTR+14.6%、曝光-11.7%、覆盖率+4.0%、流行度偏差下降。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Recall-Oriented Evaluation&quot;,&quot;Lexicographic Fairness&quot;,&quot;Worst-Case Robustness&quot;]"><strong><a href="debias_fairness_industrial/recall_robustness_and_lexicographic_evaluation/">Recall, Robustness, and Lexicographic Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>从形式化视角重新定义排序召回，证明召回等价于最坏用户/提供方鲁棒性，并提出词典序召回 lexirecall，在 3 个推荐任务与 17 个检索任务上获得更高的敏感性与标签缺失稳定性。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/离线评测&quot;,&quot;相关性感知公平&quot;,&quot;个体物品公平&quot;,&quot;度量校正&quot;]"><strong><a href="debias_fairness_industrial/relevance_aware_individual_item_fairness_measures_for_recommender_systems_l/">Relevance-aware Individual Item Fairness Measures for Recommender Systems: Limitations and Usage Guidelines</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统诊断7类相关性感知个体物品公平度量的5类理论缺陷，提出逐用户归一化、top-k指示与曝光函数校正，并在4个真实数据集上验证。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准与元研究&quot;,&quot;可复现性&quot;,&quot;实验方法论&quot;]"><strong><a href="debias_fairness_industrial/reproducibility_in_recommender_systems_a_survey/">Reproducibility in Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>对 RecSys 可复现性赛道 2020–2025 年 51 篇论文的结构化元分析，揭示其方法论高度同质化、以“扩展”替代“严格复现”的实践落差。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;推荐系统评测&quot;]"><strong><a href="debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/生成式端到端&quot;,&quot;Provider Fairness&quot;,&quot;Prompt Engineering&quot;,&quot;In-Context Learning&quot;]"><strong><a href="debias_fairness_industrial/understanding_biases_in_chatgpt_based_recommender_systems_provider_fairness/">Understanding Biases in ChatGPT-based Recommender Systems: Provider Fairness, Temporal Stability, and Recency</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以千余次 API 调用系统解构 ChatGPT 推荐器的 Prompt 设计×系统角色×ICL 对提供方公平性、覆盖度与时间新鲜度的影响。</p>
</dd>
</dl>
</section>
</div>
