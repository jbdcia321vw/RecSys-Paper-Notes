---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2026 推荐系统论文解读

19 篇 IJCAI (RecSys Track) 2026 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2026 ×14 · IJCAI ×4 · 大语言模型推荐 ×4 · 生成式推荐 ×4 · 多模态推荐 ×2 · 流行度去偏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;软情感指派&quot;,&quot;跨视图对比对齐&quot;,&quot;情感交换机制&quot;]"><strong><a href="graph_rec/sentiment_aware_rating_based_recommendation_via_semantic_enhanced_item_alig/">Sentiment-aware Rating-based Recommendation via Semantic-enhanced Item Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用经验贝叶斯软情感指派替代硬阈值，配合协同-语义图与锚引导单向对齐，以及显式正负情感对冲打分，在四个 Amazon 5-core 数据集上全库排序取得 MSE 23.08%、NDCG@10 21.51% 的提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;LLM 语义兴趣抽取&quot;,&quot;马尔可夫状态空间&quot;,&quot;生成式-判别式协同&quot;]"><strong><a href="sequential_rec/bridging_the_semantic_gap_leveraging_llms_for_hierarchical_interest_evoluti/">Bridging the Semantic Gap: Leveraging LLMs for Hierarchical Interest Evolution in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>HSIEN 用 LLM 做马尔可夫式层级兴趣抽取（局部递推+全局 SMoE），再以离线预计算+在线查表喂给判别式 CTR 模型，Amazon AUC 0.7795、工业 +0.59PP、线上 ROI +3.0PP。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Flow Matching&quot;,&quot;对比正则化&quot;]"><strong><a href="sequential_rec/contrastive_flow_matching_for_sequential_recommendation/">Contrastive Flow Matching for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过表示级与速度级对比正则化，解决流匹配推荐中的 averaged flow 问题，在 Full-Ranking 协议下显著提升长尾个性化推荐。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;降尺度卷积&quot;,&quot;属性感知序列建模&quot;]"><strong><a href="sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;生成式推荐&quot;,&quot;Draft-and-Verify&quot;,&quot;混合稀疏-稠密表征&quot;,&quot;并行验证推理&quot;]"><strong><a href="generative_rec/dav_gen_end_to_end_generative_retrieval_via_draft_and_verify/">DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成式检索重构为“起草-验证”双阶段单模型范式，以混合稀疏-稠密表征与复合损失统一召回与排序，工业搜索延迟降至70ms。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式&quot;,&quot;条件 VAE&quot;,&quot;分子图编码 (GIN)&quot;,&quot;DDI 潜空间正则&quot;]"><strong><a href="generative_rec/mgrec_structure_grounded_medication_recommendation_via_condition_aware_mole/">MGRec: Structure-Grounded Medication Recommendation via Condition-Aware Molecular Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>把药物推荐重构为条件感知的连续分子表征推断，治疗-安全解耦 CVAE 加 DDI 潜空间正则，MIMIC 双库全库排序精度与 DDI 风险双优。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端&quot;,&quot;流匹配 ODE&quot;,&quot;流行度去偏&quot;,&quot;长尾覆盖率&quot;]"><strong><a href="generative_rec/pdflow_popularity_debiased_flow_matching_for_sequential_recommendation/">PDFlow: Popularity-Debiased Flow Matching for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在流匹配生成式序列推荐中引入残差流行度向量场与跨流行度对齐损失，逐步减去流行度漂移分量，全库排序下 Beauty HR@20 提升 11.34%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;双通道语义融合&quot;]"><strong><a href="llm_rec/dual_channel_semantic_enhanced_combinatorial_medication_recommendation_via_/">Dual-Channel Semantic-Enhanced Combinatorial Medication Recommendation via Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双通道语义融合 + 注意力图蒸馏，把 PubMedBERT 临床叙述知识迁移到 ID-based 药物组合推荐，MIMIC-III Jaccard 0.5632（+3.7%）</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式解释&quot;,&quot;超球聚类&quot;,&quot;vMF 混合先验&quot;,&quot;聚类引导 MoE&quot;]"><strong><a href="llm_rec/hyperxrec_unifying_preference_clusters_and_llm_experts_for_robust_explainab/">HyperXRec: Unifying Preference Clusters and LLM Experts for Robust Explainable Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 vMF 混合先验学习超球偏好簇，并以簇后验引导 LLM 的 MoE 专家路由，提升稀疏场景解释保真度。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;学习路径分段漂移&quot;,&quot;语义-结构对齐融合&quot;]"><strong><a href="llm_rec/llm_enhanced_knowledge_and_learning_path_understanding_for_graph_based_educ/">LLM-Enhanced Knowledge and Learning Path Understanding for Graph-based Educational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 摘要资源知识与学习路径分段模式，双分支融合图结构表征，全库排序下 Recall@10 最高提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回/排序&quot;,&quot;LLM兴趣摘要&quot;,&quot;多模态自蒸馏&quot;]"><strong><a href="llm_rec/llm_summarized_interest_distillation_for_multimedia_recommendation/">LLM-Summarized Interest Distillation for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>SID用LLM摘要兴趣并以语义邻居多教师蒸馏增强稀疏交互；全库排序下Beauty/Toys Recall@20较最佳基线提升10.33%/13.75%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;多模态推荐&quot;,&quot;端到端解释生成&quot;,&quot;多兴趣画像&quot;,&quot;偏好锚定推理&quot;,&quot;多模态聚类&quot;]"><strong><a href="multimodal_rec/joint_multi_modal_multi_interest_profiling_and_preference_grounded_reasonin/">Joint Multi-Modal Multi-Interest Profiling and Preference-Grounded Reasoning for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>PRIME以兴趣感知聚类、多模态多兴趣画像与偏好锚定检索增强可解释推荐；时间戳切分下BLEURT较Reason4Rec提升7.4%–10.2%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多教师在线蒸馏&quot;,&quot;模态不平衡&quot;]"><strong><a href="multimodal_rec/three_minds_one_student_online_multi_teacher_knowledge_distillation_for_mul/">Three Minds, One Student: Online Multi-Teacher Knowledge Distillation for Multimodal Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多模态推荐重构为“三教师-单学生”在线蒸馏，在排序级与表征级选择性迁移 ID/图/文偏好，Full-Ranking 上最高相对提升 7.22%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;流匹配去噪&quot;,&quot;跨域流形对齐&quot;,&quot;社交噪声&quot;]"><strong><a href="social_group_rec/manifold_aligned_rectification_flow_for_denoised_social_recommendation/">Manifold-Aligned Rectification Flow for Denoised Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用流匹配将噪声社交表征沿社交感知偏好流形做跨域输运重整，随机 7:1:2 划分下 Recall@20 在 Ciao/Epinions 相对最优基线提升 2.34%/5.56%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文老虎机&quot;,&quot;知识追踪&quot;,&quot;课程推荐&quot;]"><strong><a href="rl_bandit_rec/pace_a_personalized_adaptive_curriculum_engine_for_9_1_1_call_taker_trainin/">PACE: A Personalized Adaptive Curriculum Engine for 9-1-1 Call-taker Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用技能图信念传播+个体学习/遗忘动力学+Thompson Sampling 上下文老虎机为 9-1-1 调度员推荐训练场景，Z2H 缩短 19.50%、终局掌握度提升 10.95%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI-ECAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;联邦推荐&quot;,&quot;基础模型个性化&quot;,&quot;语义隐私边界&quot;]"><strong><a href="debias_fairness_industrial/a_survey_of_personalized_federated_foundation_models_for_privacy_preserving/">A Survey of Personalized Federated Foundation Models for Privacy-Preserving Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>综述提出“主动语义保人格”范式，用几何正交约束与语义感知聚合解耦全局知识与用户人格，并给出五类开放问题与未来路线。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/解释评估&quot;,&quot;反事实解释&quot;,&quot;Listwise Top-k&quot;]"><strong><a href="debias_fairness_industrial/beyond_top_1_addressing_inconsistencies_in_evaluating_counterfactual_explan/">Beyond Top-1: Addressing Inconsistencies in Evaluating Counterfactual Explanations for Recommender Systems (Extended Abstract)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 CE 评估从 top-1 扩展到 listwise top-k POS，用 3 数据集×2 模型×4 检查点验证一致性。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/精排模型蒸馏部署&quot;,&quot;流行度去偏&quot;,&quot;球面测地路由&quot;,&quot;元学习门控融合&quot;]"><strong><a href="debias_fairness_industrial/geodesic_expert_routing_for_unbiased_knowledge_distillation_in_recommendati/">Geodesic Expert Routing for Unbiased Knowledge Distillation in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>指出知识蒸馏会放大流行度偏差，用按热度分层训练的专家教师+球面测地路由+元去偏门控实现精度-公平双赢。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端生成式/智能体决策支持&quot;,&quot;可信推荐&quot;,&quot;LLM Agent&quot;,&quot;关系用户建模&quot;]"><strong><a href="debias_fairness_industrial/toward_trustworthy_recommender_systems_in_the_era_of_agentic_ai_from_relati/">Toward Trustworthy Recommender Systems in the Era of Agentic AI: From Relational User Modeling to Generative Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向智能体AI的可信推荐研究议程，统一关系用户建模、生成式个性化与可信基础；原文无数值实验。</p>
</dd>
</dl>
</section>
</div>
