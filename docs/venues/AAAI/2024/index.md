---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2024 推荐系统论文解读

20 篇 AAAI (RecSys Track) 2024 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI 2024 ×16 · AAAI ×4 · 大语言模型推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="collaborative_filtering/adaptive_hardness_negative_sampling_for_collaborative_filtering/">Adaptive Hardness Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示固定难度负采样引发 FPP/FNP，提出三准则自适应难度范式 AHNS 及实例 AHNS_{p&lt;0}，4 数据集 R@20 最高 +8.2%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;重排/配料替代&quot;,&quot;FoodKG&quot;,&quot;水足迹&quot;]"><strong><a href="graph_rec/knowledge_powered_recommendation_for_an_improved_diet_water_footprint/">Knowledge-Powered Recommendation for an Improved Diet Water Footprint</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>AAAI 2024 知识图谱推荐系统演示：以 FoodKG 对齐营养与水足迹，按水足迹排序替代配料，案例水足迹降约74%，但无离线基准与消融。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱域图卷积&quot;,&quot;互补物品推荐&quot;,&quot;两阶段注意力&quot;]"><strong><a href="graph_rec/spectral_based_graph_neural_networks_for_complementary_item_recommendation/">Spectral-Based Graph Neural Networks for Complementary Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>谱域解耦互补关系为低频相关性与中频差异性，用低通/中通 GCN 滤波器加两阶段注意力建模，四个 Amazon 数据集 HR@10 提升 4.4%–14.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多轮自适应检索&quot;,&quot;表征适配器&quot;,&quot;FFT 可学习滤波&quot;]"><strong><a href="sequential_rec/ada_retrieval_an_adaptive_multi_round_retrieval_paradigm_for_sequential_rec/">Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>模型无关的多轮自适应召回：物品/用户表征适配器逐轮修正用户向量并补召回，Full-Ranking 下 NDCG@50 最高 +9.07%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;傅里叶变换&quot;,&quot;过平滑缓解&quot;]"><strong><a href="sequential_rec/an_attentive_inductive_bias_for_sequential_recommendation_beyond_the_self_a/">An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示自注意力低通滤波导致过平滑，用 DFT 注入注意力归纳偏置并重缩放频率，6 数据集全库排序平均提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;扩散模型插件&quot;,&quot;去噪重加权与负采样&quot;]"><strong><a href="sequential_rec/plug_in_diffusion_model_for_sequential_recommendation/">Plug-in Diffusion Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用时间间隔扩散模型对全库物品生成偏好，再以历史行为重加权、软正样本增广与无噪负采样三种插件策略回注序列推荐器，采样排序下 SASRec 提升最高 13.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;relevant-visit selection&quot;,&quot;target-aware attention&quot;,&quot;用药推荐&quot;]"><strong><a href="sequential_rec/vita_carefully_chosen_and_weighted_less_is_better_in_medication_recommendat/">VITA: &#x27;Carefully Chosen and Weighted Less&#x27; Is Better in Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可微 Gumbel 选访 + 目标感知注意力替代 RNN 顺序聚合，仅聚合与当前就诊相关的历史；MIMIC 上 Jaccard 相对最优基线提升 3.83%/5.56%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;简历补全&quot;,&quot;生成对抗对齐&quot;]"><strong><a href="llm_rec/enhancing_job_recommendation_through_llm_based_generative_adversarial_netwo/">Enhancing Job Recommendation through LLM-based Generative Adversarial Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用交互历史引导 LLM 补全简历，并以 GAN 对齐少样本用户低质简历表征，三数据集 map@5 平均提升约 7.9%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;Meta-path Prompt&quot;,&quot;路径去偏&quot;]"><strong><a href="llm_rec/exploring_large_language_model_for_graph_data_understanding_in_online_job_r/">Exploring Large Language Model for Graph Data Understanding in Online Job Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 meta-path prompt 把异构行为图翻译成自然语言供 LLM 理解，并设计路径打乱/软选择器消除路径权重与位置偏置，AUC 最高提升 29.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;超图学习&quot;,&quot;解耦表征&quot;]"><strong><a href="multimodal_rec/lgmrec_local_and_global_graph_learning_for_multimodal_recommendation/">LGMRec: Local and Global Graph Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>解耦协同/模态局部图表征并用模态超图捕获全局兴趣，Amazon 三数据集全库排序 R@20 最高提升 11.44%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;召回/展示素材选择&quot;,&quot;跨模态图文匹配&quot;,&quot;用户细分个性化&quot;,&quot;图库摘要&quot;]"><strong><a href="multimodal_rec/tell_me_what_is_good_about_this_property_leveraging_reviews_for_segment_per/">Tell Me What Is Good About This Property: Leveraging Reviews For Segment-Personalized Image Collection Summarization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以细分评论主题为隐式 query，MuMIC 多标签图文表征 + KMedoids 聚类约束下做跨模态贪心选图，实现无监督的房源图库个性化摘要，Coverage 较无个性化基线提升 25.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;群组共识聚合&quot;,&quot;上下文Bandit&quot;,&quot;技能覆盖&quot;]"><strong><a href="social_group_rec/promoting_research_collaboration_with_open_data_driven_team_recommendation_/">Promoting Research Collaboration with Open Data Driven Team Recommendation in Response to Call for Proposals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从 RFP 与学者主页抽取技能供需并用 ACM-CCS 归一，以随机/字符串/本体/关系型 Boosted Bandit 四级方法组队，最大化覆盖-规模加权 goodness，USC 与 IIT-R 双站点验证并上线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Temporal Feature Shift&quot;]"><strong><a href="cold_start_rec/temporally_and_distributionally_robust_optimization_for_cold_start_recommen/">Temporally and Distributionally Robust Optimization for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出时间分布鲁棒优化 TDRO，以最坏组因子与漂移因子重加权特征提取器，在全库排序上稳定提升冷启动物品推荐。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端召回与排序&quot;,&quot;掩码域建模&quot;,&quot;域偏好建模&quot;]"><strong><a href="cross_domain_multi_scenario/multi_domain_recommendation_to_attract_users_via_domain_preference_modeling/">Multi-Domain Recommendation to Attract Users via Domain Preference Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 DRIP，用掩码域建模统一学习多对多跨域映射并分解域级/物品级偏好；Amazon 双平台 Top-K 排序下显著优于 CDR 基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;Mobility Tree&quot;,&quot;时段偏好建模&quot;,&quot;Tree-LSTM&quot;]"><strong><a href="poi_spatiotemporal_rec/learning_time_slot_preferences_via_mobility_tree_for_next_poi_recommendatio/">Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 Mobility Tree 分层编码签到轨迹，用四步 IAC/IRC 节点交互与多任务学习刻画多粒度时段偏好，三数据集全指标超越 10 个 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;InfoNCE&quot;,&quot;辅助匹配任务&quot;,&quot;负样本下采样&quot;]"><strong><a href="ctr_feature_interaction/at4ctr_auxiliary_match_tasks_for_enhancing_click_through_rate_prediction/">AT4CTR: Auxiliary Match Tasks for Enhancing Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 UIM/NIP 双 InfoNCE 辅助任务缓解负采样稀疏，Taobao/美团 AUC 0.9535/0.7441，线上 RPS +1.27%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;有序回归&quot;,&quot;误差自适应离散化&quot;]"><strong><a href="ctr_feature_interaction/cread_a_classification_restoration_framework_with_error_adaptive_discretiza/">CREAD: A Classification-Restoration Framework with Error Adaptive Discretization for Watch Time Prediction in Video Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>观看时长有序二分类+误差自适应离散化，平衡学习/还原误差，离线MAE/XAUC领先，线上+0.29%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;广告创意排序&quot;,&quot;并行级联架构&quot;,&quot;联合训练与知识蒸馏&quot;]"><strong><a href="ctr_feature_interaction/parallel_ranking_of_ads_and_creatives_in_real_time_advertising_systems/">Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 Peri-CR 广告-创意并行排序架构与 JAC 离线联合蒸馏模型，配 NSCTR 指标；线上 CTR +10.12%、RPM +7.67% 且零延迟增长。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;负迁移&quot;,&quot;停梯度门控&quot;,&quot;任务专属嵌入&quot;]"><strong><a href="multi_task_reranking/stem_unleashing_the_power_of_embeddings_for_multi_task_recommendation/">STEM: Unleashing the Power of Embeddings for Multi-task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按任务反馈分桶诊断出 Comparable 子集负迁移，提出共享+任务专属双轨嵌入与「全前向任务特定反向」停梯度门控，三数据集平均 AUC 超越 PLE 约 4e-3 并落地腾讯广告 A/B。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;多阶段排序（召回/粗排/精排）&quot;,&quot;Multi-Agent RL&quot;,&quot;Cascading Information Chain&quot;]"><strong><a href="rl_bandit_rec/unex_rl_reinforcing_long_term_rewards_in_multi_stage_recommender_systems_wi/">UNEX-RL: Reinforcing Long-Term Rewards in Multi-Stage Recommender Systems with UNidirectional EXecution</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多阶段推荐建模为单向执行 MARL，用 CIC 重放全链观测并配合 SG/CQR 降方差；离线模拟器与 Kwai 在线 A/B 显著提升长期观看时长。</p>
</dd>
</dl>
</section>
</div>
