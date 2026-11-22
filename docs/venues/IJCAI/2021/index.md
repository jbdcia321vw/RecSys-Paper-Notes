---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2021 推荐系统论文解读

17 篇 IJCAI (RecSys Track) 2021 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2021 ×11 · IJCAI ×6 · CTR预估与特征交叉 ×2 · 综述与分类体系 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述与分类体系" aria-pressed="false">综述与分类体系<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图构造与信息传播&quot;,&quot;综述与分类体系&quot;]"><strong><a href="graph_rec/graph_learning_based_recommender_systems_a_review/">Graph Learning based Recommender Systems: A Review</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首篇 GLRS 系统综述：按数据形态→图结构→学习范式三层正交归类，形式化 $R=\\arg\\max_\\Theta f(M(\\Theta)|G)$，并汇总 17 个开源实现与 16 个公开数据集。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构关系学习&quot;,&quot;消息传递范式&quot;]"><strong><a href="graph_rec/recent_advances_in_heterogeneous_relation_learning_for_recommendation/">Recent Advances in Heterogeneous Relation Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首篇异构关系学习推荐综述，以「关系表征+信息融合」两问统一社交、知识图谱与多行为三条路线，并指出多模态、稀缺、时序、可解释与实时五大方向</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;端到端多标签集合预测&quot;,&quot;分子图消息传递 MPNN&quot;,&quot;药物相互作用可控损失&quot;]"><strong><a href="graph_rec/safedrug_dual_molecular_graph_encoders_for_recommending_effective_and_safe_/">SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>全局 MPNN + 局部子结构二部图双编码器刻画药物分子，配合 PID 式 DDI 可控损失，在 MIMIC-III 上 DDI 率相对降 19.43%、Jaccard 升 2.88%，参数仅 325K</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;个性化异质图&quot;,&quot;异质图池化&quot;]"><strong><a href="graph_rec/user_as_graph_user_modeling_with_heterogeneous_graph_pooling_for_news_recom/">User-as-Graph: User Modeling with Heterogeneous Graph Pooling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户行为构造成个性化异质图，以类型感知异质图池化学习用户兴趣，在 MIND 曝光排序上稳定提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;端到端预测&quot;,&quot;对抗分布正则&quot;,&quot;记忆网络多跳读取&quot;,&quot;药物相互作用DDI&quot;]"><strong><a href="self_supervised_denoising/self_supervised_adversarial_distribution_regularization_for_medication_reco/">Self-Supervised Adversarial Distribution Regularization for Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>SARMR 从原始 EHR 中自监督拟合低 DDI 率患者表征的高斯先验，用 GAN 对抗正则患者表征分布以降低药物相互作用，MIMIC-III 上 F1 0.6608、ΔDDI -2.72%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;不可靠实例过滤&quot;,&quot;不确定性建模&quot;,&quot;高斯分布表征&quot;]"><strong><a href="sequential_rec/does_every_data_instance_matter_enhancing_sequential_recommendation_by_elim/">Does Every Data Instance Matter? Enhancing Sequential Recommendation by Eliminating Unreliable Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用高斯分布同时建模实例损失与不确定性，配合不确定性感知图卷积 UGCN 与端到端过滤机制剔除「目标与输入不匹配」的不可靠实例，四数据集采样排序下 NDCG 最高提升 11.98%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;时间间隔建模&quot;,&quot;虚拟中心节点图&quot;]"><strong><a href="sequential_rec/exploring_periodicity_and_interactivity_in_multi_interest_framework_for_seq/">Exploring Periodicity and Interactivity in Multi-Interest Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>IJCAI 2021｜序列推荐论文：以时间间隔矩阵建模兴趣周期性、虚拟中心节点图建模物品交互性，再经自注意力抽取多兴趣向量；在 Amazon/Taobao 全库排序下 NDCG@50 最高提升约 26%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;自监督一致性预训练&quot;,&quot;模仿学习知识蒸馏&quot;]"><strong><a href="sequential_rec/improving_sequential_recommendation_consistency_with_self_supervised_imitat/">Improving Sequential Recommendation Consistency with Self-Supervised Imitation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用三条自监督一致性任务构造教师模型，再以 logit+表征模仿蒸馏进学生序列推荐器，1:99 采样排序下 NDCG@10 提升 2.0%–3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列去噪&quot;,&quot;强化学习策略&quot;,&quot;序列模式挖掘&quot;]"><strong><a href="sequential_rec/pattern_enhanced_contrastive_policy_learning_network_for_sequential_recomme/">Pattern-enhanced Contrastive Policy Learning Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将序列去噪建模为 MDP，用频繁序列模式作先验引导策略网络逐物品打标签，并以正/负子序列延迟奖励做对比学习，Full-Ranking 下 HR@10 相对最佳基线提升 2-4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;预训练语言模型&quot;,&quot;多粒度匹配&quot;]"><strong><a href="llm_rec/unbert_user_news_matching_bert_for_news_recommendation/">UNBERT: User-News Matching BERT for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用预训练 BERT 联合建模用户历史新闻序列与候选新闻，在词级与新闻级双粒度匹配信号上精排，MIND 上 Full 候选内 AUC 最高提升 2.81%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;Next-POI&quot;,&quot;时空群体偏好&quot;]"><strong><a href="cold_start_rec/mfnp_a_meta_optimized_model_for_few_shot_next_poi_recommendation/">MFNP: A Meta-optimized Model for Few-shot Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元学习融合用户与区域群体偏好提升少样本下一 POI 推荐；全库排序 Acc@5 较 STGCN 提升 48.0%/45.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;综述与分类体系&quot;,&quot;域定义与知识迁移&quot;,&quot;负迁移&quot;]"><strong><a href="cross_domain_multi_scenario/cross_domain_recommendation_challenges_progress_and_prospects/">Cross-Domain Recommendation: Challenges, Progress, and Prospects</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以单目标/多域/双目标/多目标四类场景重构 CDR 版图，提出含 CH1–CH6 挑战与全景观分类体系的系统性综述。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图卷积网络&quot;,&quot;共享账户隐潜用户&quot;,&quot;域感知注意力&quot;]"><strong><a href="cross_domain_multi_scenario/da_gcn_a_domain_aware_attentive_graph_convolution_network_for_shared_accoun/">DA-GCN: A Domain-aware Attentive Graph Convolution Network for Shared-account Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建跨域序列图并用域感知注意力 GCN 建模共享账户下 H 个隐潜用户，E 域 MRR@5 相对 PSJNet 提升 15.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;精排&quot;,&quot;Seq2Graph 增广&quot;,&quot;类别感知图注意力&quot;,&quot;多任务类别预测&quot;]"><strong><a href="poi_spatiotemporal_rec/discovering_collaborative_signals_for_next_poi_recommendation_with_iterativ/">Discovering Collaborative Signals for Next POI Recommendation with Iterative Seq2Graph Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将孤立 POI 序列扩增为跨序列一跳图，用类别感知图注意力与位置感知注意力联合建模，并以类别预测多任务缓解稀疏，Gowalla 上 HR@5 提升 49%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;神经架构搜索&quot;,&quot;三阶段推荐模型&quot;,&quot;One-shot 随机搜索&quot;]"><strong><a href="ctr_feature_interaction/ameir_automatic_behavior_modeling_interaction_exploration_and_mlp_investiga/">AMEIR: Automatic Behavior Modeling, Interaction Exploration and MLP Investigation in the Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出三阶段搜索空间与三步 one-shot 随机搜索，自动联合行为建模、特征交互与 MLP；Alimama AUC 0.6383，Criteo/Avazu 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉算子&quot;,&quot;用户行为建模&quot;,&quot;AutoML架构搜索&quot;]"><strong><a href="ctr_feature_interaction/deep_learning_for_click_through_rate_estimation/">Deep Learning for Click-Through Rate Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>深度 CTR 综述：以显式交叉算子、用户行为建模、自动化架构搜索三条主线梳理浅层到深度模型演进与工业落地路线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评测（评估协议）&quot;,&quot;Sampled Metrics&quot;,&quot;Consistency of Ranking Metrics&quot;,&quot;Unbiased Rank Estimator&quot;]"><strong><a href="debias_fairness_industrial/on_sampled_metrics_for_item_recommendation_extended_abstract/">On Sampled Metrics for Item Recommendation (Extended Abstract)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>证明“1 正例 + m 采样负例”的采样排序指标在期望意义下也不保序，给出采样排名的二项分布刻画、无偏排名估计与偏差-方差修正，并在 MovieLens 上验证修正可恢复真实算法排序。</p>
</dd>
</dl>
</section>
</div>
