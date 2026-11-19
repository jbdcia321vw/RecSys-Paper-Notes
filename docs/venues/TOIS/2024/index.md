---
hide:
  - toc
---

# 📖 ACM TOIS 2024 推荐系统论文解读

22 篇 ACM TOIS 2024 论文解读，覆盖 9 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×18 · TOIS 2024 ×4 · 对比学习 ×3 · 因果推断 ×2 · 图卷积网络 ×2 · 扩散模型 ×2 · 离线强化学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/精排" aria-pressed="false">召回/精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积网络" aria-pressed="false">图卷积网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;采样Softmax损失&quot;,&quot;难负样本挖掘&quot;,&quot;图卷积网络&quot;]"><strong><a href="collaborative_filtering/on_the_effectiveness_of_sampled_softmax_loss_for_item_recommendation/">On the Effectiveness of Sampled Softmax Loss for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论揭示SSM损失缓解流行度偏差、挖掘难负样本并对齐DCG，图消息传递可补偿其模长学习缺陷；LightGCN+SSM全库排序Recall@20最高提升41.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/精排&quot;,&quot;多行为推荐&quot;,&quot;级联残差GCN&quot;]"><strong><a href="graph_rec/cascading_residual_graph_convolutional_network_for_multi_behavior_recommend/">Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>无额外参数的级联残差GCN按行为顺序精化嵌入，三数据集全库排序HR@K平均较最优基线提升24.76%/27.28%/25.10%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;软节点聚类&quot;,&quot;Gumbel-Softmax&quot;,&quot;过平滑&quot;]"><strong><a href="graph_rec/cluster_based_graph_collaborative_filtering/">Cluster-based Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Gumbel-Softmax 软节点聚类构建簇特定图，将高阶图卷积限制在同兴趣簇内以滤除噪声并缓解过平滑，Recall@20 相对 SOTA 提升 2.2%–4.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;表征学习/召回&quot;,&quot;对比学习&quot;,&quot;数据增强&quot;]"><strong><a href="self_supervised_denoising/contrastive_self_supervised_learning_in_recommender_systems_a_survey/">Contrastive Self-supervised Learning in Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>综述提出统一框架与三维分类，系统梳理视图生成、对比任务与目标，并给出组件选择指南与开放问题。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比与交互去噪&quot;,&quot;召回&quot;,&quot;多视图对比学习&quot;,&quot;捆绑包推荐&quot;,&quot;图卷积网络&quot;]"><strong><a href="self_supervised_denoising/multicbr_multi_view_contrastive_learning_for_bundle_recommendation/">MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出三视图(UB/UI/BI)表征与“先融合后对比”框架，BI 双向图传播破解捆绑包稀疏，iFashion 上 Recall@20 相对提升 32.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;异质超图&quot;,&quot;双偏好学习&quot;]"><strong><a href="sequential_rec/bi_preference_learning_heterogeneous_hypergraph_networks_for_session_based_/">Bi-Preference Learning Heterogeneous Hypergraph Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用类别内逻辑斯谛价格分档与异质超图三层卷积，联合建模价格/兴趣双偏好，在三个会话数据集全库排序下稳定超越强基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;分布表征&quot;,&quot;不确定性注入&quot;]"><strong><a href="sequential_rec/diffurec_a_diffusion_model_for_sequential_recommendation/">DiffuRec: A Diffusion Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次将扩散模型引入序列推荐，用分布表征替代固定物品向量并注入不确定性，Leave-One-Out + Full-Ranking 协议下在四个数据集 HR/NDCG 最高提升 57.26%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;多关系会话图&quot;,&quot;协同过滤图&quot;]"><strong><a href="sequential_rec/spatio_temporal_contrastive_learning_enhanced_gnns_for_session_based_recomm/">Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RESTC 用空间(MGAT)与时间(SESTrans)双视图对比学习对齐会话表征，并以全局协同过滤图补足短会话稀疏性，六数据集全库排序 HR@20 最高 +27.08%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/重排评估&quot;,&quot;会话多样性&quot;,&quot;准确率-多样性关系&quot;]"><strong><a href="sequential_rec/understanding_diversity_in_session_based_recommendation/">Understanding Diversity in Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一16个会话推荐基线的准确率/多样性评测，揭示非多样化模型亦可高多样性，且准确率-多样性关系随模型与数据变化。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多模态推荐&quot;,&quot;召回/精排&quot;,&quot;动态多模态融合&quot;,&quot;元学习&quot;]"><strong><a href="multimodal_rec/dynamic_multimodal_fusion_via_meta_learning_towards_micro_video_recommendat/">Dynamic Multimodal Fusion via Meta-Learning Towards Micro-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以元学习为每个微视频生成 item-specific 融合网络，并用 CP 分解压缩；三数据集全库排序下 NDCG@10 提升 4.46%–6.54%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/排序&quot;,&quot;图提示微调&quot;,&quot;对比学习&quot;,&quot;参数高效迁移&quot;]"><strong><a href="cross_domain_multi_scenario/contrastive_graph_prompt_tuning_for_cross_domain_recommendation/">Contrastive Graph Prompt-tuning for Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>冻结GNN、仅调个性化图提示的跨域Top-K推荐；四组Amazon域对最高降74%调参，冷启动+11.41%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;三重序列建模&quot;,&quot;对比学习&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/triple_sequence_learning_for_cross_domain_recommendation/">Triple Sequence Learning for Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Tri-CDR 联合建模源域/目标域/混合三条行为序列，用三重跨域注意力与三重对比学习（粗粒度对齐+细粒度区分）实现正向迁移，六组跨域设定全库排序一致提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回/精排&quot;,&quot;扩散模型&quot;,&quot;时空双图&quot;]"><strong><a href="poi_spatiotemporal_rec/a_diffusion_model_for_poi_recommendation/">A Diffusion model for POI recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以时空双图编码用户与位置，并用用户条件扩散反向SDE采样空间后验；四个LBSN数据集全库排序下R@10较SOTA提升5.3%–11.9%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多粒度签到信号&quot;,&quot;区域感知通道加权&quot;]"><strong><a href="poi_spatiotemporal_rec/a_multi_channel_next_poi_recommendation_framework_with_multi_granularity_ch/">A Multi-Channel Next POI Recommendation Framework with Multi-Granularity Check-in Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MCMG 用 POI/区域/类目三通道自注意力 + 全局 POI 图 GCN，并按同区域/跨区域分组自适应加权融合，四城市 HR/NDCG 平均提升 9.7%/14.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;交互式策略/重排&quot;,&quot;离线强化学习&quot;,&quot;因果推断&quot;,&quot;过滤气泡&quot;]"><strong><a href="rl_bandit_rec/cirs_bursting_filter_bubbles_by_counterfactual_interactive_recommender_syst/">CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将因果推断嵌入离线RL，用曝光核生成反事实满意度奖励，在KuaiEnv全观测环境击破过滤气泡并提升长期累积满意度。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/Top-K 排序&quot;,&quot;视觉感知推荐系统投毒&quot;,&quot;引导扩散模型&quot;,&quot;对抗样本不可感知性&quot;]"><strong><a href="debias_fairness_industrial/adversarial_item_promotion_on_visually_aware_recommender_systems_by_guided_/">Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用引导扩散模型把对抗扰动注入高斯噪声，生成高保真对抗商品图以提升长尾商品曝光，全库排序下 ER@5 最高 +56.8%、FID 降 87.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;视频长度去偏&quot;,&quot;多任务学习&quot;,&quot;长度条件采样&quot;]"><strong><a href="debias_fairness_industrial/alleviating_video_length_effect_for_micro_video_recommendation/">Alleviating Video-Length Effect for Micro-video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出VLDRec，用长度分组、进度标签与长度条件采样多任务去偏视频时长效应，在快手/微信固定总时长评估上@120提升1.81%/11.32%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全库排序/端到端协同过滤&quot;,&quot;因果解耦&quot;,&quot;偏好漂移&quot;,&quot;时序VAE&quot;]"><strong><a href="debias_fairness_industrial/causal_disentangled_recommendation_against_user_preference_shifts/">Causal Disentangled Recommendation Against User Preference Shifts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出CDR，以时序VAE推断未观测因素并建模多环境偏好漂移，用可学习稀疏矩阵解耦类别偏好；全库排序下R@10较次优提升19.5%–28.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回/精排&quot;,&quot;因果推断&quot;,&quot;反事实数据增广&quot;,&quot;后门调整与 IPW/DR&quot;]"><strong><a href="debias_fairness_industrial/causal_inference_in_recommender_systems_a_survey_and_future_directions/">Causal Inference in Recommender Systems: A Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以数据偏差、数据缺失/噪声、超越精度三类病理为主轴，系统梳理 40 篇工作中后门调整、IPW/DR、反事实增广与因果发现四类工具的用法、代价与开放问题</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;事后解释与公平性诊断&quot;,&quot;反事实解释&quot;,&quot;离线强化学习&quot;,&quot;异构图属性归因&quot;]"><strong><a href="debias_fairness_industrial/counterfactual_explanation_for_fairness_in_recommendation/">Counterfactual Explanation for Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用异构图属性节点做反事实解释，以离线强化学习搜索最小属性集解释曝光不公平，擦除评估下 NDCG 与 Gini 双优。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦推荐端到端训练&quot;,&quot;图像投毒攻击&quot;,&quot;扩散模型净化与检测&quot;]"><strong><a href="debias_fairness_industrial/manipulating_visually_aware_federated_recommender_systems_and_its_counterme/">Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示视觉联邦推荐中模型投毒失效、图像投毒生效的新后门，并提出扩散模型引导净化+免训练检测的 GDMPD 防御，ER@5 由 1.0 压至 0.0。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果/信息论&quot;]"><strong><a href="debias_fairness_industrial/robust_collaborative_filtering_to_popularity_distribution_shift/">Robust Collaborative Filtering to Popularity Distribution Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>PopGo 用同构 shortcut model 量化交互级流行度捷径并以掩码预测去偏，在 OOD/ID 全库排序上同时提升。</p>
</dd>
</dl>
</section>
</div>
