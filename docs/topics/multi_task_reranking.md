---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排 (Multi-Task Learning & Listwise Re-ranking)

> **研究范畴**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

当前方向共收录 **252** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 去偏公平与可信推荐 ×21 · 大语言模型推荐 ×11 · 多行为推荐 ×9 · CTR预估与特征交叉 ×5 · 生成式推荐 ×5 · 知识蒸馏 ×5 · Actor-Critic ×4 · 双边公平 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">104</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">90</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">65</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">56</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">23</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">22</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">21</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">20</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Actor-Critic" aria-pressed="false">Actor-Critic<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双边公平" aria-pressed="false">双边公平<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式重排" aria-pressed="false">生成式重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级重排" aria-pressed="false">列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标优化" aria-pressed="false">多目标优化<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托前沿" aria-pressed="false">帕累托前沿<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixture-of-Experts" aria-pressed="false">Mixture-of-Experts<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Off-Policy Learning" aria-pressed="false">Off-Policy Learning<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏置" aria-pressed="false">位置偏置<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平性-相关性联合评测" aria-pressed="false">公平性-相关性联合评测<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级上下文建模" aria-pressed="false">列表级上下文建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态损失加权" aria-pressed="false">动态损失加权<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实推断" aria-pressed="false">反事实推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排级联" aria-pressed="false">召回+精排级联<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线知识蒸馏" aria-pressed="false">在线知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多归因学习" aria-pressed="false">多归因学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性-相关性权衡" aria-pressed="false">多样性-相关性权衡<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为建模" aria-pressed="false">多行为建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="奖励分解" aria-pressed="false">奖励分解<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排列不变性" aria-pressed="false">排列不变性<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提供者公平" aria-pressed="false">提供者公平<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光公平" aria-pressed="false">曝光公平<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="校准推荐" aria-pressed="false">校准推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="样本选择偏差" aria-pressed="false">样本选择偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="点击模型" aria-pressed="false">点击模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评估" aria-pressed="false">离线评估<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏MoE" aria-pressed="false">稀疏MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联图卷积" aria-pressed="false">级联图卷积<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放定律" aria-pressed="false">缩放定律<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观看时长建模" aria-pressed="false">观看时长建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移" aria-pressed="false">负迁移<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似算法" aria-pressed="false">近似算法<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期留存优化" aria-pressed="false">长期留存优化<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;DQN 多子动作联合决策&quot;,&quot;Dueling 价值-优势分解&quot;,&quot;广告-推荐列表混排&quot;]"><strong><a href="../../venues/AAAI/2021/rl_bandit_rec/dear_deep_reinforcement_learning_for_online_advertising_impression_in_recom/">DEAR: Deep Reinforcement Learning for Online Advertising Impression in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用单一 Dueling-DQN 同时决策「是否插广告/插哪条/插哪个槽位」，把前向复杂度从 O(|A|·L) 降到 O(|A|)，在抖音 100 万会话上累积奖励较 HDQN 提升 6.71%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;图Transformer&quot;,&quot;知识图谱&quot;]"><strong><a href="../../venues/AAAI/2021/graph_rec/knowledge_enhanced_hierarchical_graph_transformer_network_for_multi_behavio/">Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>时间感知图Transformer融合多行为与知识图谱，Leave-One-Out+99负采样评测，Retail HR@10相对提升23.1%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/策略层端到端&quot;,&quot;世界模型 (World Model)&quot;,&quot;通用价值函数 (UVF)&quot;,&quot;解耦价值估计&quot;]"><strong><a href="../../venues/AAAI/2021/rl_bandit_rec/reinforcement_learning_with_a_disentangled_universal_value_function_for_ite/">Reinforcement Learning with a Disentangled Universal Value Function for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以目标导向世界模型解耦环境动态与高方差奖励，构建通用价值函数 $Q=g^{\top}M(s,a,g)$，在三个仿真环境与网易线上场景提升累积点击 3.7%–10.7%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Cross-Task Knowledge Distillation&quot;,&quot;Quadruplet Ranking&quot;,&quot;Calibrated Distillation&quot;]"><strong><a href="../../venues/AAAI/2022/multi_task_reranking/cross_task_knowledge_distillation_in_multi_task_recommendation/">Cross-Task Knowledge Distillation in Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用四元组辅助排序任务捕获跨任务细粒度排序知识，经校准蒸馏与误差纠错同步训练，在 TikTok/微信朋友圈样本级 AUC/Multi-AUC 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;负迁移&quot;,&quot;停梯度门控&quot;,&quot;任务专属嵌入&quot;]"><strong><a href="../../venues/AAAI/2024/multi_task_reranking/stem_unleashing_the_power_of_embeddings_for_multi_task_recommendation/">STEM: Unleashing the Power of Embeddings for Multi-task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按任务反馈分桶诊断出 Comparable 子集负迁移，提出共享+任务专属双轨嵌入与「全前向任务特定反向」停梯度门控，三数据集平均 AUC 超越 PLE 约 4e-3 并落地腾讯广告 A/B。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;梯度手术&quot;,&quot;个性化门控&quot;]"><strong><a href="../../venues/AAAI/2025/multi_task_reranking/direct_routing_gradient_drgrad_a_personalized_information_surgery_for_multi/">Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用梯度余弦相似度做路由+Split-MMoE+PPNet个性化门控，在150亿样本工业数据上Click AUC提升0.0027且不增推理延迟。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;紧迫度感知&quot;,&quot;时间位置编码&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP Findings&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回+精排两阶段&quot;,&quot;联邦学习&quot;,&quot;本地差分隐私&quot;]"><strong><a href="../../venues/ACL_EMNLP/2021/debias_fairness_industrial/uni_fedrec_a_unified_privacy_preserving_news_recommendation_framework_for_m/">Uni-FedRec: A Unified Privacy-Preserving News Recommendation Framework for Model Training and Online Serving</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>联邦多兴趣召回+客户端本地精排：兴趣分解-聚合+LDP 噪声，MIND R@100 +27.7%，隐私泄漏率 17.0%→0.55%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL_EMNLP&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;重排&quot;,&quot;Denoising Attention&quot;,&quot;Query-aware User Modeling&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/self_supervised_denoising/denoising_attention_for_query_aware_user_modeling_in_personalized_search/">Denoising Attention for Query-aware User Modeling in Personalized Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ACL/EMNLP 2024 [自监督对比学习与交互去噪] 以有界对齐+阈值移位+ReLU 门控与 max-归一化重构查询感知用户建模，Top-1000 重排协议下 MAP 较最佳注意力基线提升 14–17%</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Pointwise-Pairwise&quot;,&quot;RTL Rank Aggregation&quot;,&quot;Learning-to-Rank&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/multi_task_reranking/efficient_pointwise_pairwise_learning_to_rank_for_news_recommendation/">Efficient Pointwise-Pairwise Learning-to-Rank for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GLIMPSE 以单次多任务生成式微调同时习得点式相关性与成对偏好，再用 O(K) 右到左相邻交换精排，在 MIND/Adressa 全库排序上超越 UniTRec 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;隐式最高级查询&quot;,&quot;列表级重排&quot;,&quot;深思式提示&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/generative_product_recommendations_for_implicit_superlative_queries/">Generative Product Recommendations for Implicit Superlative Queries</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 SUPERB 四级标注体系与深思式属性生成，用 LLM 列表级重排处理隐式最高级商品查询，nDCG@10 相对 BM25 提升约 18%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;多模态推荐&quot;,&quot;重排&quot;,&quot;语义匹配分数&quot;,&quot;多尺度聚合&quot;,&quot;评论有用性预测&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/multimodal_rec/premise_matching_based_prediction_for_accurate_review_recommendation/">PREMISE: Matching-based Prediction for Accurate Review Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以多尺度多字段语义匹配分数替代跨模态融合表征，用于多模态评论有用性排序，MAP 较 GBDT 提升 5–17 点且训练提速 42%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;咨询价值评估&quot;,&quot;咨询-行为交互&quot;,&quot;个性化搜索&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/similarity_value_consultation_value_assessment_and_alignment_for_personaliz/">Similarity = Value? Consultation Value Assessment and Alignment for Personalized Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>指出语义相似度无法代理咨询价值，提出场景范围/后验行为/时间衰减三维价值评估与 CAI 对齐的 VAPS，检索排序双任务全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;会话内上下文&quot;,&quot;曝光偏置建模&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/intra_session_context_aware_feed_recommendation_in_live_systems/">Intra-session Context-aware Feed Recommendation in Live Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将信息流会话建模为点击/滑动的马尔可夫事件链，以会话内曝光序列编码上下文，多任务联合学习并用 GPT 式贪心序列生成优化总曝光与总点击；线上人均曝光 +1.16%、转化 +1.56%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;早期特征交互&quot;,&quot;对比学习正则&quot;]"><strong><a href="../../venues/CIKM/2022/ctr_feature_interaction/inttower_the_next_generation_of_two_tower_model_for_pre_ranking_system/">IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Light-SE 特征选择、FE-Block 早期细粒度交互与对比交互正则，在保持双塔解耦部署的前提下逼近单塔精排效果。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;端侧推理&quot;,&quot;自适应束搜索&quot;,&quot;实时特征交叉&quot;]"><strong><a href="../../venues/CIKM/2022/multi_task_reranking/real_time_short_video_recommendation_on_mobile_devices/">Real-time Short Video Recommendation on Mobile Devices</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>端侧轻量排序模型融合实时反馈交叉特征，并以自适应束搜索做上下文感知重排，线上有效播放/点赞/关注分别 +1.28%/+8.22%/+13.6%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;校准推荐&quot;,&quot;置信度加权&quot;]"><strong><a href="../../venues/CIKM/2022/multi_task_reranking/towards_confidence_aware_calibrated_recommendation/">Towards Confidence-aware Calibrated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出置信度感知校准重排CCL，用用户画像大小加权MILP平衡相关性、校准与多样性，MovieLens上nDCG与覆盖率提升</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;粗排&quot;,&quot;排序一致性对齐&quot;,&quot;分块采样&quot;,&quot;ECPM 排序学习&quot;]"><strong><a href="../../venues/CIKM/2023/multi_task_reranking/copr_consistency_oriented_pre_ranking_for_online_advertising/">COPR: Consistency-Oriented Pre-Ranking for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将粗排↔精排的分数对齐松弛为 ECPM 排序对齐，用分块采样与 ΔNDCG 加权 pairwise 损失优化一致性，淘宝线上 +12.3% CTR、+5.6% RPM。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;任务特定底层表征&quot;,&quot;超网络条件Transformer&quot;,&quot;SENet式特征重标定&quot;]"><strong><a href="../../venues/CIKM/2023/multi_task_reranking/deep_task_specific_bottom_representation_network_for_multi_task_recommendat/">Deep Task-specific Bottom Representation Network for Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CIKM 2023 多任务推荐论文，提出 DTRN 以超网络条件 Transformer 与 SENet 式任务特定底层表征缓解负迁移；在电商与 Tenrec 样本级 AUC/LogLoss 上全面优于 MMoE/PLE/AITM，在线进入率 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;稀疏MoE路由&quot;,&quot;多模态特征融合&quot;]"><strong><a href="../../venues/CIKM/2023/multi_task_reranking/efficient_multi_task_learning_via_generalist_recommender/">Efficient Multi-Task Learning via Generalist Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GRec 用多模态 Wide&amp;Deep + 并行 Transformer + 任务-句子级稀疏 MoE 路由，在电信场景离线 AP 与线上 A/B 全面超越 MMoE 与单任务模型。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Delayed Feedback&quot;,&quot;Entire Space Modeling&quot;,&quot;多任务辅助学习&quot;]"><strong><a href="../../venues/CIKM/2023/ctr_feature_interaction/entire_space_cascade_delayed_feedback_modeling_for_effective_conversion_rat/">Entire Space Cascade Delayed Feedback Modeling for Effective Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 ECVR 分解为 CVR×(1−RFR)，用全空间 CVRFR 任务加级联延迟辅助塔一并解决 DS/SSB/CDF，线上 ECVR +5.21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排/出价&quot;,&quot;Bid Shading&quot;,&quot;Multi-task Learning&quot;]"><strong><a href="../../venues/CIKM/2023/ctr_feature_interaction/mebs_multi_task_end_to_end_bid_shading_for_multi_slot_display_advertising/">MEBS: Multi-task End-to-end Bid Shading for Multi-slot Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向多广告位展示广告，用多任务端到端出价折减最大化广告主盈余，离线 Surplus 与线上 GMV/ROI 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;投影解耦&quot;,&quot;多专家网络&quot;]"><strong><a href="../../venues/CIKM/2023/multi_task_reranking/parallel_knowledge_enhancement_based_framework_for_multi_behavior_recommend/">Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以并行流知识投影增强级联多行为 GNN，并用投影解耦多专家消除 MTL 梯度冲突，三数据集 HR@10 最高提升 95.16%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排/模型选择&quot;,&quot;帕累托前沿&quot;,&quot;多目标优化&quot;,&quot;校准推荐&quot;]"><strong><a href="../../venues/CIKM/2023/multi_task_reranking/post_hoc_selection_of_pareto_optimal_solutions_in_search_and_recommendation/">Post-hoc Selection of Pareto-Optimal Solutions in Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 PDU 从帕累托前沿事后挑出唯一可部署解：用样本级 utopia 距离替代聚合点距离，并支持用户级校准。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;结构化行列式点过程 SDPP&quot;,&quot;集合级优化准则&quot;,&quot;多样性建模&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/a_universal_sets_level_optimization_framework_for_next_set_recommendation/">A Universal Sets-level Optimization Framework for Next Set Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用结构化行列式点过程 (SDPP) 将时序集合整体视为元素并设计集合级条件似然，联合共现表征与类别感知多样性核，在四个数据集上 Recall/NDCG 提升 6–25% 且同时提升多样性。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;自校正点过程&quot;,&quot;行列式多样性&quot;,&quot;指针式序列解码&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/divnet_diversity_aware_self_correcting_sequential_recommendation_networks/">DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将整页集体推荐改写为带因果掩码的指针式序列选择，用自校正解码建模物品相互激发，并以相似度行列式注入多样性；线上 page CTR +8.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;粗排&quot;,&quot;Pareto优化&quot;,&quot;遗忘曲线&quot;]"><strong><a href="../../venues/CIKM/2024/multi_task_reranking/pareto_based_multi_objective_recommender_system_with_forgetting_curve/">Pareto-based Multi-Objective Recommender System with Forgetting Curve</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Ebbinghaus 遗忘曲线建模快滑负反馈，并以 Pareto 求解器自适应权衡粗排一致性与 recency，KuaiRec 与微信视频号 A/B 验证，GMV +1.45%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;内容无关审核&quot;,&quot;立场中立&quot;,&quot;观点极化&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/content_agnostic_moderation_for_stance_neutral_recommendation/">Content-Agnostic Moderation for Stance-Neutral Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出不依赖内容特征的立场中立化后处理审核，用双聚类离散化打散用户-物品共簇，仿真闭环中显著降低立场偏度且 CTR 仅降约 11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;有序多token预测&quot;,&quot;Rerank-DPO&quot;]"><strong><a href="../../venues/CIKM/2025/multi_task_reranking/gref_a_unified_generative_framework_for_efficient_reranking_via_ordered_mul/">GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双向编码器+动态匹配自回归解码器统一生成式重排，Rerank-DPO 端到端对齐用户偏好，OMTP 多token并行解码将延迟压至 12.97ms，快手线上转发+2.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;生成式表征对齐&quot;,&quot;多头子空间匹配&quot;]"><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/hit_model_a_hierarchical_interaction_enhanced_two_tower_model_for_pre_ranki/">HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双生成器预生成跨塔粗粒度向量+多头子空间细粒度匹配，在保持双塔预缓存效率下逼近全交互模型精度，腾讯线上GMV+1.66%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;观看时长建模&quot;,&quot;因果去偏标注&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/leveraging_watch_time_feedback_for_short_video_recommendations_a_causal_lab/">Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分位数把观看时长转成 WPR 与二值标签，并在标注阶段按视频时长分组做因果去偏，离线全指标最优、线上观看时长 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标召回&quot;,&quot;目标条件化 Transformer&quot;,&quot;动态配额分配&quot;]"><strong><a href="../../venues/CIKM/2025/multi_task_reranking/mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;因果解耦&quot;,&quot;停留时间预测&quot;,&quot;负迁移&quot;]"><strong><a href="../../venues/CIKM/2025/multi_task_reranking/orca_mitigating_over_reliance_for_multi_task_dwell_time_prediction_with_cau/">ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用因果解耦显式建模并减去CTR对停留时间预测的负迁移，恢复中时长分桶语义，DT指标平均提升10.6%且不损CTR。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;有序回归&quot;,&quot;树结构学习&quot;]"><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/personalized_tree_based_progressive_regression_model_for_watch_time_predict/">Personalized Tree-Based Progressive Regression Model for Watch-Time Prediction in Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将TPM固定满二叉树改为样本级可学习剪枝树，并用IPS校正条件建模选择偏差，离线MAE/XAUC全面领先，线上观看时长+0.428%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉&quot;,&quot;稀疏MoE&quot;,&quot;缩放定律&quot;]"><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/rankmixer_scaling_up_ranking_models_in_industrial_recommenders/">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用无参多头Token Mixing替代自注意力、Per-token FFN隔离特征子空间，配合ReLU路由稀疏MoE，在抖音万亿级数据上将MFU从4.5%提至45%，参数扩100倍而延迟不增。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多归因学习&quot;,&quot;多任务学习&quot;,&quot;辅助训练&quot;]"><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/see_beyond_a_single_view_multi_attribution_learning_leads_to_better_convers/">See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多归因标签作为辅助任务联合学习，用笛卡尔积辅助训练增强监督信号，离线GAUC+0.51%、线上ROI+2.6%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;模仿学习&quot;,&quot;用户留存&quot;,&quot;行为克隆&quot;]"><strong><a href="../../venues/CIKM/2025/rl_bandit_rec/stratified_expert_cloning_for_retention_aware_recommendation_at_scale/">Stratified Expert Cloning for Retention-Aware Recommendation at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层专家行为克隆替代 RL 探索，配合动作熵正则与自适应专家选择，在快手双平台 A/B 中提升活跃天数 0.098%/0.122%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;列表级重排&quot;,&quot;树状上下文缓存&quot;]"><strong><a href="../../venues/CIKM/2025/multi_task_reranking/you_only_evaluate_once_a_tree_based_rerank_method_at_meituan/">You Only Evaluate Once: A Tree-based Rerank Method at Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用树状多尺度上下文提取+上下文缓存实现单阶段全排列重排，离线GAUC显著提升并落地美团外卖。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多模态推荐&quot;,&quot;重排&quot;,&quot;双频图证据&quot;,&quot;候选级行为残差校准&quot;]"><strong><a href="../../venues/CIKM/2026/multimodal_rec/bridge_behavior_guided_residual_integration_with_dual_frequency_graph_evide/">BRIDGE: Behavior-Guided Residual Integration with Dual-Frequency Graph Evidence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示跨视图对齐存在有效区间与频谱共享/私有分裂，提出双频图证据+行为符号残差仅在 top-K 候选内校准，Baby/Sports/Electronics 全库排序 R@20 最高提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;重排&quot;,&quot;域感知条件化&quot;,&quot;跨域序列建模&quot;,&quot;游戏匹配&quot;]"><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/champ_cross_domain_hybrid_architecture_for_matchmaking_and_prediction_in_on/">CHAMP: Cross-domain Hybrid Architecture for Matchmaking and Prediction in Online Multi-Player Games</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 DAKE 域 token 条件化 OmniNet 编码器，单模型服务多游戏模式匹配重排，Elite 胜率预测 67.73%，新手 5 分钟击杀崩盘率降 20.73%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Off-Policy Learning&quot;,&quot;Top-k Policy Decomposition&quot;]"><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/efficient_offline_learning_of_ranking_policies_via_top_k_policy_decompositi/">Efficient Offline Learning of Ranking Policies via Top-$k$ Policy Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将排序策略分解为 top-$k$ 策略梯度与 bottom 回归两段，在 CPC 条件下无偏且大幅降方差，小样本与大规模排序空间下显著优于 IPS/DR/RIPS/IIPS。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;任务依赖图&quot;,&quot;低秩图生成&quot;,&quot;自适应渐进掩码&quot;]"><strong><a href="../../venues/CIKM/2026/multi_task_reranking/personalized_task_dependency_graphs_for_mitigating_signal_erosion_in_multi_/">Personalized Task Dependency Graphs for Mitigating Signal Erosion in Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用低秩个性化任务依赖图+硬因果掩码GCN替代刚性漏斗，缓解深层稀疏任务信号侵蚀，离线AUC最高+1.45%，线上CVR+1.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;模型合并&quot;,&quot;可控多目标推荐&quot;]"><strong><a href="../../venues/CIKM/2026/multi_task_reranking/recsoup_efficient_model_merging_approach_for_controllable_multi_objective_r/">RecSoup: Efficient Model Merging Approach for Controllable Multi-Objective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用剪枝+可学习门控做任务向量合并，实现推理时可控多目标推荐，HV 提升 8.5% 且延迟快 165–228 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;轮播界面评估&quot;,&quot;位置折扣&quot;,&quot;眼动数据&quot;]"><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/revisiting_n2dcg_an_empirically_grounded_reformulation_of_carousel_recommen/">Revisiting N2DCG: An Empirically Grounded Reformulation of Carousel Recommendation Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>修正 N2DCG 的理想排序类别约束与镜像 F 型折扣，提出类别感知归一化与行-页折扣，在眼动数据与 2 万次布局仿真中显著提升评估保真度。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;自蒸馏&quot;,&quot;推理增强&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/selfdr_self_distillation_from_reasoning_for_llm_based_recommendation/">SelfDR: Self-Distillation from Reasoning for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用下游推荐奖励训练同源 Reasoner 生成理由，构建推理增强教师，再自蒸馏出免推理的直接推荐学生，重排 HR@1 提升约 15%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;序列列表重排&quot;,&quot;离线策略优化&quot;,&quot;多模态证据累积&quot;]"><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sequential_multimodal_evidence_optimization_for_product_media_ranking_in_e_/">Sequential Multimodal Evidence Optimization for Product Media Ranking in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将商品媒体排序建模为证据累积序列决策，两阶段解耦效用学习与生存加权自回归策略，离线 DR-CVR +5.4%、MSC −15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Ranking Transformer&quot;,&quot;稀疏MoE&quot;,&quot;生成式预训练&quot;]"><strong><a href="../../venues/CIKM/2026/multi_task_reranking/sort_a_systematically_optimized_ranking_transformer_for_industrial_scale_re/">SORT: A Systematically Optimized Ranking Transformer for Industrial-scale Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以请求为中心组织样本、局部注意力+查询剪枝、生成式预训练与稀疏MoE重构工业精排Transformer，线上GMV+8.65%且延迟降62%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Anchor Token&quot;,&quot;任务边界掩码&quot;,&quot;统一主干&quot;]"><strong><a href="../../venues/CIKM/2026/multi_task_reranking/task_blind_no_more_multi_task_information_flow_in_unified_ranking_backbones/">Task-Blind No MORE: Multi-Task Information Flow in Unified Ranking Backbones</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多任务信息流下沉至统一排序主干，以共享/私有 Anchor Token 逐层共演化，离线七任务全胜、线上时长+3.0%、P99 延迟降约30%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;半自回归生成&quot;,&quot;在线知识蒸馏&quot;,&quot;个性化位置编码&quot;]"><strong><a href="../../venues/DASFAA/2026/multi_task_reranking/efficient_personalized_reranking_with_semi_autoregressive_generation_and_on/">Efficient Personalized Reranking with Semi-Autoregressive Generation and Online Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用半自回归分块生成做教师、在线蒸馏轻量打分网络做学生，并以用户画像门控与个性化位置编码深化用户-物品交互，重排精度领先且推理延迟减半。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;多任务学习与列表级重排&quot;,&quot;端到端生成式&quot;,&quot;任务级 LoRA-MoE&quot;,&quot;收敛感知采样调度&quot;]"><strong><a href="../../venues/DASFAA/2026/multi_task_reranking/generative_representational_learning_of_foundation_models_for_recommendatio/">Generative Representational Learning of Foundation Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建 3 表征+10 生成任务的统一推荐基座数据集，用任务级低秩专家 MoE、收敛感知采样调度与模型合并在 Mistral-7B 上统一多任务训练，13 任务平均排名最优。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;消费者公平&quot;,&quot;可复现性基准&quot;,&quot;Demographic Parity&quot;]"><strong><a href="../../venues/ECIR/2022/debias_fairness_industrial/consumer_fairness_in_recommender_systems_contextualizing_definitions_and_mi/">Consumer Fairness in Recommender Systems: Contextualizing Definitions and Mitigations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复现 8 个消费者公平缓解方法并在统一协议下对照，揭示去偏对效用冲击微弱、DP 与 KS 两类公平度量互不一致、且差异影响并非总伤及少数群体。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;LLM推荐&quot;,&quot;重排&quot;,&quot;Zero-Shot Ranking&quot;,&quot;Prompting &amp; Bootstrapping&quot;]"><strong><a href="../../venues/ECIR/2024/llm_rec/large_language_models_are_zero_shot_rankers_for_recommender_systems/">Large Language Models are Zero-Shot Rankers for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将推荐形式化为条件排序，用提示与自举缓解顺序/位置/流行度偏差，在 ML-1M/Games 的 20 候选零样本排序中显著优于 UniSRec。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;后处理分桶&quot;,&quot;提供者公平&quot;,&quot;类型校准&quot;]"><strong><a href="../../venues/ECIR/2024/multi_task_reranking/moregin_multi_objective_recommendation_at_the_global_and_individual_levels/">MOReGIn: Multi-Objective Recommendation at the Global and Individual Levels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>分桶后处理重排同时实现全局提供者地理公平与个体类型校准，在 Movies/Songs 双数据集上将 ΔTotal 降为 0、校准误差降一个量级。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/重排与评测协议&quot;,&quot;top-n 选择策略&quot;,&quot;超参优化偏差&quot;,&quot;显著性检验&quot;]"><strong><a href="../../venues/ECIR/2024/debias_fairness_industrial/revealing_the_hidden_impact_of_top_n_metrics_on_optimization_in_recommender/">Revealing the Hidden Impact of Top-N Metrics on Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>穷举 252 种 top-n 选择策略 × 20 数据集 × 11 算法，证明 top-n 评估截断带来的优化偏差在统计上可忽略（增益 ≤1.5%）</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;描述性货架&quot;,&quot;LLM 元数据增强&quot;,&quot;列表解释&quot;]"><strong><a href="../../venues/ECIR/2025/llm_rec/contextualizing_spotify_s_audiobook_list_recommendations_with_descriptive_s/">Contextualizing Spotify&#x27;s Audiobook List Recommendations with Descriptive Shelves</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 抽取十类书目描述符并生成多样化描述性货架标题，两轮线上 A/B 测试验证发现性与互动指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;拼团推荐&quot;,&quot;专家网络&quot;,&quot;门控&quot;]"><strong><a href="../../venues/ICDE/2023/multi_task_reranking/group_buying_recommendation_model_based_on_multi_task_learning/">Group Buying Recommendation Model Based on Multi-task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将拼团推荐拆为发起人选品与参与者选团，用三视图GCN、专家门控与双辅助损失联合建模；Task B MRR@10提升71.65%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;k-DPP&quot;,&quot;集合级排序准则&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="../../venues/ICDE/2024/collaborative_filtering/learning_k_determinantal_point_processes_for_personalized_ranking/">Learning k-Determinantal Point Processes for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 k-DPP 集合概率比较替代 BPR 的点对比较，把相关性排序与多样性排序统一进同一优化准则，全库排序下 Recall@5 相对最优基线提升 8–11%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023 Workshop&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;多任务学习&quot;,&quot;行为数据失衡&quot;]"><strong><a href="../../venues/ICDM/2023/graph_rec/alleviating_behavior_data_imbalance_for_multi_behavior_graph_collaborative_/">Alleviating Behavior Data Imbalance for Multi-Behavior Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多任务 LightGCN + 富行为表征注入稀疏行为 + 梯度路径隔离缓解点击/加购/购买数据失衡，Beibei/Taobao 上 Recall@20 最高提升 35.84%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;推荐遗忘&quot;,&quot;因果反事实解释&quot;,&quot;输出级分数干预&quot;]"><strong><a href="../../venues/ICDM/2026/debias_fairness_industrial/explain_then_forget_recommendation_unlearning_under_asymmetric_access_const/">Explain, Then Forget: Recommendation Unlearning under Asymmetric Access Constraint</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 MLaaS 非对称访问约束的免参数推荐遗忘框架，用因果解释定位最小支持集并做输出级分数干预，遗忘效果逼近全量重训且延迟降低 9–55 倍。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与工业系统&quot;,&quot;重排&quot;,&quot;反事实排序&quot;,&quot;双边市场实验设计&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2021/debias_fairness_industrial/a_b_testing_for_recommender_systems_in_a_two_sided_marketplace/">A/B Testing for Recommender Systems in a Two-sided Marketplace</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>UniCoRn 以反事实排序统一实现生产者侧 A/B 实验，α 显式权衡精度与成本，线上 viewee 指标 +0.51%/+0.57%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回+精排级联&quot;,&quot;两阶段推荐&quot;,&quot;Mixture-of-Experts&quot;,&quot;Bandit遗憾分解&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2021/multi_task_reranking/on_component_interactions_in_two_stage_recommender_systems/">On component interactions in two-stage recommender systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用遗憾分解揭示两阶段推荐器组件交互，提出MoE学习物品池分配，提升prec@5/recall@5。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;富媒体广告拍卖&quot;,&quot;单调分配规则&quot;,&quot;无政府代价 PoA&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/simple_mechanisms_for_welfare_maximization_in_rich_advertising_auctions/">Simple Mechanisms for Welfare Maximization in Rich Advertising Auctions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以绝对性价比单调分配广告空间再随机混合最大价值创意，得 3-近似诚实富媒体广告拍卖，GSP 下 PoA≤6，实测达 VCG 的 0.95 且快 9 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ICML 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Pairwise Ranking Loss&quot;,&quot;Welfare Maximization&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/pairwise_ranking_losses_of_click_through_rates_prediction_for_welfare_maxim/">Pairwise Ranking Losses of Click-Through Rates Prediction for Welfare Maximization in Ad Auctions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出面向广告拍卖福利最大化的加权 pairwise ranking loss，并借助教师网络校准；Criteo/DeepFM 上 welfare 提升 1.47%，AUC 基本持平。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;分数去噪&quot;,&quot;对抗噪声生成&quot;,&quot;多阶段一致性&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/denoising_neural_reranker_for_recommender_systems/">Denoising Neural Reranker for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将重排建模为对召回打分的去噪过程，用对抗噪声生成器与三项损失实现 retriever-aware 重排，三数据集与线上 A/B（realshow +1.089%）验证。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;约束感知决策&quot;,&quot;可持续饮食推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/cold_start_rec/eating_for_a_sustainable_planet_personalized_sustainable_diet_recommendatio/">Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将可持续饮食推荐重构为约束感知决策问题，用可学习用户约束阈值替代偏好建模，在自建 SusDiet 上兼顾偏好精度与四维可持续性。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;目标条件监督学习&quot;,&quot;多目标推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式列表排序&quot;,&quot;组相对优化&quot;,&quot;缩放定律&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goalrank_group_relative_optimization_for_a_large_ranking_model/">GoalRank: Group-Relative Optimization for a Large Ranking Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>证明单生成器排序器近似误差严格优于多生成器-评估器，提出组相对优化训练大排序模型，离线 H@6 最高 +25.4%，线上全量部署。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Semi-Bandit&quot;,&quot;Estimation-to-Decisions&quot;,&quot;对数障碍探索&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/rl_bandit_rec/optimal_and_efficient_contextual_combinatorial_semi_bandits_with_general_fu/">Optimal and Efficient Contextual Combinatorial Semi-bandits with General Function Approximation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SquareCB.Comb，用回归 oracle + 对数障碍边际优化实现通用函数逼近下 CCSB 的极小极大最优 regret，并在网页推荐与最短路任务上验证。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;端到端多标签集合预测&quot;,&quot;分子图消息传递 MPNN&quot;,&quot;药物相互作用可控损失&quot;]"><strong><a href="../../venues/IJCAI/2021/graph_rec/safedrug_dual_molecular_graph_encoders_for_recommending_effective_and_safe_/">SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>全局 MPNN + 局部子结构二部图双编码器刻画药物分子，配合 PID 式 DDI 可控损失，在 MIMIC-III 上 DDI 率相对降 19.43%、Jaccard 升 2.88%，参数仅 325K</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;列表级上下文建模&quot;,&quot;评估器-生成器范式&quot;,&quot;综述与基准库&quot;]"><strong><a href="../../venues/IJCAI/2022/multi_task_reranking/neural_re_ranking_in_multi_stage_recommender_systems_a_review/">Neural Re-ranking in Multi-stage Recommender Systems: A Review</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以“目标×监督信号”四象限梳理神经重排，给出复杂度对比、LibRerank 基准与 PRM Public 上 +68% 的量化结论。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习&quot;,&quot;混合专家&quot;,&quot;Look-alike 人群扩展&quot;]"><strong><a href="../../venues/KDD/2021/cold_start_rec/learning_to_expand_audience_via_meta_hybrid_experts_and_critics_for_recomme/">Learning to Expand Audience via Meta Hybrid Experts and Critics for Recommendation and Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用两阶段元学习模拟「理解种子—扩展人群」，并以任务驱动门控的混合专家/评论家建模任务关系，微信十亿级线上 CVR 提升约 7.9%–15.4%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多任务学习&quot;,&quot;序列依赖&quot;,&quot;信息迁移&quot;]"><strong><a href="../../venues/KDD/2021/ctr_feature_interaction/modeling_the_sequential_dependence_among_audience_multi_step_conversions_wi/">Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AITM 用自适应信息迁移模块在输出层向量空间建模多步转化的顺序依赖，配合行为期望校准损失，在美团信用卡广告与 Ali-CCP 上均取得稳定 AUC 提升，线上端到端转化率显著改善。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排+拍卖机制设计&quot;,&quot;可微排序 NeuralSort&quot;,&quot;端到端自动机制设计&quot;]"><strong><a href="../../venues/KDD/2021/ctr_feature_interaction/neural_auction_end_to_end_learning_of_auction_mechanisms_for_e_commerce_adv/">Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出可微深度神经拍卖 DNA，以 DeepSet 集合编码、部分单调 MIN-MAX 打分与 NeuralSort 松弛排序，在淘宝离线与在线 A/B 中显著提升多目标拍卖指标。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;情感计算&quot;,&quot;双塔多标签&quot;,&quot;弱监督标注&quot;]"><strong><a href="../../venues/KDD/2022/multi_task_reranking/affective_signals_in_a_social_media_recommender_system/">Affective Signals in a Social Media Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建14类情感taxonomy与CARE弱标注管线，用双塔多标签模型产出32维情感embedding注入Facebook排序打分模块，离线归一化AUC损失降&gt;8%，20M用户在线A/B违规内容曝光降&gt;0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态表征学习&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/KDD/2022/multimodal_rec/itemsage_learning_product_embeddings_for_shopping_recommendations_at_pinter/">ItemSage: Learning Product Embeddings for Shopping Recommendations at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用单层Transformer聚合PinSAGE图像与哈希文本特征，以logQ校正分离式负采样与多任务学习产出单一商品嵌入，A/B测试GMV最高+6.85%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Batch RL&quot;,&quot;Multi-Task Fusion&quot;]"><strong><a href="../../venues/KDD/2022/rl_bandit_rec/multi_task_fusion_via_reinforcement_learning_for_long_term_user_satisfactio/">Multi-Task Fusion via Reinforcement Learning for Long-Term User Satisfaction in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将多任务融合建模为会话内 MDP，用 Batch RL 学习个性化融合权重，并以保守离线估计与线上 A/B 验证长期满意度。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Task-to-Task Fusion&quot;,&quot;门控混合专家 (MoE)&quot;]"><strong><a href="../../venues/KDD/2023/multi_task_reranking/adatt_adaptive_task_to_task_fusion_network_for_multitask_learning_in_recomm/">AdaTT: Adaptive Task-to-Task Fusion Network for Multitask Learning in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用任务对级门控融合+原生专家残差构建多层 AdaTT，在工业短视频 70B 样本与 UCI Census 上稳定超越 PLE/MMoE。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;训练稳定性&quot;,&quot;自适应梯度裁剪&quot;,&quot;Adagrad 优化器&quot;,&quot;多任务共享底层&quot;]"><strong><a href="../../venues/KDD/2023/multi_task_reranking/improving_training_stability_for_multitask_ranking_models_in_recommender_sy/">Improving Training Stability for Multitask Ranking Models in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对多任务精排 loss 发散，提出基于 L∞ 更新范数与双阈值自适应裁剪的 Clippy，Large+DCN 上 AUC 72.37 且可吃 2× 学习率</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;在线原始-对偶&quot;,&quot;曝光约束&quot;,&quot;跨通道建模&quot;]"><strong><a href="../../venues/KDD/2023/multi_task_reranking/multi_channel_integrated_recommendation_with_exposure_constraints/">Multi-channel Integrated Recommendation with Exposure Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多通道信息流推荐建模为带曝光上下界的在线线性规划，用对偶曝光分配与 PCR/CAR 上下文重排联合求解，淘宝 A/B 点击提升 3.00%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;对比上下文编码&quot;,&quot;多样性推荐&quot;]"><strong><a href="../../venues/KDD/2024/multi_task_reranking/contextual_distillation_model_for_diversified_recommendation/">Contextual Distillation Model for Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用兴趣感知 MMR 作教师、Gumbel-Top-k 对比上下文编码器作学生，把 O(NK²) 的贪心多样性重排蒸馏成 O(N log K) 的 point-wise 打分，全链路可用。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差 (Duration Bias)&quot;,&quot;反事实观看时长 (CWT)&quot;]"><strong><a href="../../venues/KDD/2024/debias_fairness_industrial/counteracting_duration_bias_in_video_recommendation_via_counterfactual_watc/">Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024｜去偏与工业推荐：以“反事实观看时长(CWT)截断”重构时长偏差，用经济学边际收益推导变换函数与截断似然，KuaiRand/WeChat/Product 上 MAE 降约 25%、AUC 逼近 Oracle，线上 MWT +2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;奖励分解&quot;,&quot;列表重排&quot;]"><strong><a href="../../venues/KDD/2024/rl_bandit_rec/future_impact_decomposition_in_request_level_recommendations/">Future Impact Decomposition in Request-level Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在请求级列表 MDP 下将 A2C 的 TD 目标与优势函数按 item 分解重加权，KuaiRand 上奖励/深度较 HAC 提升 26.7%/20.9%，并完成亿级短视频平台 A/B 验证。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;梯度冲突消解&quot;,&quot;多目标优化&quot;]"><strong><a href="../../venues/KDD/2024/multi_task_reranking/gradcraft_elevating_multi_task_recommendations_through_holistic_gradient_cr/">GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出“幅度柔性对齐 + 全局方向去冲突”的两阶段梯度整形，在 Wechat/Kuaishou 上 AV-G 分别提升 1.06%/3.79%，线上 VV +0.95%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;非自回归生成&quot;,&quot;对比解码&quot;]"><strong><a href="../../venues/KDD/2024/multi_task_reranking/non_autoregressive_generative_models_for_reranking_recommendation/">Non-autoregressive Generative Models for Reranking Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>非自回归生成器一次解码全部位置，配合匹配模型、序列级 unlikelihood 训练与对比解码；Avito AUC 0.7234、快手 Recall@6 74.86%，推理约 5 倍提速。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;互惠推荐&quot;,&quot;因果推断&quot;]"><strong><a href="../../venues/KDD/2024/debias_fairness_industrial/revisiting_reciprocal_recommender_systems_metrics_formulation_and_method/">Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出覆盖/稳定性/双边均衡五项互惠推荐新指标，并以双边因果 treatment + 空位重排求解；全库排序下 CRecall 最高提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多任务学习&quot;,&quot;混合专家网络&quot;,&quot;自适应损失掩码&quot;]"><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/an_audio_centric_multi_task_learning_framework_for_streaming_ads_targeting_/">An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向音频主导多模态广告场景的 CAMoE 框架，以模态感知任务分组、自适应损失掩码与 DCN 专家实现近帕累托最优 CTR 预估，线上音频 CTR +14.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;意图多样化&quot;,&quot;贝叶斯后验更新&quot;,&quot;整页优化&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/beyond_item_dissimilarities_diversifying_by_intent_in_recommender_systems/">Beyond Item Dissimilarities: Diversifying by Intent in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用实时意图概率驱动整页贪心重排，以贝叶斯后验更新替代物品相似度，YouTube 线上 DAU +0.05%、用户享受度 +0.09%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;出价环境建模&quot;,&quot;零膨胀分布&quot;,&quot;基础模型&quot;]"><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/bid2x_revealing_dynamics_of_bidding_environment_in_online_advertising_from_/">Bid2X: Revealing Dynamics of Bidding Environment in Online Advertising from A Foundation Model Lens</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将自动出价环境建模重构为跨场景基础模型的下一token预测，用变量/时间双注意力+零膨胀投影，8数据集MAE全面领先，线上GMV+4.65%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;NDCG@K 代理损失&quot;,&quot;分位数估计&quot;,&quot;Softmax Loss&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;组合优化约束&quot;,&quot;负迁移抑制&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;生成式拍卖&quot;,&quot;排列级外部性&quot;,&quot;机制设计&quot;]"><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/contextual_generative_auction_with_permutation_level_externalities_for_onli/">Contextual Generative Auction with Permutation-level Externalities for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用自回归生成器直接生成广告位序列并建模排列级外部性，配合可微事后遗憾学习支付规则，淘宝线上 RPM +3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;序数逻辑回归&quot;,&quot;多反馈统一嵌入&quot;,&quot;嵌套优化&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Pareto 多目标优化&quot;,&quot;符号图结构学习&quot;,&quot;LLM 知识注入解释&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;离线强化学习&quot;,&quot;因果状态编码&quot;,&quot;预算分配&quot;]"><strong><a href="../../venues/KDD/2025/rl_bandit_rec/multi_task_offline_reinforcement_learning_for_online_advertising_in_recomme/">Multi-task Offline Reinforcement Learning for Online Advertising in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用因果状态编码器+因果注意力+多任务解码器把广告渠道推荐与预算分配统一进离线RL，KuaiRand/Criteo 上 F1 提升约 7.5%/16.2%，线上 CTR +0.08。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;观看时长建模&quot;,&quot;分桶统计建模&quot;,&quot;软标签分类回归&quot;]"><strong><a href="../../venues/KDD/2025/ctr_feature_interaction/swat_statistical_modeling_of_video_watch_time_through_user_behavior_analysi/">SWaT: Statistical Modeling of Video Watch Time through User Behavior Analysis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户观看行为假设翻译为分桶二项/几何分布白盒统计模型，以软标签 BCE 训练，离线与线上 A/B 均稳定超越 WLR/TPM/CREAD。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;Bradley-Terry&quot;,&quot;算法排名一致性&quot;,&quot;数据集分类学&quot;]"><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/bradley_terry_rankings_for_recommender_systems_across_dataset_taxonomies/">Bradley-Terry Rankings for Recommender Systems Across Dataset Taxonomies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Bradley-Terry 概率排序替代指标均值聚合，在 14 算法×89 数据集上给出数据集感知的推荐算法排名与免运行预测。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;图结构解码&quot;,&quot;似然陷阱&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/breaking_the_likelihood_trap_consistent_generative_recommendation_with_grap/">Breaking the Likelihood Trap: Consistent Generative Recommendation with Graph-structured Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用有向无环图解码器扩展非自回归生成空间，并以 Gumbel-Softmax 打通评估器梯度，破解似然陷阱，快手线上 Views +0.78%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;稠密监督蒸馏&quot;,&quot;累积回归&quot;]"><strong><a href="../../venues/KDD/2026/multi_task_reranking/degre_dense_supervised_generative_reranking_for_recommendation/">DeGRe: Dense-supervised Generative Reranking for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>离线束搜索挖掘前瞻序列+累积回归稠密监督蒸馏至轻量生成器，单次贪心解码逼近全局最优，线上GMV+3.75%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Uplift Modeling&quot;,&quot;多处理因果推断&quot;,&quot;多目标权衡&quot;]"><strong><a href="../../venues/KDD/2026/rl_bandit_rec/heterogeneous_multi_treatment_uplift_modeling_for_trade_off_optimization_in/">Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用异构多处理增益建模+在线动态权重，个性化权衡短视频消费与商业化指标，快手全量部署</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;GD-RTB 混合竞价&quot;,&quot;对偶变量引导&quot;,&quot;列表级生成评估&quot;]"><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/hmaf_a_hierarchical_multi_slot_gd_rtb_allocation_framework/">HMAF: A Hierarchical Multi-Slot GD-RTB Allocation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Plan-Calibrate-Execute 范式将 PV 约束对偶优化与在线列表级生成-评估排序耦合，美团线上 GD 交付率 +3.72%、广告总收入 +1.59%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多归因学习&quot;,&quot;多任务学习&quot;,&quot;MoE&quot;]"><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/mac_a_conversion_rate_prediction_benchmark_featuring_labels_under_multiple_/">MAC: A Conversion Rate Prediction Benchmark Featuring Labels Under Multiple Attribution Mechanisms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个多归因机制CVR公开基准MAC与PyMAL库，提出MoAE模型，GAUC最高提升0.39pt。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;宏图专家&quot;,&quot;多任务图合并&quot;]"><strong><a href="../../venues/KDD/2026/multi_task_reranking/macro_graph_of_experts_for_billion_scale_multi_task_recommendation/">Macro Graph of Experts for Billion-Scale Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以MTMG合并多任务宏图，结合宏图专家与级联读出；Taobao平均AUC提升13.42%，线上GMV较MMoE提升16.46%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Tokenization&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/mdl_a_unified_multi_distribution_learner_in_large_scale_industrial_recommen/">MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将场景/任务Token化并与特征逐层交叉注意，在抖音搜索0.5B模型上QAUC相对提升0.23–0.77%，线上LT30 +0.0626%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多域多目标MDMO&quot;,&quot;组合合并与知识蒸馏&quot;]"><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Transformer-Native 架构&quot;,&quot;梯度截断与任务解耦&quot;]"><strong><a href="../../venues/KDD/2026/multi_task_reranking/onerank_unified_transformer_native_ranking_architecture_for_multi_task_reco/">OneRank: Unified Transformer-Native Ranking Architecture for Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多任务推理内化进 Transformer 栈，用任务私有 token 通道 + 梯度截断 + 匹配式打分替代编码器-预测器分离，Shopee 离线与线上 A/B 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习&quot;,&quot;策略个性化&quot;,&quot;Uplift建模&quot;,&quot;因果推断&quot;,&quot;双边市场&quot;]"><strong><a href="../../venues/KDD/2026/multi_task_reranking/personalizing_marketplace_policies_with_competing_objectives_and_constraine/">Personalizing Marketplace Policies with Competing Objectives and Constrained Experiments: Evidence from a Job Marketplace</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双边市场免费额度阈值个性化：集成CATE+混合排序+线性外推，护栏风险降10%+，线上验证外推精度r=0.85</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Sparse MoE&quot;,&quot;Progressive Expert Routing&quot;,&quot;Load Balancing&quot;]"><strong><a href="../../venues/KDD/2026/multi_task_reranking/smes_towards_scalable_multi_task_recommendation_via_expert_sparsity/">SMES: Towards Scalable Multi-Task Recommendation via Expert Sparsity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>KDD 2026 多任务精排：渐进式共享/私有专家稀疏路由与全局负载均衡，工业 A/B 观看时长 +0.31%、延迟降 50%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;全空间建模&quot;,&quot;多任务学习&quot;,&quot;样本选择偏差&quot;]"><strong><a href="../../venues/RecSys/2021/ctr_feature_interaction/an_analysis_of_entire_space_multi_task_models_for_post_click_conversion_pre/">An Analysis Of Entire Space Multi-Task Models For Post-Click Conversion Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在 Twitter 广告 5B 曝光工业数据上系统消融 ESMM 的三大要素，发现仅需硬参数共享或仅需点击加权即可获得全部增益，且二者不可叠加</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;页面级序列决策&quot;,&quot;上下文 Bandit 数据采集&quot;,&quot;位置偏置 IPS&quot;]"><strong><a href="../../venues/RecSys/2021/multi_task_reranking/page_level_optimization_of_e_commerce_item_recommendations/">Page-level Optimization of e-Commerce Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将商品详情页模块的选序建模为两阶段 RNN 序列决策，用模块族嵌入压缩动作空间、束搜索+IPS+Thompson 采样实现线上个性化重排，线上 CTR +2.48%、PTR +7.34%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Search Recommendation&quot;,&quot;Results Organization&quot;,&quot;Instant Search&quot;]"><strong><a href="../../venues/RecSys/2021/multi_task_reranking/recommendations_and_results_organization_in_netflix_search/">Recommendations and Results Organization in Netflix Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>Netflix 搜索把协同过滤推荐注入检索，并把结果组织成带标签的横向分组列表做列表级重排，属无量化指标的工业系统论文。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;用户生命周期建模&quot;,&quot;动态损失加权&quot;]"><strong><a href="../../venues/RecSys/2023/multi_task_reranking/stan_stage_adaptive_network_for_multi_task_recommendation_by_learning_user_/">STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布采样刻画用户生命周期阶段向量，动态加权各任务损失，工业数据集 AUC 最高 +18.5%，线上停留时长 +3.05%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;辅助模型&quot;,&quot;上下文特征解耦&quot;,&quot;工业级部署&quot;]"><strong><a href="../../venues/RecSys/2023/ctr_feature_interaction/unleash_the_power_of_context_enhancing_large_scale_recommender_systems_with/">Unleash the Power of Context: Enhancing Large-Scale Recommender Systems with Context-Based Prediction Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用一个不含物品特征的上下文辅助模型预测上下文 CTR，作为新特征注入主 CTR 模型，离线 RIG 最高 +5.58%、线上 RPM +0.97%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;生成式检索 Generative Retrieval&quot;,&quot;多任务联合训练&quot;,&quot;原子 ID 表征正则化&quot;]"><strong><a href="../../venues/RecSys/2024/generative_rec/bridging_search_and_recommendation_in_generative_retrieval_does_one_task_he/">Bridging Search and Recommendation in Generative Retrieval: Does One Task Help the Other?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用单个 T5 联合训练搜索与推荐的生成式检索，以仿真+真实数据论证流行度与隐表征双重正则化机制，全库 R@30 平均 +16%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;多任务学习&quot;,&quot;教师偏差泄漏&quot;]"><strong><a href="../../venues/RecSys/2024/debias_fairness_industrial/bridging_the_gap_unpacking_the_hidden_challenges_in_knowledge_distillation_/">Bridging the Gap: Unpacking the Hidden Challenges in Knowledge Distillation for Online Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>针对推荐蒸馏中的数据漂移、教师选型与多学生标签共享三大工业挑战，提出持续更新教师 + 分离 logits 的辅助蒸馏框架，线上互动率最高提升 1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR预估与特征交叉&quot;,&quot;重排&quot;,&quot;后处理概率校准&quot;,&quot;Top-N 排序偏差&quot;]"><strong><a href="../../venues/RecSys/2024/ctr_feature_interaction/calibrating_the_predictions_for_top_n_recommendations/">Calibrating the Predictions for Top-N Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 Top-N 列表内存在排名相关的过/低估偏差，提出秩分组+秩折扣加权的通用校准优化方法与 ECE@N/RDECE@N 指标，KuaiRec 上 ECE@20 最高降 89%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;服务时在线学习&quot;,&quot;代理评估器&quot;,&quot;电商推荐&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/do_not_wait_learning_re_ranking_model_without_user_feedback_at_serving_time/">Do Not Wait: Learning Re-Ranking Model Without User Feedback At Serving Time in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用代理评估器在服务时刻为每个请求在线生成瞬态参数修正，无需等待延迟反馈即提升重排效果，离线 NDCG@10 +0.65%、线上购买数 +2.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文组合赌博机 (CCB)&quot;,&quot;离线策略评估 (OPE)&quot;,&quot;奖励分解&quot;]"><strong><a href="../../venues/RecSys/2024/rl_bandit_rec/effective_off_policy_evaluation_and_learning_in_contextual_combinatorial_ba/">Effective Off-Policy Evaluation and Learning in Contextual Combinatorial Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>因子化动作空间 + 主/残差效应分解的 OPCB 估计器，仅对主动作重要性采样、残差回归，KuaiRec 上 MSE 较 IPS 降 92%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;统一上下文建模&quot;,&quot;特征交叉&quot;,&quot;个性化搜索&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/joint_modeling_of_search_and_recommendations_via_an_unified_contextual_reco/">Joint Modeling of Search and Recommendations Via an Unified Contextual Recommender (UniCoRn)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Netflix 以 {user, query, country, source entity, task} 为共享上下文训练单一深度排序器，插补缺失上下文并做特征交叉，统一服务搜索与推荐，个性化后离线指标 +7%/+10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/融合权重决策&quot;,&quot;Off-Policy Learning&quot;,&quot;Counterfactual Risk Minimisation&quot;,&quot;Effective Sample Size&quot;,&quot;Multi-Objective Scalarisation&quot;]"><strong><a href="../../venues/RecSys/2024/rl_bandit_rec/multi_objective_recommendation_via_multivariate_policy_learning/">Multi-Objective Recommendation via Multivariate Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多目标融合权重视为连续动作，用 CRM/SNIPS 学习策略，提出 ESS 校正提升小样本置信区间覆盖率，线上 A/B 显著提升留存与时长。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回&quot;,&quot;Pareto 前沿近似&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/pareto_front_approximation_for_multi_objective_session_based_recommender_sy/">Pareto Front Approximation for Multi-Objective Session-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将偏好向量条件化注入 TRON 会话 Transformer，配合 KL 非均匀性正则，单模型覆盖 CTR–CVR 全 Pareto 前沿并经线上 A/B 验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;跨内容类型排序&quot;,&quot;曝光预算与多样性&quot;,&quot;线上 A/B 测试&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/ranking_across_different_content_types_the_robust_beauty_of_multinomial_ble/">Ranking Across Different Content Types: The Robust Beauty of Multinomial Blending</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式混合(MB)在打分后按内容类型采样组列表，实现可解释、可重训稳定的跨内容类型曝光预算，Amazon Music 线上 A/B 播客时长 +18.82% 且整体互动 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;计算社会选择&quot;,&quot;异构公平性&quot;,&quot;公平感知重排&quot;]"><strong><a href="../../venues/RecSys/2024/debias_fairness_industrial/social_choice_for_heterogeneous_fairness_in_recommendation/">Social Choice for Heterogeneous Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用计算社会选择的多智能体架构 SCRUF-D 将三类异构提供方公平性定义分别交由独立 Agent，在分配—聚合两阶段中实现多目标公平重排，且几乎不损 nDCG。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Hybrid Targets&quot;,&quot;梯度调节&quot;,&quot;标签嵌入&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/touch_the_core_exploring_task_dependence_among_hybrid_targets_for_recommend/">Touch the Core: Exploring Task Dependence Among Hybrid Targets for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次提出混合目标（离散转化+连续核心值）多任务学习，用标签嵌入与信息融合显式传递依赖，并设计梯度方向/量级调节策略，离线三数据集与线上 A/B 均显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;捆绑推荐公平性&quot;,&quot;曝光偏差&quot;,&quot;双层公平评估&quot;]"><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/a_reproducibility_study_of_product_side_fairness_in_bundle_recommendation/">A Reproducibility Study of Product-side Fairness in Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个捆绑推荐商品侧公平性复现研究，揭示捆绑级与物品级曝光分布不一致、指标间结论分歧，并发现用户捆绑偏好越强曝光越公平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;随机游走&quot;,&quot;规范目标分布&quot;,&quot;新闻推荐多样性&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/d_rdw_diversity_driven_random_walks_for_news_recommender_systems/">D-RDW: Diversity-Driven Random Walks for News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将规范目标分布(NTD)作为线性约束嵌入随机游走采样，实现可解释、低能耗的新闻多样化重排，能耗降低一个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;非参数图卷积&quot;,&quot;CTR 精排&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;扩散参数生成&quot;,&quot;测试时自适应&quot;,&quot;可控推荐&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/paragon_parameter_generation_for_controllable_multi_task_recommendation/">Paragon: Parameter Generation for Controllable Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散模型按偏好权重条件生成适配器参数，免重训实现测试时可控多任务推荐，响应时间降低 94.6%+。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;稀有转化预估&quot;,&quot;数据驱动任务划分&quot;,&quot;推理模型裁剪&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散模型&quot;,&quot;Slate 生成&quot;]"><strong><a href="../../venues/RecSys/2025/generative_rec/prompt_to_slate_diffusion_models_for_prompt_conditioned_slate_generation/">Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用潜空间扩散 Transformer 从纯文本 Prompt 直接生成整组 Slate，离线 NDCG 提升 17%、线上重复曝光降 13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;异步离线预排序&quot;,&quot;级联漏斗解耦&quot;,&quot;候选池扩充&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;差分隐私后处理&quot;,&quot;Margin 稳定性证书&quot;,&quot;锚点混合&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;子空间划分&quot;,&quot;多目标检索&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;轮播界面点击模型&quot;,&quot;离策略评估&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/from_click_modeling_to_offline_and_off_policy_evaluation_in_carousel_recomm/">From Click Modeling to Offline and Off-Policy Evaluation in Carousel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于 RecGaze 眼动数据推翻轮播界面四大行为假设，提出以可观测变量为核心的点击模型设计框架、离散选择点击模型与 N2DCG 重构，并规划轮播专用 OPE 路线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;整页生成&quot;,&quot;加权二分类后训练&quot;,&quot;约束解码&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/genpage_towards_end_to_end_generative_homepage_construction_at_netflix/">GenPage: Towards End-to-End Generative Homepage Construction at Netflix</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单个 Transformer 将用户上下文 token 化后自回归生成整页多行首页，WBC/RL 后训练，线上 +0.24% 核心参与度且延迟降 20%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;异构内容排序&quot;,&quot;MoE&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/heterogeneous_ranking_in_industrial_scale_recommender_systems_a_case_study/">Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 Google Discover 异构信息流的 HA-MoE 多门控专家架构，配合 LENS 可观测性与 DL-AUC 双层评估，离线与线上均获稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;GTV 预估&quot;,&quot;层级建模&quot;,&quot;代理标签学习&quot;]"><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/hierarchical_gtv_estimation_bridging_the_gap_between_container_level_rankin/">Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向容器式电商的层级 GTV 预估框架 HGE，用集合感知编码器 + 代理标签分解 + 残差融合解决粒度错配，离线 XAUC +0.63pp、线上 RPS +1.42%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评估&quot;,&quot;曝光份额预测&quot;,&quot;反事实因果推断&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/impression_share_prediction_an_offline_evaluation_task_for_ranking_systems/">Impression Share Prediction: An Offline Evaluation Task for Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出曝光份额预测作为离线评估新任务，用因果DAG证明反事实可识别，PatchTST编码器在首小时恢复+22% L1。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;辅助头&quot;,&quot;联合分布逼近&quot;,&quot;协方差建模&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;轻量排序头&quot;,&quot;在线持续学习&quot;,&quot;任务冲突隔离&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;下游奖励建模&quot;,&quot;长期留存优化&quot;]"><strong><a href="../../venues/RecSys/2026/rl_bandit_rec/long_term_user_engagement_optimization_through_model_agnostic_downstream_re/">Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 P2P 兔子洞深度、浅层关闭负奖励与新用例采纳三类下游奖励替代稀疏留存标签，在 Pinterest 多场景线上 A/B 中稳定提升会话与留存指标。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;分段加权&quot;,&quot;MMoE&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;标签偏移&quot;,&quot;规范对齐&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/normative_alignment_of_recommender_systems_via_internal_label_shift/">Normative Alignment of Recommender Systems via Internal Label Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将编辑/公平目标建模为属性层内部标签偏移，用免重训的闭式重加权实现系统级规范对齐，KL 大幅下降且 AUC 基本无损。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;可解释推荐&quot;,&quot;离线候选池选择&quot;,&quot;成对排序学习&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;LLM Agent&quot;,&quot;Tool Learning&quot;,&quot;记忆反思机制&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/personalized_recommendation_tool_learning_via_autonomous_language_agents/">Personalized Recommendation Tool Learning via Autonomous Language Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>让 LLM 智能体做个性化工具选择、传统推荐模型做全库打分，以架构设计绕开幻觉与上下文长度限制，Full-Ranking 下 R@10 最高提升 68.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;位置偏置&quot;,&quot;置换一致性&quot;,&quot;LLM 列表重排&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM Agent&quot;,&quot;语义画像&quot;,&quot;DPO 对齐&quot;]"><strong><a href="../../venues/RecSys/2026/agent_conversational_rec/shape_your_feed_an_llm_based_agentic_system_for_conversational_recommendati/">Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三层 Agentic 架构（感知/服务/自进化）实现多模态实时用户可控重排，离线对齐打分 98.85% 准确率，线上 dislike 降 2.74%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;成对比较特征&quot;,&quot;上下文感知列表重排&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/towards_full_candidate_interaction_a_comprehensive_comparison_network_for_b/">Towards Full Candidate Interaction: A Comprehensive Comparison Network for Better Route Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出比较优先范式，用非重叠路段构造显式比较张量并在成对空间直接推理，线上CR +1.2%、偏航率 -0.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;反事实推断&quot;,&quot;标题党去偏&quot;]"><strong><a href="../../venues/SIGIR/2021/debias_fairness_industrial/clicks_can_be_cheating_counterfactual_recommendation_for_mitigating_clickba/">Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在因果图中补入曝光→打分直连边，反事实扣除曝光的 NDE，仅用点击数据即把 N@10 提升 7.5%–11.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;元学习&quot;,&quot;行为依赖迁移&quot;]"><strong><a href="../../venues/SIGIR/2021/graph_rec/graph_meta_network_for_multi_behavior_recommendation/">Graph Meta Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元学习为每个用户动态生成行为异质变换与预测层参数，联合建模多行为图传播与跨类型依赖；Leave-One-Out + 1:99 采样排序下，Taobao HR@10 相对 NGCF_M 提升 31.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Metric Optimization&quot;,&quot;RBP Loss&quot;]"><strong><a href="../../venues/SIGIR/2021/multi_task_reranking/new_insights_into_metric_optimization_for_ranking_based_recommendation/">New Insights into Metric Optimization for Ranking-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>系统比较 RR/AP/nDCG/RBP 作为 pairwise 与 listwise 优化目标，提出 RBP 损失，揭示优化指标≠评估指标更优。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边公平&quot;,&quot;曝光公平&quot;,&quot;后处理启发式&quot;]"><strong><a href="../../venues/SIGIR/2021/debias_fairness_industrial/tfrom_a_two_sided_fairness_aware_recommendation_model_for_both_customers_an/">TFROM: A Two-sided Fairness-Aware Recommendation Model for Both Customers and Providers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出双边公平后处理重排模型 TFROM：位置敏感曝光 + 质量升序贪心填位，离线/在线两版在三数据集上以 &lt;10% 质量损失大幅降低提供者曝光方差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Two-sided Fairness&quot;,&quot;Constrained Re-ranking&quot;,&quot;Greedy MILP Relaxation&quot;]"><strong><a href="../../venues/SIGIR/2022/debias_fairness_industrial/cpfair_personalized_consumer_and_producer_fairness_re_ranking_for_recommend/">CPFair: Personalized Consumer and Producer Fairness Re-ranking for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用加性贪心重排把用户侧与物品侧公平偏差同时作为软惩罚，8 数据集上 mCPF 平均降 12.8%~46.9% 且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Determinantal Point Process&quot;,&quot;集合似然损失&quot;,&quot;质量-多样性权衡&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/determinantal_point_process_likelihoods_for_sequential_recommendation/">Determinantal Point Process Likelihoods for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 DPP 集合似然替换逐点/成对损失，把「下 T 项」建模为集合抽取，即插即用于 Caser/MARank/SASRec，Beauty Recall@3 最高 +37%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;set2list 交互&quot;,&quot;排列不变性&quot;]"><strong><a href="../../venues/SIGIR/2022/multi_task_reranking/multi_level_interaction_reranking_with_user_behavior_history/">Multi-Level Interaction Reranking with User Behavior History</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MIR 将候选集视为无序集合、用户历史视为时序列表，用 SLAttention 做 set2list 跨层交互并保证排列不变性，三数据集全指标超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;供应商群体公平&quot;,&quot;Hellinger 距离&quot;,&quot;最大边际相关性&quot;]"><strong><a href="../../venues/SIGIR/2022/debias_fairness_industrial/regulating_group_exposure_for_item_providers_in_recommendation/">Regulating Group Exposure for Item Providers in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SIGIR 2022 短文，将供应商按敏感属性分组，提出基于 Hellinger 距离与 MMR 的贪心重排序，使推荐曝光贴合平台指定目标分布，在几乎不损失 NDCG 的前提下显著提升少数群体曝光与列表级多样性/新颖性。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/推理阶段&quot;,&quot;反事实推断&quot;,&quot;过滤气泡&quot;,&quot;用户可控推荐&quot;]"><strong><a href="../../venues/SIGIR/2022/debias_fairness_industrial/user_controllable_recommendation_against_filter_bubbles/">User-controllable Recommendation Against Filter Bubbles</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出用户可控推荐原型 UCRS 与因果增强的推理期框架 UCI，用反事实推断扣除过时用户 ID 表征效应，并支持四类控制指令，在保持精度的同时显著降低隔离度与多数类目支配。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;排序集成&quot;,&quot;用户意图建模&quot;,&quot;误差-歧义分解&quot;]"><strong><a href="../../venues/SIGIR/2023/multi_task_reranking/intent_aware_ranking_ensemble_for_personalized_recommendation/">Intent-aware Ranking Ensemble for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 IntEL，用意图感知的物品级权重融合多目标异构列表，经误差-歧义分解理论保证，Tmall/LifeData 全目标 NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;逻辑查询嵌入 (LQE)&quot;,&quot;多任务知识共享&quot;]"><strong><a href="../../venues/SIGIR/2023/graph_rec/logicrec_recommendation_with_users_logical_requirements/">LogicRec: Recommendation with Users&#x27; Logical Requirements</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户逻辑需求形式化为 KG 结构化逻辑查询，用 LQE 联合推断缺失事实并以多任务多门专家共享 A_l/A_u 答案集，Amazon-book hit@20 相对提升约 22%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景学习&quot;,&quot;全局标签空间&quot;,&quot;对比解耦&quot;]"><strong><a href="../../venues/SIGIR/2024/cross_domain_multi_scenario/a_unified_search_and_recommendation_framework_based_on_multi_scenario_learn/">A Unified Search and Recommendation Framework Based on Multi-Scenario Learning for Ranking in E-commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用场景触发器驱动 S&amp;R 双视角兴趣/特征抽取与对比解耦，并以全局标签条件概率做多任务，AUC 提升 0.08~0.62 点并落地 7Fresh。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Mixture-of-Experts&quot;,&quot;多任务学习&quot;,&quot;AutoML 双层优化&quot;]"><strong><a href="../../venues/SIGIR/2024/cross_domain_multi_scenario/m3oe_multi_domain_multi_task_mixture_of_experts_recommendation_framework/">M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出共享/域/任务三视角 MoE 与两级融合，并用 AutoML 双层优化自适应学习融合权重，解决多域多任务联合建模的 MDMT 跷跷板问题，MovieLens/KuaiRand 平均 AUC 相对提升 2.54%/5.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;多生成器集成&quot;,&quot;列表全面性&quot;,&quot;互补生成器学习&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/comprehensive_list_generation_for_multi_generator_reranking/">Comprehensive List Generation for Multi-Generator Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出多生成器-评估器重排框架与列表全面性指标，用 CLIG 自动学习互补生成器，离线 AUC 提升 0.9%、线上观看时长 +0.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标 EBR&quot;,&quot;参数高效微调&quot;,&quot;级联选择性掩码&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动机制设计&quot;,&quot;混合拍卖&quot;,&quot;捆绑广告&quot;]"><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;兴趣生命周期&quot;,&quot;向量量化聚类&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;召回+精排统一&quot;,&quot;序列生成&quot;,&quot;排序驱动增强器&quot;,&quot;梯度自适应加权&quot;]"><strong><a href="../../venues/SIGIR/2025/generative_rec/killing_two_birds_with_one_stone_unifying_retrieval_and_ranking_with_a_sing/">Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将召回与排序统一为单生成模型的位置级序列生成任务，配合排序驱动增强器与梯度自适应加权实现协同与同步优化，全库排序下召回排序双提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;神经重排&quot;,&quot;长尾学生表征增强&quot;,&quot;习题推荐&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;眼动追踪数据集&quot;,&quot;轮播界面位置偏置&quot;,&quot;点击模型&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/recgaze_the_first_eye_tracking_and_user_interaction_dataset_for_carousel_in/">RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个轮播界面眼动+点击+光标+选择解释数据集，87用户3477次交互，揭示F型浏览与位置偏置。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;评论偏好画像&quot;,&quot;推理蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/review_driven_personalized_preference_reasoning_with_large_language_models_/">Review-driven Personalized Preference Reasoning with Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用评论抽取偏好画像并蒸馏逐步推理，评分预测与重排双任务均超 CF 与 LLM 基线，解释性显著提升</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;列表级上下文建模&quot;,&quot;一致性正则&quot;,&quot;自蒸馏式自洽&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/towards_principled_learning_for_re_ranking_in_recommender_systems/">Towards Principled Learning for Re-ranking in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出收敛一致性与对抗一致性两条重排学习原则，以自洽重排与相邻换位扰动构造额外监督，即插即用地提升多种重排基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评估&quot;,&quot;逆倾向得分&quot;,&quot;控制变量&quot;,&quot;方差缩减&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/additive_control_variates_dominate_self_normalisation_in_off_policy_evaluat/">Additive Control Variates Dominate Self-Normalisation in Off-Policy Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>证明最优加性基线 β*-IPS 在 MSE 上渐近严格优于 SNIPS，并给出精确方差差解析式与排序场景 β*⊥⊥-IPM 扩展。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排/投放分配&quot;,&quot;保量展示广告(GD)&quot;,&quot;约束优化/KKT&quot;,&quot;页面浏览(PV)约束与合同轮盘&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/beyond_single_slot_joint_optimization_for_multi_slot_guaranteed_display_adv/">Beyond Single Slot: Joint Optimization for Multi-Slot Guaranteed Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多槽位保量展示广告分配建模为页面浏览粒度的离线上限约束二分匹配，用PV约束与合同轮盘互斥机制，线上DID ARPU提升28.17%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;残差量化&quot;,&quot;Target Attention&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差量化码本近似 Target Attention，将交互式建模引入粗排，AUC 0.799、线上 GMV +7.57%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;提供者公平&quot;,&quot;异质效用建模&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/equity_vs_equality_optimizing_ranking_fairness_for_tailored_provider_needs/">Equity vs. Equality: Optimizing Ranking Fairness for Tailored Provider Needs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出以增益为基的公平框架与 EquityRank 梯度重排算法，按提供者异质偏好分配曝光与销量收益，离线/在线仿真均取得更优效果-公平权衡。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;测试时缩放&quot;,&quot;预测合并&quot;]"><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/exploring_test_time_scaling_via_prediction_merging_on_large_scale_recommend/">Exploring Test-time Scaling via Prediction Merging on Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出预测合并式测试时缩放，用异构架构与同构随机种子生成多样 logit 并平均融合，三数据集八骨干上稳定超越参数缩放。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/评估&quot;,&quot;眼动追踪&quot;,&quot;点击模型&quot;,&quot;位置偏置&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/following_the_eye_tracking_evidence_established_web_search_assumptions_fail/">Following the Eye-Tracking Evidence: Established Web-Search Assumptions Fail in Carousel Interfaces</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RecGaze 眼动数据系统证伪 F 型注意力、层级点击与标题导航等四条 Web 搜索假设，提出轮播界面特有的 L 型注视模式。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;排列不变性&quot;,&quot;列表级重排&quot;]"><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;提供方曝光公平&quot;,&quot;分层KL分解&quot;,&quot;可微排序网络&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/post_hoc_provider_fairness_adaptation_via_hierarchical_exposure_alignment/">Post-hoc Provider Fairness Adaptation via Hierarchical Exposure Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结基座+轻量公平适配器，用分层曝光对齐(HEFA)与可微NDCG联合优化，实现即插即用的提供方曝光公平。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;公平重排&quot;,&quot;流形优化&quot;,&quot;注意力经济&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/the_attention_market_interpreting_online_fair_re_ranking_as_manifold_optimi/">The Attention Market: Interpreting Online Fair Re-ranking as Manifold Optimization under Walrasian Equilibrium</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将在线公平重排建模为注意力市场瓦尔拉斯均衡，用供需双侧梯度校正对齐排序流形，7 数据集 42 设置几乎全胜。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;自适应推理&quot;,&quot;推理路由&quot;,&quot;Pareto 前沿&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/think_when_needed_model_aware_reasoning_routing_for_llm_based_ranking/">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出轻量即插即用的模型感知推理路由器，在生成前逐实例决定 Think/Non-Think，以 Pareto 前沿策略在降 token 的同时提升排序效用。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;稀疏监督对齐&quot;,&quot;在线学习&quot;]"><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LoRA+MoE 独立参数通路隔离稀疏问卷满意度信号，并以在线 DPO 对齐主排序模型，离线 NDCG@5 提升约 3%，线上 Q-Satisfied 提升 0.5–0.9%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;后验价值对齐&quot;,&quot;线性 Boosting 归因&quot;]"><strong><a href="../../venues/SIGIR/2026/multi_task_reranking/uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;梯度子空间调优&quot;,&quot;零空间投影&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/unifying_search_and_recommendation_in_llms_via_gradient_multi_subspace_tuni/">Unifying Search and Recommendation in LLMs via Gradient Multi-Subspace Tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用梯度多子空间分解+零空间投影替代 LoRA，在统一 S&amp;R 生成式建模中消解任务梯度冲突并保住通用知识，Qilin/Amazon 上大幅超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;强化学习PPO&quot;,&quot;二维网格面板&quot;,&quot;MDP动作空间设计&quot;]"><strong><a href="../../venues/SIGIR-AP/2023/multi_task_reranking/reinforcement_re_ranking_with_2d_grid_based_recommendation_panels/">Reinforcement Re-ranking with 2D Grid-based Recommendation Panels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 2D 网格面板重排建模为“时间步=输入排名、动作=(是否曝光,槽位选择)”的 MDP，PPO 训练，AUC 相对提升 6.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;级联预测&quot;]"><strong><a href="../../venues/TKDE/2021/multi_task_reranking/learning_to_recommend_with_multiple_cascading_behaviors/">Learning to Recommend with Multiple Cascading Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>NMTR 以共享嵌入、分行为交互函数与级联预测建模多行为序关系，并在两电商数据集全库排序上显著超越单/多行为基线。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;结果多样性&quot;,&quot;MMR与DPP&quot;,&quot;曝光公平&quot;]"><strong><a href="../../venues/TKDE/2024/multi_task_reranking/result_diversification_in_search_and_recommendation_a_survey/">Result Diversification in Search and Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次统一搜索与推荐的结果多样性分类体系：概念层区分个体级/系统级与外在/内在多样性，度量层分无关/含相关性两族，方法层按干预时机分预/中/后处理与在线 Bandit/RL。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;双边公平&quot;,&quot;最小效用保证&quot;,&quot;二分搜索&quot;]"><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/fairsort_learning_to_fair_rank_for_personalized_recommendations_in_two_side/">FairSort: Learning to Fair Rank for Personalized Recommendations in Two-Sided Platforms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将重排视为“跑道”而非背包，用公平提升速度+二分搜索保证用户最低效用，实现双边公平与个性化推荐的三方权衡。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全链路(数据/排序/重排)&quot;,&quot;双边公平&quot;,&quot;摊销公平&quot;,&quot;校准公平&quot;]"><strong><a href="../../venues/TOIS/2023/debias_fairness_industrial/a_survey_on_the_fairness_of_recommender_systems/">A Survey on the Fairness of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首篇系统综述：以“过程/结果×群体/个体×七概念”定义体系、三视角分类与数据-排序-重排方法学，统一 60+ 篇公平推荐文献并指明双边联合公平等空白。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;视频长度去偏&quot;,&quot;多任务学习&quot;,&quot;长度条件采样&quot;]"><strong><a href="../../venues/TOIS/2024/debias_fairness_industrial/alleviating_video_length_effect_for_micro_video_recommendation/">Alleviating Video-Length Effect for Micro-video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出VLDRec，用长度分组、进度标签与长度条件采样多任务去偏视频时长效应，在快手/微信固定总时长评估上@120提升1.81%/11.32%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/精排&quot;,&quot;多行为推荐&quot;,&quot;级联残差GCN&quot;]"><strong><a href="../../venues/TOIS/2024/graph_rec/cascading_residual_graph_convolutional_network_for_multi_behavior_recommend/">Cascading Residual Graph Convolutional Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>无额外参数的级联残差GCN按行为顺序精化嵌入，三数据集全库排序HR@K平均较最优基线提升24.76%/27.28%/25.10%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;多任务提示微调&quot;,&quot;任务信息解耦&quot;,&quot;生成对抗网络&quot;]"><strong><a href="../../venues/TOIS/2025/multi_task_reranking/efficient_multi_task_prompt_tuning_for_recommendation/">Efficient Multi-task Prompt Tuning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出两阶段 MPT-Rec：GAN 解耦任务共享/特有信息后冻结主干，仅用任务感知 Prompt 微调新任务，参数降至 10% 且性能保持 95%+。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;指令微调&quot;,&quot;混合排序&quot;]"><strong><a href="../../venues/TOIS/2025/llm_rec/recranker_instruction_tuning_large_language_model_as_ranker_for_top_k_recom/">RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以自适应用户采样、位置偏移与检索信号增强构建三类排序指令微调 LLM 重排器，并提出混合排序集成，Leave-One-Out 全库排序下 HR/NDCG 最高提升 213%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;不变风险最小化&quot;,&quot;变分自编码器&quot;]"><strong><a href="../../venues/TOIS/2025/multi_task_reranking/user_invariant_preference_learning_for_multi_behavior_recommendation/">User Invariant Preference Learning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 IRM 范式 + VAE 从多行为交互中抽取用户不变偏好以抑制辅助行为噪声，Leave-One-Out + Full-Ranking 下 HR@10 最高提升 3.98%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边公平&quot;,&quot;后悔理论&quot;,&quot;模糊规划&quot;]"><strong><a href="../../venues/TOIS/2026/debias_fairness_industrial/regret_aware_re_ranking_for_guaranteeing_two_sided_fairness_and_accuracy_in/">Regret-aware Re-ranking for Guaranteeing Two-sided Fairness and Accuracy in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将后悔理论与模糊规划引入双边公平重排，兼顾用户个体公平、平均准确率与提供方曝光。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;级联图卷积&quot;,&quot;对比偏好对齐&quot;]"><strong><a href="../../venues/TOIS/2026/multi_task_reranking/towards_multi_behavior_multi_task_recommendation_via_behavior_informed_grap/">Towards Multi-Behavior Multi-Task Recommendation via Behavior-informed Graph Embedding Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>BiGEL 在级联图范式上叠加目标→辅助门控反馈、全局上下文增强与对比偏好对齐，JD/UB 全库 Top-5 排序上目标与辅助行为双赢（UB 点击 Prec@5 +101%）。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;AutoML&quot;,&quot;搜索空间抽象化&quot;,&quot;特征交互搜索&quot;]"><strong><a href="../../venues/TORS/2024/ctr_feature_interaction/a_comprehensive_survey_on_automated_machine_learning_for_recommendations/">A Comprehensive Survey on Automated Machine Learning for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个按 DRS 组件（特征选择/嵌入/交叉/训练）细粒度划分的 AutoML 综述，提出“单→多组件、详细→抽象搜索空间、梯度主导”的三轴演进路线，系统对比各方法搜索空间量级与复杂性。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;重排&quot;,&quot;价值敏感设计&quot;,&quot;多利益相关者优化&quot;]"><strong><a href="../../venues/TORS/2024/debias_fairness_industrial/building_human_values_into_recommender_systems_an_interdisciplinary_synthes/">Building Human Values into Recommender Systems: An Interdisciplinary Synthesis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>22位跨学科作者综合：价值→指标→排序干预的七步落地流程、三类度量数据源与多利益相关者权衡防护，属立场综述而非实验论文。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;社会选择&quot;,&quot;多智能体公平&quot;,&quot;动态重排&quot;]"><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/dynamic_fairness_aware_recommendation_through_multi_agent_social_choice/">Dynamic Fairness-aware Recommendation through Multi-agent Social Choice</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多利益相关方公平推荐建模为「分配+聚合」两阶段社会选择问题，SCRUF-D 框架动态调度多个公平性智能体，在真实与合成数据上实现多维权衡。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线列表推荐/重排与探索&quot;,&quot;Contextual Bandit&quot;,&quot;Thompson Sampling&quot;,&quot;Reward-Informed Segmentation&quot;]"><strong><a href="../../venues/TORS/2026/rl_bandit_rec/a_reward_informed_semi_personalized_bandit_approach_for_enhancing_accuracy_/">A Reward-Informed Semi-Personalized Bandit Approach for Enhancing Accuracy and Serendipity in Online Slate Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>按物品学习奖励监督决策树划分用户段，并在叶级运行 Thompson Sampling 生成列表，兼顾 regret 与 serendipity。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Calibrated Recommendation&quot;,&quot;KL-Divergence&quot;,&quot;Post-processing Reranking&quot;]"><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/calibrated_recommendations_survey_and_future_directions/">Calibrated Recommendations: Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统综述 53 篇校准推荐工作，统一拆解「基线排序→候选→分布构造→散度度量→权衡重排」七步框架，并量化揭示评测协议短板（88% 仅离线、2% 线上、27.1% 未报告划分）。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;LLM零样本重排&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="../../venues/TORS/2026/multi_task_reranking/enhancing_recommendation_diversity_by_re_ranking_with_large_language_models/">Enhancing Recommendation Diversity by Re-ranking with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 8 种零样本提示让 LLM 对 MF 候选列表做多样性重排，Anime/Goodreads 上优于随机重排但逊于贪心方法，延迟高 3 个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;推荐系统评测&quot;]"><strong><a href="../../venues/TORS/2026/debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;碳足迹感知推荐&quot;,&quot;绿色度重排&quot;]"><strong><a href="../../venues/TORS/2026/multi_task_reranking/towards_carbon_footprint_aware_recommender_systems_for_greener_item_recomme/">Towards Carbon Footprint-Aware Recommender Systems for Greener Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建首个含碳足迹的食谱数据集 RecipeEmission，基准测试 9 种传统算法发现精度优化忽视绿色度，提出 α 加权重排以极小精度损失换取最高 +89.1% 绿色度提升。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;组合公平 (Compositional Fairness)&quot;,&quot;反事实 Headroom 分析&quot;,&quot;分布匹配正则&quot;]"><strong><a href="../../venues/WSDM/2021/debias_fairness_industrial/practical_compositional_fairness_understanding_fairness_in_multi_component_/">Practical Compositional Fairness: Understanding Fairness in Multi-Component Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>证明多组件乘性打分下公平性不会自动组合，给出对称性充分条件与反事实 headroom 测试框架，产线系统整体 PairAcc gap 由 0.1546 降至 0.0222。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;特权特征 (Privileged Features)&quot;,&quot;表征对齐/相似度损失&quot;]"><strong><a href="../../venues/WSDM/2023/multi_task_reranking/slate_aware_ranking_for_recommendation/">Slate-Aware Ranking for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>训练期用特权 slate 编码器与用户编码器做表征对齐，推理期仅用用户编码器注入列表互影响，精排 AUC +0.43%、线上 PV +0.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;序列多样性&quot;,&quot;近似算法&quot;,&quot;有序哈密顿路径&quot;]"><strong><a href="../../venues/WSDM/2025/multi_task_reranking/sequential_diversification_with_provable_guarantees/">Sequential Diversification with Provable Guarantees</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多样性从集合度量升级为序列期望度量，证明 MaxSSD 的 NP 难性并给出 BτI/SBτI 等常数近似算法，在 7 个数据集上稳定超越 EXPLORE/MMR/DPP。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;多任务学习&quot;,&quot;参数解耦&quot;]"><strong><a href="../../venues/WSDM/2025/cross_domain_multi_scenario/towards_personalized_federated_multi_scenario_multi_task_recommendation/">Towards Personalized Federated Multi-Scenario Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个联邦多场景多任务推荐框架 PF-MSMTrec，通过专家参数解耦、联邦批归一化与冲突协调缓解数据/模型/任务三级冲突，AliExpress 与 Tenrec 上 CTR/CTCVR AUC 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景多任务&quot;,&quot;LoRA 信息解耦&quot;,&quot;信息流剪枝&quot;]"><strong><a href="../../venues/WSDM/2026/cross_domain_multi_scenario/automated_information_flow_selection_for_multi_scenario_multi_task_recommen/">Automated Information Flow Selection for Multi-scenario Multi-task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LoRA 解耦四类场景-任务信息单元并以可学习掩码剪枝无效信息流，AUC 全面超越 HiNet/M3oE，线上 CVR 均涨 2.46%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;冷启动迁移&quot;,&quot;广告定向&quot;]"><strong><a href="../../venues/WSDM/2026/multi_task_reranking/cold_starting_podcast_ads_and_promotions_with_multi_task_learning_on_spotif/">Cold-Starting Podcast Ads and Promotions with Multi-Task Learning on Spotify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Spotify 用单一 5 任务 MTL 模型统一播客广告与推广定向，靠非对称损失掩码与源平衡采样实现跨渠道迁移，线上 eCPS 降 22%、i2s 升 18–24%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;秩差异感知采样&quot;,&quot;双向互学习&quot;]"><strong><a href="../../venues/WWW/2021/collaborative_filtering/bidirectional_distillation_for_top_k_recommender_system/">Bidirectional Distillation for Top-K Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>打破“教师恒优”单向蒸馏假设，用方向差异化的秩差异感知采样实现师生互教，Full-Ranking 下学生 H@50 相对 CD/RD 提升 3.0%–13.9%，教师自身亦涨 3.8%–20.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;行程感知注意力&quot;,&quot;双目标联合训练&quot;]"><strong><a href="../../venues/WWW/2021/collaborative_filtering/itinerary_aware_personalized_deep_matching_at_fliggy/">Itinerary-aware Personalized Deep Matching at Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以「未消费订单列表=行程」为用户侧输入，用三重行程感知注意力 + 意图/点击双目标训练的双塔深度召回，在飞猪生产数据与线上 A/B 中 CTR 较 MIND 提升 1.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/粗排&quot;,&quot;Policy Gradient&quot;,&quot;下游自适应剪枝&quot;,&quot;奖励塑形&quot;]"><strong><a href="../../venues/WWW/2021/rl_bandit_rec/optimizing_ad_pruning_of_sponsored_search_with_reinforcement_learning/">Optimizing AD Pruning of Sponsored Search with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将下游精排/拍卖视为黑盒环境，用策略梯度的序列式选品智能体替代 eCPM 规则剪枝，百度在线 A/B 取得 CPM +1.95%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Pareto 优化&quot;,&quot;多目标确定性策略梯度&quot;]"><strong><a href="../../venues/WWW/2021/rl_bandit_rec/reinforcement_recommendation_with_user_multi_aspect_preference/">Reinforcement Recommendation with User Multi-aspect Preference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将单值 reward 扩展为多维度方面级 reward 向量，用 Pareto-DPG 多 critic 单 actor 学得帕累托有效策略，并理论分析梯度偏差上界，三个数据集 F1/NDCG 相对最优基线提升 4.4%–9.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;分层抽样配比 (Stratified Composition)&quot;,&quot;列表顺序编排 (Sequencing)&quot;,&quot;使用与满足理论 (U&amp;G)&quot;]"><strong><a href="../../venues/WWW/2021/multi_task_reranking/toward_the_next_generation_of_news_recommender_systems/">Toward the Next Generation of News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 table d&#x27;hôte 新闻推荐范式：按 U&amp;G 理论对文章分层（surveillance/serendipity），再以配额配比与顺序编排生成“套餐式”列表；为纯 vision 论文，无数据集与实验。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;用户群体公平&quot;,&quot;约束重排&quot;,&quot;0-1 整数规划&quot;]"><strong><a href="../../venues/WWW/2021/debias_fairness_industrial/user_oriented_fairness_in_recommendation/">User-oriented Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>按活跃度划分优势/劣势用户群，将用户公平约束注入重排阶段，用 0-1 整数规划求解 top-K 列表，在降 75% 群体差距的同时反超整体精度。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;重排/列表级生成&quot;,&quot;条件变分自编码器&quot;,&quot;石板推荐 (Slate Recommendation)&quot;,&quot;多样性控制&quot;]"><strong><a href="../../venues/WWW/2021/generative_rec/variation_control_and_evaluation_for_generative_slaterecommendations/">Variation Control and Evaluation for Generative Slate Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>揭示 CVAE 列表推荐的「重构-集中」困境，提出两阶段 Pivot-CVAE 预生成扰动与列表方差评估协议，物品覆盖率提升数十倍而精度仅微降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;在线知识蒸馏&quot;,&quot;异构目标一致性&quot;,&quot;多任务梯度归一化&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/consensus_learning_from_heterogeneous_objectives_for_one_class_collaborativ/">Consensus Learning from Heterogeneous Objectives for One-Class Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为 OCCF 目标模型挂异构目标辅助头，用排名一致性生成动态共识做在线蒸馏，全库排序下 R@20 最高提升 30.7%，推理零额外开销</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;端到端推荐决策&quot;,&quot;分层Transformer&quot;,&quot;MOBA Draft&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/draftrec_personalized_draft_recommendation_for_winning_in_multi_player_onli/">DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层Transformer联合玩家偏好与队伍协同，阈值约束胜率最大化；LOL HR@1较SASRec提升12.1%，86.9%用户愿用。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多反馈用户建模&quot;,&quot;强到弱注意力蒸馏&quot;,&quot;参与度多任务学习&quot;]"><strong><a href="../../venues/WWW/2022/multi_task_reranking/feedrec_news_feed_recommendation_with_various_user_feedbacks/">FeedRec: News Feed Recommendation with Various User Feedbacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用异构/同构 Transformer 与强到弱注意力统一建模点击、跳过、分享、点踩、读完、秒关六类反馈，并以点击-读完-时长多任务训练，AUC 相对最强基线提升约 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;损失函数设计&quot;,&quot;用户兴趣边界&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/learning_explicit_user_interest_boundary_for_recommendation/">Learning Explicit User Interest Boundary for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为每个用户学习可学习兴趣边界 $b_u$，令正负样本分别与边界做 pairwise，融合 pointwise 灵活性与 pairwise 排序性，Leave-One-Out + 1:100 采样排序下 Hit@1 平均提升 5.22%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;双边匹配市场&quot;,&quot;双随机排序&quot;,&quot;PBM&quot;]"><strong><a href="../../venues/WWW/2022/multi_task_reranking/optimizing_rankings_for_recommendation_in_matching_markets/">Optimizing Rankings for Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双边匹配市场列表级联合排序，双随机策略优化社会福利，离线仿真匹配数显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Contextualized Transformer&quot;,&quot;Merged Cross-Attention&quot;,&quot;列表级多任务学习&quot;]"><strong><a href="../../venues/WWW/2022/multi_task_reranking/pear_personalized_re_ranking_with_contextualized_transformer_for_recommenda/">PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以特征级 MLP + 列表/历史双向交叉注意力 Transformer 重排 top-30 列表，并加列表级多任务监督，MicroVideo 与华为新闻日志上 gAUC@30 分别提升 1.07%/1.71%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果嵌入解耦&quot;,&quot;从众偏差&quot;,&quot;多任务排序&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/cam2_conformity_aware_multi_task_ranking_model_for_large_scale_recommender_/">CAM2: Conformity-Aware Multi-Task Ranking Model for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用特征分区+任务无关辅助损失在十亿级视频排序中解耦“从众”与“真实兴趣”因果嵌入，离线 NE 降 0.139%，线上互动 +0.50%、DAU +0.21%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;上下文特征&quot;,&quot;多样性校准&quot;,&quot;贪心序列重排&quot;]"><strong><a href="../../venues/WWW/2023/multi_task_reranking/caviar_context_aware_video_recommendations/">CAViaR: Context Aware Video Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用上下文特征让点式模型感知前序视频相似度，贪心式序列重排去冗余；NE 相对降低 0.85%–1.4%，线上互动 +1.7%、DAU +1.5%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;实例级高阶关系&quot;,&quot;多任务梯度解耦&quot;]"><strong><a href="../../venues/WWW/2023/graph_rec/compressed_interaction_graph_based_framework_for_multi_behavior_recommendat/">Compressed Interaction Graph based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用矩阵乘法显式构造实例级高阶交互图并以节点注意力压缩图空间，配合分离输入多专家网络解耦多任务梯度，Taobao HR@10 相对提升 20.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回+精排级联&quot;,&quot;难负例采样&quot;,&quot;排序序蒸馏&quot;,&quot;渐近无偏KL估计&quot;]"><strong><a href="../../venues/WWW/2023/multi_task_reranking/cooperative_retriever_and_ranker_in_deep_recommenders/">Cooperative Retriever and Ranker in Deep Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CoRR 联合训练召回与精排：用自适应采样器从召回分布抽硬负例训精排，用渐近无偏 KL 估计做排序序一致性蒸馏回灌召回，四个大规模数据集 NDCG@10 提升 5–18%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;意图分解&quot;,&quot;触发诱导推荐&quot;,&quot;多任务联合学习&quot;]"><strong><a href="../../venues/WWW/2023/ctr_feature_interaction/deep_intention_aware_network_for_click_through_rate_prediction/">Deep Intention-Aware Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将小程序触发诱导推荐 CTR 分解为「进入意图 × 双条件 CTR」，用后验标注监督意图网络，淘宝聚划算 AUC 0.812、线上 CTR +4.74%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;易到难课程学习&quot;,&quot;异构模型集成&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/distillation_from_heterogeneous_models_for_top_k_recommendation/">Distillation from Heterogeneous Models for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用教师训练轨迹构造由易到难的排序知识课程，将异构推荐模型集成蒸馏进轻量学生，R@10 最高提升 24.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;聚类分治难负例&quot;,&quot;Prompt 式多任务学习&quot;,&quot;双塔 ANN 检索&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/divide_and_conquer_towards_better_embedding_based_retrieval_for_recommender/">Divide and Conquer: Towards Better Embedding-based Retrieval for Recommender Systems From a Multi-task Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将候选库语义聚类后分簇并行 EBR，仅用同簇难负例训练并配合 Prompt 式多任务适配，KuaiRand Recall 最高提升 42.7%，线上 A/B 全指标正向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;级联图卷积&quot;,&quot;行为链依赖&quot;]"><strong><a href="../../venues/WWW/2023/graph_rec/multi_behavior_recommendation_with_cascading_graph_convolution_networks/">Multi-Behavior Recommendation with Cascading Graph Convolution Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用级联 LightGCN + 特征变换逐跳传递行为链特征、仅用目标行为监督，Beibei/Tmall 全库排序下 Recall@10 相对最强基线提升 23.2%/44.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;动态损失加权&quot;,&quot;会话级 MDP&quot;]"><strong><a href="../../venues/WWW/2023/multi_task_reranking/multi_task_recommendations_with_reinforcement_learning/">Multi-Task Recommendations with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将会话级交互建模为 MDP，用多 Critic 网络输出 Q 值动态调节各任务 BCE 权重，AUC 相对 SOTA 提升约 0.3–0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;长期留存优化&quot;,&quot;无限视界 MDP&quot;,&quot;Actor-Critic&quot;]"><strong><a href="../../venues/WWW/2023/rl_bandit_rec/reinforcing_user_retention_in_a_billion_scale_short_video_recommender_syste/">Reinforcing User Retention in a Billion Scale Short Video Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频留存优化建模为无限视界 request-based MDP，以最小化累积回归间隔为目标，用奖励归一化、活跃度分组策略与软行为正则改造 Actor-Critic，快手全量 A/B 中 DAU +0.2%、次日留存 +0.053%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;停留时间重加权&quot;,&quot;有效阅读筛选&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/WWW/2023/self_supervised_denoising/reweighting_clicks_with_dwell_time_in_recommendation/">Reweighting Clicks with Dwell Time in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用停留时间定义有效阅读并以归一化 Sigmoid 重加权点击信号，工业级双塔多任务框架在微信看一看离线 AUC+5.62%、线上 CTR+2.58%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Constrained MDP&quot;,&quot;Actor-Critic&quot;,&quot;多目标软约束&quot;]"><strong><a href="../../venues/WWW/2023/rl_bandit_rec/two_stage_constrained_actor_critic_for_short_video_recommendation/">Two-Stage Constrained Actor-Critic for Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将短视频推荐建模为 CMDP，用多 Critic 分离评估稠密/稀疏奖励，两阶段软 KL 约束 Actor，离线 WatchTime +2.23%、线上 +0.379%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多目标优化&quot;,&quot;数据重采样&quot;,&quot;PID 控制&quot;]"><strong><a href="../../venues/WWW/2024/multi_task_reranking/a_data_centric_multi_objective_learning_framework_for_responsible_recommend/">A Data-Centric Multi-Objective Learning Framework for Responsible Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出三层数据采样框架 MoRec，用符号梯度重加权与 PI 控制器统一优化准确率、收入、公平与对齐，最大精度损失 2.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;结果缓存 Router&quot;,&quot;特征函数学习 Critic 解耦&quot;]"><strong><a href="../../venues/WWW/2024/rl_bandit_rec/cache_aware_reinforcement_learning_in_large_scale_recommender_systems/">Cache-Aware Reinforcement Learning in Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将结果缓存的路由决策显式建模为 MDP 的 cache state，提出特征函数学习解耦双 critic，Kwai 线上 session 观看时长 +0.545%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;离散扩散&quot;,&quot;条件生成&quot;]"><strong><a href="../../venues/WWW/2024/multi_task_reranking/discrete_conditional_diffusion_for_reranking_in_recommendation/">Discrete Conditional Diffusion for Reranking in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出离散条件扩散重排DCDR，以交换/替换加噪和用户反馈条件去噪生成序列，离线提升显著并由300M DAU线上验证。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨阶段数据复用&quot;,&quot;Set-to-Set 共交互建模&quot;,&quot;SimHash 近邻选择&quot;]"><strong><a href="../../venues/WWW/2024/ctr_feature_interaction/recall_augmented_ranking_enhancing_click_through_rate_prediction_accuracy_w/">Recall-Augmented Ranking: Enhancing Click-Through Rate Prediction Accuracy with Cross-Stage Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RAR 用 SimHash 从跨阶段召回池选出相似用户集与召回物品集，经 set-to-set 匹配矩阵共交互聚合增强排序表征，即插即用；在 KKBox/MovieLens/CandiCTR-Pub 上 AUC 相对提升最高 4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;准确率-多样性权衡&quot;,&quot;消费约束&quot;,&quot;γ-同质性&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/reconciling_the_accuracy_diversity_trade_off_in_recommendations/">Reconciling the accuracy-diversity trade-off in recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用消费约束（用户只看 top-k）解释准确率-多样性权衡，证明计入 top-k 目标时最优解自然多样、忽略时完全同质，并以 γ-同质性给出由分布尾部决定的闭式多样性指数。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;个性化列表长度&quot;,&quot;概率校准&quot;,&quot;期望用户效用&quot;]"><strong><a href="../../venues/WWW/2024/multi_task_reranking/top_personalized_k_recommendation/">Top-Personalized-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出按用户效用自适应决定推荐列表长度的新任务，用用户级 Platt 校准概率估计期望效用并选最优 k，全库排序下 F1 相对 SOTA 截断法提升 3.7%–16.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;跨阶段协同&quot;,&quot;Margin InfoNCE&quot;,&quot;样本选择偏差&quot;]"><strong><a href="../../venues/WWW/2025/multi_task_reranking/a_hybrid_cross_stage_coordination_pre_ranking_model_for_online_recommendati/">A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 HCCP 粗排模型，融合上游召回与下游精排多级未曝光样本，用 Margin InfoNCE 联合优化一致性与长尾精度，JD 线上 UCVR+14.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;原生广告分配&quot;,&quot;近似算法&quot;,&quot;注意力衰减&quot;]"><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;重排&quot;]"><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/joint_evaluation_of_fairness_and_relevance_in_recommender_systems_with_pare/">Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 DPFR：以测试集构造相关性-公平性帕累托前沿，用模型得分到前沿的欧氏距离作为联合评测指标，揭示现有联合指标排序与帕累托最优解不一致。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;LLM 零样本重排&quot;,&quot;Chain-of-Thought 节点图&quot;,&quot;多准则均衡&quot;]"><strong><a href="../../venues/WWW/2025/multi_task_reranking/llm4rerank_llm_based_auto_reranking_framework_for_recommendations/">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将准确性/多样性/公平性抽象为全连接图节点，用零样本 LLM 以 CoT 多跳自动重排，三数据集上多准则均衡超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;邻居列表反事实训练&quot;,&quot;非自回归采样生成&quot;]"><strong><a href="../../venues/WWW/2025/multi_task_reranking/nlgr_utilizing_neighbor_lists_for_generative_rerank_in_personalized_recomme/">NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用邻居列表相对奖励对齐评估器-生成器目标，并以采样式非自回归生成灵活跳转组合空间，美团线上 CTR +3.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Prompt Tuning&quot;,&quot;Continual Learning&quot;]"><strong><a href="../../venues/WWW/2025/multi_task_reranking/pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;条件概率融合&quot;,&quot;个性化相关性激励&quot;,&quot;语义一致性正则&quot;]"><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;不确定性量化&quot;,&quot;不确定性分解&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/uncertainty_quantification_and_decomposition_for_llm_based_recommendation/">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Plackett-Luce 单次推理估计列表排序预测熵，并分解为推荐/提示不确定性，指导不确定性感知提示提升 N@20 约 17%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;多任务融合&quot;,&quot;强化学习&quot;,&quot;单调融合单元&quot;]"><strong><a href="../../venues/WWW/2025/multi_task_reranking/xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Target-Aware 蒸馏&quot;,&quot;Generate-then-Rank&quot;,&quot;分解因果自注意力&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多任务与列表级重排&quot;,&quot;重排&quot;,&quot;约束多目标优化&quot;,&quot;LLM 智能体编排&quot;,&quot;演化算法&quot;]"><strong><a href="../../venues/WWW/2026/multi_task_reranking/llms_as_orchestrators_constraint_compliant_multi_agent_optimization_for_rec/">LLMs as Orchestrators: Constraint-Compliant Multi-Agent Optimization for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双种群演化搜索把公平/卖家覆盖/新品曝光当作硬约束，LLM 每 10 代调度探索-利用配比 α，配合 ε-松弛保证 100% 可行性与 HV 提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;梯度协调对比学习&quot;,&quot;难度感知模型路由&quot;]"><strong><a href="../../venues/WWW/2026/multi_task_reranking/not_all_candidates_are_created_equal_a_heterogeneity_aware_approach_to_pre_/">Not All Candidates are Created Equal: A Heterogeneity-Aware Approach to Pre-ranking in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>针对粗排异构候选的梯度冲突与算力错配，提出 GHCL 分组对比损失与难度感知双模型路由，AUC 全面超越 SOTA 且线上降本 6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;多目标 Pareto 优化&quot;,&quot;Look-ahead 序列重排&quot;,&quot;沉浸式交互推荐&quot;]"><strong><a href="../../venues/WWW/2026/multi_task_reranking/stcrank_spatio_temporal_collaborative_ranking_for_interactive_recommender_s/">STCRank: Spatio-temporal Collaborative Ranking for Interactive Recommender System at Kuaishou E-shop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向快手电商沉浸式交互推荐的时空协同排序框架，用标签/权重重构消解 vtr-cvr-sdr 目标干扰，并以看前式序列级重排破解时序贪心陷阱，线上 A/B 全链路 IPV +9.65%。</p>
</dd>
</dl>
