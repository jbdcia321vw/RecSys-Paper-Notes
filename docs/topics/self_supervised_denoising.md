---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪 (Self-Supervised, Contrastive & Denoising Rec)

> **研究范畴**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

当前方向共收录 **264** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 多模态推荐 ×24 · 多行为推荐 ×11 · 知识蒸馏 ×9 · 大语言模型推荐 ×8 · 序列推荐 ×6 · 冷启动 ×5 · 双层优化 ×5 · 图对比学习 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">179</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">68</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">57</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">40</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">33</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">25</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">24</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">22</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图对比学习" aria-pressed="false">图对比学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式反馈去噪" aria-pressed="false">隐式反馈去噪<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="监督对比学习" aria-pressed="false">监督对比学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="InfoNCE" aria-pressed="false">InfoNCE<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列建模" aria-pressed="false">多行为序列建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐与均匀性" aria-pressed="false">对齐与均匀性<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列去噪" aria-pressed="false">序列去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图对比学习" aria-pressed="false">意图对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增广" aria-pressed="false">数据增广<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督预训练" aria-pressed="false">自监督预训练<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解耦表征学习" aria-pressed="false">解耦表征学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视图对比学习" aria-pressed="false">跨视图对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Graph Transformer" aria-pressed="false">Graph Transformer<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不变风险最小化" aria-pressed="false">不变风险最小化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元网络" aria-pressed="false">元网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分自编码器" aria-pressed="false">变分自编码器<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/精排" aria-pressed="false">召回/精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回与精排" aria-pressed="false">召回与精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤" aria-pressed="false">图协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗训练" aria-pressed="false">对抗训练<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列数据增强" aria-pressed="false">序列数据增强<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异质信息网络" aria-pressed="false">异质信息网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦表征" aria-pressed="false">意图解耦表征<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型去噪" aria-pressed="false">扩散模型去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示微调" aria-pressed="false">提示微调<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交去噪" aria-pressed="false">社交去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督去噪" aria-pressed="false">自监督去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督学习" aria-pressed="false">自监督学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习" aria-pressed="false">自监督对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督数据增广" aria-pressed="false">自监督数据增广<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征对齐" aria-pressed="false">表征对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域对比学习" aria-pressed="false">跨域对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态对比学习" aria-pressed="false">跨模态对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态对齐" aria-pressed="false">跨模态对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视图对齐" aria-pressed="false">跨视图对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软标签" aria-pressed="false">软标签<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过平滑" aria-pressed="false">过平滑<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负例挖掘" aria-pressed="false">难负例挖掘<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负采样" aria-pressed="false">难负采样<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域去噪" aria-pressed="false">频域去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="鲁棒推荐" aria-pressed="false">鲁棒推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨服务对比预训练&quot;,&quot;用户表征缩放定律&quot;]"><strong><a href="../../venues/AAAI/2023/self_supervised_denoising/scaling_law_for_recommendation_models_towards_general_purpose_user_represen/">Scaling Law for Recommendation Models: Towards General-purpose User Representations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以层级双 Transformer + 跨服务 CLIP 式对比学习预训练通用用户表征 CLUE，在文本化物品描述上对齐异构服务，首次验证推荐用户建模的幂律缩放律，特征级迁移在离线与线上 CTR 上全面超越任务专属模型。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦训练（服务器聚合）&quot;,&quot;无目标投毒攻击&quot;,&quot;对比学习均匀性正则&quot;,&quot;拜占庭鲁棒聚合&quot;]"><strong><a href="../../venues/AAAI/2023/debias_fairness_industrial/untargeted_attack_against_federated_recommendation_systems_via_poisonous_it/">Untargeted Attack against Federated Recommendation Systems via Poisonous Item Embeddings and the Defense</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 ClusterAttack 用簇内方差最小化的恶意 item 梯度塌缩嵌入空间打乱排序，并给出基于对比学习均匀性正则的 UNION 防御与服务器端梯度过滤。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="../../venues/AAAI/2024/collaborative_filtering/adaptive_hardness_negative_sampling_for_collaborative_filtering/">Adaptive Hardness Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示固定难度负采样引发 FPP/FNP，提出三准则自适应难度范式 AHNS 及实例 AHNS_{p&lt;0}，4 数据集 R@20 最高 +8.2%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;InfoNCE&quot;,&quot;辅助匹配任务&quot;,&quot;负样本下采样&quot;]"><strong><a href="../../venues/AAAI/2024/ctr_feature_interaction/at4ctr_auxiliary_match_tasks_for_enhancing_click_through_rate_prediction/">AT4CTR: Auxiliary Match Tasks for Enhancing Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 UIM/NIP 双 InfoNCE 辅助任务缓解负采样稀疏，Taobao/美团 AUC 0.9535/0.7441，线上 RPS +1.27%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;Mixup&quot;,&quot;表征级融合&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/augmenting_sequential_recommendation_with_balanced_relevance_and_diversity/">Augmenting Sequential Recommendation with Balanced Relevance and Diversity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出即插即用平衡增强插件 BASRec，以单序列随机算子+表征级 mixup 与跨序列非线性 mixup 兼顾相关性与多样性，全库排序下 SASRec 平均提升 33.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;对比学习&quot;,&quot;样本效率&quot;,&quot;交互式推荐&quot;]"><strong><a href="../../venues/AAAI/2025/rl_bandit_rec/contrastive_representation_for_interactive_recommendation/">Contrastive Representation for Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用偏好排序对比学习(PRCL)为 DRL 交互推荐提炼状态表征，冷启动下样本效率显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;解耦表征学习&quot;,&quot;社交超图&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/AAAI/2025/social_group_rec/disentangled_modeling_of_preferences_and_social_influence_for_group_recomme/">Disentangled Modeling of Preferences and Social Influence for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>解耦用户偏好与社会影响的双通道(超)图传播，配合社交重要性引导的对比学习，缓解群组推荐的偏好偏差与数据稀疏。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;伪排序构造&quot;,&quot;排序损失&quot;,&quot;梯度置信度&quot;]"><strong><a href="../../venues/AAAI/2025/collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;软标签&quot;,&quot;困难负采样&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/future_sight_and_tough_fights_revolutionizing_sequential_recommendation_wit/">Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间相关软标签挖掘未来交互、用锚-负混合构造持久困难负样本，四数据集全库排序平均提升 6.16%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;共享账户推荐&quot;,&quot;图胶囊网络&quot;,&quot;子空间对齐&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/lightweight_yet_fine_grained_a_graph_capsule_convolutional_network_with_sub/">Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用线性注意力胶囊图+低秩子空间对齐，在共享账户混合序列上做细粒度潜在用户解耦，精度大幅领先且参数/耗时更低。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跨模态对齐&quot;,&quot;自监督对比学习&quot;,&quot;物品语义图&quot;]"><strong><a href="../../venues/AAAI/2025/multimodal_rec/mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域建模&quot;,&quot;跨会话信息&quot;,&quot;频域一致性损失&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;重排&quot;,&quot;群组画像&quot;,&quot;群组感知强化学习&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="../../venues/AAAI/2026/self_supervised_denoising/g_ubs_towards_robust_understanding_of_implicit_feedback_via_group_aware_use/">G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 群组画像 + 群组感知 GRPO 强化学习为个体隐式反馈去噪，构建 IF-VR 多模态基准，推理准确率提升 14.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;困难-噪声混淆&quot;,&quot;LLM 辅助信号&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/AAAI/2026/self_supervised_denoising/hard_vs_noise_resolving_hard_noisy_sample_confusion_in_recommender_systems_/">Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 语义/逻辑双相关性信号区分困难样本与噪声样本，配合目标对齐与图对比抑制幻觉，全库排序下 Recall@20 平均提升约 45%。</p>
</dd>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;召回与精排&quot;,&quot;预训练&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2023/sequential_rec/punr_pre_training_with_user_behavior_modeling_for_news_recommendation/">PUNR: Pre-training with User Behavior Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Findings of EMNLP 2023 序列与会话推荐：提出用户行为掩码与生成双预训练任务，结合孪生编码器优化新闻推荐用户表征。</p>
</dd>
<dt data-paper-tags="[&quot;ACL_EMNLP&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;重排&quot;,&quot;Denoising Attention&quot;,&quot;Query-aware User Modeling&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/self_supervised_denoising/denoising_attention_for_query_aware_user_modeling_in_personalized_search/">Denoising Attention for Query-aware User Modeling in Personalized Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ACL/EMNLP 2024 [自监督对比学习与交互去噪] 以有界对齐+阈值移位+ReLU 门控与 max-归一化重构查询感知用户建模，Top-1000 重排协议下 MAP 较最佳注意力基线提升 14–17%</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;协同过滤与双塔召回&quot;,&quot;排序&quot;,&quot;模块化新闻推荐&quot;,&quot;监督对比学习&quot;,&quot;方面定制&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/collaborative_filtering/train_once_use_flexibly_a_modular_framework_for_multi_aspect_neural_news_re/">Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用监督对比学习训练内容/方面专用新闻编码器，推理时线性组合相似度实现按需个性化与多样化，MIND上nDCG@10相对提升2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导模态去噪&quot;,&quot;多视图图卷积&quot;,&quot;行为感知融合&quot;]"><strong><a href="../../venues/ACM_MM/2023/multimodal_rec/multi_view_graph_convolutional_network_for_multimedia_recommendation/">Multi-View Graph Convolutional Network for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为引导净化模态噪声 + U-I/I-I 双视图编码 + 行为感知自适应融合与互信息自监督，Amazon 三集全排序 Recall@20 最高提升 22.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布表征&quot;]"><strong><a href="../../venues/ACM_MM/2023/sequential_rec/personalized_behavior_aware_transformer_for_multi_behavior_sequential_recom/">Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布刻画物品/行为/位置并借 Wasserstein 距离重构自注意力，把个性化多行为模式与行为-时间协同注入 Transformer；Leave-One-Out + HR/NDCG 上 Taobao HR@5 相对最优基线提升 6.37%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;属性驱动解耦表征&quot;,&quot;跨模态对比学习&quot;]"><strong><a href="../../venues/ACM_MM/2024/multimodal_rec/attribute_driven_disentangled_representation_learning_for_multimodal_recomm/">Attribute-driven Disentangled Representation Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 price/brand/category/popularity 显式绑定到表征 chunk，用模态内分类、模态间对比与属性值预测实现属性级解耦，Recall@20 提升 4.8%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回/排序&quot;,&quot;图扩散&quot;,&quot;跨模态对比&quot;]"><strong><a href="../../venues/ACM_MM/2024/multimodal_rec/diffmm_multi_modal_diffusion_model_for_recommendation/">DiffMM: Multi-Modal Diffusion Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用模态感知图扩散生成用户-物品图，并以跨模态对比对齐多模态与协同信号；Amazon 全库排序 R@20 相对 MMSSL 提升约 3.9%–5.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;MLLM 负样本生成&quot;,&quot;因果学习&quot;]"><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/generating_negative_samples_for_multi_modal_recommendation/">Generating Negative Samples for Multi-Modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 三步提示生成兼具凝聚力与难度的属性级负样本，并以因果学习解耦关键特征，全库排序下 Recall@10 最高提升 9.02%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;不变风险最小化&quot;,&quot;信息瓶颈&quot;]"><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/i_3_mrec_invariant_learning_with_information_bottleneck_for_incomplete_moda/">I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将各模态视为独立语义环境做 IRM 不变学习，并用 IB 压缩模态特有信息，在模态缺失场景下 Full-Ranking Recall@20 稳定提升 2%–4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态对比学习精炼&quot;,&quot;同构图关系挖掘&quot;]"><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/refining_contrastive_learning_and_homography_relations_for_multi_modal_reco/">Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用元网络+正交约束精炼多模态对比学习，并补建用户兴趣图与物品共现图，全库排序下 R@20 最高提升约 9.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;正交净化&quot;,&quot;混合专家路由&quot;]"><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回（序列召回）&quot;,&quot;自注意力校准&quot;,&quot;对抗扰动去噪&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/attention_calibration_for_transformer_based_sequential_recommendation/">Attention Calibration for Transformer-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐中注意力权重错配现象，提出空间校准器与对抗校准器直接修正注意力分布，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 8.65%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平与工业系统&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;对比学习&quot;,&quot;反事实采样&quot;]"><strong><a href="../../venues/CIKM/2023/debias_fairness_industrial/contrastive_counterfactual_learning_for_causality_aware_interpretable_recom/">Contrastive Counterfactual Learning for Causality-aware Interpretable Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用因果图定位曝光混杂，提出CCL以反事实正采样做对比学习去偏，在Coat/Yahoo! R3无偏测试集全库排序上超越IPS/CVIB。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负样本采样&quot;,&quot;Mixup 数据增广&quot;]"><strong><a href="../../venues/CIKM/2023/collaborative_filtering/dimension_independent_mixup_for_hard_negative_sample_in_collaborative_filte/">Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>从采样区域视角重塑隐式反馈负采样，提出首个 Area-wise 方法 DINS（边界定义+逐维独立 Mixup+多层池化），在 Full-Ranking 协议下 MF backbone 上 Recall@20 最高提升 36.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;对齐与一致性&quot;,&quot;高阶邻域聚合&quot;,&quot;轻量图卷积&quot;]"><strong><a href="../../venues/CIKM/2023/graph_rec/graph_based_alignment_and_uniformity_for_recommendation/">Graph-based Alignment and Uniformity for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以高阶邻域聚合压缩稠密表示并对齐用户/物品，层间池化缓解稀疏，Office R@20 较 DirectAU 提升 16.64%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;随机播放&quot;,&quot;会话推荐&quot;,&quot;转移增广&quot;]"><strong><a href="../../venues/CIKM/2023/self_supervised_denoising/muse_music_recommender_system_with_shuffle_play_recommendation_enhancement/">MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对音乐随机播放会话唯一转移率过高的问题，提出转移频次引导的会话增广与细粒度匹配，在 MSSD 全库排序上稳定超越 12 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;超图推荐&quot;,&quot;一致性感知增强&quot;,&quot;购物篮推荐&quot;]"><strong><a href="../../venues/CIKM/2023/self_supervised_denoising/robust_basket_recommendation_via_noise_tolerated_graph_contrastive_learning/">Robust Basket Recommendation via Noise-tolerated Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>BNCL 以双行为图/超图与一致性感知对比学习抑制购物篮噪声，全库排序下 R@60 最高 21.10。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数统一分析&quot;,&quot;对齐与均匀性&quot;,&quot;流行度去偏边缘&quot;]"><strong><a href="../../venues/CIKM/2023/collaborative_filtering/toward_a_better_understanding_of_loss_functions_for_collaborative_filtering/">Toward a Better Understanding of Loss Functions for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用统一数学框架把 7 种 CF 损失还原为 SSM 的极限特例，并提出边缘感知对齐+加权均匀性损失 MAWU，MF/LightGCN 全库排序下稳定超越 DirectAU 与 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回与排序&quot;,&quot;表征对齐&quot;,&quot;对比学习&quot;,&quot;视觉语言预训练&quot;]"><strong><a href="../../venues/CIKM/2024/multimodal_rec/alignrec_aligning_and_training_in_multimodal_recommendations/">AlignRec: Aligning and Training in Multimodal Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多模态推荐目标拆解为内容内、内容-类别、用户-物品三重对齐，配合两阶段解耦训练与中间评估协议，全库排序 R@20 相对提升 4.9%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;医疗处方推荐&quot;,&quot;意图对比学习&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/contrastive_learning_on_medical_intents_for_sequential_prescription_recomme/">Contrastive Learning on Medical Intents for Sequential Prescription Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ARCI 用跨就诊 Transformer 抽取多条“时间路径”，并以意图对比学习正则化多头注意力，在 MIMIC-III/AKI 全库排序上命中率较 SOTA 提升 5–7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;缺失模态插补&quot;,&quot;图特征传播&quot;,&quot;Personalized PageRank&quot;]"><strong><a href="../../venues/CIKM/2024/multimodal_rec/do_we_really_need_to_drop_items_with_missing_modalities_in_multimodal_recom/">Do We Really Need to Drop Items with Missing Modalities in Multimodal Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用免训练插补替代“丢弃缺失模态物品”，提出 NeighMean/MultiHop/PersPageRank 三种图感知插补，在 Amazon 三域 Recall@20 最高提升 20.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤&quot;,&quot;反馈分图对比学习&quot;,&quot;宏观聚类建模&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/feedback_reciprocal_graph_collaborative_filtering/">Feedback Reciprocal Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按用户反馈将交互图拆为 I&amp;F/I&amp;U 双图分别做 GCF，用冲动性互惠对比与宏观聚类建模交换信息，全库排序下 Recall@20 平均提升 6.28%，淘宝 A/B GMV +1.61%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Triplet Shapley&quot;,&quot;负采样与三元组重加权&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/interpretable_triplet_importance_for_personalized_ranking/">Interpretable Triplet Importance for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用三元组 Shapley 值可解释地度量 BPR 三元组重要性，蒙特卡洛+控制变量稳定估计并指导重采样，六数据集 R@20 平均提升约 9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;扩散模型去噪&quot;,&quot;社交关系重校准&quot;]"><strong><a href="../../venues/CIKM/2024/social_group_rec/recdiff_diffusion_model_for_social_recommendation/">RecDiff: Diffusion Model for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在隐空间用扩散模型对社交图嵌入做多步加噪-去噪，任务感知训练剔除虚假社交边，Yelp Full-Ranking Recall@20 达 0.0597。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨模态兴趣对齐&quot;,&quot;序列去噪&quot;,&quot;LLM2Vec&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;生成式数据增强&quot;,&quot;物品互补矩阵&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态对齐&quot;,&quot;空洞卷积注意力&quot;,&quot;MMD 分布正则&quot;]"><strong><a href="../../venues/CIKM/2025/multimodal_rec/modality_alignment_with_multi_scale_bilateral_attention_for_multimodal_reco/">Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 MambaRec，用多尺度空洞卷积+双注意力做局部图文对齐、MMD+InfoNCE 做全局分布对齐，Amazon 三数据集 Recall@20 最高提升约 5.4%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;多查询注意力&quot;,&quot;查询窗口矩阵&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/multi_item_query_attention_for_stable_sequential_recommendation/">Multi-Item-Query Attention for Stable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多物品查询窗口+虚拟物品替代单查询掩码注意力，缓解序列噪声敏感，ML-1M HR@5 提升 21.2%，LastFM 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱融合&quot;,&quot;对抗扰动不稳定性估计&quot;,&quot;自适应对比学习&quot;]"><strong><a href="../../venues/CIKM/2026/graph_rec/do_all_nodes_benefit_equally_from_knowledge_graphs_adaptive_node_aware_kg_f/">Do All Nodes Benefit Equally from Knowledge Graphs? Adaptive Node-Aware KG Fusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以对抗扰动估计节点级 CF 不稳定性，自适应门控 KG 注入强度并配合自适应对比对齐，四数据集 Full-Ranking 平均排名第一。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;显式负反馈&quot;]"><strong><a href="../../venues/CIKM/2026/collaborative_filtering/dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;注意力校准&quot;,&quot;MoE&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/from_overlooked_to_explored_recovering_item_relations_via_mixture_of_perspe/">From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示自注意力相似性偏置导致异构关系被系统性忽略，提出 K 个视角透镜（亲和/对比双视图）校准注意力，7 个基准全库排序平均提升 3–11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;组件级嫁接&quot;,&quot;对比蒸馏&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/god_enhancing_generalization_via_deep_grafting_for_sequential_recommendatio/">GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「嫁接」构造教师条件混合源模型，对学生嵌入表/编码器做组件级蒸馏，Full-Ranking 下最高相对提升 13.92%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;信息论引导对齐&quot;,&quot;行为校准图精炼&quot;,&quot;共享兴趣子空间&quot;]"><strong><a href="../../venues/CIKM/2026/multimodal_rec/information_guided_selective_modality_interest_alignment_for_multimodal_rec/">Information-Guided Selective Modality-Interest Alignment for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以互信息/条件熵分解为理论视角，用行为校准图精炼+兴趣对比+共享子空间选择性对齐，三数据集 Full-Ranking 全面 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;曲率感知传播深度&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/CIKM/2026/graph_rec/neural_tree_collaborative_filtering_rethinking_graph_collaborative_filterin/">Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以局部度不平衡曲率代理为每节点分配自适应传播深度，将 GCF 重释为节点根树集成，Full-Ranking 下 R@20 稳定超越 LightGCN 与 SSL 骨干。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;可学习数据增强&quot;,&quot;对抗训练&quot;]"><strong><a href="../../venues/DASFAA/2023/self_supervised_denoising/adversarial_learning_data_augmentation_for_graph_contrastive_learning_in_re/">Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 InfoMin/InfoMax 对抗框架学习可微的边增删图增强算子，四数据集 Full-Ranking 下 Recall@10 最高相对 NCL 提升 23.5%</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;反事实数据增强&quot;,&quot;捆绑推荐&quot;,&quot;对比约束&quot;]"><strong><a href="../../venues/DASFAA/2023/graph_rec/data_augmented_counterfactual_learning_for_bundle_recommendation/">Data-Augmented Counterfactual Learning for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用启发式采样器生成反事实用户-捆绑包-物品图视图，并以自适应阈值反事实损失约束训练，缓解捆绑推荐双向稀疏，全库排序下 R@20 最高提升约 9.9%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;物品侧对比学习&quot;,&quot;多粒度数据增广&quot;]"><strong><a href="../../venues/DASFAA/2023/self_supervised_denoising/multi_granularity_item_based_contrastive_recommendation/">Multi-granularity Item-based Contrastive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为双塔召回设计特征/语义/会话三级物品侧对比任务，工业级全库 HIT@500 相对 SOTA 提升 2.8%，线上 ACC +3.29%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;软标签解耦&quot;,&quot;标签传播&quot;]"><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/learning_recommender_systems_with_soft_target_a_decoupled_perspective/">Learning Recommender Systems with Soft Target: A Decoupled Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Softmax 软标签目标解耦为目标置信度与未观测项兴趣分布两项，配合 K-means 邻居标签传播生成软标签，在四个骨干模型三个数据集上稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向 Transformer&quot;,&quot;多正样本对比学习&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/contrastive_learning_with_bidirectional_transformers_for_sequential_recomme/">Contrastive Learning with Bidirectional Transformers for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用双向 Transformer + 滑窗切分 + 完形/ dropout 混合增广构造多正样本对比学习，并动态重加权双损失；在 Beauty/Toys/ML-1M 全库排序下 NDCG@10 最高提升 21.04%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/下一物品排序&quot;,&quot;自对比学习&quot;,&quot;表示均匀性&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/self_contrastive_learning_for_session_based_recommendation/">Self Contrastive Learning for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将自对比损失作为物品表示均匀性正则，无需数据增广/正负样本构造，在三个会话推荐基准全库排序下显著提升 P@10/MRR@10。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;LLM 提示去噪&quot;,&quot;后训练用户画像清洗&quot;]"><strong><a href="../../venues/ECIR/2026/self_supervised_denoising/post_training_denoising_of_user_profiles_with_llms_in_collaborative_filteri/">Post-Training Denoising of User Profiles with LLMs in Collaborative Filtering Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 零样本/少样本提示在推理前清洗用户画像，无需重训 CF 模型即获最高 13% NDCG/HR 提升。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（训练期负采样）&quot;,&quot;假阴性辨识&quot;,&quot;贝叶斯后验风险最小化&quot;,&quot;负采样&quot;]"><strong><a href="../../venues/ICDE/2023/collaborative_filtering/bayesian_negative_sampling_for_recommendation/">Bayesian Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>由排序关系导出真/假阴性类条件密度，构造模型无关贝叶斯后验负信号与风险最小化采样准则；MF/LightGCN 双主干 Top-20 NDCG 最高 +5.1%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;频域滤波&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/ICDE/2023/sequential_rec/contrastive_enhanced_slide_filter_mixer_for_sequential_recommendation/">Contrastive Enhanced Slide Filter Mixer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用频域滑动滤波与对比学习替代自注意力，在5个基准全库排序下 HR@10 最高相对提升21.39%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排表示增强&quot;,&quot;意图解耦&quot;,&quot;标签对比对齐&quot;]"><strong><a href="../../venues/ICDE/2023/self_supervised_denoising/intent_aware_multi_source_contrastive_alignment_for_tag_enhanced_recommenda/">Intent-aware Multi-source Contrastive Alignment for Tag-enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户/物品表征拆成K个意图子向量，与标签簇端到端聚类对齐，并用集合到集合正样本增强；7数据集Top-20排序最高R@20 +6.7%、NDCG@20 +10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;过平滑缓解&quot;,&quot;层精炼 Layer Refinement&quot;,&quot;度敏感边丢弃&quot;]"><strong><a href="../../venues/ICDE/2023/graph_rec/layer_refined_graph_convolutional_networks_for_recommendation/">Layer-refined Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ego 层余弦相似度动态精炼每层传播表征，配合度敏感边剪枝，破解 GCN 推荐过平滑困境，全库排序下 Recall@20 最高提 10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;Graph Information Bottleneck&quot;,&quot;Mixhop 图对比学习&quot;]"><strong><a href="../../venues/ICDE/2024/self_supervised_denoising/graph_augmentation_for_recommendation/">Graph Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 GIB 正则化的可学习图增强器生成去噪对比视图，配合 Mixhop 混阶编码器缓解过平滑，Recall@20 较 NCL 提升 5.9%–8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;大语言模型推荐&quot;,&quot;召回/表征增强&quot;,&quot;LLM 图增强&quot;,&quot;去噪鲁棒训练&quot;]"><strong><a href="../../venues/ICDE/2024/llm_rec/llmrec_large_language_models_with_graph_augmentation_for_recommendation/">LLMRec: Large Language Models with Graph Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>LLM 从自然语言侧增强 U-I 交互边、物品属性与用户画像，并以噪声剪枝+MAE 鲁棒化，全库排序 R@10 提升 13.95%/4.88%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;生成式视图增强&quot;,&quot;变分自编码器&quot;,&quot;元优化两步训练&quot;]"><strong><a href="../../venues/ICDE/2024/self_supervised_denoising/meta_optimized_joint_generative_and_contrastive_learning_for_sequential_rec/">Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 VAE Seq2Seq 生成器替代手工增强生成对比视图，双 ELBO 导出互信息项并落地 InfoNCE，元优化两步训练自适应视图，全库排序 HR@10 最高提升 20.8%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;序列推荐&quot;,&quot;序列去噪&quot;,&quot;自增强&quot;,&quot;全局关系编码&quot;]"><strong><a href="../../venues/ICDE/2024/self_supervised_denoising/ssdrec_self_augmented_sequence_denoising_for_sequential_recommendation/">SSDRec: Self-Augmented Sequence Denoising for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ICDE 2024 自监督交互去噪 | 序列增强后再去噪以缓解过度/不足去噪(OUPs)，Full-Ranking 提升 SASRec 平均 23%–110%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;表征解耦对齐&quot;,&quot;结构对齐&quot;]"><strong><a href="../../venues/ICDE/2025/llm_rec/darec_a_disentangled_alignment_framework_for_large_language_model_and_recom/">DaRec: A Disentangled Alignment Framework for Large Language Model and Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从信息论证明零间隙对齐次优，提出解耦共享/特有表征+全局局部结构对齐的即插即用框架，全库排序下 Recall@20 最高提升 5.33%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排&quot;,&quot;对抗负采样&quot;,&quot;难例挖掘&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/self_supervised_denoising/empowering_collaborative_filtering_with_principled_adversarial_contrastive_/">Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 AdvInfoNCE：用可学习难度 δ_j 对抗重加权负样本，证明等价 KL-DRO，在四个 Top-K 基准上显著提升 OOD 泛化。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;SVD 图增强&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/self_supervised_denoising/lightgcl_simple_yet_effective_graph_contrastive_learning_for_recommendation/">LightGCL: Simple Yet Effective Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以截断 SVD 重构交互图生成全局协同视图，并与主视图直接对比，在五个基准的全库排序中 Recall@20 最高提升 23%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;端到端可学习聚类&quot;,&quot;意图对比学习&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/sequential_rec/end_to_end_learnable_clustering_for_intent_learning_in_recommendation/">End-to-end Learnable Clustering for Intent Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将意图聚类中心设为可学习神经元，端到端优化聚类与对比学习，全库排序下NDCG@5较ICLRec提升8.9%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;无监督聚类&quot;,&quot;自监督学习&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/social_group_rec/identify_then_recommend_towards_unsupervised_group_recommendation/">Identify Then Recommend: Towards Unsupervised Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>无监督识别动态群组并用双重自监督 pretext 推荐，免组数/组标签；两数据集 HR/NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数设计&quot;,&quot;假阴性鲁棒&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/collaborative_filtering/psl_rethinking_and_improving_softmax_loss_from_pairwise_perspective_for_rec/">PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 Softmax Loss 改写为成对形式并把 exp 换成 tanh/atan/ReLU 有界激活，兼具更紧 DCG 上界、可控样本权重与 DRO 化 BPR 三重性质，IID/OOD/噪声三场景 NDCG@20 稳定提升 1.1%–5.0%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;LLM语义检索&quot;,&quot;可学习样本合成&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommenda/">Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义检索构建用户/物品候选池并配可学习样本合成器，替代随机增广与 K-Means 硬规则，全库排序下 HR@20 最高提升 11.82%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;生成式推荐&quot;,&quot;递归自我改进&quot;,&quot;保真度质量控制&quot;,&quot;数据稠密化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图信息瓶颈&quot;,&quot;跨行为对比学习&quot;,&quot;多行为推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图协同过滤&quot;,&quot;对比学习损失&quot;,&quot;邻居对加权&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;社交与群组推荐&quot;,&quot;召回与精排&quot;,&quot;去噪自监督&quot;,&quot;跨视图对齐&quot;]"><strong><a href="../../venues/IJCAI/2023/social_group_rec/denoised_self_augmented_learning_for_social_recommendation/">Denoised Self-Augmented Learning for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出双视图GCN与交互引导去噪自监督，在Ciao/Epinions/Yelp的Leave-One-Out全库排序上显著超越MHCN/SMIN。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;意图解耦表征&quot;,&quot;意图级对比学习&quot;,&quot;码率约简正则&quot;]"><strong><a href="../../venues/IJCAI/2023/graph_rec/intent_aware_recommendation_via_disentangled_graph_contrastive_learning/">Intent-aware Recommendation via Disentangled Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户-物品-概念图为基础，用概念软聚类语义基引导行为意图解耦，配合意图级对比学习与码率约简正则，在 ML-100k/ML-1M/MtBusiness 全库排序下 Recall@20 最高相对提升 8.52%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;用户行为采样&quot;,&quot;强化学习策略梯度&quot;,&quot;多目标奖励&quot;]"><strong><a href="../../venues/IJCAI/2025/sequential_rec/towards_automatic_sampling_of_user_behaviors_for_sequential_recommender_sys/">Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 AutoSAM，用轻量采样器+强化学习多目标奖励自适应丢弃低信息历史行为，在四个真实数据集全库排序下 NDCG@10 平均提升约 8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自适应视图生成&quot;,&quot;图边去噪&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/KDD/2023/self_supervised_denoising/adaptive_graph_contrastive_learning_for_recommendation/">Adaptive Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 VGAE 生成视图与可学习边去噪视图替代手工图增广，为 LightGCN 提供自适应对比信号，全库排序下 Recall@20 相对提升约 5–8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;一致性训练&quot;,&quot;双向KL散度&quot;]"><strong><a href="../../venues/KDD/2023/sequential_rec/ct4rec_simple_yet_effective_consistency_training_for_sequential_recommendat/">CT4Rec: Simple yet Effective Consistency Training for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>仅加双向KL一致性正则，在四数据集采样排序上超CL4SRec且线上CTR提升2.717%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;双层优化&quot;,&quot;梯度匹配&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="../../venues/KDD/2023/self_supervised_denoising/efficient_bi_level_optimization_for_recommendation_denoising/">Efficient Bi-Level Optimization for Recommendation Denoising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将隐式反馈去噪建模为双层优化，用权重生成器+一步梯度匹配高效求解，LightGCN 上 Recall@20 最高提升 10.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱合理化&quot;,&quot;掩码自编码器&quot;,&quot;跨视图对比学习&quot;]"><strong><a href="../../venues/KDD/2023/graph_rec/knowledge_graph_self_supervised_rationalization_for_recommendation/">Knowledge Graph Self-Supervised Rationalization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KGRec 用注意力合理化分数驱动掩码重建生成式任务与去噪跨视图对比学习，Full-Ranking 下 Recall@20 较 SOTA 提升 2.3%–4.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;自监督预训练&quot;,&quot;特征损坏与恢复&quot;,&quot;噪声对比估计&quot;]"><strong><a href="../../venues/KDD/2023/ctr_feature_interaction/map_a_model_agnostic_pretraining_framework_for_click_through_rate_predictio/">MAP: A Model-agnostic Pretraining Framework for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向多域类别数据提出特征损坏-恢复预训练框架，派生 MFP 与 RFD 两种自监督任务，在 Avazu/Criteo 上为 DCNv2 等基座带来最高 1.22% AUC 提升，且预训练提速约 18×</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;数据侧（训练集重构）&quot;,&quot;数据集再生&quot;,&quot;双层优化&quot;,&quot;序列模式挖掘&quot;]"><strong><a href="../../venues/KDD/2024/sequential_rec/dataset_regeneration_for_sequential_recommendation/">Dataset Regeneration for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出数据中心的 DR4SR/DR4SR+ 框架，用多样性促进的 Transformer 再生器把原始序列再生为高质量转移模式数据集，配合双层优化个性化打分，在 4 个数据集 5 类骨干上 Full-Ranking NDCG@20 最高提升 21.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;全链路训练/学习范式&quot;,&quot;交互去噪&quot;,&quot;隐式反馈&quot;]"><strong><a href="../../venues/KDD/2024/self_supervised_denoising/double_correction_framework_for_denoising_recommendation/">Double Correction Framework for Denoising Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024交互去噪推荐针对单次损失判定噪声的不稳定性，提出DCF双重校正框架：确认损失+硬样本搜索与渐进式标签翻转，显著提升推荐精度。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;决策边界感知扰动&quot;,&quot;对抗图对比学习&quot;]"><strong><a href="../../venues/KDD/2024/self_supervised_denoising/towards_robust_recommendation_via_decision_boundary_aware_graph_contrastive/">Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>决策边界感知扰动、关系感知对抗图对比与间隔最大化；五数据集全库排序最高相对提升 14.14%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图卷积等价&quot;,&quot;简单对比协同过滤&quot;]"><strong><a href="../../venues/KDD/2024/self_supervised_denoising/unifying_graph_convolution_and_contrastive_learning_in_collaborative_filter/">Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论证明对比损失等价双图卷积，提出无GCN的SCCF，在4数据集全库排序最高提升6.4%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段数据投毒&quot;,&quot;对比学习推荐&quot;,&quot;谱分析&quot;,&quot;双层优化&quot;]"><strong><a href="../../venues/KDD/2024/debias_fairness_industrial/unveiling_vulnerabilities_of_contrastive_recommender_systems_to_poisoning_a/">Unveiling Vulnerabilities of Contrastive Recommender Systems to Poisoning Attacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从谱分析揭示 InfoNCE 的全局离散效应正是 CL 推荐易受定向投毒的根源，提出双目标 CLeaR 攻击，Hit-Ratio@50 平均提升 1.5–3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;文本-协同语义融合&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/bridging_textual_collaborative_gap_through_semantic_codes_for_sequential_re/">Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多视图文本量化语义码作桥梁，以码为 Query 交叉注意力融合文本，配合双重掩码自监督，Full-Ranking 下 R@10 最高提升 20.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;离散码向量量化&quot;,&quot;虚拟邻居增强&quot;]"><strong><a href="../../venues/KDD/2025/self_supervised_denoising/enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱推荐&quot;,&quot;轻量化GNN&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/KDD/2025/graph_rec/lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;虚拟三元组&quot;,&quot;预热阈值策略&quot;,&quot;数据稀疏&quot;]"><strong><a href="../../venues/KDD/2025/multimodal_rec/mdvt_enhancing_multimodal_recommendation_with_model_agnostic_multimodal_dri/">MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多模态融合表征动态构造虚拟三元组作为额外监督信号，配合三种预热阈值策略与增强 pairwise 损失，即插即用缓解数据稀疏，稀疏用户组 NDCG@10 最高提升 12.85%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;异构图表征&quot;,&quot;多任务对比学习&quot;,&quot;大规模图引擎&quot;]"><strong><a href="../../venues/KDD/2025/self_supervised_denoising/omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;元学习&quot;,&quot;流行度感知任务切分&quot;,&quot;在线流式训练&quot;]"><strong><a href="../../venues/KDD/2025/cold_start_rec/online_item_cold_start_recommendation_with_popularity_aware_meta_learning/">Online Item Cold-Start Recommendation with Popularity-Aware Meta-Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>按流行度固定切分元学习任务并配冷启动增强器，流式在线场景下冷启动召回相对 SOTA 提升 11%–74%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;数据剪枝/重训&quot;,&quot;Shapley Value&quot;,&quot;FastSHAP&quot;,&quot;交互价值评估&quot;]"><strong><a href="../../venues/KDD/2025/self_supervised_denoising/shapley_value_driven_data_pruning_for_recommender_systems/">Shapley Value-driven Data Pruning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 FastSHAP 实时估计交互对训练损失的边际贡献，按 Shapley 值剪枝低效用交互并重训，四数据集 Recall/NDCG 平均提升 3.72%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;语义因子路由&quot;,&quot;对齐与均匀性&quot;]"><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;伪标签纠正&quot;,&quot;多模态内容相似&quot;]"><strong><a href="../../venues/KDD/2026/cold_start_rec/denoising_implicit_feedback_for_cold_start_recommendation/">Denoising Implicit Feedback for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用内容相似热物品协同表征为冷物品生成置信加权伪标签，并以相对熵×冷启动状态估计样本不确定性做自适应软标签纠正，快手十亿级 A/B 全指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;特征频域谱滤波&quot;,&quot;可靠性感知融合&quot;]"><strong><a href="../../venues/KDD/2026/self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;画像外推&quot;,&quot;不变性学习&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/proex_a_unified_framework_leveraging_large_language_model_with_profile_extr/">ProEx: A Unified Framework Leveraging Large Language Model with Profile Extrapolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用四步 CoT 为每个用户/物品生成多面画像并做环境不变性融合，在三个数据集上对 6 个基模型平均提升 5%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;虚拟交互增强&quot;,&quot;模态相似度&quot;]"><strong><a href="../../venues/KDD/2026/multimodal_rec/vi_mmrec_similarity_aware_training_cost_free_virtual_user_item_interactions/">VI-MMRec: Similarity-Aware Training Cost-free Virtual User-Item Interactions for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态相似度构造虚拟交互矩阵并以统计权重融合，零训练开销即插即用地缓解多模态推荐数据稀疏，Full-Ranking 下 Recall@10 最高提升 22.25%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;对抗训练&quot;,&quot;Sharpness-aware Minimization&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/adversarial_collaborative_filtering_for_free/">Adversarial Collaborative Filtering for Free</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SharpCF 以轨迹损失隐式替代对抗训练，免二次反传；全库排序下均超 BPR 18.1%、APR 6.82%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;解耦表征&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/augmented_negative_sampling_for_collaborative_filtering/">Augmented Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>解耦难/易因子后受控扰动易因子合成负样本，并以增强增益采样，5 数据集全库排序显著超越 DNS/MixGCF/DENS。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;等变对比学习&quot;,&quot;替换项检测&quot;,&quot;数据增广&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/equivariant_contrastive_learning_for_sequential_recommendation/">Equivariant Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用生成器-条件判别器实现替换项检测损失，使序列编码器对侵入式增广等变、对温和增广不变；全库排序下 ML-1m/Yelp Recall@10 较 SASRec 提升 19.5%/10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;物品相关性加权损失&quot;,&quot;多未来物品评估协议&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/integrating_item_relevance_in_training_loss_for_sequential_recommender_syst/">Integrating Item Relevance in Training Loss for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多未来物品评估协议与相关性加权损失替代单目标假设，SASRec 在 4 个数据集上 NDCG@10 提升约 1.2%–1.63%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负反馈建模&quot;,&quot;反事实模拟&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/learning_from_negative_user_feedback_and_measuring_responsiveness_for_seque/">Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 not-to-recommend 损失将显式不喜欢与隐式跳过纳入序列召回训练，并用反事实模拟量化响应性，线上不喜欢率降 2.44%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为序列推荐&quot;,&quot;跨关系对比学习&quot;,&quot;动态记忆网络&quot;]"><strong><a href="../../venues/RecSys/2023/self_supervised_denoising/multi_relational_contrastive_learning_for_recommendation/">Multi-Relational Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多行为序列推荐框架 RCL，用关系感知图编码器 + 跨关系记忆网络 + 行为级 InfoNCE，全库 99 负采样下 HR@10 最高提升 24.4%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;平均池化嵌入&quot;,&quot;序统计量闭式分析&quot;,&quot;表征自洽性&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/on_the_consistency_of_average_embeddings_for_item_recommendation/">On the Consistency of Average Embeddings for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 Consistency_k 期望精度指标与正态近似闭式解，证明理想 i.i.d. 嵌入下平均向量自洽，而 Deezer 真实嵌入的 Consistency 不足 14%/6%/2%，揭示工业平均池化表征的系统性失配。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与工业架构&quot;,&quot;端到端评测基准&quot;,&quot;Shilling Attack&quot;,&quot;攻击者知识约束&quot;,&quot;可复现性&quot;]"><strong><a href="../../venues/RecSys/2023/debias_fairness_industrial/recad_towards_a_unified_library_for_recommender_attack_and_defense/">RecAD: Towards A Unified Library for Recommender Attack and Defense</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一推荐攻击-防御基准库，标准化数据/模型/工作流三模块与双轨评估协议，揭示攻击者知识比例与检测失效现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣学习&quot;,&quot;兴趣感知难负采样&quot;,&quot;路由正则化&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/rethinking_multi_interest_learning_for_candidate_matching_in_recommender_sy/">Rethinking Multi-Interest Learning for Candidate Matching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示多兴趣召回中均匀采样负例失效与路由坍缩两大训练缺陷，用兴趣感知难负采样与路由方差正则修复，Recall@20 相对提升最高 19.7%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;生成式端到端&quot;,&quot;对比量化&quot;,&quot;语义 Tokenization&quot;,&quot;RQ-VAE&quot;]"><strong><a href="../../venues/RecSys/2024/generative_rec/cost_contrastive_quantization_based_semantic_tokenization_for_generative_re/">CoST: Contrastive Quantization based Semantic Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用批内 InfoNCE 替代 RQ-VAE 逐物品重建损失来学习语义 ID，MIND 全库排序 Recall@5 相对提升 43.3%。</p>
</dd>
<dt data-paper-tags="[&quot;Knowledge-Based Systems (KBS)&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;多面向解耦表征&quot;,&quot;门控融合&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/modeling_multi_aspect_preferences_and_intents_for_multi_behavioral_sequenti/">Modeling Multi-aspect Preferences and Intents for Multi-behavioral Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>多面向投影+行为增强LSTM+精炼注意力，联合建模偏好/意图并按面向门控融合，采样排序下较最强基线显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;注意力权重细化&quot;,&quot;高阶依赖建模&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/pay_attention_to_attention_for_sequential_recommendation/">Pay Attention to Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AWRSR，把自注意力权重矩阵本身当作表征再做一次注意力细化，以极低开销捕获高阶 item 转移；在 Beauty/ML-1M/Anime 上对 SASRec 与 STOSA 普遍带来 2%–14% 相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;隐式反馈负采样&quot;,&quot;中性类集合级排序&quot;,&quot;半监督伪标签&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/unlocking_the_hidden_treasures_enhancing_recommendations_with_unlabeled_dat/">Unlocking the Hidden Treasures: Enhancing Recommendations with Unlabeled Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出正-中性-负 (PNN) 三段式集合级排序范式，配合用户感知注意力半监督分类与钳制机制质心排序损失，在 4 个数据集上使 MF/GNN/对比学习骨干一致提升 3%–30%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;实体表征解耦&quot;,&quot;对比自监督&quot;,&quot;上游-下游范式&quot;]"><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;邻接层对比视图&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;统一损失&quot;,&quot;免增广训练&quot;]"><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;分布级对比监督&quot;,&quot;软量化语义对齐&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/distribution_level_contrastive_supervision_for_generative_recommendation/">Distribution-Level Contrastive Supervision for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分词器多层码本概率分布作软语义目标，以负对称KL+BPR对比监督对齐推荐器表征，即插即用提升多种生成式骨干。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;方向性扩散噪声&quot;,&quot;线性图 Transformer&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/a_directional_diffusion_graph_transformer_for_recommendation/">A Directional Diffusion Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以各向异性方向高斯噪声替代等方向噪声做嵌入级扩散，并用条件线性 Transformer 反向去噪，全库排序 Recall@20 相对 SOTA 提升约 3–4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习&quot;,&quot;嵌入掩码&quot;,&quot;CVR预估&quot;]"><strong><a href="../../venues/SIGIR/2023/ctr_feature_interaction/contrastive_learning_for_conversion_rate_prediction/">Contrastive Learning for Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用嵌入掩码替代特征掩码构造对比视图，配合假负样本剔除与转化正样本纳入，缓解CVR稀疏，工业/公开集AUC+0.0079/+0.0066</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;状态增广&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2023/rl_bandit_rec/contrastive_state_augmentations_for_reinforcement_learning_based_recommende/">Contrastive State Augmentations for Reinforcement Learning-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用四种局部状态增广平滑离线 RL 的 Q 值面，并对增广视图施加对比损失，在三个数据集与模拟在线环境中稳定提升序列推荐收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;意图解缠&quot;,&quot;图对比学习&quot;,&quot;自适应增广&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/disentangled_contrastive_collaborative_filtering/">Disentangled Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 K 个全局意图原型解缠用户/物品表征，并用解缠相似度驱动参数化图掩码做自适应对比增广，全库排序下 Recall@20 相对最优基线提升 2.8%–6.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;集成学习&quot;,&quot;对比知识蒸馏&quot;,&quot;掩码物品预测&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/ensemble_modeling_with_contrastive_knowledge_distillation_for_sequential_re/">Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 N 个并行双向 Transformer 做序列编码器集成，并以 ICL/CCL 表征对比蒸馏 + logits 协同 KL 蒸馏打通网络间知识迁移；5-core + Leave-One-Out + 全库排序下 NDCG@10 相对提升 20.7%~33.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;频域注意力&quot;,&quot;自相关周期建模&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/frequency_enhanced_hybrid_attention_network_for_sequential_recommendation/">Frequency Enhanced Hybrid Attention Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户行为序列迁移到频域，用频率斜坡采样改造自注意力并叠加自相关周期注意力，Leave-One-Out 全库排序下在四个基准上实现 3.6%–10.9% 的稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回（全库排序）&quot;,&quot;变分图重构&quot;,&quot;聚类感知对比学习&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/generative_contrastive_graph_learning_for_recommendation/">Generative-Contrastive Graph Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以变分图重构采样替代数据增广生成对比视图，联合节点/簇级对比，在三个数据集 Full-Ranking 下 NDCG@20 较 LightGCN 提升 8.61%–28.17%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图结构重连 (Graph Rewiring)&quot;,&quot;双边 Top-K 去噪与增广&quot;,&quot;用户-用户/物品-物品相关性&quot;]"><strong><a href="../../venues/SIGIR/2023/graph_rec/graph_collaborative_signals_denoising_and_augmentation_for_recommendation/">Graph Collaborative Signals Denoising and Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>预训练后双边 Top-K 重连二部图并补入用户-用户/物品-物品对称相关块，Amazon 全库排序 NDCG@20 相对提升 6.9%–13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图掩码自编码器&quot;,&quot;自适应数据增广&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/graph_masked_autoencoder_for_sequential_recommendation/">Graph Masked Autoencoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MAERec 用以任务自适应路径掩码驱动的图掩码自编码器替代手工对比增广，在留一法全库排序下 HR@20 相对最强基线提升最高 11.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;Graph Transformer&quot;,&quot;Masked Autoencoding&quot;,&quot;Rationale Discovery&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/graph_transformer_for_recommendation/">Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GFormer 用拓扑感知图 Transformer 参数化发现协同 rationale 子图，并以 rationale-aware 掩码自编码替代随机对比增广；全库排序下 Recall@20 较 SGL 提升 5.2%–7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为对齐&quot;,&quot;KL散度去噪&quot;,&quot;隐变量推断&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/improving_implicit_feedback_based_recommendation_through_multi_behavior_ali/">Improving Implicit Feedback-Based Recommendation through Multi-Behavior Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户真实偏好设为隐变量，用 KL 散度对齐点击/购买两条行为链并交替去噪，三数据集 Recall@20 相对最优基线提升 0.5%–6.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与排序表征学习&quot;,&quot;知识图谱剪枝&quot;,&quot;对比去噪&quot;,&quot;伯努利掩码&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/knowledge_refined_denoising_network_for_robust_recommendation/">Knowledge-refined Denoising Network for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KRDN 用伯努利掩码剪枝 KG 任务无关三元组，并以协同/知识信号分歧对比式剔除噪声交互；全库排序下 NDCG@20 较 SOTA 提升 4.6%–12.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;可学习模型增强&quot;,&quot;元学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/meta_optimized_contrastive_learning_for_sequential_recommendation/">Meta-optimized Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习模型增强器与元学习优化对比学习，在 LOO+Full-Ranking 下 HR/NDCG 相对最强基线提升 2.5–8.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为建模&quot;,&quot;梯度冲突矫正&quot;]"><strong><a href="../../venues/SIGIR/2023/self_supervised_denoising/multi_behavior_self_supervised_learning_for_recommendation/">Multi-behavior Self-supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为感知GNN+双层自判别对比+梯度混合矫正，Leave-One-Out全库排序下Recall@10平均提升约19%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排/训练阶段负采样&quot;,&quot;难负例挖掘&quot;,&quot;加权物品转移图&quot;,&quot;课程学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/neighborhood_based_hard_negative_mining_for_sequential_recommendation/">Neighborhood-based Hard Negative Mining for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在全局加权物品转移图上以 Jaccard 邻域重叠度构造难负采样分布，并用课程学习调度假阴性阈值；在三个 Amazon 数据集 Leave-One-Out + 1000 候选采样排序下，相对最强负采样基线 NDCG@20 稳定提升 1.0–2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/排序&quot;,&quot;不确定性伪交互生成&quot;,&quot;教师-学生一致性学习&quot;]"><strong><a href="../../venues/SIGIR/2023/cold_start_rec/uncertainty_aware_consistency_learning_for_cold_start_item_recommendation/">Uncertainty-aware Consistency Learning for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>UCC 用不确定性伪交互生成与教师-学生一致性学习对齐冷/热物品分布，10-core/7:1:2、K=20 下 Recall 最高提升 27.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;兴趣解耦&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/when_search_meets_recommendation_learning_disentangled_search_representatio/">When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SESRec 用查询-物品对齐 + 三元组对比自监督把搜索/推荐行为解耦为相似与相异兴趣，在快手与 Amazon 上 NDCG@10 相对提升 7.05%/2.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/排序&quot;,&quot;负迁移过滤&quot;,&quot;用户表征变换&quot;,&quot;监督对比学习&quot;]"><strong><a href="../../venues/SIGIR/2024/cross_domain_multi_scenario/aiming_at_the_target_filter_collaborative_information_for_cross_domain_reco/">Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以目标域用户相似度为显式约束，结合用户变换层与监督对比损失过滤源域协同信息，缓解跨域负迁移。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;物品级偏好建模&quot;,&quot;图卷积预训练&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/behavior_contextualized_item_preference_modeling_for_multi_behavior_recomme/">Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 BCIPM：用「行为上下文化物品偏好网络」只从目标行为聚合物品级偏好做推荐，辅助行为仅参与参数训练以规避噪声；四个数据集 Leave-One-Out + Full-Ranking 下 HR@10 相对提升 12.15%–26.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户行为模式&quot;,&quot;自监督去噪&quot;,&quot;Target Pattern Attention&quot;]"><strong><a href="../../venues/SIGIR/2024/ctr_feature_interaction/deep_pattern_network_for_click_through_rate_prediction/">Deep Pattern Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 Target Attention 从 item 级拓宽到 pattern 级，用 target-aware 检索 + 自监督去噪精炼 + Target Pattern Attention 挖掘用户行为模式，三数据集 AUC 相对 CAN 提升 1.53%–2.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散模型去噪&quot;,&quot;隐式反馈噪声&quot;,&quot;模型无关插件&quot;]"><strong><a href="../../venues/SIGIR/2024/self_supervised_denoising/denoising_diffusion_recommender_model/">Denoising Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用扩散前向-逆向过程对任意后端模型的用户/物品嵌入做多步去噪，以协同信息为引导、以历史偏好均值为推理起点，三数据集全排序 R@10 最高提升 25.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果推断解耦&quot;,&quot;会话推荐&quot;,&quot;多模态对齐&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/disentangling_id_and_modality_effects_for_session_based_recommendation/">Disentangling ID and Modality Effects for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DIMO 物品级把共现矩阵显式注入 ID 表征并将图文统一对齐到文本空间，会话级用代理机制与反事实推断无监督解耦 ID/模态成因，Cellphones Prec@20 较最优基线提升 55.12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;长尾罕见病公平性&quot;,&quot;自监督预训练&quot;]"><strong><a href="../../venues/SIGIR/2024/debias_fairness_industrial/leave_no_patient_behind_enhancing_medication_recommendation_for_rare_diseas/">Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RAREMed 用统一疾病-手术序列 Transformer + 序列匹配/自重建双自监督预训练，显著抬升罕见病患者的用药推荐精度并压缩群体间公平性方差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;对比学习序列增广&quot;,&quot;多兴趣表征&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/modeling_user_fatigue_for_sequential_recommendation/">Modeling User Fatigue for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以兴趣感知相似度矩阵刻画目标-历史细粒度重复度，经 Cross 网络与疲劳门控 GRU 注入长短期兴趣，并用序列增广构造疲劳对比监督；Taobao AUC 提升 0.026，线上 EVTR 与类目多样性同升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化自增强学习&quot;,&quot;短期协作图&quot;,&quot;序列去噪&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/selfgnn_self_supervised_graph_neural_networks_for_sequential_recommendation/">SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按时间切片构建短期协作图并用个性化自增强学习以长期兴趣矫正短期噪声，1:999 采样排序下四个数据集全面超越 17 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;序列数据增强&quot;,&quot;免训练推理&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态解耦&quot;,&quot;缺失模态生成&quot;,&quot;互信息对比学习&quot;]"><strong><a href="../../venues/SIGIR/2025/multimodal_rec/disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;意图解耦表征&quot;,&quot;跨模态对齐&quot;,&quot;动量蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/intent_representation_learning_with_large_language_model_for_recommendation/">Intent Representation Learning with Large Language Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 思维链抽取文本意图，与交互意图双塔对齐+动量蒸馏匹配，作为模型无关插件在三个数据集上稳定提升 Recall/NDCG。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图原型聚类&quot;,&quot;条件扩散增强&quot;,&quot;意图感知对比学习&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图不变学习&quot;,&quot;社交去噪&quot;,&quot;对抗环境生成&quot;]"><strong><a href="../../venues/SIGIR/2025/social_group_rec/invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;SENet特征重加权&quot;]"><strong><a href="../../venues/SIGIR/2025/self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自适应数据增强&quot;,&quot;Sinkhorn 可微投影&quot;]"><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;可微语义ID&quot;,&quot;Gumbel-Softmax 探索&quot;,&quot;码本坍缩&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，Leave-One-Out 全库排序下 R@10 相对 TIGER 提升约 12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;知识蒸馏&quot;,&quot;难负例挖掘&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;批采样&quot;,&quot;不可能负样本&quot;,&quot;用户表征学习&quot;]"><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/don_t_contrast_the_impossible_region_constrained_batching_for_contrastive_u/">Don&#x27;t Contrast the Impossible: Region-Constrained Batching for Contrastive User Modeling on a Local Community Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别地理约束下 in-batch 负样本中的“不可能负样本”问题，提出区域同质批采样 RCBS，仅改批构造即显著提升用户表征与线上指标。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;模块化集成&quot;,&quot;引导式互学习&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;基础模型特征增强&quot;,&quot;跨视图对比学习&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/from_raw_features_to_effective_embeddings_a_three_stage_approach_for_multim/">From Raw Features to Effective Embeddings: A Three-Stage Approach for Multimodal Recipe Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 VLM/LLM 摘要 + LightGCN 传播 + 跨视图对比学习三阶段精炼食谱多模态特征，Full-Ranking 下 Recall@10 提升 7–15%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;物品关系推断&quot;,&quot;自监督去噪&quot;,&quot;元路径编码&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/multi_modal_relational_item_representation_learning_for_inferring_substitut/">Multi-modal Relational Item Representation Learning for Inferring Substitutable and Complementary Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>融合多模态基础模型与元路径行为编码，以自监督对比去噪与 LLM 校验弱监督推断可替代/互补关系，M@10 平均提升 26.1%/39.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;问答推荐&quot;,&quot;多项式图滤波&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2026/graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将问答对解耦为用户-问题/用户-答案双协同图与问题-答案/问题-问题双语义图，融合为全局异质图后用 Jacobi 多项式图滤波器分离高低频信号，配合噪声对比学习，在知乎与商业数据集 Full-Ranking 下 Recall@10 提升 5.16%–5.34%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;精排&quot;,&quot;空间约束对比学习&quot;,&quot;长尾表征增强&quot;]"><strong><a href="../../venues/SIGIR/2026/poi_spatiotemporal_rec/rest_a_plug_and_play_spatially_constrained_representation_enhancement_frame/">ReST: A Plug-and-Play Spatially-Constrained Representation Enhancement Framework for Local-Life Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以物品为中心，用属性语义预热+空间约束难负采样对比学习增强长尾物品ID表征，离线AUC最高+5.90%，线上GMV+2.804%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;α-entmax 稀疏对比学习&quot;,&quot;内容空间知识蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2026/cold_start_rec/sparse_contrastive_learning_for_content_based_cold_item_recommendation/">Sparse Contrastive Learning for Content-Based Cold Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>抛弃 CF 对齐、纯内容空间 item-item 相似度建模，用 α-entmax 稀疏化 sampled softmax 并支持内容空间蒸馏，冷物品排序与曝光公平性双赢。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;下一购物篮推荐&quot;,&quot;意图解耦&quot;,&quot;Hawkes 过程&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TIDE 用 Hawkes 增强傅里叶时间编码刻画物品级非单调补货周期，并以双专家解耦习惯与探索意图，Leave-One-Out 全库排序下 R@10 最高提升 18.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;解耦表征学习&quot;,&quot;会话图神经网络&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/disentangled_graph_neural_networks_for_session_based_recommendation/">Disentangled Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将物品嵌入解耦为 K 个因子并用因子级相似度 GGNN 传播、距离相关解耦，按因子注意力聚合会话表征，三数据集 P@20 最高 +16.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对抗变分推断&quot;,&quot;多峰后验嵌入&quot;]"><strong><a href="../../venues/TKDE/2023/self_supervised_denoising/dual_adversarial_variational_embedding_for_robust_recommendation/">Dual Adversarial Variational Embedding for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双侧 VAE + 对抗式隐式后验正则实现个性化降噪与多峰偏好建模，Leave-One-Out + 1:99 采样排序下 NDCG@5 提升约 1.3%–8.3%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;意图解耦&quot;,&quot;特征自监督&quot;]"><strong><a href="../../venues/TKDE/2023/cold_start_rec/intent_disentanglement_and_feature_self_supervision_for_novel_recommendatio/">Intent Disentanglement and Feature Self-supervision for Novel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户意图解耦为从众（热度）与偏好两因子并配以协同-内容特征自监督，统一优化精度/新颖性/覆盖度，F1@5 相对提升 32%+</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;图结构学习&quot;,&quot;对比多模态融合&quot;]"><strong><a href="../../venues/TKDE/2023/multimodal_rec/latent_structure_mining_with_contrastive_modality_fusion_for_multimedia_rec/">Latent Structure Mining with Contrastive Modality Fusion for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TKDE 2023 多模态推荐：模态感知图学习+对比融合，Amazon 全排序 R@20 提升 9.2%–24.1%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Actor-Critic&quot;,&quot;临时兴趣建模&quot;,&quot;非典型交互&quot;]"><strong><a href="../../venues/TKDE/2023/rl_bandit_rec/learning_from_atypical_behavior_temporary_interest_aware_recommendation_bas/">Learning from Atypical Behavior: Temporary Interest Aware Recommendation Based on Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双智能体 Actor-Critic 将&quot;非典型交互&quot;从噪声重构为临时兴趣，无监督分类器教推荐器平衡长期偏好与短期兴趣，全库排序下 HR@20 提升最高 15.19%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;曝光置信度建模&quot;,&quot;随机游走信息量采样&quot;]"><strong><a href="../../venues/TKDE/2023/collaborative_filtering/samwalker_recommendation_with_informative_sampling_strategy/">SamWalker++: recommendation with informative sampling strategy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用伪社交网络图扩散核自适应分配隐式反馈置信度，并沿网络随机游走采样正比于置信度的信息量实例，稀疏数据 Pre@5 提升 67.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;贝叶斯图增广&quot;,&quot;多视图注意力&quot;,&quot;鲁棒推荐&quot;]"><strong><a href="../../venues/TKDE/2024/poi_spatiotemporal_rec/bayes_enhanced_multi_view_attention_networks_for_robust_poi_recommendation/">Bayes-enhanced Multi-view Attention Networks for Robust POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用贝叶斯后验引导的节点复制图增广加三视图注意力建模个人/语义/距离依赖，在签到缺失与噪声下 R@5 最高提升 16.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;监督对比损失&quot;,&quot;协同邻居&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/TKDE/2024/self_supervised_denoising/neighborhood_enhanced_supervised_contrastive_learning_for_collaborative_fil/">Neighborhood-Enhanced Supervised Contrastive Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 ItemKNN 最近邻与交互邻居纳入对比损失正样本，缓解假阴性；三数据集全库排序 NDCG@20 较 SGL 提升 10.09%/7.09%/35.36%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自监督预训练&quot;,&quot;数据增广&quot;,&quot;InfoNCE&quot;,&quot;SELFRec&quot;]"><strong><a href="../../venues/TKDE/2024/self_supervised_denoising/self_supervised_learning_for_recommender_systems_a_survey/">Self-Supervised Learning for Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次系统界定自监督推荐 (SSR)，提出对比/生成/预测/混合四分类与 JL/PF/IL 三训练范式，并开源 SELFRec 在 Full-Ranking 下给出受控实证结论。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/全库排序&quot;,&quot;图对比学习&quot;,&quot;噪声嵌入增广&quot;,&quot;跨层对比&quot;]"><strong><a href="../../venues/TKDE/2024/self_supervised_denoising/xsimgcl_towards_extremely_simple_graph_contrastive_learning_for_recommendat/">XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用噪声嵌入增广与单编码器跨层对比替代图增广，四数据集全库排序显著提升且训练近 LightGCN 轻量。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;测试时训练&quot;,&quot;OOD 推荐&quot;,&quot;自蒸馏聚类&quot;]"><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/dual_test_time_training_for_out_of_distribution_recommender_system/">Dual Test-time Training for Out-of-distribution Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个将测试时训练引入 OOD 推荐，用自蒸馏+对比双任务在测试阶段适配用户/物品特征漂移，全库排序下 R@50 最高提升 11.56%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;关系链对比学习&quot;]"><strong><a href="../../venues/TKDE/2025/graph_rec/dual_channel_multiplex_graph_neural_networks_for_recommendation/">Dual-Channel Multiplex Graph Neural Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双通道多路图网络显式建模行为模式、隐式挖掘关系链依赖，三数据集全指标平均提升 10%+</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;图感知对比学习&quot;,&quot;多跳提示构造&quot;]"><strong><a href="../../venues/TKDE/2025/llm_rec/enhancing_collaborative_semantics_of_language_model_driven_recommendations_/">Enhancing Collaborative Semantics of Language Model-Driven Recommendations via Graph-Aware Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多跳提示+图感知对比学习把 GNN 聚合意图注入 LLM，在三个 Amazon/Yelp 数据集 Full-Ranking 下 HR@10 最高提升约 15%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;对抗数据增强&quot;,&quot;群体公平性&quot;]"><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/fairdgcl_fairness_aware_recommendation_with_dynamic_graph_contrastive_learn/">FairDgcl: Fairness-aware Recommendation with Dynamic Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用对抗式视图生成器+判别器动态生成公平增强视图，在四个数据集上同时提升准确率与群体公平性。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向时序数据增强&quot;,&quot;伪先验物品生成&quot;,&quot;知识增强微调&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/improving_sequential_recommendations_via_bidirectional_temporal_data_augmen/">Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>BARec 用双向时序增强预训练生成保偏好伪先验物品，并以 KL 知识增强微调对齐原/增广序列分布，全库排序下 Beauty Recall@5 提升 16.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;意图解耦&quot;,&quot;对比学习&quot;,&quot;高阶关系&quot;]"><strong><a href="../../venues/TKDE/2025/collaborative_filtering/intent_propagation_contrastive_collaborative_filtering/">Intent Propagation Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双螺旋消息传播融合高阶关系与意图解耦，对比学习对齐结构/意图视图，Full-Ranking 下 NDCG 平均提升 9.36%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;元路径意图建模&quot;,&quot;信息瓶颈去噪&quot;]"><strong><a href="../../venues/TKDE/2025/graph_rec/intent_guided_heterogeneous_graph_contrastive_learning_for_recommendation/">Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用元路径意图构建双对比学习视图，并以信息瓶颈掩码自编码器去噪，六数据集全库排序 Recall@20 最高提升 12.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;门控专家网络&quot;,&quot;潜在因子解耦&quot;]"><strong><a href="../../venues/TKDE/2025/collaborative_filtering/latent_factor_modeling_with_expert_network_for_multi_behavior_recommendatio/">Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用门控专家网络显式建模共享潜在因子，配合对比学习保证因子独立与一致，在三个多行为数据集 Full-Ranking 下 HR@10 最高提升 3.95%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;概率嵌入&quot;,&quot;逻辑合取算子&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/pattern_wise_transparent_sequential_recommendation/">Pattern-wise Transparent Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多尺度滑窗切出点级/联合级模式，以概率嵌入+逻辑合取建模模式，KL散度+距离权重实现可解释且超越黑盒的序列推荐。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;分数匹配扩散模型&quot;,&quot;社交去噪&quot;,&quot;跨域对比学习&quot;]"><strong><a href="../../venues/TKDE/2025/social_group_rec/score_based_generative_diffusion_models_for_social_recommendations/">Score-based Generative Diffusion Models for Social Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 SDE 扩散模型在隐空间直接生成协同一致的社交表征，配合课程训练与跨域对比学习，Full-Ranking 下 Recall@10 相对提升 1.85%–3.07%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;托攻击&quot;,&quot;鲁棒推荐&quot;,&quot;对抗防御&quot;]"><strong><a href="../../venues/TKDE/2025/debias_fairness_industrial/towards_robust_recommendation_a_review_and_an_adversarial_robustness_evalua/">Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统综述推荐系统对抗/非对抗鲁棒性，提出 ShillingREC 评测库统一攻击与防御模型评估。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣聚类&quot;,&quot;稳定性-可塑性解耦&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;多级图神经网络&quot;,&quot;超图&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TMM/2023/sequential_rec/contrastive_multi_level_graph_neural_networks_for_session_based_recommendat/">Contrastive Multi-Level Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 L/G/H 三级图与跨层级 InfoNCE，将高阶超图监督注入成对会话表示，在 Tmall/Diginetica/Nowplaying 全库排序上多数指标领先。</p>
</dd>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;引导扩散模型&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TMM/2025/multimodal_rec/knowledge_aware_diffusion_enhanced_multimedia_recommendation/">Knowledge-aware Diffusion-Enhanced Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用重启随机游走 Jaccard 注意力矩阵替代可学习注意力，并以用户嵌入引导扩散模型重建低噪知识图谱构建对比视图，三数据集 Recall@20 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;TOMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;递归跨模态注意力&quot;,&quot;双图对比学习&quot;]"><strong><a href="../../venues/TMM/2026/multimodal_rec/cross_modal_attention_network_with_dual_graph_learning_in_multimodal_recomm/">Cross-Modal Attention Network with Dual Graph Learning in Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>递归跨模态注意力 RCA 迭代对齐图文特征，配合用户-物品异构图与物品-物品语义图双图 + 对比学习，四数据集平均提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/表征预训练&quot;,&quot;自监督图预训练&quot;,&quot;图对比学习&quot;]"><strong><a href="../../venues/TOIS/2023/cold_start_rec/a_multi_strategy_based_pre_training_method_for_cold_start_recommendation/">A Multi-Strategy based Pre-Training Method for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MPT 在 PT-GNN 上并联 Transformer 路径编码器与对比学习借口任务，元学习模拟冷启动，外延 Recall@20 相对提升 12%–20%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;异质信息网络&quot;,&quot;课程式预训练&quot;,&quot;子图对比学习&quot;]"><strong><a href="../../venues/TOIS/2023/graph_rec/curriculum_pre_training_heterogeneous_subgraph_transformer_for_top_n_recomm/">Curriculum Pre-Training Heterogeneous Subgraph Transformer for Top-N Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>交互特定异质子图 + 复合槽位嵌入 Transformer + 由易到难课程式预训练，Leave-One-Out/1000 负采样下 HR@20 较最强基线提升 9.9%–25.0%</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端侧个性化排序&quot;,&quot;去中心化协同学习&quot;,&quot;自监督POI表征&quot;,&quot;差分隐私&quot;]"><strong><a href="../../venues/TOIS/2023/poi_spatiotemporal_rec/decentralized_collaborative_learning_framework_for_next_poi_recommendation/">Decentralized Collaborative Learning Framework for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>端侧去中心化 DCLR：地理/类别自监督预训练 + 双邻居 MIM 融合，缓解稀疏并逼近集中式 STAN。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理/模型压缩&quot;,&quot;组合编码&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/efficient_on_device_session_based_recommendation/">Efficient On-Device Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以组合编码压缩物品嵌入并用双向自监督蒸馏保容，30×压缩下实现6–8×端侧推理加速且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;偏好编辑自监督&quot;,&quot;多偏好表征&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/improving_transformer_based_sequential_recommenders_through_preference_edit/">Improving Transformer-based Sequential Recommenders through Preference Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 K 个特殊 token 表征多偏好，并以“偏好分离-重组”自监督构造跨序列信号，Beauty/ML-100k 上 Recall@20 提升显著</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回/精排&quot;,&quot;因果推断&quot;,&quot;反事实数据增广&quot;,&quot;后门调整与 IPW/DR&quot;]"><strong><a href="../../venues/TOIS/2024/debias_fairness_industrial/causal_inference_in_recommender_systems_a_survey_and_future_directions/">Causal Inference in Recommender Systems: A Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以数据偏差、数据缺失/噪声、超越精度三类病理为主轴，系统梳理 40 篇工作中后门调整、IPW/DR、反事实增广与因果发现四类工具的用法、代价与开放问题</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;表征学习/召回&quot;,&quot;对比学习&quot;,&quot;数据增强&quot;]"><strong><a href="../../venues/TOIS/2024/self_supervised_denoising/contrastive_self_supervised_learning_in_recommender_systems_a_survey/">Contrastive Self-supervised Learning in Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>综述提出统一框架与三维分类，系统梳理视图生成、对比任务与目标，并给出组件选择指南与开放问题。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比与交互去噪&quot;,&quot;召回&quot;,&quot;多视图对比学习&quot;,&quot;捆绑包推荐&quot;,&quot;图卷积网络&quot;]"><strong><a href="../../venues/TOIS/2024/self_supervised_denoising/multicbr_multi_view_contrastive_learning_for_bundle_recommendation/">MultiCBR: Multi-view Contrastive Learning for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出三视图(UB/UI/BI)表征与“先融合后对比”框架，BI 双向图传播破解捆绑包稀疏，iFashion 上 Recall@20 相对提升 32.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;多关系会话图&quot;,&quot;协同过滤图&quot;]"><strong><a href="../../venues/TOIS/2024/sequential_rec/spatio_temporal_contrastive_learning_enhanced_gnns_for_session_based_recomm/">Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RESTC 用空间(MGAT)与时间(SESTrans)双视图对比学习对齐会话表征，并以全局协同过滤图补足短会话稀疏性，六数据集全库排序 HR@20 最高 +27.08%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;偏好解耦&quot;,&quot;一致性约束&quot;,&quot;双目标跨域推荐&quot;]"><strong><a href="../../venues/TOIS/2025/cross_domain_multi_scenario/coherence_guided_preference_disentanglement_for_cross_domain_recommendation/">Coherence-guided Preference Disentanglement for Cross-domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用全局-局部图 + 一致性约束抽取跨域共享物品属性，并以流行度加权 Softmax 将共享用户偏好解耦为兴趣/从众，六组双目标 CDR 任务 HR@10 最高提升约 37%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;软标签&quot;,&quot;知识蒸馏&quot;,&quot;去噪鲁棒性&quot;]"><strong><a href="../../venues/TOIS/2025/sequential_rec/learning_robust_sequential_recommenders_through_confident_soft_labels/">Learning Robust Sequential Recommenders through Confident Soft Labels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模型级/数据级/训练级三种教师模块生成高置信软标签，与 one-hot 联合训练学生序列推荐器，全库排序下 Recall@10 最高提升约 11%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;噪声视图&quot;,&quot;对称损失&quot;]"><strong><a href="../../venues/TOIS/2025/self_supervised_denoising/symmetric_graph_contrastive_learning_against_noisy_views_for_recommendation/">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>定义图增广产生的噪声视图，用对称指数对比损失替代 InfoNCE，使 GCL 对噪声视图鲁棒，全库排序下 NDCG@20 最高提升 12.25%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;不变风险最小化&quot;,&quot;变分自编码器&quot;]"><strong><a href="../../venues/TOIS/2025/multi_task_reranking/user_invariant_preference_learning_for_multi_behavior_recommendation/">User Invariant Preference Learning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 IRM 范式 + VAE 从多行为交互中抽取用户不变偏好以抑制辅助行为噪声，Leave-One-Out + Full-Ranking 下 HR@10 最高提升 3.98%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图对比学习&quot;,&quot;相似就诊检索&quot;]"><strong><a href="../../venues/TOIS/2026/graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;输出扰动&quot;,&quot;无负采样&quot;]"><strong><a href="../../venues/TORS/2023/self_supervised_denoising/selfcf_a_simple_framework_for_self_supervised_collaborative_filtering/">SelfCF: A Simple Framework for Self-supervised Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TORS 2023 自监督去噪：输出扰动替代负采样，四数据集全库排序下较 BUIR 平均提升 17.79%，训练快 2–4×。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;解耦表征学习&quot;,&quot;元网络&quot;]"><strong><a href="../../venues/TORS/2024/graph_rec/disentangled_cascaded_graph_convolution_networks_for_multi_behavior_recomme/">Disentangled Cascaded Graph Convolution Networks for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>解耦因子块+元网络个性化迁移+行为级注意力，在Beibei/Tmall全库排序平均提升7.07%/9.00%</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入坍缩&quot;,&quot;谱均衡&quot;,&quot;有效秩&quot;]"><strong><a href="../../venues/TORS/2025/collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;冷启动&quot;,&quot;综述&quot;,&quot;数据稀缺&quot;,&quot;知识迁移&quot;,&quot;自监督学习&quot;]"><strong><a href="../../venues/TORS/2025/cold_start_rec/data_scarcity_in_recommendation_systems_a_survey/">Data Scarcity in Recommendation Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理数据稀缺下五类缓解范式（增广/自监督/迁移/广度学习/知识图谱），并给出挑战与未来方向。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;对比学习&quot;,&quot;高阶关联&quot;]"><strong><a href="../../venues/TORS/2025/collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TWEB&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构图神经网络&quot;,&quot;多行为会话建模&quot;]"><strong><a href="../../venues/TWEB/2024/sequential_rec/heterogeneous_information_crossing_on_graphs_for_session_based_recommender_/">Heterogeneous Information Crossing on Graphs for Session-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on the Web">ACM TWEB</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>会话多行为异构图 + 跨行为类型边 + 双层注意力交叉，全库排序下 HR@20 相对最优基线提升约 3.5%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习&quot;,&quot;特征表征学习&quot;,&quot;长尾低频特征&quot;]"><strong><a href="../../venues/WSDM/2023/ctr_feature_interaction/cl4ctr_a_contrastive_learning_framework_for_ctr_prediction/">CL4CTR: A Contrastive Learning Framework for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用随机/特征/维度三种嵌入扰动构造正对，叠加同域对齐与跨域均匀性三项自监督信号正则化嵌入层，模型无关且零推理开销，四数据集 AUC/LogLoss 全面超越 DCN-V2。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异构图对比学习&quot;,&quot;元网络&quot;]"><strong><a href="../../venues/WSDM/2023/graph_rec/heterogeneous_graph_contrastive_learning_for_recommendation/">Heterogeneous Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出元网络个性化异构图对比学习，在 Ciao/Epinions/Yelp 的 1:99 采样排序上 HR@10 较最强基线提升 2.0%–3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;知识图谱对比学习&quot;,&quot;InfoNCE&quot;]"><strong><a href="../../venues/WSDM/2023/self_supervised_denoising/knowledge_enhancement_for_contrastive_multi_behavior_recommendation/">Knowledge Enhancement for Contrastive Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>KMCLR 以多行为 InfoNCE + 知识图谱双视图对比 + 一致性调制子图采样去噪，三数据集 HR@10 相对最佳基线提升 1.4%–9.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱扩散&quot;,&quot;对比学习去噪&quot;]"><strong><a href="../../venues/WSDM/2024/graph_rec/diffkg_knowledge_graph_diffusion_model_for_recommendation/">DiffKG: Knowledge Graph Diffusion Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KG 扩散生成任务相关子图，并以协同卷积与对比增广去噪，在三个数据集 Full-Ranking 下 Recall@20 最高提升 43.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图对比学习&quot;,&quot;跨子序列监督&quot;]"><strong><a href="../../venues/WSDM/2024/sequential_rec/intent_contrastive_learning_with_cross_subsequences_for_sequential_recommen/">Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用同一目标物品的跨序列子序列构造粗/细粒度意图对比监督，全库排序 NDCG@20 最高提升 44.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排&quot;,&quot;逆对偶损失&quot;,&quot;逆梯度元学习&quot;]"><strong><a href="../../venues/WSDM/2024/multimodal_rec/inverse_learning_with_extremely_sparse_feedback_for_recommendation/">Inverse Learning with Extremely Sparse Feedback for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用逆对偶损失对无标签样本做双侧自动标注，再用元学习逆梯度仲裁更新方向，Micro Video/ML1M 上 AUC 最高提升 9.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱传播&quot;,&quot;多样性-精度权衡&quot;,&quot;对齐与均匀性&quot;]"><strong><a href="../../venues/WSDM/2024/graph_rec/knowledge_graph_context_enhanced_diversified_recommendation/">Knowledge Graph Context-Enhanced Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KG-Diverse，以实体/关系覆盖度为细粒度多样性度量，用关系感知 KG 传播、多样性感知用户聚合与条件对齐-均匀性正则，在三个数据集上大幅提升覆盖度而精度损失有限。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;模体(Motif)采样与超图编码&quot;,&quot;Prompt Tuning&quot;,&quot;对比预训练&quot;]"><strong><a href="../../venues/WSDM/2024/cross_domain_multi_scenario/motif_based_prompt_learning_for_universal_cross_domain_recommendation/">Motif-Based Prompt Learning for Universal Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MOP 用 butterfly/triangle/random-walk 三类模体构造跨域共享嵌入，并以「预训练+Prompt Tuning」把 SSL 与推荐统一为模体相似度学习，在四类 CDR 场景上取得稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序评测&quot;,&quot;模块化基准平台&quot;,&quot;自监督数据增广&quot;]"><strong><a href="../../venues/WSDM/2024/self_supervised_denoising/sslrec_a_self_supervised_learning_framework_for_recommendation/">SSLRec: A Self-Supervised Learning Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出统一模块化 SSL 推荐基准，覆盖五大场景约30个模型，并以全库排序协议复现评测。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;维度坍缩&quot;,&quot;率失真编码长度&quot;,&quot;非对比学习&quot;]"><strong><a href="../../venues/WSDM/2024/collaborative_filtering/towards_mitigating_dimensional_collapse_of_representations_in_collaborative/">Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以率失真编码长度替代 InfoNCE，消除维度坍缩，免增广免负采样，ML10M Recall@10 提升 14.21%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/预训练&quot;,&quot;任务超图&quot;,&quot;过渡注意力&quot;]"><strong><a href="../../venues/WSDM/2024/graph_rec/unified_pretraining_for_recommendation_via_task_hypergraphs/">Unified Pretraining for Recommendation via Task Hypergraphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多类预训练任务统一为任务超图超边预测，过渡注意力判别迁移至推荐，全库排序 Recall@10 最高提升 107%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;评论增广对比学习&quot;,&quot;用户-物品-评论三方对齐&quot;]"><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/a_contrastive_framework_with_user_item_and_review_alignment_for_recommendat/">A Contrastive Framework with User, Item and Review Alignment for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论从“特征”升级为“对比信号”，用评论增广+三方对齐双对比任务缓解稀疏与评论缺失，Full-Ranking 下 NDCG@5 最高提升 7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图剪枝&quot;,&quot;知识蒸馏&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/WSDM/2025/graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质超图&quot;,&quot;意图解耦&quot;,&quot;多行为对比学习&quot;]"><strong><a href="../../venues/WSDM/2025/graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Oracle 引导对齐&quot;,&quot;频域低通去噪&quot;,&quot;两阶段训练&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;套装兼容性建模&quot;,&quot;监督对比学习&quot;,&quot;自适应间隔损失&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;分数生成模型&quot;,&quot;难负采样&quot;]"><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;多模态去噪&quot;,&quot;知识蒸馏&quot;,&quot;最优传输&quot;]"><strong><a href="../../venues/WSDM/2025/multimodal_rec/teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果自注意力&quot;,&quot;注意力矩阵谱分析&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性注意力&quot;,&quot;持续学习&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;离散扩散&quot;,&quot;一致性模型&quot;]"><strong><a href="../../venues/WSDM/2026/generative_rec/continuous_time_discrete_space_diffusion_model_for_recommendation/">Continuous-time Discrete-space Diffusion Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将推荐历史交互建模为连续时间离散空间（吸收态掩码）扩散，配合流行度感知噪声调度、一致性参数化与多跳对比引导，在三个数据集 Full-Ranking 下 Recall@10 相对提升 4.2%–5.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散数据增强&quot;,&quot;难负采样&quot;]"><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;侧信息融合&quot;,&quot;频域去噪&quot;,&quot;时间跨度划分&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自监督序列校正&quot;,&quot;逆序生成插入&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/a_self_correcting_sequential_recommender/">A Self-Correcting Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>STEAM 用 item-wise 校正器（keep/delete/insert）+ 两个自监督纠错任务先修序列再推荐，三个数据集 5-core/LOO/99 负采样下 HR@5 最高 +5.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排（训练数据级预处理）&quot;,&quot;深度强化学习策略网络&quot;,&quot;实例级去噪与数据子集迁移&quot;]"><strong><a href="../../venues/WWW/2023/self_supervised_denoising/autodenoise_automatic_data_instance_denoising_for_recommendations/">AutoDenoise: Automatic Data Instance Denoising for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 DRL 策略网络逐实例做 select/deselect 去噪，配合搜索-验证两阶段交替训练，产出可迁移的干净子集，DeepFM 等骨干 AUC 稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;可学习图掩码&quot;,&quot;掩码图自编码器&quot;,&quot;Infomax 语义相关性&quot;]"><strong><a href="../../venues/WWW/2023/self_supervised_denoising/automated_self_supervised_learning_for_recommendation/">Automated Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>AutoCF 用可学习掩码 + 图自注意力掩码自编码器替代手工图增广，在全库排序下相对最优 SSL 基线提升 7.8%–33.7% Recall@20。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;自监督对比学习&quot;,&quot;免负采样训练&quot;,&quot;潜在表征 dropout&quot;]"><strong><a href="../../venues/WWW/2023/multimodal_rec/bootstrap_latent_representations_for_multi_modal_recommendation/">Bootstrap Latent Representations for Multi-modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>BM3 用隐层 dropout 自举对比视图、免负采样免辅助图，重构交互图并对齐模态特征，全库排序下 R@10 提升 3.7–20.4% 且训练快 2–9 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;特征级数据增广&quot;,&quot;人群定向&quot;]"><strong><a href="../../venues/WWW/2023/cold_start_rec/bootstrapping_contrastive_learning_enhanced_music_cold_start_matching/">Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用动态自举的特征相关性矩阵驱动特征级对比增广，学习鲁棒歌曲表征，并结合聚类人群定向完成冷启歌曲快速加热，全库排序 Recall@50 提升 4.74%，线上有效播放 +38.47%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;模糊协同嵌入矫正&quot;,&quot;内容-共现双视图&quot;]"><strong><a href="../../venues/WWW/2023/cold_start_rec/contrastive_collaborative_filtering_for_cold_start_item_recommendation/">Contrastive Collaborative Filtering for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用内容 CF 与共现 CF 双模块的对比学习把共现协同信号隐式&quot;背&quot;进内容编码器，矫正冷启动物品的模糊协同嵌入，全库用户排序下 HR@5 提升 6.1%/29.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;从众权重解耦&quot;,&quot;去偏对比学习&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/debiased_contrastive_learning_for_sequential_recommendation/">Debiased Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 DCRec，用三通道从众权重网络解耦用户真实兴趣与流行度从众，并以加权 InfoNCE 实现去偏对比学习，四数据集 LOO 全库排序 HR@1 最高提升 10.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;提示微调&quot;,&quot;图重构去噪&quot;]"><strong><a href="../../venues/WWW/2023/self_supervised_denoising/denoising_and_prompt_tuning_for_multi_behavior_recommendation/">Denoising and Prompt-Tuning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DPT 以三阶段范式先用重构难度无监督去噪辅助行为，再用重初始化与连续提示轻量微调，在 IJCAI/Tmall 上 HR@10 提升 1.58%–2.62%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;因果解耦嵌入&quot;,&quot;对比学习样本增广&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/disentangled_causal_embedding_with_contrastive_learning_for_recommender_sys/">Disentangled Causal Embedding With Contrastive Learning For Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用流行度加权双通道对比学习把交互解耦为兴趣与从众两类因果嵌入，OOD 干预测试集 HR@20 较 MACR 提升 93%，快手线上 EVTR +7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因子化多头注意力&quot;,&quot;双兴趣解耦&quot;,&quot;负反馈序列建模&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/dual_interest_factorization_heads_attention_for_sequential_recommendation/">Dual-interest Factorization-heads Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用头对分解注意力与反馈标签掩码建模跳/不跳混合序列，双兴趣解耦塔加 BPR 成对对比，Micro-video GAUC 提升 2.19%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;对比学习&quot;,&quot;图神经网络&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/graph_less_collaborative_filtering/">Graph-less Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双层知识蒸馏把 GCN 教师压缩为 MLP 学生，配合梯度自适应对比正则，在 Full-Ranking 协议下兼顾精度与推理效率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;数据增广&quot;,&quot;双层优化&quot;,&quot;群组公平性&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/improving_recommendation_fairness_via_data_augmentation/">Improving Recommendation Fairness via Data Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 FDA 框架，用嵌入扰动生成跨组“伪交互”并以双层优化做数据增广，无需预定义公平指标即可将 DP/EO 不公平度降低 4%–12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;对抗式图增广&quot;,&quot;跨模态对比学习&quot;]"><strong><a href="../../venues/WWW/2023/multimodal_rec/multi_modal_self_supervised_learning_for_recommendation/">Multi-Modal Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Gumbel+WGAN-GP 对抗生成模态感知交互图，再以跨模态对比学习对齐模态偏好，All-Rank 全库排序下 Recall@20 最高提升 16.04%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;Wasserstein 距离&quot;,&quot;自监督数据增广&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/mutual_wasserstein_discrepancy_minimization_for_sequential_recommendation/">Mutual Wasserstein Discrepancy Minimization for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 2-Wasserstein 距离替代 KL 散度重建对比互信息估计，四数据集全指标相对提升 2.24%–20.10%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;停留时间重加权&quot;,&quot;有效阅读筛选&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/WWW/2023/self_supervised_denoising/reweighting_clicks_with_dwell_time_in_recommendation/">Reweighting Clicks with Dwell Time in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用停留时间定义有效阅读并以归一化 Sigmoid 重加权点击信号，工业级双塔多任务框架在微信看一看离线 AUC+5.62%、线上 CTR+2.58%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;联邦推荐&quot;,&quot;协同聚类&quot;,&quot;监督对比学习&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/co_clustering_for_federated_recommender_system/">Co-clustering for Federated Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出服务器端按物品类别协同聚类用户并聚合相似组模型，结合全局物品成员监督对比学习，在4个数据集上以1:99采样排序评测提升HR/NDCG。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多行为序列建模&quot;,&quot;频域去噪&quot;,&quot;噪声解耦对比学习&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/end4rec_efficient_noise_decoupling_for_multi_behavior_sequential_recommenda/">END4Rec: Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 FFT 频域算子+双粒度（硬/软）去噪与噪声解耦对比学习做多行为序列推荐，Leave-One-Out+99 负采样评测，三数据集 HR@10 相对提升 5.8%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;反事实样本合成&quot;,&quot;数值特征单调性&quot;]"><strong><a href="../../venues/WWW/2024/self_supervised_denoising/enhancing_interpretability_and_effectiveness_in_recommendation_with_numeric/">Enhancing Interpretability and Effectiveness in Recommendation with Numerical Features via Learning to Contrast the Counterfactual samples</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以反事实样本合成+对比排序损失显式建模数值特征与模型输出的单调性，模型无关即插即用，KuaiRand/工业集 AUC 最高 +31.2%，线上 collect_rate +3.93%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;序列数据增强&quot;,&quot;对比学习&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/WWW/2024/self_supervised_denoising/is_contrastive_learning_necessary_a_study_of_data_augmentation_vs_contrasti/">Is Contrastive Learning Necessary? A Study of Data Augmentation vs Contrastive Learning in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统评测8种序列数据增强与3种对比学习SRS，发现slide-window等直接增强可媲美CL且更省算力。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;Graph Transformer&quot;,&quot;Knowledge Contrastive Denoising&quot;]"><strong><a href="../../venues/WWW/2024/graph_rec/knowledge_enhanced_multi_intent_transformer_network_for_recommendation/">Knowledge Enhanced Multi-intent Transformer Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可学习多意图与图Transformer注入全局信号，再用意图引导知识采样和局部-全局对比去噪；三基准与阿里线上A/B均稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;在线原型聚类&quot;,&quot;自蒸馏&quot;,&quot;梯度反转对抗&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/leave_no_one_behind_online_self_supervised_self_distillation_for_sequential/">Leave No One Behind: Online Self-Supervised Self-Distillation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用在线原型聚类+头尾对抗+簇感知自蒸馏，把长序列知识蒸馏给短序列用户，四数据集全库排序 HR@5 最高提升 9.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;异质信息网络&quot;,&quot;对比学习&quot;,&quot;知识点推荐&quot;]"><strong><a href="../../venues/WWW/2024/graph_rec/modeling_balanced_explicit_and_implicit_relations_with_contrastive_learning/">Modeling Balanced Explicit and Implicit Relations with Contrastive Learning for Knowledge Concept Recommendation in MOOCs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CL-KCRec 用基向量关系嵌入 GCN + 多通道软选边 GNN 分离建模 MOOC 异质网的显式/隐式关系，并以原型图对比学习做平衡增强；MOOCCube1819 上 N@5 相对最优基线提升 10.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;提示微调&quot;]"><strong><a href="../../venues/WWW/2024/multimodal_rec/promptmm_multi_modal_knowledge_distillation_for_recommendation_with_prompt_/">PromptMM: Multi-Modal Knowledge Distillation for Recommendation with Prompt-Tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以软提示增强教师、三层知识蒸馏压缩轻量学生，在三个数据集 Full-Ranking 下 R@20 提升 3.5%–14.8%，参数仅教师的 2.70%–7.83%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;双对比学习&quot;,&quot;Barlow Twins&quot;,&quot;特征去冗余&quot;]"><strong><a href="../../venues/WWW/2024/self_supervised_denoising/recdcl_dual_contrastive_learning_for_recommendation/">RecDCL: Dual Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出用户-物品双对比框架，融合特征维去冗余与批维输出增广，在四个公开集和支付工业集全库排序 Recall@20 最高提升5.65%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回/表征学习&quot;,&quot;跨视图对齐&quot;,&quot;互信息最大化&quot;]"><strong><a href="../../venues/WWW/2024/llm_rec/representation_learning_with_large_language_models_for_recommendation/">Representation Learning with Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RLMRec 用互信息最大化把 LLM 语义表征与协同表征对齐，模型无关地增强 6 个 CF 骨干，全库排序下 Recall@20 最高提升 7.6%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;数据混合增强&quot;,&quot;双混合对比学习&quot;]"><strong><a href="../../venues/WWW/2025/self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;个人损失分布重采样&quot;]"><strong><a href="../../venues/WWW/2025/self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;表征对齐&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/rallrec_improving_retrieval_augmented_large_language_model_recommendation_w/">RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成物品详述并做文本-协同表征对比对齐，配合时间感知重排器提升 RAG 式 LLM 推荐的检索质量。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域对比学习&quot;,&quot;课程学习&quot;,&quot;Stop-Gradient&quot;]"><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/separated_contrastive_learning_for_matching_in_cross_domain_recommendation_/">Separated Contrastive Learning for Matching in Cross-domain Recommendation with Curriculum Scheduling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 intra-CL 与 inter-CL 解耦为两阶段课程训练，配合 stop-gradient 与基于 Katz 中心性的负样本难度调度，稳定跨域召回训练并显著提升 HIT@100。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;数据蒸馏/训练加速&quot;,&quot;Tucker 分解&quot;,&quot;双层优化&quot;,&quot;特征空间对齐&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;信息瓶颈&quot;,&quot;图 CoT 推理&quot;]"><strong><a href="../../venues/WWW/2025/self_supervised_denoising/unleashing_the_power_of_large_language_model_for_denoising_recommendation/">Unleashing the Power of Large Language Model for Denoising Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>LLaRD 用 LLM 生成偏好知识与图 CoT 关系知识，再以信息瓶颈对齐去噪图，Full-Ranking 下 Recall@20 最高提升约 14%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;无用户嵌入&quot;,&quot;对比学习增强&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/alphafree_recommendation_free_from_users_ids_and_gnns/">AlphaFree: Recommendation Free from Users, IDs, and GNNs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语言表征替代ID、在线推断用户偏好、以相似物品增强+对比对齐替代GNN，实现无用户/无ID/无GNN的轻量推荐，内存降69%、冷启动提升26–38%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图结构注入&quot;,&quot;对比学习&quot;,&quot;事件序列建模&quot;]"><strong><a href="../../venues/WWW/2026/self_supervised_denoising/beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;k-DPP 多样性采样&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;训练加速/数据侧&quot;,&quot;Coreset Selection&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="../../venues/WWW/2026/self_supervised_denoising/efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果去偏&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/mclmr_a_model_agnostic_causal_learning_framework_for_multi_behavior_recomme/">MCLMR: A Model-Agnostic Causal Learning Framework for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用后门干预+MoE自适应聚合+偏差感知对比学习，作为即插即用框架增强多行为推荐，三数据集全库排序显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;参数高效用户建模&quot;,&quot;重叠社区检测&quot;,&quot;自监督对比正则&quot;]"><strong><a href="../../venues/WWW/2026/social_group_rec/pulse_socially_aware_user_representation_modeling_toward_parameter_efficien/">PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用社区结构与社交邻居物品信号替代显式用户嵌入，参数降 50% 且 NDCG@20 提升 8-9%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;扩散增广&quot;,&quot;非对称对比&quot;]"><strong><a href="../../venues/WWW/2026/self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
</dl>
