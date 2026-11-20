---
hide:
  - toc
---

# 🏛️ ICDE 2023 推荐系统论文解读

10 篇 ICDE 2023 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（训练期负采样）&quot;,&quot;假阴性辨识&quot;,&quot;贝叶斯后验风险最小化&quot;,&quot;负采样&quot;]"><strong><a href="collaborative_filtering/bayesian_negative_sampling_for_recommendation/">Bayesian Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>由排序关系导出真/假阴性类条件密度，构造模型无关贝叶斯后验负信号与风险最小化采样准则；MF/LightGCN 双主干 Top-20 NDCG 最高 +5.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;动态异构图&quot;,&quot;邻域扰动&quot;,&quot;增量学习&quot;]"><strong><a href="graph_rec/instant_representation_learning_for_recommendation_over_large_dynamic_graph/">Instant Representation Learning for Recommendation over Large Dynamic Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SUPA 以采样-更新-传播架构替代邻居聚合，缓解动态多重异构图中的邻域扰动，配合单遍增量训练工作流 InsLearn，在六个真实数据集全库排序下 H@50 最高相对提升 54%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;过平滑缓解&quot;,&quot;层精炼 Layer Refinement&quot;,&quot;度敏感边丢弃&quot;]"><strong><a href="graph_rec/layer_refined_graph_convolutional_networks_for_recommendation/">Layer-refined Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ego 层余弦相似度动态精炼每层传播表征，配合度敏感边剪枝，破解 GCN 推荐过平滑困境，全库排序下 Recall@20 最高提 10.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排表示增强&quot;,&quot;意图解耦&quot;,&quot;标签对比对齐&quot;]"><strong><a href="self_supervised_denoising/intent_aware_multi_source_contrastive_alignment_for_tag_enhanced_recommenda/">Intent-aware Multi-source Contrastive Alignment for Tag-enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户/物品表征拆成K个意图子向量，与标签簇端到端聚类对齐，并用集合到集合正样本增强；7数据集Top-20排序最高R@20 +6.7%、NDCG@20 +10.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;频域滤波&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/contrastive_enhanced_slide_filter_mixer_for_sequential_recommendation/">Contrastive Enhanced Slide Filter Mixer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用频域滑动滤波与对比学习替代自注意力，在5个基准全库排序下 HR@10 最高相对提升21.39%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;影响路径生成&quot;,&quot;个性化注意力掩码&quot;]"><strong><a href="sequential_rec/influential_recommender_system/">Influential Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出影响力推荐新范式 IRS 与 Transformer 模型 IRN，用个性化易受影响性掩码 PIM 生成引导用户走向目标物品的影响路径，SR20 最高提升约 3.5 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;解耦表征学习&quot;,&quot;异构图表征&quot;,&quot;记忆网络&quot;]"><strong><a href="social_group_rec/disentangled_graph_social_recommendation/">Disentangled Graph Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以可微记忆单元把社交边、交互边、物品语义边解耦到多潜因子空间，构建异构社交 GNN，采样排序 HR@10 最高提升 4.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;图匹配&quot;,&quot;部分重叠跨域迁移&quot;,&quot;用户表征补全&quot;]"><strong><a href="cross_domain_multi_scenario/neural_node_matching_for_multi_target_cross_domain_recommendation/">Neural Node Matching for Multi-Target Cross Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出节点匹配框架 NMCDR，用域内-域间全连接用户图匹配加虚拟交互补全，在 0.1% 极小重叠下 Amazon 平均提升 24.84%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;拼团推荐&quot;,&quot;专家网络&quot;,&quot;门控&quot;]"><strong><a href="multi_task_reranking/group_buying_recommendation_model_based_on_multi_task_learning/">Group Buying Recommendation Model Based on Multi-task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将拼团推荐拆为发起人选品与参与者选团，用三视图GCN、专家门控与双辅助损失联合建模；Task B MRR@10提升71.65%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;匹配/派单&quot;,&quot;Contextual Bandit&quot;,&quot;Capacity-Aware Assignment&quot;]"><strong><a href="rl_bandit_rec/towards_capacity_aware_broker_matching_from_recommendation_to_assignment/">Towards Capacity-Aware Broker Matching: From Recommendation to Assignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将经纪人匹配从 top-k 推荐转为容量感知指派，NN-UCB 在线估容量，MDP 价值函数+KM 全局分配，CBS 无损加速。</p>
</dd>
</dl>
</section>
</div>
