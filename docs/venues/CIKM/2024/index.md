---
hide:
  - toc
---

# 🏛️ CIKM 2024 推荐系统论文解读

8 篇 CIKM 2024 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;文本侧信息融合&quot;]"><strong><a href="sequential_rec/sequence_level_semantic_representation_fusion_for_recommender_systems/">Sequence-level Semantic Representation Fusion for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>频域互滤波实现文本-ID 序列级语义融合，等价于上下文卷积；全库排序下 ML-1M/OR 提升 14%/38%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端召回&quot;,&quot;生成式标识 GID&quot;,&quot;内容-协同对齐&quot;]"><strong><a href="generative_rec/content_based_collaborative_generation_for_recommender_systems/">Content-Based Collaborative Generation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ColaRec 用 LightGCN 层次聚类构造协同 GID，T5 序列到序列直接生成物品标识，并以物品索引任务+对比损失对齐内容与协同空间；四数据集 8:1:1 全库排序 Recall@20 提升 3.5%–6.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;LLM推荐&quot;,&quot;召回+精排&quot;,&quot;互补知识图&quot;,&quot;E-E-I权重决策&quot;]"><strong><a href="llm_rec/breaking_the_barrier_utilizing_large_language_models_for_industrial_recomme/">Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CIKM 2024[LLM推荐] 用 LLM 离线构建互补知识图，结合 U2E2I 召回与 E-E-I 精排校准；在支付宝三场景 AUC 与线上转化显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;评分预测/精排&quot;,&quot;LoRA 插件&quot;,&quot;ID-内容对比对齐&quot;]"><strong><a href="llm_rec/enhancing_content_based_recommendation_via_large_language_model/">Enhancing Content-based Recommendation via Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>CIKM 2024｜LLM4Rec：LoRA 跨域语义插件经 DARE 合并，再用 ID-内容对比对齐做评分预测；Amazon 11 域 8:1:1 划分下 MSE 显著优于 RGCL。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端策略学习&quot;,&quot;离线强化学习&quot;,&quot;用户模拟器&quot;,&quot;以用户为中心偏好建模&quot;]"><strong><a href="agent_conversational_rec/reformulating_conversational_recommender_systems_as_tri_phase_offline_polic/">Reformulating Conversational Recommender Systems as Tri-Phase Offline Policy Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 CRS 重构为「用户建模—离线策略—独立可控仿真评估」三阶段解耦范式，Yelp SR@15 达 0.415 且 AT 降至 11.62。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/端到端策略学习&quot;,&quot;Offline RL&quot;,&quot;Reward Shaping&quot;]"><strong><a href="rl_bandit_rec/roler_effective_reward_shaping_in_offline_reinforcement_learning_for_recomm/">ROLeR: Effective Reward Shaping in Offline Reinforcement Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用免训练 kNN 奖励整形与簇内距离不确定性惩罚修正 world model 奖励估计，在 KuaiRec 全观测模拟器上累积奖励较 DORL 提升 62.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/精排共用特征底座&quot;,&quot;NeighborHash 哈希表&quot;,&quot;批量点查 KV 存储&quot;,&quot;版本一致性协议&quot;]"><strong><a href="debias_fairness_industrial/an_enhanced_batch_query_architecture_in_real_time_recommendation/">An Enhanced Batch Query Architecture in Real-time Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 NeighborHash（邻域探测+内联偏移）替代传统哈希，配 NVMe 分层 KV 与版本强一致滚动更新，批量点查吞吐近随机访存上界 90%，线上 CTR +2.2%~6.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/序列排序&quot;,&quot;模型水印&quot;,&quot;自回归OOD序列&quot;]"><strong><a href="debias_fairness_industrial/watermarking_recommender_systems/">Watermarking Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AOW 自回归 OOD 水印，为序列推荐植入黑盒可验证版权序列；四数据集 Full-Ranking 达 100% R@1，兼顾效用与抗蒸馏/微调。</p>
</dd>
</dl>
</section>
</div>
