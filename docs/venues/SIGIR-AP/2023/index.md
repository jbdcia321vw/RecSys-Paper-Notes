---
hide:
  - toc
---

# 🏛️ SIGIR-AP 2023 推荐系统论文解读

4 篇 SIGIR-AP 2023 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR-AP&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣解耦&quot;,&quot;强化学习序列分解&quot;,&quot;时间间隔建模&quot;]"><strong><a href="sequential_rec/sequential_recommendation_with_user_evolving_preference_decomposition/">Sequential Recommendation with User Evolving Preference Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 RL 智能体把用户行为序列自适应拆成多条演化偏好线程，ε-softmax 决定新建线程、时间衰减刻画间隔，六数据集 NDCG@5 平均 +10.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR-AP&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;物品索引&quot;,&quot;谱聚类协同ID&quot;,&quot;约束解码&quot;]"><strong><a href="generative_rec/how_to_index_item_ids_for_recommendation_foundation_models/">How to Index Item IDs for Recommendation Foundation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统对比随机/标题/独立/顺序/协同/语义/混合七类物品ID索引方案，用谱聚类协同索引+独立令牌在P5上取得HR@10最高+64.8%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR-AP 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;强化学习PPO&quot;,&quot;二维网格面板&quot;,&quot;MDP动作空间设计&quot;]"><strong><a href="multi_task_reranking/reinforcement_re_ranking_with_2d_grid_based_recommendation_panels/">Reinforcement Re-ranking with 2D Grid-based Recommendation Panels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 2D 网格面板重排建模为“时间步=输入排名、动作=(是否曝光,槽位选择)”的 MDP，PPO 训练，AUC 相对提升 6.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR-AP 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;对抗训练&quot;,&quot;特征公平&quot;,&quot;自适应扰动&quot;]"><strong><a href="debias_fairness_industrial/automatic_feature_fairness_in_recommendation_via_adversaries/">Automatic Feature Fairness in Recommendation via Adversaries</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出以特征域为粒度的公平性目标，用自适应对抗扰动按「组合多样性」定扰动强度、按「特征频率」定对抗权重，在 FM 主干上同时提升准确率与组间公平性。</p>
</dd>
</dl>
</section>
</div>
