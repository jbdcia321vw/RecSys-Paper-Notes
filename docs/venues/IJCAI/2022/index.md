---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2022 推荐系统论文解读

8 篇 IJCAI (RecSys Track) 2022 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2022 ×6 · IJCAI ×2 · 图对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="图对比学习" aria-pressed="false">图对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;负采样与去偏&quot;]"><strong><a href="collaborative_filtering/trading_hard_negatives_and_true_negatives_a_debiased_contrastive_collaborat/">Trading Hard Negatives and True Negatives: A Debiased Contrastive Collaborative Filtering Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用集合级对比目标自适应挖掘硬负例，并以频次感知批内采样 + 去偏项消除假负例偏差，MovieLens/Yelp/Gowalla 上 NDCG@10 提升 4.9%–9.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质图层次注意力&quot;,&quot;集合 Transformer&quot;,&quot;图对比学习&quot;]"><strong><a href="graph_rec/reciperec_a_heterogeneous_graph_learning_model_for_recipe_recommendation/">RecipeRec: A Heterogeneous Graph Learning Model for Recipe Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建用户-食谱-食材异质图 URI-Graph，用层次注意力异质 GNN、食材集合 Transformer 与图对比增强联合建模内容与高阶协同信号，Leave-One-Out 采样排序下 HR@10 达 45.70。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;加权物品转移图&quot;,&quot;图对比学习&quot;,&quot;MMD表征对齐&quot;]"><strong><a href="sequential_rec/enhancing_sequential_recommendation_with_graph_contrastive_learning/">Enhancing Sequential Recommendation with Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建全用户加权物品转移图，以子图采样+共享GNN生成增广视图，结合图对比损失与MMD个性化对齐，Full-Ranking下HR@10最高提升12.9%</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（Next-Item 检索）&quot;,&quot;MLP-Mixer&quot;,&quot;三向信息融合&quot;]"><strong><a href="sequential_rec/mlp4rec_a_pure_mlp_architecture_for_sequential_recommendations/">MLP4Rec: A Pure MLP Architecture for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用纯 MLP 的三向 Mixer（序列/通道/特征）替代自注意力，免位置编码、线性复杂度，Beauty 上 HR@10 提升 6.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排（广告位拍卖/机制层）&quot;,&quot;GSP拍卖&quot;,&quot;组级公平分配&quot;,&quot;EF1/EFX&quot;,&quot;无遗憾学习&quot;]"><strong><a href="ctr_feature_interaction/fair_equilibria_in_sponsored_search_auctions_the_advertisers_perspective/">Fair Equilibria in Sponsored Search Auctions: The Advertisers&#x27; Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>在 GSP 后接组级 EF1/EFX 公平分配并补偿受损广告主，证明无遗憾学习收敛到高社会福利均衡；真实日志上公平损失≤10%，补偿占收入20–40%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;列表级上下文建模&quot;,&quot;评估器-生成器范式&quot;,&quot;综述与基准库&quot;]"><strong><a href="multi_task_reranking/neural_re_ranking_in_multi_stage_recommender_systems_a_review/">Neural Re-ranking in Multi-stage Recommender Systems: A Review</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以“目标×监督信号”四象限梳理神经重排，给出复杂度对比、LibRerank 基准与 PRM Public 上 +68% 的量化结论。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦训练&quot;,&quot;模型投毒&quot;,&quot;梯度攻击&quot;,&quot;联邦推荐&quot;]"><strong><a href="debias_fairness_industrial/poisoning_deep_learning_based_recommender_model_in_federated_learning_scena/">Poisoning Deep Learning Based Recommender Model in Federated Learning Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>无先验知识下用高斯近似良性用户嵌入与硬用户挖掘生成毒化梯度，0.1% 恶意用户即把冷门目标物品 ER@10 推至 1.0。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;逆倾向加权 IPS&quot;,&quot;对抗域适应去偏&quot;]"><strong><a href="debias_fairness_industrial/towards_resolving_propensity_contradiction_in_offline_recommender_learning/">Towards Resolving Propensity Contradiction in Offline Recommender Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出倾向无关泛化误差上界与对抗式矩阵分解 DAMF，无需 MCAR 数据与倾向估计，Yahoo! R3 上 MSE 相对降 26.3%。</p>
</dd>
</dl>
</section>
</div>
