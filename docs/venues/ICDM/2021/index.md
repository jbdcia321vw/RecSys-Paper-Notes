---
hide:
  - toc
---

# 🏛️ ICDM 2021 推荐系统论文解读

6 篇 ICDM 2021 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 协同过滤 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJDSA&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;Item-KNN&quot;,&quot;贝叶斯热门度解耦&quot;]"><strong><a href="collaborative_filtering/a_probabilistic_perspective_on_nearest_neighbor_for_implicit_recommendation/">A probabilistic perspective on nearest neighbor for implicit recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将 Item-KNN 相似度重解释为条件概率，用 Beta 可信区间选邻居、概率 OR 聚合，并用贝叶斯推断解耦热门度，全库排序 NDCG@50 相对最优 KNN 提升 1.2%–5.6%</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2021&quot;,&quot;协同过滤&quot;,&quot;召回（Embedding 表征层）&quot;,&quot;混合维度 Embedding&quot;,&quot;AutoML/DARTS 双层优化&quot;,&quot;流式推荐&quot;]"><strong><a href="collaborative_filtering/autoemb_automated_embedding_dimensionality_search_in_streaming_recommendati/">AutoEmb: Automated Embedding Dimensionality Search in Streaming Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用控制器网络为每个用户/物品按流行度软加权选择 2/16/128 维 embedding，并在流式设定下以 DARTS 式双层优化动态搜索，MSE 相对统一维度基线降 2.0%–2.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;超元路径&quot;,&quot;图对比学习&quot;,&quot;多行为推荐&quot;]"><strong><a href="self_supervised_denoising/hyper_meta_path_contrastive_learning_for_multi_behavior_recommendation/">Hyper Meta-Path Contrastive Learning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出超元路径/超元图显式刻画多行为依赖，用相邻超元图 InfoNCE 对比替代固定融合，Taobao Recall@5 相对提升 24.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;对比预测编码 (CPC)&quot;,&quot;记忆网络与多实例负采样&quot;]"><strong><a href="sequential_rec/memory_augmented_multi_instance_contrastive_predictive_coding_for_sequentia/">Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>记忆增强 CPC + 多实例 NCE：用记忆残差捕捉长期偏好、Dropout 语义正样本缓解监督稀疏，四数据集全库排序 HR@5 最高提升 152.7%</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列推荐&quot;,&quot;MOBA 数据集&quot;,&quot;RNN&quot;]"><strong><a href="sequential_rec/sequential_item_recommendation_in_the_moba_game_dota_2/">Sequential Item Recommendation in the MOBA Game Dota 2</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 Dota-350k 购买序列数据集并系统评测 SIR 模型，发现 GRU 以 Rec@3 0.7363 超越 Transformer。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;随机门控冷启动模拟&quot;,&quot;冷热协调 (Cold-Warm Harmonization)&quot;,&quot;多视角内容表征&quot;]"><strong><a href="cold_start_rec/cold_item_integration_in_deep_hybrid_recommenders_via_tunable_stochastic_ga/">Cold Item Integration in Deep Hybrid Recommenders via Tunable Stochastic Gates</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用可调伯努利随机门在训练中模拟完全冷物品，配双路内容表征补偿缺失的 CF 向量，实现冷/热物品可调平衡。</p>
</dd>
</dl>
</section>
</div>
