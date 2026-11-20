---
hide:
  - toc
---

# 🏛️ DASFAA 2023 推荐系统论文解读

8 篇 DASFAA 2023 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;反事实数据增强&quot;,&quot;捆绑推荐&quot;,&quot;对比约束&quot;]"><strong><a href="graph_rec/data_augmented_counterfactual_learning_for_bundle_recommendation/">Data-Augmented Counterfactual Learning for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用启发式采样器生成反事实用户-捆绑包-物品图视图，并以自适应阈值反事实损失约束训练，缓解捆绑推荐双向稀疏，全库排序下 R@20 最高提升约 9.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图对比学习&quot;,&quot;可学习数据增强&quot;,&quot;对抗训练&quot;]"><strong><a href="self_supervised_denoising/adversarial_learning_data_augmentation_for_graph_contrastive_learning_in_re/">Adversarial Learning Data Augmentation for Graph Contrastive Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 InfoMin/InfoMax 对抗框架学习可微的边增删图增强算子，四数据集 Full-Ranking 下 Recall@10 最高相对 NCL 提升 23.5%</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;物品侧对比学习&quot;,&quot;多粒度数据增广&quot;]"><strong><a href="self_supervised_denoising/multi_granularity_item_based_contrastive_recommendation/">Multi-granularity Item-based Contrastive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为双塔召回设计特征/语义/会话三级物品侧对比任务，工业级全库 HIT@500 相对 SOTA 提升 2.8%，线上 ACC +3.29%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;双专家门控混合&quot;,&quot;动态知识蒸馏&quot;,&quot;用户行为偏置&quot;]"><strong><a href="cold_start_rec/cold_warm_net_addressing_cold_start_users_in_recommender_systems/">Cold &amp; Warm Net: Addressing Cold-Start Users in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>召回阶段用冷启/升温双专家+门控+动态知识蒸馏+行为偏置网建模用户冷启动，冷启用户 HR@100 提升 9.5%–23.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景CTR预估&quot;,&quot;冷启动记忆网络&quot;,&quot;场景感知投影&quot;]"><strong><a href="cross_domain_multi_scenario/cold_start_based_multi_scenario_ranking_model_for_click_through_rate_predic/">Cold-Start based Multi-Scenario Ranking Model for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以场景感知正交投影分离多场景用户兴趣，并用外部记忆网络让冷启动用户借相似画像富行为用户兴趣，离线 AUC 0.7001、线上 CTR +3.85%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;轻量化GCN&quot;,&quot;外部注意力&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/towards_lightweight_cross_domain_sequential_recommendation_via_external_att/">Towards Lightweight Cross-domain Sequential Recommendation via External Attention-enhanced Graph Convolution Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>LEA-GCN 用单层聚合 GCN 与双通道外部注意力实现轻量跨域序列推荐，参数量降 6 倍仍全面超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;频域FFT&quot;,&quot;隐式负反馈&quot;]"><strong><a href="ctr_feature_interaction/fan_fatigue_aware_network_for_click_through_rate_prediction_in_e_commerce_r/">FAN: Fatigue-Aware Network for Click-Through Rate Prediction in E-commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 FFT 把非点击序列变换到频域建模用户疲劳，并以类目调制与门控置信度融入 CTR 精排，离线 AUC 0.7249、在线 PCTR +1.63%、曝光类目数 +11.13%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;Accuracy-Bias Tradeoff&quot;,&quot;Self-Sampling&quot;,&quot;tIPS&quot;,&quot;Self-Training&quot;,&quot;Self-Evaluation&quot;]"><strong><a href="debias_fairness_industrial/self_sampling_training_and_evaluation_for_the_accuracy_bias_tradeoff_in_rec/">Self-Sampling Training and Evaluation for the Accuracy-Bias Tradeoff in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出SSTE，以自采样构造多偏差环境，联合训练与鲁棒自评估实现有益/有害偏差权衡；在Yahoo! R3/Product无偏测试与腾讯理财通在线A/B中均获提升。</p>
</dd>
</dl>
</section>
</div>
