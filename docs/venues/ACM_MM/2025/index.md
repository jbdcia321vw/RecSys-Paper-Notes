---
hide:
  - toc
---

# 🏛️ ACM MM 2025 推荐系统论文解读

9 篇 ACM MM 2025 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;端侧重排&quot;,&quot;混合精度量化&quot;,&quot;设备-云协同&quot;,&quot;超网络&quot;]"><strong><a href="sequential_rec/chord_customizing_hybrid_precision_on_device_model_for_sequential_recommend/">CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端超网络按用户画像生成逐通道混合精度量化策略（2-bit 编码），端侧单次前向完成个性化适配与推理，精度反超全精度基线且传输降 173.8×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;游玩时长建模&quot;,&quot;随机游走多样性&quot;]"><strong><a href="multimodal_rec/dual_phase_playtime_guided_recommendation_interest_intensity_exploration_an/">Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双 Beta 分布解耦游玩时长强弱兴趣，并以多模态随机游走做跨类目探索，在 Steam 数据上同时提升准确率与覆盖度。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;MLLM 负样本生成&quot;,&quot;因果学习&quot;]"><strong><a href="multimodal_rec/generating_negative_samples_for_multi_modal_recommendation/">Generating Negative Samples for Multi-Modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 三步提示生成兼具凝聚力与难度的属性级负样本，并以因果学习解耦关键特征，全库排序下 Recall@10 最高提升 9.02%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;不变风险最小化&quot;,&quot;信息瓶颈&quot;]"><strong><a href="multimodal_rec/i_3_mrec_invariant_learning_with_information_bottleneck_for_incomplete_moda/">I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将各模态视为独立语义环境做 IRM 不变学习，并用 IB 压缩模态特有信息，在模态缺失场景下 Full-Ranking Recall@20 稳定提升 2%–4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态对比学习精炼&quot;,&quot;同构图关系挖掘&quot;]"><strong><a href="multimodal_rec/refining_contrastive_learning_and_homography_relations_for_multi_modal_reco/">Refining Contrastive Learning and Homography Relations for Multi-Modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用元网络+正交约束精炼多模态对比学习，并补建用户兴趣图与物品共现图，全库排序下 R@20 最高提升约 9.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;冷启动&quot;,&quot;端侧推理&quot;,&quot;参数编辑&quot;,&quot;原型聚类&quot;,&quot;端云协同&quot;]"><strong><a href="cold_start_rec/tackling_device_data_distribution_real_time_shift_via_prototype_based_param/">Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端神经适配器免反传生成参数编辑矩阵，多原型聚类+跨层知识迁移实现端侧实时分布偏移适配，1s 内完成个性化且 NDCG@5 显著超越微调。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;扩散模型生成式迁移&quot;,&quot;多模态大表征模型&quot;,&quot;边侧用户分布学习&quot;]"><strong><a href="cross_domain_multi_scenario/leveraging_multimodal_data_and_side_users_for_diffusion_cross_domain_recomm/">Leveraging Multimodal Data and Side Users for Diffusion Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 抽取多模态特征做初步对齐，再以边侧用户+重叠用户两阶段扩散生成目标域冷启动用户向量，Amazon 三任务 MAE 最高降 13.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;GFlowNet&quot;,&quot;短视频预取&quot;,&quot;多候选生成&quot;]"><strong><a href="rl_bandit_rec/generative_flow_networks_for_personalized_multimedia_systems_a_case_study_o/">Generative Flow Networks for Personalized Multimedia Systems: A Case Study on Short Video Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 GFlowNet 多候选生成替代单动作 RL 策略，在短视频预取场景实现个性化 QoE 与带宽双优化。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;后处理/嵌入层&quot;,&quot;机器遗忘 (Machine Unlearning)&quot;,&quot;互信息最小化&quot;,&quot;属性推断攻击&quot;]"><strong><a href="debias_fairness_industrial/lego_a_lightweight_and_efficient_multiple_attribute_unlearning_framework_fo/">LEGO: A Lightweight and Efficient Multiple-Attribute Unlearning Framework for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多属性遗忘拆为并行嵌入校准与权重组合两步，以互信息最小化保证同时遗忘，动态需求下近零额外开销。</p>
</dd>
</dl>
</section>
</div>
