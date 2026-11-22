---
hide:
  - toc
---

# 🏛️ ACM MM 2026 推荐系统论文解读

8 篇 ACM MM 2026 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 多模态推荐 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/物品表征&quot;,&quot;侧适配器&quot;,&quot;流感知路由&quot;]"><strong><a href="sequential_rec/stream_aware_side_adaptation_for_large_pre_trained_multimodal_embedding_mod/">Stream-aware Side Adaptation for Large Pre-trained Multimodal Embedding Models in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结Qwen3-VL侧适配新框架SHAF/ReSA，Leave-One-Out全库排序，HR@10最高提升7.4%</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba/状态空间对偶&quot;,&quot;多模态迁移学习&quot;,&quot;时间感知结构化掩码&quot;]"><strong><a href="sequential_rec/towards_transfer_efficient_multi_modal_sequential_recommendation_with_state/">Towards Transfer-Efficient Multi-modal Sequential Recommendation with State Space Duality</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 SSD 时间衰减+跨模态共享掩码对齐+双通道傅里叶滤波融合，替代对比学习式多模态预训练，跨域微调收敛加速约 10 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;正交净化&quot;,&quot;混合专家路由&quot;]"><strong><a href="multimodal_rec/beyond_modality_harmony_orthogonal_purification_and_topology_guided_moe_for/">Beyond Modality Harmony: Orthogonal Purification and Topology-Guided MoE for Conflict-Aware Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以纯协同锚点为几何参考轴正交分解并保能截断模态噪声，配合解耦 Sigmoid 路由 MoE 与冲突感知 safe-SSL，在 Amazon 三数据集 8:1:1 划分下 R@20 最高提升 6.55%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图信号低通滤波&quot;,&quot;频域序列去噪&quot;]"><strong><a href="multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;多模态推荐&quot;,&quot;多模态表征与端到端&quot;,&quot;Mixture-of-Experts&quot;,&quot;联邦跨模态对齐&quot;]"><strong><a href="multimodal_rec/fm_2_unified_federated_foundation_models_for_heterogeneous_multimodal_medic/">FM2: Unified Federated Foundation Models for Heterogeneous Multimodal Medical Imaging</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双 MoE 解耦类级个性化与模态级共识，近端正则 HMA 保证 O(1/√T) 收敛，LLM 描述作跨模态语义桥 CEL，联邦医疗三任务全面领先</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;物品-物品图&quot;,&quot;三元闭包重加权&quot;]"><strong><a href="multimodal_rec/one_graph_multiple_gains_single_high_quality_item_item_graph_for_multimodal/">One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建一次高质量物品-物品图，经三元闭包重加权后复用于表征、交互图与优化三阶段，四数据集全指标SOTA且训练更快更省显存。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/精排&quot;,&quot;联邦协作预训练&quot;,&quot;语义软交叉熵&quot;,&quot;LoRA 适配&quot;]"><strong><a href="cross_domain_multi_scenario/from_transfer_to_collaboration_a_federated_framework_for_cross_market_seque/">From Transfer to Collaboration: A Federated Framework for Cross-Market Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>FeCoSR 以联邦多对多协作与语义软交叉熵 S2CE 缓解跨市场异质性，在 XMarket 八市场 Leave-One-Out Full-Ranking 上 NDCG@10 最高提升 17.28%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型遗忘/隐私删除&quot;,&quot;多模态推荐&quot;,&quot;反向更新&quot;,&quot;参数组隔离&quot;]"><strong><a href="debias_fairness_industrial/tru_targeted_reverse_update_for_efficient_multimodal_recommendation_unlearn/">TRU: Targeted Reverse Update for Efficient Multimodal Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示多模态推荐遗忘中排序/模态/模块三层非均匀性，提出门控+分支缩放+参数组隔离的靶向反向更新，逼近重训练前沿且收敛更快。</p>
</dd>
</dl>
</section>
</div>
