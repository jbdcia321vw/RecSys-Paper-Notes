# 🧩 多模态推荐与内容理解

**AAAI (RecSys Track) 2025** · **5** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** AAAI 2025 ×5 · 多模态推荐 ×5

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP替代GCN" aria-pressed="false">MLP替代GCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层排序损失" aria-pressed="false">分层排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层图卷积" aria-pressed="false">双层图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑剪枝" aria-pressed="false">拓扑剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态遗忘" aria-pressed="false">模态遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品语义图" aria-pressed="false">物品语义图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束嵌入更新" aria-pressed="false">约束嵌入更新<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导特征适配&quot;,&quot;参数高效微调&quot;]"><strong><a href="befa_a_general_behavior_driven_feature_adapter_for_multimedia_recommendatio/">BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为特征引导的轻量适配器解耦-过滤-重构预训练内容特征，修复信息漂移与遗漏，全库排序下平均提升约 9-11%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;拓扑剪枝&quot;,&quot;邻域对齐损失&quot;,&quot;MLP替代GCN&quot;]"><strong><a href="beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跳过行为建模&quot;,&quot;双层图卷积&quot;,&quot;分层排序损失&quot;]"><strong><a href="exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跨模态对齐&quot;,&quot;自监督对比学习&quot;,&quot;物品语义图&quot;]"><strong><a href="mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;逐步图卷积&quot;,&quot;模态遗忘&quot;,&quot;约束嵌入更新&quot;]"><strong><a href="stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
</dl>
