---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**ACM MM 2023** · **5** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [ACM MM 2022 (1)](../../2022/multimodal_rec/) · [ACM MM 2024 (3)](../../2024/multimodal_rec/) · [ACM MM 2025 (4)](../../2025/multimodal_rec/) · [ACM MM 2026 (3)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/multimodal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [CIKM 2024 (5)](../../../CIKM/2024/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2022 (1)](../../../KDD/2022/multimodal_rec/) · [KDD 2024 (1)](../../../KDD/2024/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2022 (2)](../../../RecSys/2022/multimodal_rec/) · [RecSys 2024 (2)](../../../RecSys/2024/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2023 (4)](../../../SIGIR/2023/multimodal_rec/) · [SIGIR 2024 (1)](../../../SIGIR/2024/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2023 (2)](../../../TMM/2023/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/multimodal_rec/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/multimodal_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

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
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图结构冻结&quot;,&quot;度数敏感边剪枝&quot;]"><strong><a href="a_tale_of_two_graphs_freezing_and_denoising_graph_structures_for_multimodal/">A Tale of Two Graphs: Freezing and Denoising Graph Structures for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>冻结多模态物品-物品图并对用户-物品图做度数敏感剪枝去噪，全排序协议下相对 LATTICE 平均提升 19.07%，内存最高降 6×。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;数据与特征工程&quot;,&quot;多模态特征抽取&quot;,&quot;可复现框架&quot;]"><strong><a href="ducho_a_unified_framework_for_the_extraction_of_multimodal_features_in_reco/">Ducho: A Unified Framework for the Extraction of Multimodal Features in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一 TensorFlow/PyTorch/Transformers 三后端的多模态特征抽取框架，以 YAML 配置解耦模态、来源与抽取层，提供 Docker 化可复现管线。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导模态去噪&quot;,&quot;多视图图卷积&quot;,&quot;行为感知融合&quot;]"><strong><a href="multi_view_graph_convolutional_network_for_multimedia_recommendation/">Multi-View Graph Convolutional Network for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为引导净化模态噪声 + U-I/I-I 双视图编码 + 行为感知自适应融合与互信息自监督，Amazon 三集全排序 Recall@20 最高提升 22.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;不变风险最小化&quot;,&quot;帕累托多目标优化&quot;]"><strong><a href="pareto_invariant_representation_learning_for_multimedia_recommendation/">Pareto Invariant Representation Learning for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 IID-OOD 权衡建模为多目标优化，梯度自适应加权 ERM/IRM 学不变掩码，三数据集 IID 与 OOD 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;语义引导特征提取&quot;,&quot;模态互补融合&quot;]"><strong><a href="semantic_guided_feature_distillation_for_multimodal_recommendation/">Semantic-Guided Feature Distillation for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用语义标签监督的 DNN 教师 + 浅层学生知识蒸馏替代联合训练的特征提取器，缓解稀疏下的提取失效，三类骨干最高提升 8.56%</p>
</dd>
</dl>
