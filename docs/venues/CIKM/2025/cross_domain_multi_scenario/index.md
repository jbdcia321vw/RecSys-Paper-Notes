---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**CIKM 2025** · **4** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [CIKM 2021 (4)](../../2021/cross_domain_multi_scenario/) · [CIKM 2022 (4)](../../2022/cross_domain_multi_scenario/) · [CIKM 2023 (2)](../../2023/cross_domain_multi_scenario/) · [CIKM 2024 (2)](../../2024/cross_domain_multi_scenario/) · [CIKM 2026 (5)](../../2026/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/cross_domain_multi_scenario/) · [DASFAA 2023 (2)](../../../DASFAA/2023/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ECIR 2023 (3)](../../../ECIR/2023/cross_domain_multi_scenario/) · [ICDE 2022 (1)](../../../ICDE/2022/cross_domain_multi_scenario/) · [ICDE 2023 (1)](../../../ICDE/2023/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2023 (1)](../../../ICDM/2023/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/cross_domain_multi_scenario/) · [KDD 2021 (1)](../../../KDD/2021/cross_domain_multi_scenario/) · [KDD 2022 (1)](../../../KDD/2022/cross_domain_multi_scenario/) · [KDD 2023 (1)](../../../KDD/2023/cross_domain_multi_scenario/) · [KDD 2024 (1)](../../../KDD/2024/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (7)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2024 (2)](../../../RecSys/2024/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2021 (1)](../../../SIGIR/2021/cross_domain_multi_scenario/) · [SIGIR 2023 (1)](../../../SIGIR/2023/cross_domain_multi_scenario/) · [SIGIR 2024 (5)](../../../SIGIR/2024/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (5)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [ACM TKDD 2022 (1)](../../../TKDD/2022/cross_domain_multi_scenario/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/cross_domain_multi_scenario/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/cross_domain_multi_scenario/) · [ACM TORS 2026 (1)](../../../TORS/2026/cross_domain_multi_scenario/) · [WSDM 2022 (2)](../../../WSDM/2022/cross_domain_multi_scenario/) · [WSDM 2024 (4)](../../../WSDM/2024/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/cross_domain_multi_scenario/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;软分区动态参数&quot;,&quot;半监督伪标签迁移&quot;]"><strong><a href="a_soft_partitioned_semi_supervised_collaborative_transfer_learning_approach/">A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用软分区动态参数缓解主导域压制、用主导域伪标签加权迁移缓解稀疏域过拟合，离线全库AUC与美团线上A/B双验证。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;搜索增强推荐&quot;,&quot;向量量化码本&quot;,&quot;用户-码本图消息传递&quot;]"><strong><a href="benefit_from_rich_tackling_search_interaction_sparsity_in_search_enhanced_r/">Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM+RQ-VAE 将用户搜索/推荐偏好离散为码本，构建 User-Code 图做消息传递，把搜索富用户信号传给稀疏用户，Qilin 上 HR@1 相对 UniSAR 提升 12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;变分推断&quot;,&quot;场景特定表征&quot;,&quot;全局先验分布&quot;]"><strong><a href="global_distribution_aware_scenario_specific_variational_representation_lear/">Global-Distribution Aware Scenario-Specific Variational Representation Learning Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用变分推断为每个场景下的用户/物品学习分布表征，并以全局行为模式与物品属性构造先验，缓解稀疏场景表征欠学习，即插即用提升 PEPNet/MARIA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景 CTR 预估&quot;,&quot;基准复现&quot;,&quot;场景共享-特化建模&quot;]"><strong><a href="scenario_wise_rec_a_multi_scenario_recommendation_benchmark/">Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个多场景推荐统一基准，整合 6 个公开数据集与 12 个 MSR 模型，统一数据处理/训练/评测管线并附工业广告数据验证。</p>
</dd>
</dl>
