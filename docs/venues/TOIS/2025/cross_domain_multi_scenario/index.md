---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**ACM TOIS 2025** · **3** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [ACM TOIS 2026 (1)](../../2026/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (4)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (4)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (4)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cross_domain_multi_scenario/)

🔥 **高频主题：** TOIS ×3 · 双目标跨域推荐 ×2

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双目标跨域推荐" aria-pressed="false">双目标跨域推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="FFT 频域滤波" aria-pressed="false">FFT 频域滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="一致性约束" aria-pressed="false">一致性约束<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好解耦" aria-pressed="false">偏好解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去混杂" aria-pressed="false">因果去混杂<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混杂因子解耦" aria-pressed="false">混杂因子解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义聚类" aria-pressed="false">语义聚类<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;因果去混杂&quot;,&quot;混杂因子解耦&quot;,&quot;双目标跨域推荐&quot;]"><strong><a href="causal_deconfounding_via_confounder_disentanglement_for_dual_target_cross_d/">Causal Deconfounding via Confounder Disentanglement for Dual-Target Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>显式解耦单域/跨域观测混杂因子，经后门调整保留正向效应并消除负向偏差，七数据集 HR@10 平均提升 6.17%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;偏好解耦&quot;,&quot;一致性约束&quot;,&quot;双目标跨域推荐&quot;]"><strong><a href="coherence_guided_preference_disentanglement_for_cross_domain_recommendation/">Coherence-guided Preference Disentanglement for Cross-domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用全局-局部图 + 一致性约束抽取跨域共享物品属性，并以流行度加权 Softmax 将共享用户偏好解耦为兴趣/从众，六组双目标 CDR 任务 HR@10 最高提升约 37%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;联邦学习&quot;,&quot;语义聚类&quot;,&quot;FFT 频域滤波&quot;]"><strong><a href="federated_semantic_learning_for_privacy_preserving_cross_domain_recommendat/">Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 FFMSR，用多层 PLM+MoE 提取语义、服务器端聚类融合跨域语义、FFT 滤波去噪，在完全非重叠联邦场景下 Full-Ranking 提升 1.4%–5.6%。</p>
</dd>
</dl>
  </div>
</div>
