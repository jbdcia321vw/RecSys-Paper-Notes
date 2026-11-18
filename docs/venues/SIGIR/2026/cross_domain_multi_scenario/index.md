---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**SIGIR 2026** · **4** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [SIGIR 2025 (6)](../../2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (4)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (4)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cross_domain_multi_scenario/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Double Q-learning" aria-pressed="false">Double Q-learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM数据增广" aria-pressed="false">LLM数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双阶段训练" aria-pressed="false">双阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实对比学习" aria-pressed="false">反事实对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="场景门控对比学习" aria-pressed="false">场景门控对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多市场CTR" aria-pressed="false">多市场CTR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知语义" aria-pressed="false">时间感知语义<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差向量量化" aria-pressed="false">残差向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经 ODE" aria-pressed="false">神经 ODE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;神经 ODE&quot;,&quot;时间感知语义&quot;,&quot;反事实对比学习&quot;]"><strong><a href="bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用神经 ODE 解耦长短期行为、LLM 离散时间 token 反事实增强语义、门控自适应域迁移，1:999 采样排序下 MRR 最高提升 30.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;残差向量量化&quot;,&quot;多市场CTR&quot;]"><strong><a href="fedmm_federated_collaborative_signal_quantization_for_multi_market_ctr_pred/">FedMM: Federated Collaborative Signal Quantization for Multi-Market CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双层 RQ-VAE 码本作联邦传输介质，对齐跨市场不相交 ID 空间，AUC 全面超越本地/联邦基线且通信量降 20×。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;双阶段训练&quot;,&quot;LLM数据增广&quot;]"><strong><a href="llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可迁移物品增广+双阶段训练+域感知画像，缓解CDSR域不平衡与域迁移难题，全库排序下H@10最高提升26.96%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Double Q-learning&quot;,&quot;场景门控对比学习&quot;]"><strong><a href="reinforcing_user_interest_evolution_in_multi_scenario_learning_for_recommen/">Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Double Q-learning 估计场景级行为置信度作为门控，加权三元组对比损失以抑制多场景负迁移，KuaiSAR 全库排序 N@20 提升约 3.3 倍。</p>
</dd>
</dl>
  </div>
</div>
