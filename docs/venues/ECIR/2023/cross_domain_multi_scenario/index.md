---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**ECIR 2023** · **3** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/cross_domain_multi_scenario/) · [CIKM 2021 (4)](../../../CIKM/2021/cross_domain_multi_scenario/) · [CIKM 2022 (4)](../../../CIKM/2022/cross_domain_multi_scenario/) · [CIKM 2023 (2)](../../../CIKM/2023/cross_domain_multi_scenario/) · [CIKM 2024 (2)](../../../CIKM/2024/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (5)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2023 (2)](../../../DASFAA/2023/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ICDE 2022 (1)](../../../ICDE/2022/cross_domain_multi_scenario/) · [ICDE 2023 (1)](../../../ICDE/2023/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2023 (1)](../../../ICDM/2023/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/cross_domain_multi_scenario/) · [KDD 2021 (1)](../../../KDD/2021/cross_domain_multi_scenario/) · [KDD 2022 (1)](../../../KDD/2022/cross_domain_multi_scenario/) · [KDD 2023 (1)](../../../KDD/2023/cross_domain_multi_scenario/) · [KDD 2024 (1)](../../../KDD/2024/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (7)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2024 (2)](../../../RecSys/2024/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2021 (1)](../../../SIGIR/2021/cross_domain_multi_scenario/) · [SIGIR 2023 (1)](../../../SIGIR/2023/cross_domain_multi_scenario/) · [SIGIR 2024 (5)](../../../SIGIR/2024/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (5)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [ACM TKDD 2022 (1)](../../../TKDD/2022/cross_domain_multi_scenario/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/cross_domain_multi_scenario/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/cross_domain_multi_scenario/) · [ACM TORS 2026 (1)](../../../TORS/2026/cross_domain_multi_scenario/) · [WSDM 2022 (2)](../../../WSDM/2022/cross_domain_multi_scenario/) · [WSDM 2024 (4)](../../../WSDM/2024/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/cross_domain_multi_scenario/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排/CTR&quot;,&quot;多域推荐&quot;,&quot;图神经网络&quot;,&quot;域感知消息传递&quot;]"><strong><a href="exploiting_graph_structured_cross_domain_representation_for_multi_domain_re/">Exploiting Graph Structured Cross-Domain Representation for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用域感知图卷积与全局图池化建模多域交互图，CTR 负采样评测在 Amazon 2–13 域稳定超越 MDR/MTL 基线。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;市场嵌入 (Market Embedding)&quot;,&quot;跨市场推荐 CMR&quot;,&quot;元学习替代方案&quot;]"><strong><a href="market_aware_models_for_efficient_cross_market_recommendation/">Market-Aware Models for Efficient Cross-Market Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用市场嵌入 o_l 逐元素调制物品表征替代元学习，在 XMarket 上以 15% 训练时间取得或超越 MAML/FOREC 的 nDCG@10，全局建模下 6/7 市场夺冠。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;在线服务/召回&quot;,&quot;API-centric 平台&quot;,&quot;多域推荐系统&quot;]"><strong><a href="uptrendz_api_centric_real_time_recommendations_in_multi_domain_settings/">Uptrendz: API-Centric Real-time Recommendations in Multi-Domain Settings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 API 为中心、schema 驱动的实时多域推荐平台，用系统级/物品级双层域划分与场景化算法配置实现一套服务支撑多业务，MovieLens-100k 与创业辅导两域实证可部署性。</p>
</dd>
</dl>
