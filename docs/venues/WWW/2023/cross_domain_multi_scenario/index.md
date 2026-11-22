---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**WWW (TheWebConf) 2023** · **2** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2022 (3)](../../2022/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2024 (4)](../../2024/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (8)](../../2026/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/cross_domain_multi_scenario/) · [CIKM 2021 (4)](../../../CIKM/2021/cross_domain_multi_scenario/) · [CIKM 2022 (4)](../../../CIKM/2022/cross_domain_multi_scenario/) · [CIKM 2023 (2)](../../../CIKM/2023/cross_domain_multi_scenario/) · [CIKM 2024 (2)](../../../CIKM/2024/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (5)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2023 (2)](../../../DASFAA/2023/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ECIR 2023 (3)](../../../ECIR/2023/cross_domain_multi_scenario/) · [ICDE 2022 (1)](../../../ICDE/2022/cross_domain_multi_scenario/) · [ICDE 2023 (1)](../../../ICDE/2023/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2023 (1)](../../../ICDM/2023/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/cross_domain_multi_scenario/) · [KDD 2021 (1)](../../../KDD/2021/cross_domain_multi_scenario/) · [KDD 2022 (1)](../../../KDD/2022/cross_domain_multi_scenario/) · [KDD 2023 (1)](../../../KDD/2023/cross_domain_multi_scenario/) · [KDD 2024 (1)](../../../KDD/2024/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (7)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2024 (2)](../../../RecSys/2024/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2021 (1)](../../../SIGIR/2021/cross_domain_multi_scenario/) · [SIGIR 2023 (1)](../../../SIGIR/2023/cross_domain_multi_scenario/) · [SIGIR 2024 (5)](../../../SIGIR/2024/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (5)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [ACM TKDD 2022 (1)](../../../TKDD/2022/cross_domain_multi_scenario/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/cross_domain_multi_scenario/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/cross_domain_multi_scenario/) · [ACM TORS 2026 (1)](../../../TORS/2026/cross_domain_multi_scenario/) · [WSDM 2022 (2)](../../../WSDM/2022/cross_domain_multi_scenario/) · [WSDM 2024 (4)](../../../WSDM/2024/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;持续迁移学习&quot;,&quot;双塔适配器&quot;,&quot;GLU 门控&quot;]"><strong><a href="continual_transfer_learning_for_cross_domain_click_through_rate_prediction_/">Continual Transfer Learning for Cross-Domain Click-Through Rate Prediction at Taobao</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 CTL 新任务与双塔 CTNet，冻结演进中的源域塔、持续训练目标塔并用 GLU 适配器做逐层注入，仅用增量数据即在淘宝 A→B/A→C 提升 GAUC 1.0%/3.6%，线上 CTR +2.5%/+12.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域异构图消息传递&quot;,&quot;用户兴趣对齐&quot;,&quot;梯度对齐&quot;]"><strong><a href="cross_domain_recommendation_via_user_interest_alignment/">Cross-domain recommendation via user interest alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>WWW 2023｜跨域多场景推荐：构建统一跨域异构图并重定义消息传递，再用对比聚类与梯度对齐约束重叠用户兴趣一致；双域留一法 1:99 采样排序下最高相对提升 10.23%。</p>
</dd>
</dl>
