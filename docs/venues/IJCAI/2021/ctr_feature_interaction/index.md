---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**IJCAI (RecSys Track) 2021** · **2** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [IJCAI (RecSys Track) 2022 (1)](../../2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [CIKM 2021 (5)](../../../CIKM/2021/ctr_feature_interaction/) · [CIKM 2022 (9)](../../../CIKM/2022/ctr_feature_interaction/) · [CIKM 2023 (13)](../../../CIKM/2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../../CIKM/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../../CIKM/2026/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2021 (5)](../../../KDD/2021/ctr_feature_interaction/) · [KDD 2022 (2)](../../../KDD/2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2021 (5)](../../../SIGIR/2021/ctr_feature_interaction/) · [SIGIR 2022 (6)](../../../SIGIR/2022/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2021 (3)](../../../WWW/2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../../WWW/2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×2 · IJCAI 2021 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;神经架构搜索&quot;,&quot;三阶段推荐模型&quot;,&quot;One-shot 随机搜索&quot;]"><strong><a href="ameir_automatic_behavior_modeling_interaction_exploration_and_mlp_investiga/">AMEIR: Automatic Behavior Modeling, Interaction Exploration and MLP Investigation in the Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出三阶段搜索空间与三步 one-shot 随机搜索，自动联合行为建模、特征交互与 MLP；Alimama AUC 0.6383，Criteo/Avazu 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉算子&quot;,&quot;用户行为建模&quot;,&quot;AutoML架构搜索&quot;]"><strong><a href="deep_learning_for_click_through_rate_estimation/">Deep Learning for Click-Through Rate Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>深度 CTR 综述：以显式交叉算子、用户行为建模、自动化架构搜索三条主线梳理浅层到深度模型演进与工业落地路线。</p>
</dd>
</dl>
