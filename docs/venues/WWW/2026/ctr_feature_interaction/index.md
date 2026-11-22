---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**WWW (TheWebConf) 2026** · **8** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2021 (3)](../../2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../2025/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [CIKM 2021 (5)](../../../CIKM/2021/ctr_feature_interaction/) · [CIKM 2022 (9)](../../../CIKM/2022/ctr_feature_interaction/) · [CIKM 2023 (13)](../../../CIKM/2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../../CIKM/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../../CIKM/2026/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [KDD 2021 (5)](../../../KDD/2021/ctr_feature_interaction/) · [KDD 2022 (2)](../../../KDD/2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2021 (5)](../../../SIGIR/2021/ctr_feature_interaction/) · [SIGIR 2022 (6)](../../../SIGIR/2022/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/)

🔥 **高频主题：** WWW 2026 ×5 · WWW ×3 · CTR预估与特征交叉 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;冷启动表征&quot;,&quot;对比学习&quot;]"><strong><a href="coins_semantic_ids_enhanced_cold_item_representation_for_click_through_rate/">COINS: Semantic Ids Enhanced Cold Item Representation for Click-through Rate Prediction in E-commerce Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RQ-OPQ 语义 ID 做门控双向迁移与残差对比学习，强化电商搜索冷启动物品表征；日志级 AUC 冷启 +0.93pp，线上订单量 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;延迟反馈建模&quot;,&quot;复购感知双分支&quot;,&quot;在线流式训练&quot;]"><strong><a href="delayed_feedback_modeling_for_post_click_gross_merchandise_volume_predictio/">Delayed Feedback Modeling for Post-Click Gross Merchandise Volume Prediction: Benchmark, Insights and Approaches</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个后点击GMV延迟反馈基准TRACE，提出复购感知双分支READER，在线流式训练下ACC相对提升2.19%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;字段级语义增强&quot;,&quot;LLM知识蒸馏&quot;,&quot;LoRA自监督微调&quot;]"><strong><a href="field_matters_a_lightweight_llm_enhanced_method_for_ctr_prediction/">Field Matters: A Lightweight LLM-enhanced Method for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出字段级 LLM 增强范式 LLaCTR：用自监督字段-特征微调蒸馏字段语义，再以 KL 对齐与字段交互矩阵增强特征表示与特征交叉，6 个骨干模型平均 AUC 相对提升 2.24%，训练成本比现有 LLM 增强方法低 10–100 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;生成式用户意图&quot;,&quot;语义ID兴趣群组&quot;,&quot;召回排序一致性&quot;]"><strong><a href="genci_generative_modeling_of_user_interest_shift_via_cohort_based_intent_le/">GenCI: Generative Modeling of User Interest Shift via Cohort-based Intent Learning for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用RQ-VAE语义ID+NTP生成兴趣群组，再以分层候选感知注意力注入精排，弥合召回-排序意图鸿沟，三数据集AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;级联延迟反馈&quot;,&quot;净转化率&quot;,&quot;在线持续学习&quot;]"><strong><a href="modeling_cascaded_delay_feedback_for_online_net_conversion_rate_prediction_/">Modeling Cascaded Delay Feedback for Online Net Conversion Rate Prediction: Benchmark, Insights and Solutions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个淘宝净转化率流式基准 CASCADE，提出 CVR-RFR 级联+分阶段去偏+延迟感知排序损失的 TESLA，NetCVR 的 RI-AUC 绝对提升 12.41%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;统一Transformer骨干&quot;,&quot;混合参数化&quot;,&quot;KV缓存&quot;]"><strong><a href="onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;空间外部性&quot;,&quot;机制设计&quot;,&quot;近似算法&quot;]"><strong><a href="online_advertising_with_spatial_interactions/">Online Advertising with Spatial Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次将广告位视为度量空间点，用空间折扣函数建模邻近广告互斥外部性，给出 1/18 近似真实机制、2D 欧氏 PTAS 及乘积距离模型的 m^{1-ε} 硬度。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长期行为序列建模&quot;]"><strong><a href="unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
</dl>
