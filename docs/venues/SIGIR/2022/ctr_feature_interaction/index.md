---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2022** · **6** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [SIGIR 2021 (5)](../../2021/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [CIKM 2021 (5)](../../../CIKM/2021/ctr_feature_interaction/) · [CIKM 2022 (9)](../../../CIKM/2022/ctr_feature_interaction/) · [CIKM 2023 (13)](../../../CIKM/2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../../CIKM/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../../CIKM/2026/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [KDD 2021 (5)](../../../KDD/2021/ctr_feature_interaction/) · [KDD 2022 (2)](../../../KDD/2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2021 (3)](../../../WWW/2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../../WWW/2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">4</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长时行为序列建模&quot;,&quot;对抗训练与行为去噪&quot;]"><strong><a href="adversarial_filtering_modeling_on_long_term_user_behavior_sequences_for_cli/">Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用分层聚合压缩重复行为、自注意力打分门选 top-k 行为，并以对抗训练（IEU 最小化、BSU 最大化剩余序列损失）剔除长序列噪声行为，工业集 AUC +0.0047、线上 CTR +4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征精炼&quot;,&quot;上下文感知表示&quot;,&quot;比特级门控&quot;]"><strong><a href="enhancing_ctr_prediction_with_context_aware_feature_representation_learning/">Enhancing CTR Prediction with Context-Aware Feature Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>FRNet 通过比特级门控融合原始特征与互补特征，为每条样本生成上下文感知的特征表示，即插即用并显著提升多种 CTR 模型。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;属性树聚合&quot;,&quot;层次意图建模&quot;]"><strong><a href="hien_hierarchical_intention_embedding_network_for_click_through_rate_predic/">HIEN: Hierarchical Intention Embedding Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用属性树自底向上聚合 + 层次注意力双向意图建模精修Embedding，作为即插即用输入模块，在阿里/Tmall/腾讯三个数据集上AUC较DG-ENN最高提升0.86%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构信息网络&quot;,&quot;Graph-Masked Transformer&quot;,&quot;一致性正则&quot;]"><strong><a href="neighbour_interaction_based_click_through_rate_prediction_via_graph_masked_/">Neighbour Interaction based Click-Through Rate Prediction via Graph-masked Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 CTR 重建为异构信息图上 user-item 局部邻域连边预测，用四类交互图 + 头分组图掩码 Transformer 编码邻域，微信线上 CTR +21.9%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;概率校准&quot;,&quot;保序回归&quot;,&quot;自适应分箱&quot;]"><strong><a href="posterior_probability_matters_doubly_adaptive_calibration_for_neural_predic/">Posterior Probability Matters: Doubly-Adaptive Calibration for Neural Predictions in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 AdaCalib，用后验统计引导的分段线性保序函数族做字段级概率校准，并以频率驱动的自适应分箱匹配字段取值，Field-RCE 相对降 10%~70%，线上 CVR +5.05%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR预估与特征交叉&quot;,&quot;召回&quot;,&quot;嵌入维度搜索&quot;,&quot;单次剪枝&quot;,&quot;混合维度嵌入&quot;]"><strong><a href="single_shot_embedding_dimension_search_in_recommender_system/">Single-shot Embedding Dimension Search in Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SSEDS 用一次前向-反向的梯度显著性打分完成字段级嵌入维度剪枝，90% 参数下 AUC 反升，并在微信订阅号 A/B 提升约 4% CTR。</p>
</dd>
</dl>
