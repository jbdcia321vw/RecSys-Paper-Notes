---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**KDD 2021** · **5** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [KDD 2022 (2)](../../2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [CIKM 2021 (5)](../../../CIKM/2021/ctr_feature_interaction/) · [CIKM 2022 (9)](../../../CIKM/2022/ctr_feature_interaction/) · [CIKM 2023 (13)](../../../CIKM/2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../../CIKM/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../../CIKM/2026/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2021 (5)](../../../SIGIR/2021/ctr_feature_interaction/) · [SIGIR 2022 (6)](../../../SIGIR/2022/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2021 (3)](../../../WWW/2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../../WWW/2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;数值特征嵌入&quot;,&quot;软离散化&quot;,&quot;元嵌入&quot;]"><strong><a href="an_embedding_learning_framework_for_numerical_features_in_ctr_prediction/">An Embedding Learning Framework for Numerical Features in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AutoDis 用字段元嵌入 + 可微软离散化 + 加权聚合，为数值特征端到端学习“连续但可区分”的嵌入，三数据集 AUC 提升 0.17–0.23%，线上 CTR +2.1%、eCPM +2.7%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;图神经网络Embedding增强&quot;,&quot;特征稀疏&quot;,&quot;课程学习&quot;]"><strong><a href="dual_graph_enhanced_embedding_neural_network_for_ctr_prediction/">Dual Graph enhanced Embedding Neural Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用属性图+协同图双图卷积精炼Embedding，以分治式字段卷积与课程学习式组织学习缓解特征/行为稀疏，Alimama AUC 提升 8.83%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多任务学习&quot;,&quot;序列依赖&quot;,&quot;信息迁移&quot;]"><strong><a href="modeling_the_sequential_dependence_among_audience_multi_step_conversions_wi/">Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>AITM 用自适应信息迁移模块在输出层向量空间建模多步转化的顺序依赖，配合行为期望校准损失，在美团信用卡广告与 Ali-CCP 上均取得稳定 AUC 提升，线上端到端转化率显著改善。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排+拍卖机制设计&quot;,&quot;可微排序 NeuralSort&quot;,&quot;端到端自动机制设计&quot;]"><strong><a href="neural_auction_end_to_end_learning_of_auction_mechanisms_for_e_commerce_adv/">Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出可微深度神经拍卖 DNA，以 DeepSet 集合编码、部分单调 MIN-MAX 打分与 NeuralSort 松弛排序，在淘宝离线与在线 A/B 中显著提升多目标拍卖指标。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;广告主侧策略推荐（出价/定向决策）&quot;,&quot;Contextual Bandit&quot;,&quot;Dropout as Thompson Sampling&quot;,&quot;偏好向量学习&quot;]"><strong><a href="we_know_what_you_want_an_advertising_strategy_recommender_system_for_online/">We Know What You Want: An Advertising Strategy Recommender System for Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 MLP 将广告主特征映射为 KPI 偏好向量 w，以 σ(w^T·v) 预测策略采纳率，Dropout 近似 Thompson Sampling 平衡探索，仿真累计后悔较随机策略降 61.2%</p>
</dd>
</dl>
