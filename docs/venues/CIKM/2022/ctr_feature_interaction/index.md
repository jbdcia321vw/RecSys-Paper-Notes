---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**CIKM 2022** · **9** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [CIKM 2021 (5)](../../2021/ctr_feature_interaction/) · [CIKM 2023 (13)](../../2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [KDD 2021 (5)](../../../KDD/2021/ctr_feature_interaction/) · [KDD 2022 (2)](../../../KDD/2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2021 (5)](../../../SIGIR/2021/ctr_feature_interaction/) · [SIGIR 2022 (6)](../../../SIGIR/2022/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2021 (3)](../../../WWW/2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../../WWW/2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构图神经网络&quot;,&quot;双查询注意力融合&quot;,&quot;距离协方差&quot;]"><strong><a href="aminergnn_heterogeneous_graph_neural_network_for_paper_click_through_rate_p/">AMinerGNN: Heterogeneous Graph Neural Network for Paper Click-through Rate Prediction with Fusion Query</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建 user-paper-keyword 异构图统一表征，用距离协方差注意力动态融合 userID 与 keyword 双查询，AMiner_KRS 上 AUC 0.899（较 MGDSPR +3.2%），线上 UV-CTR 提升 9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;图检索中心节点&quot;,&quot;多场景短期意图&quot;,&quot;个性化门控融合&quot;]"><strong><a href="graph_based_long_term_and_short_term_interest_model_for_click_through_rate_/">Graph Based Long-Term And Short-Term Interest Model for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用图检索结构替代长行为截断/硬类目过滤，结合多场景 GRU 短期意图与个性化门控融合，离线 AUC 达 0.9830，线上 CTR +4.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长短期兴趣解耦&quot;,&quot;自监督对比学习&quot;]"><strong><a href="hierarchically_fusing_long_and_short_term_user_interests_for_click_through_/">Hierarchically Fusing Long and Short-Term User Interests for Click-Through Rate Prediction in Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HIFN 以查询/目标/因果三路编码短期兴趣、注意力聚合长期兴趣、门控自适应融合并用自监督代理对比解耦长短期，Fliggy 线上 CTR +5.13%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;早期特征交互&quot;,&quot;对比学习正则&quot;]"><strong><a href="inttower_the_next_generation_of_two_tower_model_for_pre_ranking_system/">IntTower: the Next Generation of Two-Tower Model for Pre-Ranking System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Light-SE 特征选择、FE-Block 早期细粒度交互与对比交互正则，在保持双塔解耦部署的前提下逼近单塔精排效果。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Hawkes Process&quot;,&quot;Hierarchical RNN&quot;,&quot;Entire Space Multi-Task&quot;]"><strong><a href="multi_scale_user_behavior_network_for_entire_space_multi_task_learning/">Multi-Scale User Behavior Network for Entire Space Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层循环网络 HEROES 用 Hawkes 过程与门控跨层通道，在 observation→click→conversion 全空间多尺度建模用户行为，CVR AUC 最高相对提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;召回/精排嵌入层&quot;,&quot;嵌入表剪枝&quot;,&quot;神经架构搜索&quot;]"><strong><a href="optembed_learning_optimal_embedding_table_for_click_through_rate_prediction/">OptEmbed: Learning Optimal Embedding Table for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可学习场级阈值剪冗余嵌入 + 均匀采样超网与进化搜索定场级维度，在 Criteo/Avazu/KDD12 上省 39%–97% 参数且 AUC 最高 +0.15%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;LSH哈希采样&quot;,&quot;长序列用户兴趣建模&quot;,&quot;端到端训练&quot;]"><strong><a href="sampling_is_all_you_need_on_modeling_long_term_user_behaviors_for_ctr_predi/">Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用多轮 SimHash 碰撞概率替代 target attention，直接 gather 同签名行为构造长序列用户兴趣，AUC 媲美全注意力且训练快 5–11×，美团线上升 +2.98% CTR。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长用户行为序列建模&quot;,&quot;稀疏注意力记忆网络&quot;,&quot;实时推理效率&quot;]"><strong><a href="sparse_attentive_memory_network_for_click_through_rate_prediction_with_long/">Sparse Attentive Memory Network for Click-through Rate Prediction with Long Sequences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以目标物品为 Query、千长行为序列为知识库，用目标+记忆双查询稀疏注意力与 O(1) 次迭代记忆更新，实现 O(L) 复杂度长序列 CTR 建模，线上 A/B 提升 CTR 7.30%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;对比学习视觉编码&quot;,&quot;样本选择偏差去偏&quot;]"><strong><a href="visual_encoding_and_debiasing_for_ctr_prediction/">Visual Encoding and Debiasing for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用自监督+点击监督两阶段对比学习微调视觉编码器，并在 CTR 塔内加对比去偏网络与门控融合，淘宝十亿级数据 AUC +6.8%、线上 CTR +4~5%。</p>
</dd>
</dl>
