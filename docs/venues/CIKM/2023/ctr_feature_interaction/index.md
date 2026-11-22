---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**CIKM 2023** · **13** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [CIKM 2021 (5)](../../2021/ctr_feature_interaction/) · [CIKM 2022 (9)](../../2022/ctr_feature_interaction/) · [CIKM 2024 (2)](../../2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../2025/ctr_feature_interaction/) · [CIKM 2026 (6)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/ctr_feature_interaction/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [DASFAA 2026 (2)](../../../DASFAA/2026/ctr_feature_interaction/) · [ICDE 2021 (1)](../../../ICDE/2021/ctr_feature_interaction/) · [ICDE 2022 (1)](../../../ICDE/2022/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDE 2026 (2)](../../../ICDE/2026/ctr_feature_interaction/) · [ICDM 2022 (1)](../../../ICDM/2022/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [KDD 2021 (5)](../../../KDD/2021/ctr_feature_interaction/) · [KDD 2022 (2)](../../../KDD/2022/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (14)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2021 (1)](../../../RecSys/2021/ctr_feature_interaction/) · [RecSys 2022 (1)](../../../RecSys/2022/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2021 (5)](../../../SIGIR/2021/ctr_feature_interaction/) · [SIGIR 2022 (6)](../../../SIGIR/2022/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (10)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [ACM TORS 2026 (1)](../../../TORS/2026/ctr_feature_interaction/) · [WSDM 2021 (4)](../../../WSDM/2021/ctr_feature_interaction/) · [WSDM 2022 (2)](../../../WSDM/2022/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WSDM 2026 (2)](../../../WSDM/2026/ctr_feature_interaction/) · [WWW (TheWebConf) 2021 (3)](../../../WWW/2021/ctr_feature_interaction/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/ctr_feature_interaction/) · [WWW (TheWebConf) 2023 (5)](../../../WWW/2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** CTR预估与特征交叉 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;增量更新&quot;,&quot;特征先验&quot;,&quot;贝叶斯模型先验&quot;,&quot;长尾物品&quot;]"><strong><a href="an_incremental_update_framework_for_online_recommenders_with_data_driven_pr/">An Incremental Update Framework for Online Recommenders with Data-Driven Prior</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用特征级 CTR 先验 (FP) 与贝叶斯输出级模型先验 (MP) 约束滑窗增量训练，流式切分评估下 Criteo/CIKM2019 AUC 超 ASMG，线上 CTR +1.99%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;展示上下文&quot;,&quot;位置偏置&quot;,&quot;用户行为建模&quot;]"><strong><a href="deep_context_interest_network_for_click_through_rate_prediction/">Deep Context Interest Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以点击级展示上下文与位置注意力建模用户兴趣，31天工业数据AUC 0.66475，线上CTR/RPM各+1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;属性感知激活&quot;,&quot;动态参数生成&quot;,&quot;相似性与多样性偏好&quot;]"><strong><a href="dpan_dynamic_preference_based_and_attribute_aware_network_for_relevant_reco/">DPAN: Dynamic Preference-based and Attribute-aware Network for Relevant Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>属性级双激活+双维压缩重表达分离相似/多样表征，条件动态参数生成融合，AUC 0.6098（RelaImpr +10.69%），线上 CTR +7.62%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Delayed Feedback&quot;,&quot;Entire Space Modeling&quot;,&quot;多任务辅助学习&quot;]"><strong><a href="entire_space_cascade_delayed_feedback_modeling_for_effective_conversion_rat/">Entire Space Cascade Delayed Feedback Modeling for Effective Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 ECVR 分解为 CVR×(1−RFR)，用全空间 CVRFR 任务加级联延迟辅助塔一并解决 DS/SSB/CDF，线上 ECVR +5.21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩特征交互&quot;,&quot;SENet位级注意力&quot;,&quot;模型压缩&quot;]"><strong><a href="fibinet_reducing_model_size_by_low_rank_feature_interaction_layer_for_ctr_p/">FiBiNet++: Reducing Model Size by Low Rank Feature Interaction Layer for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用内积双线性+低秩交互层与分组位级SENet+重构FiBiNet，非嵌入参数压缩12–16倍且AUC全面反超。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;实例级嵌入维度搜索&quot;,&quot;伯努利门与极化正则&quot;]"><strong><a href="ihas_instance_wise_hierarchical_architecture_search_for_deep_learning_recom/">iHAS: Instance-wise Hierarchical Architecture Search for Deep Learning Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用带极化正则的伯努利门为每个样本搜出字段级嵌入维度，再聚类并重训簇专属 DLRM；Avazu AUC 0.7815（+0.0038），推理最快。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多域学习&quot;,&quot;动态加权损失&quot;]"><strong><a href="masked_multi_domain_network_multi_type_and_multi_scenario_conversion_rate_p/">Masked Multi-Domain Network: Multi-Type and Multi-Scenario Conversion Rate Prediction with a Single Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用参数共享组合、自动掩码与动态加权损失，在单模型单数据集中实现多转化类型×多展示场景 CVR 预测，AUC 与线上 CVR/CPM 均显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排/出价&quot;,&quot;Bid Shading&quot;,&quot;Multi-task Learning&quot;]"><strong><a href="mebs_multi_task_end_to_end_bid_shading_for_multi_slot_display_advertising/">MEBS: Multi-task End-to-end Bid Shading for Multi-slot Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>面向多广告位展示广告，用多任务端到端出价折减最大化广告主盈余，离线 Surplus 与线上 GMV/ROI 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多哈希码本&quot;,&quot;交叉特征记忆&quot;]"><strong><a href="memonet_memorizing_all_cross_features_representations_efficiently_via_multi/">MemoNet: Memorizing All Cross Features&#x27; Representations Efficiently via Multi-Hash Codebook Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出多哈希码本HCNet与MemoNet，以独立记忆存储全部交叉特征，在三大CTR数据集和线上A/B均显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;自适应特征选择&quot;,&quot;混合专家(MoE)&quot;,&quot;重要性分数退火&quot;]"><strong><a href="mvfs_multi_view_feature_selection_for_recommender_system/">MvFS: Multi-view Feature Selection for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多子网+门控的 MoE 式控制器为每条样本自适应选择特征域，并以逐域 tanh 退火替代跨域重加权，在 Avazu/Criteo 四个 CTR 骨干上 AUC 全面超越 AdaFS。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;样本选择偏差 (SSB)&quot;,&quot;解耦表征&quot;,&quot;工业级部署&quot;]"><strong><a href="rec4ad_a_free_lunch_to_mitigate_sample_selection_bias_for_ads_ctr_predictio/">Rec4Ad: A Free Lunch to Mitigate Sample Selection Bias for Ads CTR Prediction in Taobao</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>借推荐样本作免费午餐，用伪样本映射+SABN+对抗对齐/去相关解耦系统混杂与用户兴趣，缓解广告CTR样本选择偏差，线上CTR +6.6%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;门控交叉网络&quot;,&quot;场级维度优化&quot;,&quot;高阶特征交叉&quot;]"><strong><a href="towards_deeper_lighter_and_interpretable_cross_network_for_ctr_prediction/">Towards Deeper, Lighter and Interpretable Cross Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>GDCN 用逐层信息门控筛选高阶交叉特征使深层交叉不退化，FDO 后验分配场级压缩维度，Criteo AUC 0.8161 且参数仅 23%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;生成式检索&quot;,&quot;稠密检索&quot;,&quot;查询改写&quot;]"><strong><a href="unified_generative_dense_retrieval_for_query_rewriting_in_sponsored_search/">Unified Generative &amp; Dense Retrieval for Query Rewriting in Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 CLOVER-Unity，将生成式与稠密检索共享编码器统一到赞助搜索查询改写，在 Bing 线上提升点击 0.89% 与收入 1.27%。</p>
</dd>
</dl>
