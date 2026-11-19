---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**KDD 2024** · **7** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [KDD 2025 (8)](../../2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [RecSys 2024 (4)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR预估与特征交叉" aria-pressed="false">CTR/CVR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨域终身序列&quot;,&quot;对比学习&quot;]"><strong><a href="cross_domain_lifelong_sequential_modeling_for_online_click_through_rate_pre/">Cross-Domain LifeLong Sequential Modeling for Online Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>跨域终身序列CTR；CRP对比学习对齐跨域物品表征，LAP三级注意力金字塔；工业A/B CTR +2.93%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排后校准/线上推理&quot;,&quot;Post-hoc Calibration&quot;,&quot;Basis Function Ensemble&quot;,&quot;Multi-Field Calibration&quot;]"><strong><a href="deep_ensemble_shape_calibration_multi_field_post_hoc_calibration_in_online_/">Deep Ensemble Shape Calibration: Multi-Field Post-hoc Calibration in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多字段后置校准解耦为基函数形状校准与全字段数值校准，免分箱提升稀疏字段数据利用率，工业线上 CVR +2.5%、GMV +4.0%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征选择&quot;,&quot;AUKC&quot;,&quot;基准评测&quot;]"><strong><a href="erase_benchmarking_feature_selection_methods_for_deep_recommender_systems/">ERASE: Benchmarking Feature Selection Methods for Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一评测 11 种特征选择方法×4 个 DRS backbone，提出 AUKC 稳定性指标，工业在线裁剪 30% 特征降 20% 延迟且效果不降。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;广告位分配/拍卖机制&quot;,&quot;自动机制设计&quot;,&quot;联合拍卖&quot;,&quot;JRegNet&quot;]"><strong><a href="joint_auction_in_the_online_advertising_market/">Joint Auction in the Online Advertising Market</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出门店-品牌联合拍卖与JRegNet，在合成/美团日志上以近DSIC/IR约束提升收入15.9%–19.8%，推理1–3ms。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排后广告拍卖/计费&quot;,&quot;OCPC&quot;,&quot;转化报告操纵&quot;,&quot;拍卖博弈&quot;]"><strong><a href="optimized_cost_per_click_in_online_advertising_a_theoretical_analysis/">Optimized Cost Per Click in Online Advertising: A Theoretical Analysis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用两广告主扩展式博弈比较 OCPC/CPC/CPA，揭示站外转化操纵与外部选项下的定价优势。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;排序损失&quot;,&quot;负样本梯度消失&quot;]"><strong><a href="understanding_the_ranking_loss_for_recommendation_with_sparse_user_feedback/">Understanding the Ranking Loss for Recommendation with Sparse User Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从负样本梯度消失视角解释辅助排序损失，Criteo_x1全样本AUC/BCE评测与腾讯线上GMV提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩分解&quot;,&quot;模型压缩&quot;]"><strong><a href="unified_low_rank_compression_framework_for_click_through_rate_prediction/">Unified Low-rank Compression Framework for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出AFM统一低秩压缩框架，对MLP输出特征与Embedding表做PCA低秩近似；在Criteo/Avazu/XYZ上以3–5×压缩实现更高AUC与1.35–2.7×吞吐。</p>
</dd>
</dl>
