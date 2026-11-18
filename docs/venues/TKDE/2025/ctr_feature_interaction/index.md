# 🎯 CTR/CVR 预估与特征交叉

**IEEE TKDE 2025** · **2** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

🔥 **高频主题：** TKDE ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性校准" aria-pressed="false">不确定性校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="倾向分校准" aria-pressed="false">倾向分校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双模型协同训练" aria-pressed="false">双模型协同训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入参数效率" aria-pressed="false">嵌入参数效率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应特征选择" aria-pressed="false">自适应特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆倾向加权去偏" aria-pressed="false">逆倾向加权去偏<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自适应特征选择&quot;,&quot;嵌入参数效率&quot;,&quot;双模型协同训练&quot;]"><strong><a href="aefs_adaptive_early_feature_selection_for_deep_recommender_systems/">AEFS: Adaptive Early Feature Selection for Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 mini-AdaFS 辅助模型在嵌入层前做实例级自适应特征选择，配合双对齐损失，精度持平 AdaFS 且激活嵌入参数降 37.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;倾向分校准&quot;,&quot;逆倾向加权去偏&quot;,&quot;不确定性校准&quot;]"><strong><a href="uncertainty_calibration_for_counterfactual_propensity_estimation_in_recomme/">Uncertainty Calibration for Counterfactual Propensity Estimation in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示倾向分估计严重失准问题，提出以ECE为质量度量的模型无关校准框架，理论证明校准倾向分降低IPS偏差，在Coat/Yahoo/KuaiRand上显著提升CVR去偏效果。</p>
</dd>
</dl>
