# 🎯 CTR/CVR 预估与特征交叉

**CIKM 2026** · **5** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [CIKM 2025 (5)](../../2025/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Budgeted Allocation" aria-pressed="false">Budgeted Allocation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Funnel Composition" aria-pressed="false">Funnel Composition<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM语义推理" aria-pressed="false">LLM语义推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Uplift Modeling" aria-pressed="false">Uplift Modeling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Uplift建模" aria-pressed="false">Uplift建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全空间样本建模" aria-pressed="false">全空间样本建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏CTR" aria-pressed="false">去偏CTR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;竞价感知建模&quot;,&quot;多阶段一致性&quot;]"><strong><a href="bidding_aware_retrieval_for_multi_stage_consistency_in_online_advertising/">Bidding-Aware Retrieval for Multi-Stage Consistency in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将竞价信号注入召回打分，用单调性约束+多任务蒸馏+任务注意力精炼实现召回-精排 eCPM 一致性，线上收入 +4.32%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;转化归因&quot;,&quot;触点选择&quot;,&quot;LLM语义推理&quot;]"><strong><a href="can_large_language_models_identify_meaningful_touchpoints_in_conversion_att/">Can Large Language Models Identify Meaningful Touchpoints in Conversion Attribution?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建SILVA触点选择基准，揭示隐式触点语义鸿沟，系统评测8个LLM的成对/列表提示能力，并以LLM归因标签增强工业CVR训练获+0.35pp GAUC。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Uplift Modeling&quot;,&quot;Funnel Composition&quot;,&quot;Budgeted Allocation&quot;]"><strong><a href="funnelcausalnet_funnel_aware_joint_conversion_revenue_uplift_for_multi_tier/">FunnelCausalNet: Funnel-aware Joint Conversion-Revenue Uplift for Multi-tier Coupon Allocation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以漏斗恒等式 μ_gmv=μ_conv·μ_val 耦合双头 uplift 估计，配 RCT 锚定拉格朗日预算分配与 Bonferroni 联合保形审计带，工业多档券 RCT 上 7/7 锚点 ΔROI 最优。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多值因果干预&quot;,&quot;去偏CTR&quot;,&quot;Uplift建模&quot;]"><strong><a href="jointly_optimizing_debiased_ctr_and_uplift_for_coupons_marketing_a_unified_/">Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>UniMVT 用 MoE 解耦混淆表征 + logit 空间单调线性假设，统一估计去偏 Base CTR 与多值券面额 Uplift，线上收入 +18.14%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;全空间样本建模&quot;,&quot;异构特征交互&quot;,&quot;数据-模型协同缩放&quot;]"><strong><a href="uniscale_synergistic_entire_space_data_and_model_scaling_for_search_ranking/">UniScale: Synergistic Entire Space Data and Model Scaling for Search Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据-架构协同缩放框架：ES³全空间样本系统扩展训练信号，HHSFT异构分层Transformer吸收跨域分布，离线AUC+1.14%、线上GMV+2.04%。</p>
</dd>
</dl>
