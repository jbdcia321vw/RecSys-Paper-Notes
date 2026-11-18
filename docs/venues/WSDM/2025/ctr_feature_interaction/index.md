# 🎯 CTR/CVR 预估与特征交叉

**WSDM 2025** · **2** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="LLM解释生成" aria-pressed="false">LLM解释生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA三阶段迭代" aria-pressed="false">LoRA三阶段迭代<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PPO双奖励对齐" aria-pressed="false">PPO双奖励对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经架构搜索" aria-pressed="false">神经架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="融合学习" aria-pressed="false">融合学习<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM解释生成&quot;,&quot;PPO双奖励对齐&quot;,&quot;LoRA三阶段迭代&quot;]"><strong><a href="explainable_ctr_prediction_via_llm_reasoning/">Explainable CTR Prediction via LLM Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PPO 双奖励（LC 用户意图对齐 + IC 与 ID-CTR 模型对齐）把 LLM 解释生成嵌入 CTR 流程，免解释数据集，AUC 最高提升 18.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;融合学习&quot;,&quot;神经架构搜索&quot;,&quot;特征交叉&quot;]"><strong><a href="fusion_matters_learning_fusion_in_deep_click_through_rate_prediction_models/">Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将CTR模型的融合连接与融合算子统一建模为可微搜索空间，用one-shot联合选择替代手工stacked/parallel，三数据集AUC稳定超越EDCN与NASRec。</p>
</dd>
</dl>
