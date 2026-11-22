---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**KDD 2026** · **7** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [KDD 2021 (1)](../../2021/cross_domain_multi_scenario/) · [KDD 2022 (1)](../../2022/cross_domain_multi_scenario/) · [KDD 2023 (1)](../../2023/cross_domain_multi_scenario/) · [KDD 2024 (1)](../../2024/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/cross_domain_multi_scenario/) · [CIKM 2021 (4)](../../../CIKM/2021/cross_domain_multi_scenario/) · [CIKM 2022 (4)](../../../CIKM/2022/cross_domain_multi_scenario/) · [CIKM 2023 (2)](../../../CIKM/2023/cross_domain_multi_scenario/) · [CIKM 2024 (2)](../../../CIKM/2024/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (5)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2023 (2)](../../../DASFAA/2023/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ECIR 2023 (3)](../../../ECIR/2023/cross_domain_multi_scenario/) · [ICDE 2022 (1)](../../../ICDE/2022/cross_domain_multi_scenario/) · [ICDE 2023 (1)](../../../ICDE/2023/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2023 (1)](../../../ICDM/2023/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/cross_domain_multi_scenario/) · [RecSys 2024 (2)](../../../RecSys/2024/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2021 (1)](../../../SIGIR/2021/cross_domain_multi_scenario/) · [SIGIR 2023 (1)](../../../SIGIR/2023/cross_domain_multi_scenario/) · [SIGIR 2024 (5)](../../../SIGIR/2024/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (5)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [ACM TKDD 2022 (1)](../../../TKDD/2022/cross_domain_multi_scenario/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/cross_domain_multi_scenario/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/cross_domain_multi_scenario/) · [ACM TORS 2026 (1)](../../../TORS/2026/cross_domain_multi_scenario/) · [WSDM 2022 (2)](../../../WSDM/2022/cross_domain_multi_scenario/) · [WSDM 2024 (4)](../../../WSDM/2024/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2024 (4)](../../../WWW/2024/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/cross_domain_multi_scenario/)

🔥 **高频主题：** 模型合并 ×2 · 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型合并" aria-pressed="false">模型合并<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端序列推荐&quot;,&quot;扩散模型&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="align_for_fusion_harmonizing_triple_preferences_via_dual_oriented_diffusion/">Align-for-Fusion: Harmonizing Triple Preferences via Dual-oriented Diffusion for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出align-for-fusion：检索混合域行为噪声并对源/目标双域扩散，四设置Last-item排序HR@10最高+8.22%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;原子意图缓存&quot;,&quot;目标感知意图检索&quot;]"><strong><a href="atomic_intent_reasoning_bringing_llm_semantics_to_industrial_cross_domain_r/">Atomic Intent Reasoning: Bringing LLM Semantics to Industrial Cross-Domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 意图推理离线化为原子行为-意图对缓存，在线经意图树构建与目标感知检索实现约 400× 加速，快手 A/B 测试 GMV +3.446%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Tokenization&quot;,&quot;多任务学习&quot;]"><strong><a href="mdl_a_unified_multi_distribution_learner_in_large_scale_industrial_recommen/">MDL: A Unified Multi-Distribution Learner in Large-scale Industrial Recommendation through Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将场景/任务Token化并与特征逐层交叉注意，在抖音搜索0.5B模型上QAUC相对提升0.23–0.77%，线上LT30 +0.0626%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;模型合并&quot;,&quot;任务向量&quot;]"><strong><a href="mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多域多目标MDMO&quot;,&quot;组合合并与知识蒸馏&quot;]"><strong><a href="meta_lattice_model_space_redesign_for_cost_effective_industry_scale_ads_rec/">Meta Lattice: Model Space Redesign for Cost-Effective Industry-Scale Ads Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以模型空间重构统一碎片化广告组合，融合归因窗口混合、帕累托特征选择与推理期知识迁移，线上Topline +10%、容量省20%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;LLM Persona 建模&quot;,&quot;Doppelganger 跨域迁移&quot;]"><strong><a href="multi_tap_multi_criteria_target_adaptive_persona_modeling_for_cross_domain_/">Multi-TAP: Multi-criteria Target Adaptive Persona Modeling for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 将用户偏好拆成五类语义 Persona 表征域内异质性，再用 doppelganger 交叉注意力做目标域自适应的选择性迁移；Amazon 时间切分全库排序下 HR@5 最高提升 36.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;模型合并&quot;,&quot;锐度感知优化&quot;,&quot;LoRA 重参数化&quot;]"><strong><a href="sharpness_aware_model_merging_with_salience_recovery_for_llm_based_cross_do/">Sharpness-aware Model Merging with Salience Recovery for LLM-based Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用锐度感知平坦极小值对齐 + 重尾重参数化恢复显著偏好信号，破解 LLM 跨域序列推荐模型合并中的参数冲突与多域饱和瓶颈。</p>
</dd>
</dl>
