---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**WWW (TheWebConf) 2026** · **8** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2022 (3)](../../2022/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2023 (2)](../../2023/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2024 (4)](../../2024/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2022 (1)](../../../ACM_MM/2022/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/cross_domain_multi_scenario/) · [CIKM 2021 (4)](../../../CIKM/2021/cross_domain_multi_scenario/) · [CIKM 2022 (4)](../../../CIKM/2022/cross_domain_multi_scenario/) · [CIKM 2023 (2)](../../../CIKM/2023/cross_domain_multi_scenario/) · [CIKM 2024 (2)](../../../CIKM/2024/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (5)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2023 (2)](../../../DASFAA/2023/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ECIR 2023 (3)](../../../ECIR/2023/cross_domain_multi_scenario/) · [ICDE 2022 (1)](../../../ICDE/2022/cross_domain_multi_scenario/) · [ICDE 2023 (1)](../../../ICDE/2023/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2023 (1)](../../../ICDM/2023/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2021 (2)](../../../IJCAI/2021/cross_domain_multi_scenario/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/cross_domain_multi_scenario/) · [KDD 2021 (1)](../../../KDD/2021/cross_domain_multi_scenario/) · [KDD 2022 (1)](../../../KDD/2022/cross_domain_multi_scenario/) · [KDD 2023 (1)](../../../KDD/2023/cross_domain_multi_scenario/) · [KDD 2024 (1)](../../../KDD/2024/cross_domain_multi_scenario/) · [KDD 2025 (5)](../../../KDD/2025/cross_domain_multi_scenario/) · [KDD 2026 (7)](../../../KDD/2026/cross_domain_multi_scenario/) · [RecSys 2024 (2)](../../../RecSys/2024/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2021 (1)](../../../SIGIR/2021/cross_domain_multi_scenario/) · [SIGIR 2023 (1)](../../../SIGIR/2023/cross_domain_multi_scenario/) · [SIGIR 2024 (5)](../../../SIGIR/2024/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (5)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [ACM TKDD 2022 (1)](../../../TKDD/2022/cross_domain_multi_scenario/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/cross_domain_multi_scenario/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/cross_domain_multi_scenario/) · [ACM TORS 2026 (1)](../../../TORS/2026/cross_domain_multi_scenario/) · [WSDM 2022 (2)](../../../WSDM/2022/cross_domain_multi_scenario/) · [WSDM 2024 (4)](../../../WSDM/2024/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/)

🔥 **高频主题：** WWW 2026 ×7

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域因果建模&quot;,&quot;部分标签学习&quot;,&quot;门控特征融合&quot;]"><strong><a href="causality_enhancement_for_cross_domain_recommendation/">Causality Enhancement for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将跨域推荐重构为因果图，用相似度启发式构造因果感知数据并推导无偏部分标签因果损失，作为模型无关插件在四个域与线上 A/B 全面涨点</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;LoRA 方向分解&quot;,&quot;个性化聚合权重&quot;]"><strong><a href="fedecider_an_llm_based_framework_for_federated_cross_domain_recommendation/">FeDecider: An LLM-Based Framework for Federated Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>联邦跨域 LLM 推荐中，服务器只广播 LoRA 更新的方向分量以消除尺度噪声，客户端用可学习个性化权重做数据感知融合，Goodreads 平均 Hit@5 提升 3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端生成式&quot;,&quot;Flow Matching&quot;,&quot;高斯混合先验&quot;,&quot;多域序列推荐&quot;]"><strong><a href="gaussian_mixture_flow_matching_with_domain_alignment_for_multi_domain_seque/">Gaussian Mixture Flow Matching with Domain Alignment for Multi-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双掩码单骨干 Transformer 解耦域不变/域特定意图，再以域对齐先验引导高斯混合流匹配生成多峰转移轨迹，JD/Amazon 1:999 采样排序下 NDCG@5 最高提升 44%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;数据重生成&quot;,&quot;对比解码&quot;,&quot;序列推荐&quot;]"><strong><a href="generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;ODE 免训练&quot;]"><strong><a href="s2cdr_smoothing_sharpening_process_model_for_cross_domain_recommendation/">S2CDR: Smoothing-Sharpening Process Model for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用无噪声热方程平滑+锐化ODE替代扩散模型做跨域冷启动，免训练即在三场景六跨域对上 HR/NDCG 平均提升 7.07%/7.42%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;LLM 语义增强&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="semacdr_llm_powered_transferable_semantics_for_cross_domain_sequential_reco/">SemaCDR: LLM-Powered Transferable Semantics for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成领域无关语义并与领域特定表征对比对齐，配合跨域行为融合与自适应加权，在 Amazon 六个跨域序列场景 HR@10 最高提升 14.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域公平性&quot;,&quot;信息论增益重分配&quot;]"><strong><a href="the_double_edged_sword_of_knowledge_transfer_diagnosing_and_curing_fairness/">The Double-Edged Sword of Knowledge Transfer: Diagnosing and Curing Fairness Pathologies in Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论揭示跨域推荐中“差异迁移”与“信息增益不均”两类公平性病理，提出模型无关的 CDFA 框架，在 Tenrec QB/QK 上 UGF 最高降 98.97% 且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;模型合并 (Model Merging)&quot;,&quot;LoRA 混合域交叉训练&quot;]"><strong><a href="weaverec_an_llm_based_cross_domain_sequential_recommendation_framework_with/">WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「源域+目标域」混合数据交叉训练多个 LoRA 再权重平均合并，既避免直接数据合并的梯度冲突，又让合并成员均强于目标域，实现零额外推理开销的跨域序列推荐。</p>
</dd>
</dl>
