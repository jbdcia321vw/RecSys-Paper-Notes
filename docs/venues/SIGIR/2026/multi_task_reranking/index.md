# ⚖️ 多任务学习与列表级重排

**SIGIR 2026** · **4** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [SIGIR 2025 (4)](../../2025/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 多任务学习 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Target Attention" aria-pressed="false">Target Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级重排" aria-pressed="false">列表级重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="后验价值对齐" aria-pressed="false">后验价值对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线学习" aria-pressed="false">在线学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排列不变性" aria-pressed="false">排列不变性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差量化" aria-pressed="false">残差量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏监督对齐" aria-pressed="false">稀疏监督对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性 Boosting 归因" aria-pressed="false">线性 Boosting 归因<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;残差量化&quot;,&quot;Target Attention&quot;,&quot;知识蒸馏&quot;]"><strong><a href="equip_pre_ranking_with_target_attention_by_residual_quantization/">Equip Pre-ranking with Target Attention by Residual Quantization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差量化码本近似 Target Attention，将交互式建模引入粗排，AUC 0.799、线上 GMV +7.57%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;排列不变性&quot;,&quot;列表级重排&quot;]"><strong><a href="one_pass_any_order_position_invariant_listwise_reranking_for_llm_based_reco/">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用结构化注意力掩码+共享位置框架在架构层消除候选顺序依赖，单次前向实现近完美排列不变性，nDCG 仅微降。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;稀疏监督对齐&quot;,&quot;在线学习&quot;]"><strong><a href="towards_end_to_end_alignment_of_user_satisfaction_via_questionnaire_in_vide/">Towards End-to-End Alignment of User Satisfaction via Questionnaire in Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LoRA+MoE 独立参数通路隔离稀疏问卷满意度信号，并以在线 DPO 对齐主排序模型，离线 NDCG@5 提升约 3%，线上 Q-Satisfied 提升 0.5–0.9%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;后验价值对齐&quot;,&quot;线性 Boosting 归因&quot;]"><strong><a href="uniboost_global_coordination_with_value_alignment_for_fair_and_efficient_tr/">Uniboost: Global Coordination with Value Alignment for Fair and Efficient Traffic Allocation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用后验价值对齐把抽象混合分映射到有效完播率语义空间，再以独立线性 boosting 解耦多业务加权，线上 A/B 全指标正向。</p>
</dd>
</dl>
