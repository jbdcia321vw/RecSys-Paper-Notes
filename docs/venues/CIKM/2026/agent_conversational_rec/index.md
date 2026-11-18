# 🗣️ 对话推荐与智能体仿真

**CIKM 2026** · **5** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [CIKM 2025 (3)](../../2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR 2025 (7)](../../../SIGIR/2025/agent_conversational_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** 生成式推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DPO 偏好优化" aria-pressed="false">DPO 偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM User Simulator" aria-pressed="false">LLM User Simulator<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 摘要" aria-pressed="false">LLM 摘要<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 重排器" aria-pressed="false">LLM 重排器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Page-level Quality Anchor" aria-pressed="false">Page-level Quality Anchor<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Trait Interference" aria-pressed="false">Trait Interference<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实解释" aria-pressed="false">反事实解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;LLM User Simulator&quot;,&quot;Trait Interference&quot;,&quot;Page-level Quality Anchor&quot;]"><strong><a href="a_behavioral_trait_leaks_into_preferences_diagnosing_trait_interference_in_/">A Behavioral Trait Leaks into Preferences: Diagnosing Trait Interference in LLM User Simulators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 LLM 用户模拟器中活跃度特质干扰偏好边界并导致评估失效，提出页面级质量锚点 PQA 恢复特质独立性。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;熵减奖励&quot;,&quot;多轮偏好引导&quot;]"><strong><a href="ask_to_be_sure_informative_interactions_for_confident_multi_turn_llm_recomm/">Ask to Be Sure: Informative Interactions for Confident Multi-Turn LLM Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以推荐列表采样熵的下降量作为信息增益奖励，免真值微调 LLM 多轮对话策略，INSPIRED/ReDial 上 Hit 与对话效率双升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;可解释推荐&quot;,&quot;反事实解释&quot;,&quot;LLM 摘要&quot;,&quot;DPO 偏好优化&quot;]"><strong><a href="from_counterfactual_edits_to_human_narratives_llm_driven_interpretations_fo/">From Counterfactual Edits to Human Narratives: LLM-Driven Interpretations for Recommender System Explanations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 将冗长反事实编辑列表摘要为高层偏好叙事，并以人类校准指标评估正确性/信息量/语言质量。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;对话数据增强&quot;,&quot;知识图谱实体建模&quot;,&quot;对话-实体匹配&quot;]"><strong><a href="on_mitigating_data_sparsity_in_conversational_recommender_systems/">On Mitigating Data Sparsity in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DACRS 用 LLM 两阶段对话增强 + KG 实体替换与相似度约束 + 对话引导注意力融合，缓解对话与实体双重稀疏，ReDial R@50 提升 11.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM 重排器&quot;,&quot;评估协议敏感性&quot;,&quot;解码稳定性&quot;]"><strong><a href="retrieval_scoring_and_decoding_shape_performance_and_stability_in_llm_based/">Retrieval, Scoring, and Decoding Shape Performance and Stability in LLM-based Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 ReDial 上系统解耦候选池规模、一阶段检索器与解码温度对 LLM 重排器的影响，揭示协议选择可移动 NDCG@10 超过模型差距。</p>
</dd>
</dl>
