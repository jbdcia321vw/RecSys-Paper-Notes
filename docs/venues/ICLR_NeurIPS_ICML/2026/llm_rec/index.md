# 🧠 大语言模型推荐 (LLM4Rec)

**ICLR / NeurIPS / ICML (RecSys) 2026** · **5** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×5 · ICLR ×2 · NeurIPS ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交叉熵损失" aria-pressed="false">交叉熵损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练反思更新" aria-pressed="false">免训练反思更新<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双画像同质性门控" aria-pressed="false">双画像同质性门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成数据课程" aria-pressed="false">合成数据课程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="弱监督强化学习" aria-pressed="false">弱监督强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序模仿" aria-pressed="false">排序模仿<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索式推荐" aria-pressed="false">检索式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双画像同质性门控&quot;,&quot;免训练反思更新&quot;]"><strong><a href="harp_training_free_dual_profile_agentic_communication_for_llm_based_recomme/">HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 推荐重构为四阶段智能体通信，用双画像、同质性门控检索与单次反思三个免训练算子补齐缺失环节，Hit@1 相对提升 6.9%–24.0%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;证据高亮&quot;,&quot;弱监督强化学习&quot;]"><strong><a href="learning_evidence_highlighting_for_frozen_llms/">Learning Evidence Highlighting for Frozen LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用轻量 Emphasis Actor 以 RL 在冻结 LLM 的原始上下文中插入高亮标签，解耦证据选择与推理，序列推荐 HR@10 相对最优基线提升 5.38%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;合成数据课程&quot;,&quot;缩放定律&quot;]"><strong><a href="principled_synthetic_data_enables_the_first_scaling_laws_for_llms_in_recomm/">Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分层合成数据课程（Item-Text/CF/UIH）替代噪声交互日志，首次在 0.6B–8B LLM CPT 上拟合出推荐域幂律缩放律，SasRec 在合成数据上 Recall@100 提升 130%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;交叉熵损失&quot;,&quot;排序模仿&quot;]"><strong><a href="rejuvenating_cross_entropy_loss_in_knowledge_distillation_for_recommender_s/">Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示CE蒸馏损失与NDCG的闭包假设矛盾，提出师生协同子集拆分与自适应加权的RCE-KD，全库排序下NDCG@20最高提升11.76%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;视觉表征&quot;,&quot;检索式推荐&quot;]"><strong><a href="token_efficient_item_representation_via_images_for_llm_recommender_systems/">Token-Efficient Item Representation via Images for LLM Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单张物品图像替代冗长文本描述作为 LLM 推荐输入，配合 RISA 对齐与 RERI 检索式推荐，在 Amazon 四域上兼顾效率与效果。</p>
</dd>
</dl>
