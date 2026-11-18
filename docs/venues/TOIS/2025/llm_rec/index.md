---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**ACM TOIS 2025** · **9** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** TOIS ×9 · 大语言模型推荐 ×9 · LLM4Rec ×2

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="dX-privacy 序列扰动" aria-pressed="false">dX-privacy 序列扰动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ID-based Recommendation" aria-pressed="false">ID-based Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 微调" aria-pressed="false">LoRA 微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="专家发现" aria-pressed="false">专家发现<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化预训练" aria-pressed="false">个性化预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全链路架构" aria-pressed="false">全链路架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分类体系" aria-pressed="false">分类体系<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实推断" aria-pressed="false">反事实推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域序列推荐" aria-pressed="false">多域序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域预训练" aria-pressed="false">多域预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体强化学习" aria-pressed="false">多智能体强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实例级提示" aria-pressed="false">实例级提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习知识迁移" aria-pressed="false">对比学习知识迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="投票导向任务" aria-pressed="false">投票导向任务<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指令微调" aria-pressed="false">指令微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码物品文本预测" aria-pressed="false">掩码物品文本预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示词个性化" aria-pressed="false">提示词个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本表征替代ID" aria-pressed="false">文本表征替代ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合排序" aria-pressed="false">混合排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="维度坍缩" aria-pressed="false">维度坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述" aria-pressed="false">综述<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦序列推荐" aria-pressed="false">联邦序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为-文本适配" aria-pressed="false">行为-文本适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱编码器" aria-pressed="false">谱编码器<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;数据增强&quot;,&quot;LLM4Rec&quot;,&quot;ID-based Recommendation&quot;,&quot;LoRA 微调&quot;]"><strong><a href="enhancing_id_based_recommendation_with_large_language_models/">Enhancing ID-based Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LoRA 微调 LLM 仅凭纯 ID 序列生成伪交互数据，经双重过滤后注入任意 ID 基推荐模型，全库排序下平均提升 3%–9%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;联邦序列推荐&quot;,&quot;dX-privacy 序列扰动&quot;,&quot;对比学习知识迁移&quot;]"><strong><a href="fellas_enhancing_federated_sequential_recommendation_with_llm_as_external_s/">FELLAS: Enhancing Federated Sequential Recommendation with LLM as External Services</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 LLM 作为外部服务为联邦序列推荐提供物品级与序列级表征，配合 dX-privacy 扰动与对比学习，在 Full-Ranking 下显著超越 FedSeqRec 并逼近集中式训练。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;全链路架构&quot;,&quot;LLM4Rec&quot;,&quot;分类体系&quot;,&quot;综述&quot;]"><strong><a href="how_can_recommender_systems_benefit_from_large_language_models_a_survey/">How Can Recommender Systems Benefit from Large Language Models: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以“WHERE×HOW”双正交视角系统梳理 LLM 融入推荐全链路的五阶段角色与四象限适配范式，并给出效率/效果/伦理三大挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;反事实推断&quot;,&quot;维度坍缩&quot;,&quot;谱编码器&quot;]"><strong><a href="mitigating_propensity_bias_of_large_language_models_for_recommender_systems/">Mitigating Propensity Bias of Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用谱条件 iVAE 编码 LLM 侧信息避免维度坍缩，并以反事实推断消除 LLM 倾向性偏差，在 6 个 CF 骨干上全库排序一致提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;多域序列推荐&quot;,&quot;文本表征替代ID&quot;]"><strong><a href="one_model_for_all_large_language_models_are_domain_agnostic_recommendation_/">One Model for All: Large Language Models are Domain-Agnostic Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 编码跨域混合标题序列实现域无关推荐，采样 1000 负例评测，SPIAO 上较最优基线相对提升约 19%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;个性化预训练&quot;,&quot;专家发现&quot;,&quot;投票导向任务&quot;]"><strong><a href="pept_expert_finding_meets_personalized_pre_training/">PEPT: Expert Finding Meets Personalized Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将专家历史问答、投票分与专家ID统一进BERT预训练，用问题级MLM与投票预测双任务学习个性化兴趣与专长，六域CQA全库排序MRR全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;指令微调&quot;,&quot;混合排序&quot;]"><strong><a href="recranker_instruction_tuning_large_language_model_as_ranker_for_top_k_recom/">RecRanker: Instruction Tuning Large Language Model as Ranker for Top-k Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以自适应用户采样、位置偏移与检索信号增强构建三类排序指令微调 LLM 重排器，并提出混合排序集成，Leave-One-Out 全库排序下 HR/NDCG 最高提升 213%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;提示词个性化&quot;,&quot;多智能体强化学习&quot;,&quot;实例级提示&quot;]"><strong><a href="reinforced_prompt_personalization_for_recommendation_with_large_language_mo/">Reinforced Prompt Personalization for Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将提示词拆为四类句式模式，用多智能体强化学习为每个用户逐句选择并组合个性化提示，在三个数据集上 NDCG@10 相对任务级提示提升 107%–132%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;行为-文本适配&quot;,&quot;掩码物品文本预测&quot;,&quot;多域预训练&quot;]"><strong><a href="thoroughly_modeling_multi_domain_pre_trained_recommendation_as_language/">Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多域序列推荐彻底转为文本生成任务，用五项指标指导行为-文本双向适配，7 数据集微调与零样本均最优，NDCG@1 最高提升 23%。</p>
</dd>
</dl>
  </div>
</div>
