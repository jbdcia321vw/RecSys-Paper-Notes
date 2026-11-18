# 🗣️ 对话推荐与智能体仿真

**SIGIR 2026** · **8** 篇论文解读

> **方向说明**：多轮主动追问对话推荐 (CRS)、基于 LLM Agent 的用户行为模拟沙盒、可解释性推荐理由生成

📌 **同领域跨会议浏览：** [SIGIR 2025 (7)](../../2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/agent_conversational_rec/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (8)](../../../ACL_EMNLP/2025/agent_conversational_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (9)](../../../ACL_EMNLP/2026/agent_conversational_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/agent_conversational_rec/) · [CIKM 2026 (5)](../../../CIKM/2026/agent_conversational_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/agent_conversational_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/agent_conversational_rec/) · [ICDM 2025 (3)](../../../ICDM/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/agent_conversational_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/agent_conversational_rec/) · [KDD 2025 (2)](../../../KDD/2025/agent_conversational_rec/) · [KDD 2026 (2)](../../../KDD/2026/agent_conversational_rec/) · [RecSys 2025 (2)](../../../RecSys/2025/agent_conversational_rec/) · [RecSys 2026 (9)](../../../RecSys/2026/agent_conversational_rec/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/agent_conversational_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/agent_conversational_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/agent_conversational_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/agent_conversational_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2025 (6)](../../../WWW/2025/agent_conversational_rec/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/agent_conversational_rec/)

🔥 **高频主题：** 生成式推荐 ×2 · 用户模拟 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 零样本/少样本" aria-pressed="false">LLM 零样本/少样本<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好引导策略" aria-pressed="false">偏好引导策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="决策策略挖掘" aria-pressed="false">决策策略挖掘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双过程融合" aria-pressed="false">双过程融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实因果映射" aria-pressed="false">反事实因果映射<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实解释" aria-pressed="false">反事实解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性评测" aria-pressed="false">可复现性评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可持续旅游推荐" aria-pressed="false">可持续旅游推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体LLM" aria-pressed="false">多智能体LLM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多视角效用对齐" aria-pressed="false">多视角效用对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗分布对齐" aria-pressed="false">对抗分布对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话交互" aria-pressed="false">对话交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="游戏化数据采集" aria-pressed="false">游戏化数据采集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户效用指标" aria-pressed="false">用户效用指标<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户领域知识估计" aria-pressed="false">用户领域知识估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动化用户画像生成" aria-pressed="false">自动化用户画像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义离散化去偏" aria-pressed="false">语义离散化去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆优化偏好推断" aria-pressed="false">逆优化偏好推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复捷径去重" aria-pressed="false">重复捷径去重<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="金融对话推荐基准" aria-pressed="false">金融对话推荐基准<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;可复现性评测&quot;,&quot;重复捷径去重&quot;,&quot;用户效用指标&quot;]"><strong><a href="a_standardized_re_evaluation_of_conversational_recommender_systems_on_the_r/">A Standardized Re-evaluation of Conversational Recommender Systems on the ReDial Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一复现 7 个 CRS 模型，揭示 Recall@1 复现粒度鸿沟、约 50% 精度来自重复捷径、增益多源于 LLM 骨干而非架构创新。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;对抗分布对齐&quot;,&quot;语义离散化去偏&quot;]"><strong><a href="bridging_passive_and_active_enhancing_conversation_starter_recommendation_v/">Bridging Passive and Active: Enhancing Conversation Starter Recommendation via Active Expression Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用对抗分布对齐+RQ-VAE语义离散化把用户主动查询引入对话开场白推荐，打破曝光-点击回音室，线上活跃天数+0.04%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;多视角效用对齐&quot;,&quot;逆优化偏好推断&quot;,&quot;金融对话推荐基准&quot;]"><strong><a href="conv_finre_a_conversational_and_longitudinal_benchmark_for_utility_grounded/">Conv-FinRe: A Conversational and Longitudinal Benchmark for Utility-Grounded Financial Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用逆优化从纵向决策轨迹反推用户风险偏好，构建四视图多视角金融推荐基准，揭示 LLM 效用对齐与行为模仿的根本张力。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;双过程融合&quot;,&quot;决策策略挖掘&quot;]"><strong><a href="meituan_merchant_business_diagnosis_via_policy_guided_dual_process_user_sim/">Meituan Merchant Business Diagnosis via Policy-Guided Dual-Process User Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>从行为轨迹挖掘决策策略作为共享对齐层，锚定 LLM 推理与 GBDT 拟合双过程分支，群体仿真误差降至 8.80%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;对话交互&quot;,&quot;用户领域知识估计&quot;,&quot;游戏化数据采集&quot;,&quot;LLM 零样本/少样本&quot;]"><strong><a href="recquest_towards_estimating_user_domain_knowledge_in_conversational_recomme/">RecQuest: Towards Estimating User Domain Knowledge in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出游戏化数据采集协议 RecQuest 与知识估计新任务，用客观问卷替代自评建立 ground truth，并给出 LLM 零/少样本与微调基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;自动化用户画像生成&quot;,&quot;反事实因果映射&quot;]"><strong><a href="task_aware_automated_user_profile_generation_for_recommendation_simulation_/">Task-Aware Automated User Profile Generation for Recommendation Simulation Using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出免训练三阶段画像生成框架 APG4RecSim，用语义合并+反事实因果映射将交互历史转为任务对齐画像，在判别/排序/评分三任务上取得最佳综合表现。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;多智能体LLM&quot;,&quot;反事实解释&quot;,&quot;可持续旅游推荐&quot;]"><strong><a href="trace_a_conversational_framework_for_sustainable_tourism_recommendation_wit/">TRACE: A Conversational Framework for Sustainable Tourism Recommendation with Agentic Counterfactual Explanations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>多智能体 LLM 框架以澄清提问+反事实解释实现可持续旅游推荐的非强制绿色引导，24 人用户研究验证。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;偏好引导策略&quot;,&quot;MoE&quot;]"><strong><a href="when_and_how_to_ask_dynamic_preference_elicitation_strategies_for_conversat/">When and How to Ask: Dynamic Preference Elicitation Strategies for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 InPE 标注数据集与 COPE 分层 MoE 架构，显式建模“何时问、怎么问”的阶段感知偏好引导策略，Recall@10 相对最优基线提升 14.6%。</p>
</dd>
</dl>
