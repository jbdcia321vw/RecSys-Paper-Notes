---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**ICLR / NeurIPS / ICML (RecSys) 2026** · **11** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (4)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (9)](../../../ACL_EMNLP/2024/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (4)](../../../DASFAA/2026/llm_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2024 (1)](../../../ICDM/2024/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/llm_rec/) · [IJCAI (RecSys Track) 2026 (4)](../../../IJCAI/2026/llm_rec/) · [KDD 2022 (1)](../../../KDD/2022/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (10)](../../../KDD/2026/llm_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/llm_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2021 (1)](../../../SIGIR/2021/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (24)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (4)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×9 · ICLR 2026 ×5 · 端到端生成式 ×3 · ICLR ×2 · LLM4Rec ×2 · NeurIPS ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;Mixture-of-Experts&quot;,&quot;Item-ID Token&quot;]"><strong><a href="catalog_native_llm_speaking_item_id_dialect_with_less_entanglement_for_reco/">Catalog-Native LLM: Speaking Item-ID Dialect with Less Entanglement for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出IDIOMoE，以token-type MoE拆分文本/物品FFN，在全库排序与工业数据上兼顾协同推荐与语言理解。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;LoRA 持续学习&quot;,&quot;近端正则化&quot;,&quot;语义 ID 自回归生成&quot;]"><strong><a href="continual_low_rank_adapters_for_llm_based_generative_recommender_systems/">Continual Low-Rank Adapters for LLM-based Generative Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 PESO：以单支持续进化 LoRA + 逐模块 softmax–KL 近端正则平衡稳定性与可塑性，在 Amazon 三域按时间增量切分的生成式推荐上全面超越累积式 LoRA。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双画像同质性门控&quot;,&quot;免训练反思更新&quot;]"><strong><a href="harp_training_free_dual_profile_agentic_communication_for_llm_based_recomme/">HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 推荐重构为四阶段智能体通信，用双画像、同质性门控检索与单次反思三个免训练算子补齐缺失环节，Hit@1 相对提升 6.9%–24.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;LLM4Rec&quot;,&quot;推理时解码控制&quot;,&quot;激活干预&quot;,&quot;个性化安全对齐&quot;]"><strong><a href="inference_time_personalized_safety_control_via_paired_difference_in_means_i/">Inference-Time Personalized Safety Control via Paired Difference-in-Means Intervention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>免训练推理时激活干预做个性化安全控制；PCMS 配对均值差方向估计理论无偏且一致，LLaMA-3.1-8B 上四类敏感内容伤害分 3.36→1.73（-48.5%）而效用仅降 0.56。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;证据高亮&quot;,&quot;弱监督强化学习&quot;]"><strong><a href="learning_evidence_highlighting_for_frozen_llms/">Learning Evidence Highlighting for Frozen LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用轻量 Emphasis Actor 以 RL 在冻结 LLM 的原始上下文中插入高亮标签，解耦证据选择与推理，序列推荐 HR@10 相对最优基线提升 5.38%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;偏好对齐DPO&quot;,&quot;可解释推荐&quot;]"><strong><a href="more_than_what_was_chosen_llm_based_explainable_recommendation_beyond_noisy/">More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出连贯偏好CP与冲突感知DPO(C-APO)，在Amazon五域Leave-One-Out协议下HR@1最高提升61.4%，在线CTR相对ML提升60.9%</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;合成数据课程&quot;,&quot;缩放定律&quot;]"><strong><a href="principled_synthetic_data_enables_the_first_scaling_laws_for_llms_in_recomm/">Principled Synthetic Data Enables the First Scaling Laws for LLMs in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分层合成数据课程（Item-Text/CF/UIH）替代噪声交互日志，首次在 0.6B–8B LLM CPT 上拟合出推荐域幂律缩放律，SasRec 在合成数据上 Recall@100 提升 130%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR/NeurIPS/ICML 2026&quot;,&quot;LLM4Rec&quot;,&quot;生成式端到端&quot;,&quot;Latent Reasoning&quot;,&quot;GRPO&quot;,&quot;强化学习微调&quot;]"><strong><a href="reinforced_latent_reasoning_for_llm_based_recommendation/">Reinforced Latent Reasoning for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用连续隐空间 Token 替代显式 CoT，结合两阶段 SFT+改进 GRPO 无 CoT 监督地学习推理，D3 底座全库排序相对提升 8.4% 且推理延迟仅约 +1 token。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;交叉熵损失&quot;,&quot;排序模仿&quot;]"><strong><a href="rejuvenating_cross_entropy_loss_in_knowledge_distillation_for_recommender_s/">Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示CE蒸馏损失与NDCG的闭包假设矛盾，提出师生协同子集拆分与自适应加权的RCE-KD，全库排序下NDCG@20最高提升11.76%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;视觉表征&quot;,&quot;检索式推荐&quot;]"><strong><a href="token_efficient_item_representation_via_images_for_llm_recommender_systems/">Token-Efficient Item Representation via Images for LLM Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单张物品图像替代冗长文本描述作为 LLM 推荐输入，配合 RISA 对齐与 RERI 检索式推荐，在 Amazon 四域上兼顾效率与效果。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排/评分预测&quot;,&quot;长短期兴趣草图&quot;,&quot;推理内化&quot;]"><strong><a href="token_efficient_long_term_interest_sketching_and_internalized_reasoning_for/">Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以话题级长短期兴趣草图压缩提示、GRPO 显式推理 + 隐藏态对齐内化 CoT，实现单 token 答案解码，MAE 较次优降 12.3% 且延迟降 100 倍以上。</p>
</dd>
</dl>
