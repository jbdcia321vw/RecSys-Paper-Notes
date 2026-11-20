---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**KDD 2024** · **10** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [KDD 2023 (6)](../../2023/debias_fairness_industrial/) · [KDD 2025 (10)](../../2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2023 (5)](../../../AAAI/2023/debias_fairness_industrial/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2023 (2)](../../../ACM_MM/2023/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2023 (14)](../../../CIKM/2023/debias_fairness_industrial/) · [CIKM 2024 (4)](../../../CIKM/2024/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2023 (1)](../../../DASFAA/2023/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2023 (2)](../../../ECIR/2023/debias_fairness_industrial/) · [ECIR 2024 (4)](../../../ECIR/2024/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDE 2024 (3)](../../../ICDE/2024/debias_fairness_industrial/) · [ICDM 2023 (3)](../../../ICDM/2023/debias_fairness_industrial/) · [ICDM 2024 (1)](../../../ICDM/2024/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (7)](../../../ICLR_NeurIPS_ICML/2023/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (7)](../../../ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2024 (2)](../../../IJCAI/2024/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [RecSys 2023 (11)](../../../RecSys/2023/debias_fairness_industrial/) · [RecSys 2024 (16)](../../../RecSys/2024/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2023 (9)](../../../SIGIR/2023/debias_fairness_industrial/) · [SIGIR 2024 (5)](../../../SIGIR/2024/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/debias_fairness_industrial/) · [ACM TKDD 2024 (1)](../../../TKDD/2024/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/debias_fairness_industrial/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2023 (9)](../../../TOIS/2023/debias_fairness_industrial/) · [ACM TOIS 2024 (7)](../../../TOIS/2024/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2023 (1)](../../../TORS/2023/debias_fairness_industrial/) · [ACM TORS 2024 (4)](../../../TORS/2024/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2023 (3)](../../../WSDM/2023/debias_fairness_industrial/) · [WSDM 2024 (4)](../../../WSDM/2024/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2023 (11)](../../../WWW/2023/debias_fairness_industrial/) · [WWW (TheWebConf) 2024 (20)](../../../WWW/2024/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/多任务&quot;,&quot;表征学习&quot;,&quot;维度坍缩&quot;,&quot;兴趣解缠&quot;]"><strong><a href="ads_recommendation_in_a_collapsed_and_entangled_world/">Ads Recommendation in a Collapsed and Entangled World</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>腾讯广告系统从特征编码、维度坍缩与任务兴趣纠缠三方面重构表征学习，并以在线 A/B 验证多嵌入、解缠与训练策略的 GMV 收益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端生成式&quot;,&quot;对抗攻击&quot;,&quot;LLM Agent&quot;,&quot;Prompt Tuning&quot;]"><strong><a href="cheatagent_attacking_llm_empowered_recommender_systems_via_llm_agent/">CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用掩码定位插入位 + LLM 智能体生成对抗扰动并以前缀提示自反思优化，在黑盒下显著击穿 P5/TALLRec 推荐系统。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差 (Duration Bias)&quot;,&quot;反事实观看时长 (CWT)&quot;]"><strong><a href="counteracting_duration_bias_in_video_recommendation_via_counterfactual_watc/">Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>KDD 2024｜去偏与工业推荐：以“反事实观看时长(CWT)截断”重构时长偏差，用经济学边际收益推导变换函数与截断似然，KuaiRand/WeChat/Product 上 MAE 降约 25%、AUC 逼近 Oracle，线上 MWT +2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;双稳健估计&quot;,&quot;噪声反馈&quot;]"><strong><a href="debiased_recommendation_with_noisy_feedback/">Debiased Recommendation with Noisy Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次联合建模 MNAR 与结果测量误差 (OME)，提出 OME-EIB/IPS/DR 估计量、弱可分误差参数辨识与交替去噪训练，在 Coat/Music/KuaiRec 上 AUC 相对提升约 1.5–2.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;用户关系图&quot;,&quot;个性化物品嵌入&quot;]"><strong><a href="gpfedrec_graph_guided_personalization_for_federated_recommendation/">GPFedRec: Graph-guided Personalization for Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>服务端用客户端物品嵌入构建用户关系图引导联邦聚合，Leave-One-Out 全库排序下 HR@10 最高相对提升 3.38%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/排序策略层&quot;,&quot;用户偏好动态&quot;,&quot;危害缓解&quot;,&quot;隐函数定理梯度&quot;]"><strong><a href="harm_mitigation_in_recommender_systems_under_user_preference_dynamics/">Harm Mitigation in Recommender Systems under User Preference Dynamics</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 MNL 选择+吸引力偏好动态建模“推荐→兴趣漂移→站外有害消费”路径，在平稳点用隐函数定理求梯度优化 CTR-危害权衡，CTR 与危害同时优于基线，最高 77%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;互惠推荐&quot;,&quot;因果推断&quot;]"><strong><a href="revisiting_reciprocal_recommender_systems_metrics_formulation_and_method/">Revisiting Reciprocal Recommender Systems: Metrics, Formulation, and Method</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出覆盖/稳定性/双边均衡五项互惠推荐新指标，并以双边因果 treatment + 空位重排求解；全库排序下 CRecall 最高提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段数据投毒&quot;,&quot;对比学习推荐&quot;,&quot;谱分析&quot;,&quot;双层优化&quot;]"><strong><a href="unveiling_vulnerabilities_of_contrastive_recommender_systems_to_poisoning_a/">Unveiling Vulnerabilities of Contrastive Recommender Systems to Poisoning Attacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从谱分析揭示 InfoNCE 的全局离散效应正是 CL 推荐易受定向投毒的根源，提出双目标 CLeaR 攻击，Hit-Ratio@50 平均提升 1.5–3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（冷启动内容匹配）/供给生态&quot;,&quot;创作者竞争博弈&quot;,&quot;用户福利优化&quot;,&quot;乘法权重更新&quot;]"><strong><a href="user_welfare_optimization_in_recommender_systems_with_competing_content_cre/">User Welfare Optimization in Recommender Systems with Competing Content Creators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户满意度动态重加权 UIR/SMT/HMT 引导创作者竞争，三周线上 A/B 使 LTR +1.13%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回 (Next-POI 全库打分)&quot;,&quot;Membership Inference Attack (LiRA)&quot;,&quot;Training Data Extraction&quot;,&quot;DP-SGD / Selective DP&quot;]"><strong><a href="where_have_you_been_a_study_of_privacy_risk_for_point_of_interest_recommend/">Where Have You Been? A Study of Privacy Risk for Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个面向 POI 推荐模型的位置/轨迹级隐私攻击套件（抽取+成员推断），揭示全库排序模型可被黑盒查询大幅记忆训练签到，并证明现有 DP 防御效用崩塌。</p>
</dd>
</dl>
