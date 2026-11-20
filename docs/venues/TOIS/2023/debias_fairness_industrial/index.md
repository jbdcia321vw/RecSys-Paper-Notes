---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**ACM TOIS 2023** · **9** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ACM TOIS 2024 (7)](../../2024/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2023 (5)](../../../AAAI/2023/debias_fairness_industrial/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2023 (2)](../../../ACM_MM/2023/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2023 (14)](../../../CIKM/2023/debias_fairness_industrial/) · [CIKM 2024 (4)](../../../CIKM/2024/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2023 (1)](../../../DASFAA/2023/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2023 (2)](../../../ECIR/2023/debias_fairness_industrial/) · [ECIR 2024 (4)](../../../ECIR/2024/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDE 2024 (3)](../../../ICDE/2024/debias_fairness_industrial/) · [ICDM 2023 (3)](../../../ICDM/2023/debias_fairness_industrial/) · [ICDM 2024 (1)](../../../ICDM/2024/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (7)](../../../ICLR_NeurIPS_ICML/2023/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (7)](../../../ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2024 (2)](../../../IJCAI/2024/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2023 (6)](../../../KDD/2023/debias_fairness_industrial/) · [KDD 2024 (10)](../../../KDD/2024/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2023 (11)](../../../RecSys/2023/debias_fairness_industrial/) · [RecSys 2024 (16)](../../../RecSys/2024/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2023 (9)](../../../SIGIR/2023/debias_fairness_industrial/) · [SIGIR 2024 (5)](../../../SIGIR/2024/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/debias_fairness_industrial/) · [ACM TKDD 2024 (1)](../../../TKDD/2024/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/debias_fairness_industrial/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TORS 2023 (1)](../../../TORS/2023/debias_fairness_industrial/) · [ACM TORS 2024 (4)](../../../TORS/2024/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2023 (3)](../../../WSDM/2023/debias_fairness_industrial/) · [WSDM 2024 (4)](../../../WSDM/2024/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2023 (11)](../../../WWW/2023/debias_fairness_industrial/) · [WWW (TheWebConf) 2024 (20)](../../../WWW/2024/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** TOIS ×8 · 去偏公平与工业架构 ×2 · 因果推断去偏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断去偏" aria-pressed="false">因果推断去偏<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估协议&quot;,&quot;全局时间线&quot;,&quot;数据泄漏&quot;,&quot;非采样排序&quot;]"><strong><a href="a_critical_study_on_data_leakage_in_recommender_system_offline_evaluation/">A Critical Study on Data Leakage in Recommender System Offline Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TOIS 2023｜去偏公平/工业评测｜系统揭示 leave-last-one-out 忽略全局时间线造成未来交互泄漏，量化四模型四数据集上精度与排序失稳，并提出 timeline 增量评测方案。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全链路(数据/排序/重排)&quot;,&quot;双边公平&quot;,&quot;摊销公平&quot;,&quot;校准公平&quot;]"><strong><a href="a_survey_on_the_fairness_of_recommender_systems/">A Survey on the Fairness of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首篇系统综述：以“过程/结果×群体/个体×七概念”定义体系、三视角分类与数据-排序-重排方法学，统一 60+ 篇公平推荐文献并指明双边联合公平等空白。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果后门调整&quot;,&quot;混合专家 MoE&quot;]"><strong><a href="addressing_confounding_feature_issue_for_causal_recommendation/">Addressing Confounding Feature Issue for Causal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用后门调整 + MoE 专家架构重估 P(Y|U,do(X))，消除视频时长等混杂特征对交互的虚假直接影响，在 Kwai/Wechat 上相对最优基线平均提升 26.9%/5.0%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;倾向性得分 IPS&quot;,&quot;反馈回路&quot;]"><strong><a href="bias_and_debias_in_recommender_system_a_survey_and_future_directions/">Bias and Debias in Recommender System: A Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以反馈回路为骨架统一 7 类推荐偏差的因果定义，给出 IPS/插补/因果图等去偏方法分类学与开放挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/模型训练&quot;,&quot;随机数据集&quot;,&quot;上界最小化&quot;,&quot;系统诱导偏差&quot;]"><strong><a href="bounding_system_induced_biases_in_recommender_systems_with_a_randomized_dat/">Bounding System-Induced Biases in Recommender Systems with A Randomized Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TOIS 2023 去偏推荐：以随机策略完整反馈定义理想损失，推导三角不等式与可分性泛化上界，提出 DUB 直接优化可解上界项；在 Yahoo! R3 与 Product 上 AUC 稳定超越强基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;混淆变量平衡&quot;,&quot;对抗学习&quot;]"><strong><a href="debiased_recommendation_with_user_feature_balancing/">Debiased Recommendation with User Feature Balancing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户特征投影+IPM分布平衡替代IPS实现因果去偏，配合裁剪/采样/对抗三种平衡项策略与潜混淆建模，在6个数据集上NDCG@10最高提升6.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;学习哈希&quot;,&quot;离散优化&quot;]"><strong><a href="lightfr_lightweight_federated_recommendation_with_privacy_preserving_matrix/">LightFR: Lightweight Federated Recommendation with Privacy-preserving Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用学习哈希把联邦 MF 的用户/物品表征压成二值码，配联邦离散坐标下降与梯度聚合，精度追平联邦 SOTA，存储与推理开销降约一个量级。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信隐私&quot;,&quot;召回&quot;,&quot;隐私推断攻击&quot;,&quot;曝光数据保护&quot;]"><strong><a href="on_the_user_behavior_leakage_from_recommender_system_exposure/">On the User Behavior Leakage from Recommender System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示系统曝光序列可反推用户历史点击的隐私漏洞，用编码器-解码器黑盒攻击在 Zhihu/MIND 上实现最高 87.72% Recall@10，并提出两阶段曝光替换保护机制。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;工业架构与评测基准&quot;,&quot;隐私计算&quot;,&quot;数据披露机制&quot;,&quot;多智能体模拟&quot;]"><strong><a href="studying_the_impact_of_data_disclosure_mechanism_in_recommender_systems_via/">Studying the Impact of Data Disclosure Mechanism in Recommender Systems via Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出隐私感知推荐框架，以隐私计算建模用户细粒度数据披露决策，并用多智能体 bandit 模拟证明细粒度机制优于“全有或全无”。</p>
</dd>
</dl>
