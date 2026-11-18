# 🛡️ 去偏公平、可信隐私与工业架构

**WWW (TheWebConf) 2025** · **9** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (10)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/)

🔥 **高频主题：** WWW 2025 ×9 · 去偏公平与可信推荐 ×9 · OOD 泛化 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="OOD 泛化" aria-pressed="false">OOD 泛化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Embedding Pruning" aria-pressed="false">Embedding Pruning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="On-device Recommendation" aria-pressed="false">On-device Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Shapley Value" aria-pressed="false">Shapley Value<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SUTVA违背" aria-pressed="false">SUTVA违背<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平性-相关性联合评测" aria-pressed="false">公平性-相关性联合评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平正则化" aria-pressed="false">公平正则化<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;创作者侧推荐&quot;,&quot;用户可用性计算&quot;,&quot;双边市场公平性&quot;]"><strong><a href="creator_side_recommender_system_challenges_designs_and_applications/">Creator-Side Recommender System: Challenges, Designs, and Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出创作者侧推荐系统 DualRec，将用户侧召回/排序镜像为物品找用户，并用用户可用性计算 UAC 解决用户可达性问题，快手线上 DAC 提升 2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;敏感内容标签&quot;,&quot;算法放大&quot;,&quot;数据集构建&quot;]"><strong><a href="datasets_for_navigating_sensitive_topics_in_recommendation_systems/">Datasets for Navigating Sensitive Topics in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 ML-DDD 与 AO3 两个带敏感内容警告标签的推荐数据集，提出 Warning Amplification 指标，发现个性化 CF 反而去放大敏感内容。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测协议&quot;,&quot;无偏估计&quot;,&quot;随机曝光数据集&quot;,&quot;去偏推荐&quot;]"><strong><a href="debias_can_be_unreliable_mitigating_bias_issue_in_evaluating_debiasing_reco/">Debias Can be Unreliable: Mitigating Bias Issue in Evaluating Debiasing Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示随机曝光数据集上传统 Recall 评估与全曝光真值不一致，提出 URE 无偏估计方案并重估去偏方法。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;分布鲁棒优化&quot;,&quot;扩散去噪&quot;,&quot;OOD 泛化&quot;]"><strong><a href="distributionally_robust_graph_out_of_distribution_recommendation_via_diffus/">Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 DRO 推荐对噪声分布过度加权之弊，用 VGAE+扩散去噪、Sinkhorn 距离与熵正则重构最坏情形优化，OOD 场景最高提升 75.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果推断&quot;,&quot;扩散模型&quot;,&quot;OOD 泛化&quot;]"><strong><a href="graph_representation_learning_via_causal_diffusion_for_out_of_distribution_/">Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构因果模型定位环境混杂因子，以变分推断+扩散模型学习环境不变图表征，OOD 场景最高提升 36.73%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;重排&quot;]"><strong><a href="joint_evaluation_of_fairness_and_relevance_in_recommender_systems_with_pare/">Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 DPFR：以测试集构造相关性-公平性帕累托前沿，用模型得分到前沿的欧氏距离作为联合评测指标，揭示现有联合指标排序与帕累托最优解不一致。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Shapley Value&quot;,&quot;Embedding Pruning&quot;,&quot;On-device Recommendation&quot;]"><strong><a href="on_device_content_based_recommendation_with_single_shot_embedding_pruning_a/">On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Shapley 值一次性剪枝 CTR 嵌入表，配合场感知码本替代零填充，无需重训即可适配任意端侧存储预算。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;在线实验/因果评估&quot;,&quot;共生偏差&quot;,&quot;SUTVA违背&quot;,&quot;聚类随机化&quot;]"><strong><a href="reducing_symbiosis_bias_through_better_a_b_tests_of_recommendation_algorith/">Reducing Symbiosis Bias Through Better A/B Tests of Recommendation Algorithms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示推荐算法共享训练数据导致的共生偏差，理论建模并仿真对比三种实验设计，用国家级分流实验验证偏差存在。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;类别感知公平度量&quot;,&quot;公平正则化&quot;,&quot;性别偏差&quot;]"><strong><a href="unmasking_gender_bias_in_recommendation_systems_and_enhancing_category_awar/">Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出6个类别感知的性别偏差度量（含排序折扣）与GBS聚合，并以类别覆盖差作正则项，在3数据集上偏差降50–77%而性能损失有限。</p>
</dd>
</dl>
