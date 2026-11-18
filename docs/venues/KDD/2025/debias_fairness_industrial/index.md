# 🛡️ 去偏公平、可信隐私与工业架构

**KDD 2025** · **10** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [KDD 2026 (7)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×10

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="C2C 推荐数据集" aria-pressed="false">C2C 推荐数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三塔 Transformer 多步召回" aria-pressed="false">三塔 Transformer 多步召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交错实验" aria-pressed="false">交错实验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布式图学习" aria-pressed="false">分布式图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态子网络" aria-pressed="false">动态子网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实评估" aria-pressed="false">反事实评估<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;选择偏差&quot;,&quot;潜在外生变量&quot;,&quot;蒙特卡洛似然&quot;]"><strong><a href="addressing_correlated_latent_exogenous_variables_in_debiased_recommender_sy/">Addressing Correlated Latent Exogenous Variables in Debiased Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>放松外生变量独立假设，用双变量正态潜变量建模选择与偏好相关性，蒙特卡洛似然估计实现去偏，Coat/Yahoo!R3/KuaiRec 全指标领先。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;捷径偏置&quot;,&quot;用户聚类&quot;,&quot;多塔网络&quot;]"><strong><a href="breaker_removing_shortcut_cues_with_user_clustering_for_single_slot_recomme/">Breaker: Removing Shortcut Cues with User Clustering for Single-slot Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用用户表征聚类辅助任务+多塔簇专属偏好建模，破除单槽推荐中用户固有倾向的捷径偏置，离线Recall@1提升7%并上线美团支付。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据版权保护&quot;,&quot;序列推荐&quot;,&quot;数据水印&quot;,&quot;感受野&quot;]"><strong><a href="data_watermarking_for_sequential_recommender_systems/">Data Watermarking for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次定义序列推荐数据集/单用户双场景数据水印，用感受野约束插入连续物品序列，5 模型 3 数据集上水印 NDCG 近 1.0 且效用无损。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据生成/基准构建&quot;,&quot;概率生成模型&quot;,&quot;长尾分布混合&quot;,&quot;可控合成数据&quot;]"><strong><a href="flexible_generation_of_preference_data_for_recommendation_analysis/">Flexible Generation of Preference Data for Recommendation Analysis</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HYDRA 以用户-物品匹配、物品流行度、用户活跃度三因子混合分布生成可控合成偏好数据，复现真实基准的度分布与算法排序。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;端侧推荐&quot;,&quot;动态子网络&quot;,&quot;超网络参数生成&quot;]"><strong><a href="forward_once_for_all_structural_parameterized_adaptation_for_efficient_clou/">Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构控制器+结构引导映射器为每台设备动态组装子网络，一次前向即完成端侧适配，兼顾隐私、带宽与算力。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;分布式图学习&quot;,&quot;子图采样表格化&quot;,&quot;工业级 GNN 系统&quot;]"><strong><a href="gigl_large_scale_graph_neural_networks_at_snapchat/">GiGL: Large-Scale Graph Neural Networks at Snapchat</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Snapchat 开源十亿级图学习库 GiGL，以表格化子图采样+PyG 解耦实现 900M 节点/16.8B 边图 12 小时内端到端训练推理，支撑 35+ 线上发布。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;联邦训练通信&quot;,&quot;梯度聚类压缩&quot;,&quot;自适应动作共享&quot;,&quot;联邦推荐&quot;]"><strong><a href="gradients_as_an_action_towards_communication_efficient_federated_recommende/">Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将物品梯度聚类为少量“动作”共享传输，配合自适应聚类与带宽感知压缩，在 ML-100K/1M/Lastfm-2K 上以最高 96.88% 压缩率不损性能。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;交错实验&quot;,&quot;反事实评估&quot;]"><strong><a href="harnessing_the_power_of_interleaving_and_counterfactual_evaluation_for_airb/">Harnessing the Power of Interleaving and Counterfactual Evaluation for Airbnb Search Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出竞争对式 Team Draft 交错实验与在线反事实评估，用双层随机化与位置差增益估计器将 Airbnb 搜索排序实验灵敏度提升 50–100 倍。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;C2C 推荐数据集&quot;,&quot;三塔 Transformer 多步召回&quot;,&quot;无 SKU 商品表征&quot;]"><strong><a href="merrec_a_large_scale_multipurpose_mercari_dataset_for_consumer_to_consumer_/">MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个大规模 C2C 推荐数据集 MerRec（5.56M 用户/83M 商品/1.27B 交互），并配套三塔 Mercatran 模型与四任务基准。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果干预&quot;,&quot;时序演化&quot;]"><strong><a href="taming_recommendation_bias_with_causal_intervention_on_evolving_personal_po/">Taming Recommendation Bias with Causal Intervention on Evolving Personal Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出演化个人流行度度量与因果去混杂训练，并在推理期用移动平均趋势外推做干预，兼顾用户-物品演化一致性，Full-Ranking 下 Recall@20 最高提升 20.4%。</p>
</dd>
</dl>
