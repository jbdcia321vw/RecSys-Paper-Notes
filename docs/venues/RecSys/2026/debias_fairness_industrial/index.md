# 🛡️ 去偏公平、可信隐私与工业架构

**RecSys 2026** · **14** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [RecSys 2025 (9)](../../2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×14 · 生成式推荐 ×2 · 评测基准 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DPP 多样性正则" aria-pressed="false">DPP 多样性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="In-Context Learning" aria-pressed="false">In-Context Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 红队攻防" aria-pressed="false">LLM 红队攻防<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Margin 稳定性证书" aria-pressed="false">Margin 稳定性证书<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="云边协同" aria-pressed="false">云边协同<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;多智能体协同过滤&quot;,&quot;拓扑连通性安全&quot;,&quot;LLM 红队攻防&quot;]"><strong><a href="attacking_and_defending_multi_agent_collaborative_filtering_systems_through/">Attacking and Defending Multi-Agent Collaborative Filtering Systems Through Connectivity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 AgentCF 上系统复现 6 类攻击与 4 类防御，揭示候选数 k 与目录集中度 ρ 对用户/物品分区的非对称、非单调调控规律。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;差分隐私后处理&quot;,&quot;Margin 稳定性证书&quot;,&quot;锚点混合&quot;]"><strong><a href="decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;工业级全链路架构与评测基准&quot;,&quot;新闻推荐&quot;,&quot;可解释性&quot;,&quot;定性访谈研究&quot;]"><strong><a href="do_we_care_about_personalization_and_explainability_an_interview_study_with/">Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>对 9 家新闻机构 15 位工程师的半结构化访谈，揭示个性化采纳受编辑控制/隐私/资源约束，可解释性几乎只用于内部调试而非终端用户。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;轮播界面点击模型&quot;,&quot;离策略评估&quot;]"><strong><a href="from_click_modeling_to_offline_and_off_policy_evaluation_in_carousel_recomm/">From Click Modeling to Offline and Off-Policy Evaluation in Carousel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于 RecGaze 眼动数据推翻轮播界面四大行为假设，提出以可观测变量为核心的点击模型设计框架、离散选择点击模型与 N2DCG 重构，并规划轮播专用 OPE 路线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评估&quot;,&quot;曝光份额预测&quot;,&quot;反事实因果推断&quot;]"><strong><a href="impression_share_prediction_an_offline_evaluation_task_for_ranking_systems/">Impression Share Prediction: An Offline Evaluation Task for Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出曝光份额预测作为离线评估新任务，用因果DAG证明反事实可识别，PatchTST编码器在首小时恢复+22% L1。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;成员推理攻击&quot;,&quot;In-Context Learning&quot;,&quot;隐私泄露&quot;]"><strong><a href="membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;标签偏移&quot;,&quot;规范对齐&quot;]"><strong><a href="normative_alignment_of_recommender_systems_via_internal_label_shift/">Normative Alignment of Recommender Systems via Internal Label Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将编辑/公平目标建模为属性层内部标签偏移，用免重训的闭式重加权实现系统级规范对齐，KL 大幅下降且 AUC 基本无损。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;收敛效度&quot;,&quot;候选集采样&quot;,&quot;稀疏反馈偏差&quot;]"><strong><a href="on_the_convergent_validity_of_offline_evaluation_designs_for_recommender_sy/">On the Convergent Validity of Offline Evaluation Designs for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以稠密真值反馈为参照，用 Kendall&#x27;s τ 检验稀疏离线评测设计的收敛效度，发现无普适最优设计且 KuaiRec 上排序近乎反转。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;云边协同&quot;,&quot;隐私保护推荐&quot;,&quot;DPP 多样性正则&quot;]"><strong><a href="pricorec_a_privacy_aware_cloud_device_collaborative_framework_for_ad_recomm/">PriCoRec: A Privacy-Aware Cloud-Device Collaborative Framework for Ad Recommendation under Feature Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将敏感特征留在端侧，云侧预排加 DPP 多样性正则、端侧轻量排序加云端 logit 蒸馏，三数据集 gAUC/R@10 全面领先且时延可控。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;概率用户聚类&quot;]"><strong><a href="probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;训练种子方差&quot;,&quot;模型选择稳定性&quot;,&quot;可复现性&quot;]"><strong><a href="training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;位置偏差控制函数&quot;,&quot;流行度去偏&quot;,&quot;图消息传递&quot;]"><strong><a href="unbiased_recommender_systems_with_implicit_feedback/">Unbiased Recommender Systems with Implicit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>博士论文愿景：以控制函数法纠偏排序位置偏差、以事后投影与消息传递加权纠偏 GNN 协同过滤的流行度偏差。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;在线实验平台&quot;,&quot;曝光感知日志&quot;,&quot;双表面策略解耦&quot;,&quot;视频推荐用户研究&quot;]"><strong><a href="watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;曝光公平性审计&quot;,&quot;固定效应回归&quot;,&quot;自定义Feed生态&quot;]"><strong><a href="whose_posts_get_ranked_identical_text_exposure_gaps_in_bluesky_custom_feeds/">Whose Posts Get Ranked: Identical-Text Exposure Gaps in Bluesky Custom Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于自然重复帖的曝光审计：同文本跨作者在250个Bluesky自定义Feed中曝光差异达33%，作者历史主导排名</p>
</dd>
</dl>
