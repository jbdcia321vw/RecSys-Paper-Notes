# 🛡️ 去偏公平、可信隐私与工业架构

**AAAI (RecSys Track) 2026** · **7** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (4)](../../2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×6 · AAAI 2026 ×4 · 算法审计 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="算法审计" aria-pressed="false">算法审计<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="AAAI/ACM AIES" aria-pressed="false">AAAI/ACM AIES<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ICWSM" aria-pressed="false">ICWSM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 广告" aria-pressed="false">LLM 广告<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM智能体仿真" aria-pressed="false">LLM智能体仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="人口统计公平性" aria-pressed="false">人口统计公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="党派倾斜与谣言曝光" aria-pressed="false">党派倾斜与谣言曝光<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单义神经元" aria-pressed="false">单义神经元<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业审计" aria-pressed="false">去偏公平与工业审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实保真度" aria-pressed="false">反事实保真度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/服务端黑盒查询" aria-pressed="false">召回/服务端黑盒查询<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="广告投放审计" aria-pressed="false">广告投放审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="心理健康推荐" aria-pressed="false">心理健康推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成员推理攻击" aria-pressed="false">成员推理攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合推荐系统隐私" aria-pressed="false">混合推荐系统隐私<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相对成员度量" aria-pressed="false">相对成员度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏自编码器" aria-pressed="false">稀疏自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="路径积分归因" aria-pressed="false">路径积分归因<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排/事后干预" aria-pressed="false">重排/事后干预<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="马甲账号审计" aria-pressed="false">马甲账号审计<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/事后干预&quot;,&quot;稀疏自编码器&quot;,&quot;单义神经元&quot;,&quot;可解释推荐&quot;]"><strong><a href="extracting_interaction_aware_monosemantic_concepts_in_recommender_systems/">Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预测感知 SAE 从双塔嵌入中提取单义神经元，在 ML1M/Last.FM 上实现高纯度概念发现与事后可控干预。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;路径积分归因&quot;,&quot;反事实保真度&quot;]"><strong><a href="fidelity_aware_recommendation_explanations_via_stochastic_path_integration/">Fidelity-Aware Recommendation Explanations via Stochastic Path Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将路径积分归因引入推荐解释，用随机基线采样适配稀疏二值交互，在三种模型三数据集上刷新反事实保真度基准。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;参数高效微调&quot;,&quot;向量量化&quot;]"><strong><a href="plug_and_play_parameter_efficient_tuning_of_embeddings_for_federated_recomm/">Plug-and-Play Parameter-Efficient Tuning of Embeddings for Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结预训练全量物品嵌入、仅微调并上传压缩嵌入（LoRA/Hash/RQ-VAE），在联邦推荐中同时降通信与提精度。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/服务端黑盒查询&quot;,&quot;成员推理攻击&quot;,&quot;混合推荐系统隐私&quot;,&quot;相对成员度量&quot;]"><strong><a href="reference_recommendation_based_membership_inference_attack_against_hybrid_b/">Reference Recommendation based Membership Inference Attack against Hybrid-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>针对混合推荐系统提出参考推荐相对成员度量 MIA，仅需两次黑盒查询，ASR 达 0.93 且比影子训练快 93 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ICWSM&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;算法审计&quot;,&quot;心理健康推荐&quot;,&quot;LLM智能体仿真&quot;]"><strong><a href="seeking_help_facing_harm_auditing_tiktok_s_mental_health_recommendations/">Seeking Help, Facing Harm: Auditing TikTok&#x27;s Mental Health Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>30个LLM智能体7天控制审计TikTok推荐流，揭示交互行为主导心理健康内容曝光、求助意图无法形成安全边界。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI/ACM AIES&quot;,&quot;去偏公平与可信推荐&quot;,&quot;广告投放审计&quot;,&quot;马甲账号审计&quot;,&quot;人口统计公平性&quot;,&quot;LLM 广告&quot;]"><strong><a href="the_beginning_of_chatgpt_ads/">The Beginning of ChatGPT Ads</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个 ChatGPT 广告投放审计：91 个马甲账号 3×3 种族×收入设计，发现低收入账号更易被投放广告、种族无显著差异。</p>
</dd>
<dt data-paper-tags="[&quot;ICWSM 2026&quot;,&quot;去偏公平与工业审计&quot;,&quot;重排&quot;,&quot;算法审计&quot;,&quot;党派倾斜与谣言曝光&quot;]"><strong><a href="the_role_of_follow_networks_and_twitter_s_content_recommender_on_partisan_s/">The Role of Follow Networks and Twitter&#x27;s Content Recommender on Partisan Skew and Rumor Exposure during the 2022 U.S. Midterm Election</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 18 个自动化审计账号在 2022 美国中期选举期间对比算法流与时间流，量化推荐算法与关注网络对党派倾斜、谣言与低质信息曝光的交互效应。</p>
</dd>
</dl>
