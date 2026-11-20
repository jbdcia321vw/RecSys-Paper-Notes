---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**RecSys 2023** · **11** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [RecSys 2024 (16)](../../2024/debias_fairness_industrial/) · [RecSys 2025 (9)](../../2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2023 (5)](../../../AAAI/2023/debias_fairness_industrial/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2023 (2)](../../../ACM_MM/2023/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2023 (14)](../../../CIKM/2023/debias_fairness_industrial/) · [CIKM 2024 (4)](../../../CIKM/2024/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2023 (1)](../../../DASFAA/2023/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2023 (2)](../../../ECIR/2023/debias_fairness_industrial/) · [ECIR 2024 (4)](../../../ECIR/2024/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDE 2024 (3)](../../../ICDE/2024/debias_fairness_industrial/) · [ICDM 2023 (3)](../../../ICDM/2023/debias_fairness_industrial/) · [ICDM 2024 (1)](../../../ICDM/2024/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (7)](../../../ICLR_NeurIPS_ICML/2023/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (7)](../../../ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2024 (2)](../../../IJCAI/2024/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2023 (6)](../../../KDD/2023/debias_fairness_industrial/) · [KDD 2024 (10)](../../../KDD/2024/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [SIGIR 2023 (9)](../../../SIGIR/2023/debias_fairness_industrial/) · [SIGIR 2024 (5)](../../../SIGIR/2024/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/debias_fairness_industrial/) · [ACM TKDD 2024 (1)](../../../TKDD/2024/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/debias_fairness_industrial/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2023 (9)](../../../TOIS/2023/debias_fairness_industrial/) · [ACM TOIS 2024 (7)](../../../TOIS/2024/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2023 (1)](../../../TORS/2023/debias_fairness_industrial/) · [ACM TORS 2024 (4)](../../../TORS/2024/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2023 (3)](../../../WSDM/2023/debias_fairness_industrial/) · [WSDM 2024 (4)](../../../WSDM/2024/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2023 (11)](../../../WWW/2023/debias_fairness_industrial/) · [WWW (TheWebConf) 2024 (20)](../../../WWW/2024/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 离线评测协议 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测协议" aria-pressed="false">离线评测协议<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;位置偏置建模&quot;,&quot;因果去偏与离线评估&quot;,&quot;曝光概率校准&quot;]"><strong><a href="a_probabilistic_position_bias_model_for_short_video_recommendation_feeds/">A Probabilistic Position Bias Model for Short-Video Recommendation Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Yule-Simon 滚动预算的生存函数给出闭式曝光概率，仅 3 参数的 C-3PO 刷新 NLL 并让离线-在线相关性较 DCG 提升 16%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;TU Matching&quot;,&quot;互惠推荐&quot;]"><strong><a href="fast_and_examination_agnostic_reciprocal_recommendation_in_matching_markets/">Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Choo–Siow 可转移效用匹配做互惠推荐重排，摆脱检查函数依赖并用 (2d+2) 维内积实现 ANN 检索，匹配总量超越 Naive/Reciprocal 且与 SW 持平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据采集与反馈&quot;,&quot;评分通胀&quot;,&quot;随机对照实验&quot;]"><strong><a href="interface_design_to_mitigate_inflation_in_recommender_systems/">Interface Design to Mitigate Inflation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用Piki音乐App自然实验与RCT证明评分通胀源于用户异质与个性化曝光，计时器界面可提升评分信息量。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（评分预测）&quot;,&quot;差分隐私&quot;,&quot;集合矩阵分解&quot;,&quot;公共物品特征&quot;]"><strong><a href="private_matrix_factorization_with_public_item_features/">Private Matrix Factorization with Public Item Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出DP-CMF，将公共物品元数据/相似度/公开评分以集合矩阵分解融入用户级DP矩阵分解，在ML10M评分预测上显著缩小隐私-效用差距。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;表征中立性&quot;,&quot;冷启动新用户&quot;]"><strong><a href="providing_previously_unseen_users_fair_recommendations_using_variational_au/">Providing Previously Unseen Users Fair Recommendations Using Variational Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在 Mult-VAE 召回上切分隐状态 z/b 并用解析高斯 KL 解耦敏感属性，使用户不相交划分下的新用户也能获得公平推荐</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与工业架构&quot;,&quot;端到端评测基准&quot;,&quot;Shilling Attack&quot;,&quot;攻击者知识约束&quot;,&quot;可复现性&quot;]"><strong><a href="recad_towards_a_unified_library_for_recommender_attack_and_defense/">RecAD: Towards A Unified Library for Recommender Attack and Defense</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>统一推荐攻击-防御基准库，标准化数据/模型/工作流三模块与双轨评估协议，揭示攻击者知识比例与检测失效现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/重排&quot;,&quot;最优传输&quot;,&quot;拥堵缓解&quot;,&quot;职位推荐&quot;]"><strong><a href="recon_reducing_congestion_in_job_recommendation_using_optimal_transport/">ReCon: Reducing Congestion in Job Recommendation using Optimal Transport</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可微 Sinkhorn 最优传输与 CNE 联合训练，在职位推荐中同时优化拥堵与相关性，取得更优 Pareto 权衡。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;Explainable Recommendation&quot;,&quot;Adversarial Robustness&quot;]"><strong><a href="stability_of_explainable_recommendation/">Stability of Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>实证揭示特征级可解释推荐器在参数随机/FGSM扰动下解释稳定性显著下降，采样排序评估显示对抗噪声危害更大。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;可复现性审计&quot;,&quot;GRU4Rec&quot;,&quot;离线评测协议&quot;]"><strong><a href="the_effect_of_third_party_implementations_on_reproducibility/">The Effect of Third Party Implementations on Reproducibility</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>审计 GRU4Rec 六个第三方复现：特征缺失与 bug 使 recall@20 相对官方下降 13%–99%，训练慢 2–336 倍，呼吁复现须对齐官方实现。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;时长偏差去偏&quot;,&quot;噪声观看校正&quot;,&quot;因果分解&quot;]"><strong><a href="uncovering_user_interest_from_biased_and_noised_watch_time_in_video_recomme/">Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将观看时长因果分解为兴趣、时长偏差与噪声三项，用分时长 GMM + 敏感度控制校正生成去偏标签，KuaiRand/WeChat 上 GAUC 相对提升约 10%，线上 VV +8.35%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;评测基准&quot;,&quot;离线评测协议&quot;,&quot;时间泄漏&quot;,&quot;负采样偏差&quot;]"><strong><a href="widespread_flaws_in_offline_evaluation_of_recommender_systems/">Widespread Flaws in Offline Evaluation of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统梳理离线评测四大缺陷（数据-任务错配、过度预处理、时间泄漏、负采样），用 GRU4Rec 实证证明其可致模型排序反转与数十个百分点虚高。</p>
</dd>
</dl>
