---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**RecSys 2022** · **10** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [RecSys 2021 (8)](../../2021/sequential_rec/) · [RecSys 2023 (10)](../../2023/sequential_rec/) · [RecSys 2024 (11)](../../2024/sequential_rec/) · [RecSys 2025 (5)](../../2025/sequential_rec/) · [RecSys 2026 (7)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2021 (4)](../../../AAAI/2021/sequential_rec/) · [AAAI (RecSys Track) 2022 (2)](../../../AAAI/2022/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [ACM MM 2026 (2)](../../../ACM_MM/2026/sequential_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/sequential_rec/) · [CIKM 2022 (9)](../../../CIKM/2022/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (10)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/sequential_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/sequential_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/sequential_rec/) · [ICDE 2022 (7)](../../../ICDE/2022/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2021 (2)](../../../ICDM/2021/sequential_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/sequential_rec/) · [IJCAI (RecSys Track) 2022 (2)](../../../IJCAI/2022/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/sequential_rec/) · [KDD 2022 (4)](../../../KDD/2022/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (11)](../../../KDD/2026/sequential_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/sequential_rec/) · [SIGIR 2022 (12)](../../../SIGIR/2022/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (4)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TORS 2026 (3)](../../../TORS/2026/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/sequential_rec/) · [WSDM 2022 (4)](../../../WSDM/2022/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (4)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/sequential_rec/) · [WWW (TheWebConf) 2022 (8)](../../../WWW/2022/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** 重复消费建模 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复消费建模" aria-pressed="false">重复消费建模<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;可复现性研究&quot;,&quot;BERT4Rec&quot;,&quot;掩码自监督训练&quot;]"><strong><a href="a_systematic_review_and_replicability_study_of_bert4rec_for_sequential_reco/">A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复盘 134 次 BERT4Rec vs SASRec 对比，指出原始实现默认配置严重欠拟合（需 30× 训练时长），并给出 Hugging Face 高效复现与 ALBERT4Rec 改进。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;重排/序列生成&quot;,&quot;音乐 Tour 生成&quot;,&quot;用户中心评估&quot;]"><strong><a href="a_user_centered_investigation_of_personal_music_tours/">A User-Centered Investigation of Personal Music Tours</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首次用半结构化访谈评估 Greedy 与 Optimal 两种音乐 Tour 生成算法，发现 Greedy 因 segue 多样性更受偏好，揭示多样性、编排与熟悉度为关键属性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排/排序&quot;,&quot;会话内自适应&quot;,&quot;实时搜索推荐&quot;]"><strong><a href="augmenting_netflix_search_with_in_session_adapted_recommendations/">Augmenting Netflix Search with In-Session Adapted Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向Netflix搜索预查询画布，融合当前会话实时行为与长期偏好，离线排序指标相对生产模型提升6%，并讨论实时不缓存等工程约束。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可微二值掩码&quot;,&quot;Jacobian 正则&quot;,&quot;ARM 无偏梯度估计&quot;]"><strong><a href="denoising_self_attentive_sequential_recommendation/">Denoising Self-attentive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微二值掩码剪枝自注意力的噪声项并叠加 Jacobian 谱正则，Leave-One-Out + 101 候选采样排序下比各 Transformer 骨干相对提升 5.05%–19.55%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/曝光策略&quot;,&quot;mere exposure effect&quot;,&quot;probit 回归&quot;,&quot;重复消费建模&quot;]"><strong><a href="discovery_dynamics_leveraging_repeated_exposure_for_user_and_music_characte/">Discovery Dynamics: Leveraging Repeated Exposure for User and Music Characterization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 9M 条曝光日志拟合 probit 二次模型，验证新曲曝光—收听呈倒 U 型（峰值约 12 次），并以峰值位置刻画曲目熟悉度与用户年龄差异。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;训练目标设计&quot;,&quot;列表损失 λRank&quot;]"><strong><a href="effective_and_efficient_training_for_sequential_recommendation_using_recenc/">Effective and Efficient Training for Sequential Recommendation using Recency Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出近期性序列采样 RSS 训练目标，替代序列延续与掩码任务，配合 λRank 列表损失，1 小时训练即可超越 16 小时 BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话行为建模&quot;,&quot;物品稀缺域推荐&quot;]"><strong><a href="learning_recommendations_from_user_actions_in_the_item_poor_insurance_domai/">Learning Recommendations from User Actions in the Item-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首个保险领域跨会话推荐模型：以 3 元组网站行为序列为输入、GRU 编码多次会话，预测会话之外发生的购买行为，Full-Ranking HR@3 较 SKNN_EB 提升 3.04%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;重复消费建模&quot;,&quot;指针网络&quot;]"><strong><a href="modeling_user_repeat_consumption_behavior_for_online_novel_recommendation/">Modeling User Repeat Consumption Behavior for Online Novel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>NovelNet 用 7 维细粒度交互属性编码每次阅读、BiGRU 建模序列、指针网络+pointwise 损失统一推荐已读与新小说，MRR@1 达 47.02%（+4.65%）。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Transformer&quot;,&quot;长短期兴趣建模&quot;]"><strong><a href="rethinking_personalized_ranking_at_pinterest_an_end_to_end_approach/">Rethinking Personalized Ranking at Pinterest: An End-to-End Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>PinnerFormer 用稠密全动作损失编码长期兴趣，实时序列捕捉短期意图，线上 Repin +12.5%、Ads CTR +14.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态偏好建模&quot;,&quot;反馈回路仿真&quot;]"><strong><a href="towards_psychologically_grounded_dynamic_preference_models/">Towards Psychologically-Grounded Dynamic Preference Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出心理学接地方法论，将单纯曝光、操作性条件作用、享乐适应形式化为偏好向量增量算子，仿真揭示振荡动态与参与度/多样性指标失效</p>
</dd>
</dl>
