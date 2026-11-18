# 🛡️ 去偏公平、可信隐私与工业架构

**ICLR / NeurIPS / ICML (RecSys) 2025** · **6** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../2026/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×6

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="arXiv" aria-pressed="false">arXiv<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Jensen Gap" aria-pressed="false">Jensen Gap<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐公平性" aria-pressed="false">LLM推荐公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM检索式推荐" aria-pressed="false">LLM检索式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Max-Min Fairness" aria-pressed="false">Max-Min Fairness<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="保形预测" aria-pressed="false">保形预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全流程数据集" aria-pressed="false">全流程数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="函数秘密共享 FSS" aria-pressed="false">函数秘密共享 FSS<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布偏移" aria-pressed="false">分布偏移<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Max-Min Fairness&quot;,&quot;Jensen Gap&quot;,&quot;对偶优化&quot;]"><strong><a href="bridging_jensen_gap_for_max_min_group_fairness_optimization_in_recommendati/">Bridging Jensen Gap for Max-Min Group Fairness Optimization in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示MMF约束下小批量采样引发Jensen Gap，提出对偶重加权算法FairDual，在三个数据集六种骨干上同时提升准确率与公平性。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;保形预测&quot;,&quot;提示工程去偏&quot;,&quot;LLM推荐公平性&quot;]"><strong><a href="facter_fairness_aware_conformal_thresholding_and_prompt_engineering_for_ena/">FACTER: Fairness-Aware Conformal Thresholding and Prompt Engineering for Enabling Fair LLM-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用保形分位数阈值+违规触发提示工程，在黑盒 LLM 推荐器上闭环降低人口统计偏见，MovieLens 违规数降 95.5% 且精度基本不掉。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;可复现评估&quot;,&quot;隐私保护数据构建&quot;,&quot;网页推荐&quot;,&quot;LLM检索式推荐&quot;]"><strong><a href="orbit_open_recommendation_benchmark_for_reproducible_research_with_hidden_t/">ORBIT: Open Recommendation Benchmark for Reproducible Research with Hidden Tests</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一可复现推荐基准，含5个公开数据集标准化划分与ClueWeb-Reco隐私安全网页推荐隐藏测试集，评测12个模型并引入LLM-QueryGen基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;全流程数据集&quot;,&quot;分布偏移&quot;,&quot;级联一致性&quot;]"><strong><a href="recflow_an_industrial_full_flow_recommendation_dataset/">RecFlow: An Industrial Full Flow Recommendation Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个覆盖工业六阶段漏斗全流程的推荐数据集，含1.9B阶段样本，揭示训练-服务分布偏移并验证阶段样本作为难负例与级联蒸馏的显著收益。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平与可信推荐&quot;,&quot;端侧训练/联邦召回&quot;,&quot;函数秘密共享 FSS&quot;,&quot;稀疏安全聚合&quot;,&quot;联邦推荐&quot;]"><strong><a href="secemb_sparsity_aware_secure_federated_learning_of_on_device_recommender_sy/">SecEmb: Sparsity-Aware Secure Federated Learning of On-Device Recommender System with Large Embedding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 FSS 点函数对稀疏物品嵌入更新做无损安全聚合，通信降 90x、端侧计算降 70x 且不泄露评分物品索引。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;机制设计&quot;,&quot;拍卖策略防伪&quot;,&quot;重要性采样&quot;]"><strong><a href="truthful_aggregation_of_llms_with_an_application_to_online_advertising/">Truthful Aggregation of LLMs with an Application to Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>MOSAIC 用重要性采样+循环单调性支付实现 LLM 广告拍卖的真实报价与最优分布收敛，20 候选即近最优。</p>
</dd>
</dl>
