---
hide:
  - toc
---

# 🛡️ 去偏公平、可信隐私与工业架构

**WSDM 2024** · **4** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [WSDM 2023 (3)](../../2023/debias_fairness_industrial/) · [WSDM 2025 (1)](../../2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2023 (5)](../../../AAAI/2023/debias_fairness_industrial/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/debias_fairness_industrial/) · [ACL / EMNLP (RecSys & CRS) 2026 (3)](../../../ACL_EMNLP/2026/debias_fairness_industrial/) · [ACM MM 2023 (2)](../../../ACM_MM/2023/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2023 (14)](../../../CIKM/2023/debias_fairness_industrial/) · [CIKM 2024 (4)](../../../CIKM/2024/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2023 (1)](../../../DASFAA/2023/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2023 (2)](../../../ECIR/2023/debias_fairness_industrial/) · [ECIR 2024 (4)](../../../ECIR/2024/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDE 2024 (3)](../../../ICDE/2024/debias_fairness_industrial/) · [ICDM 2023 (3)](../../../ICDM/2023/debias_fairness_industrial/) · [ICDM 2024 (1)](../../../ICDM/2024/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (7)](../../../ICLR_NeurIPS_ICML/2023/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (7)](../../../ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2024 (2)](../../../IJCAI/2024/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2023 (6)](../../../KDD/2023/debias_fairness_industrial/) · [KDD 2024 (10)](../../../KDD/2024/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2023 (11)](../../../RecSys/2023/debias_fairness_industrial/) · [RecSys 2024 (16)](../../../RecSys/2024/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2023 (9)](../../../SIGIR/2023/debias_fairness_industrial/) · [SIGIR 2024 (5)](../../../SIGIR/2024/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/debias_fairness_industrial/) · [ACM TKDD 2024 (1)](../../../TKDD/2024/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/debias_fairness_industrial/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2023 (9)](../../../TOIS/2023/debias_fairness_industrial/) · [ACM TOIS 2024 (7)](../../../TOIS/2024/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2023 (1)](../../../TORS/2023/debias_fairness_industrial/) · [ACM TORS 2024 (4)](../../../TORS/2024/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WWW (TheWebConf) 2023 (11)](../../../WWW/2023/debias_fairness_industrial/) · [WWW (TheWebConf) 2024 (20)](../../../WWW/2024/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Exposure Bias&quot;,&quot;SNIPS&quot;]"><strong><a href="debiasing_sequential_recommenders_through_distributionally_robust_optimizat/">Debiasing Sequential Recommenders through Distributionally Robust Optimization over System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以系统曝光模拟器构造 DRO 名义分布，叠加高曝光惩罚与 SNIPS 评估器，SNIPS 全库排序下 Recall@20 最高 +25.9%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段防护&quot;,&quot;模型抽取攻击&quot;,&quot;可微交换矩阵&quot;]"><strong><a href="defense_against_model_extraction_attacks_on_recommender_systems/">Defense Against Model Extraction Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个推荐模型抽取防御 GRO：以可微交换矩阵反传 top-k 梯度，用学生模型模拟攻击者并最大化其损失，在全库排序下压低代理模型效用且保持目标模型效用。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;稀疏训练&quot;,&quot;模型压缩&quot;,&quot;彩票假设&quot;]"><strong><a href="dynamic_sparse_learning_a_novel_paradigm_for_efficient_recommendation/">Dynamic Sparse Learning: A Novel Paradigm for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>从零训练固定预算稀疏嵌入表，周期性剪枝/再生权重，训练与推理 MACs 同降 30%–75% 且性能持平。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回/生成式表征&quot;,&quot;因果去偏&quot;,&quot;预训练跨域推荐&quot;]"><strong><a href="pre_trained_recommender_systems_a_causal_debiasing_perspective/">Pre-trained Recommender Systems: A Causal Debiasing Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以层次贝叶斯因果去偏框架 PreRec 分离域内流行度偏差与跨域隐式偏差，实现跨市场/跨平台零样本与小样本推荐，Recall@K% 最高提升 87.7%。</p>
</dd>
</dl>
