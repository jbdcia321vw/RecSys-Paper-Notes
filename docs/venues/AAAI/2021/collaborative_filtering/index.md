---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**AAAI (RecSys Track) 2021** · **3** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2022 (1)](../../2022/collaborative_filtering/) · [AAAI (RecSys Track) 2024 (1)](../../2024/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2021 (4)](../../../CIKM/2021/collaborative_filtering/) · [CIKM 2022 (5)](../../../CIKM/2022/collaborative_filtering/) · [CIKM 2023 (3)](../../../CIKM/2023/collaborative_filtering/) · [CIKM 2024 (4)](../../../CIKM/2024/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2021 (1)](../../../ECIR/2021/collaborative_filtering/) · [ECIR 2022 (1)](../../../ECIR/2022/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICDE 2021 (2)](../../../ICDE/2021/collaborative_filtering/) · [ICDE 2023 (1)](../../../ICDE/2023/collaborative_filtering/) · [ICDE 2024 (2)](../../../ICDE/2024/collaborative_filtering/) · [ICDM 2021 (2)](../../../ICDM/2021/collaborative_filtering/) · [ICDM 2023 (1)](../../../ICDM/2023/collaborative_filtering/) · [ICDM 2024 (1)](../../../ICDM/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/collaborative_filtering/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/collaborative_filtering/) · [KDD 2021 (6)](../../../KDD/2021/collaborative_filtering/) · [KDD 2022 (3)](../../../KDD/2022/collaborative_filtering/) · [KDD 2023 (1)](../../../KDD/2023/collaborative_filtering/) · [KDD 2024 (1)](../../../KDD/2024/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2021 (4)](../../../RecSys/2021/collaborative_filtering/) · [RecSys 2022 (2)](../../../RecSys/2022/collaborative_filtering/) · [RecSys 2023 (7)](../../../RecSys/2023/collaborative_filtering/) · [RecSys 2024 (4)](../../../RecSys/2024/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2021 (7)](../../../SIGIR/2021/collaborative_filtering/) · [SIGIR 2022 (4)](../../../SIGIR/2022/collaborative_filtering/) · [SIGIR 2023 (6)](../../../SIGIR/2023/collaborative_filtering/) · [SIGIR 2024 (4)](../../../SIGIR/2024/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/collaborative_filtering/) · [IEEE TKDE 2021 (2)](../../../TKDE/2021/collaborative_filtering/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/collaborative_filtering/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/collaborative_filtering/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/collaborative_filtering/) · [ACM TOIS 2022 (2)](../../../TOIS/2022/collaborative_filtering/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/collaborative_filtering/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/collaborative_filtering/) · [ACM TORS 2023 (1)](../../../TORS/2023/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [ACM TWEB 2022 (1)](../../../TWEB/2022/collaborative_filtering/) · [WSDM 2021 (1)](../../../WSDM/2021/collaborative_filtering/) · [WSDM 2022 (1)](../../../WSDM/2022/collaborative_filtering/) · [WSDM 2023 (1)](../../../WSDM/2023/collaborative_filtering/) · [WSDM 2024 (3)](../../../WSDM/2024/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/collaborative_filtering/) · [WWW (TheWebConf) 2022 (6)](../../../WWW/2022/collaborative_filtering/) · [WWW (TheWebConf) 2023 (6)](../../../WWW/2023/collaborative_filtering/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/collaborative_filtering/)

🔥 **高频主题：** AAAI ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排/自适应选题&quot;,&quot;p-VAE&quot;,&quot;信息增益&quot;]"><strong><a href="educational_question_mining_at_scale_prediction_analysis_and_personalizatio/">Educational Question Mining At Scale: Prediction, Analysis and Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 p-VAE 补全稀疏学生-题目答题矩阵，并以信息论指标量化题目质量、难度与个性化选题。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/评分（矩阵分解）与事后解释&quot;,&quot;影响力估计&quot;,&quot;知识图谱补全&quot;,&quot;梯度回滚&quot;,&quot;可解释推荐&quot;]"><strong><a href="explaining_neural_matrix_factorization_with_gradient_rollback/">Explaining Neural Matrix Factorization with Gradient Rollback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>训练中累积每个三元组引起的参数更新量，测试时回滚该更新即可免重训估计样本影响力；ROAR 重训验证 PD 提升 15–39 点，开销仅 3h|D| 内存。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;逆强化学习&quot;,&quot;合成反馈闭环&quot;,&quot;生成式用户模拟&quot;]"><strong><a href="learning_to_recommend_from_sparse_data_via_generative_user_feedback/">Learning to Recommend from Sparse Data via Generative User Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用「虚拟用户」奖励估计器+反馈生成器与 CF 推荐器构成闭环，以 IRL 对抗正则在稀疏交互上稳定提升 VAE/MF 类召回 2%~9%。</p>
</dd>
</dl>
