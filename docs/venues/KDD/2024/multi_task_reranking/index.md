---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**KDD 2024** · **3** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [KDD 2022 (1)](../../2022/multi_task_reranking/) · [KDD 2023 (3)](../../2023/multi_task_reranking/) · [KDD 2025 (4)](../../2025/multi_task_reranking/) · [KDD 2026 (5)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [CIKM 2022 (2)](../../../CIKM/2022/multi_task_reranking/) · [CIKM 2023 (5)](../../../CIKM/2023/multi_task_reranking/) · [CIKM 2024 (1)](../../../CIKM/2024/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [DASFAA 2026 (2)](../../../DASFAA/2026/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/multi_task_reranking/) · [RecSys 2021 (2)](../../../RecSys/2021/multi_task_reranking/) · [RecSys 2023 (1)](../../../RecSys/2023/multi_task_reranking/) · [RecSys 2024 (5)](../../../RecSys/2024/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2021 (1)](../../../SIGIR/2021/multi_task_reranking/) · [SIGIR 2022 (1)](../../../SIGIR/2022/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/multi_task_reranking/) · [ACM TORS 2026 (2)](../../../TORS/2026/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2021 (1)](../../../WWW/2021/multi_task_reranking/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/multi_task_reranking/) · [WWW (TheWebConf) 2023 (3)](../../../WWW/2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multi_task_reranking/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;对比上下文编码&quot;,&quot;多样性推荐&quot;]"><strong><a href="contextual_distillation_model_for_diversified_recommendation/">Contextual Distillation Model for Diversified Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用兴趣感知 MMR 作教师、Gumbel-Top-k 对比上下文编码器作学生，把 O(NK²) 的贪心多样性重排蒸馏成 O(N log K) 的 point-wise 打分，全链路可用。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;梯度冲突消解&quot;,&quot;多目标优化&quot;]"><strong><a href="gradcraft_elevating_multi_task_recommendations_through_holistic_gradient_cr/">GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出“幅度柔性对齐 + 全局方向去冲突”的两阶段梯度整形，在 Wechat/Kuaishou 上 AV-G 分别提升 1.06%/3.79%，线上 VV +0.95%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;非自回归生成&quot;,&quot;对比解码&quot;]"><strong><a href="non_autoregressive_generative_models_for_reranking_recommendation/">Non-autoregressive Generative Models for Reranking Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>非自回归生成器一次解码全部位置，配合匹配模型、序列级 unlikelihood 训练与对比解码；Avito AUC 0.7234、快手 Recall@6 74.86%，推理约 5 倍提速。</p>
</dd>
</dl>
