---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**ACM TORS 2026** · **2** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [CIKM 2022 (2)](../../../CIKM/2022/multi_task_reranking/) · [CIKM 2023 (5)](../../../CIKM/2023/multi_task_reranking/) · [CIKM 2024 (1)](../../../CIKM/2024/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [DASFAA 2026 (2)](../../../DASFAA/2026/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/multi_task_reranking/) · [KDD 2022 (1)](../../../KDD/2022/multi_task_reranking/) · [KDD 2023 (3)](../../../KDD/2023/multi_task_reranking/) · [KDD 2024 (3)](../../../KDD/2024/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (5)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2021 (2)](../../../RecSys/2021/multi_task_reranking/) · [RecSys 2023 (1)](../../../RecSys/2023/multi_task_reranking/) · [RecSys 2024 (5)](../../../RecSys/2024/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2021 (1)](../../../SIGIR/2021/multi_task_reranking/) · [SIGIR 2022 (1)](../../../SIGIR/2022/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2021 (1)](../../../TKDE/2021/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2021 (1)](../../../WWW/2021/multi_task_reranking/) · [WWW (TheWebConf) 2022 (3)](../../../WWW/2022/multi_task_reranking/) · [WWW (TheWebConf) 2023 (3)](../../../WWW/2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** TORS ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;LLM零样本重排&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="enhancing_recommendation_diversity_by_re_ranking_with_large_language_models/">Enhancing Recommendation Diversity by Re-ranking with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 8 种零样本提示让 LLM 对 MF 候选列表做多样性重排，Anime/Goodreads 上优于随机重排但逊于贪心方法，延迟高 3 个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;碳足迹感知推荐&quot;,&quot;绿色度重排&quot;]"><strong><a href="towards_carbon_footprint_aware_recommender_systems_for_greener_item_recomme/">Towards Carbon Footprint-Aware Recommender Systems for Greener Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建首个含碳足迹的食谱数据集 RecipeEmission，基准测试 9 种传统算法发现精度优化忽视绿色度，提出 α 加权重排以极小精度损失换取最高 +89.1% 绿色度提升。</p>
</dd>
</dl>
