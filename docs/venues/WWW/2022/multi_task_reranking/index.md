---
hide:
  - toc
---

# ⚖️ 多任务学习与列表级重排

**WWW (TheWebConf) 2022** · **3** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (3)](../../2023/multi_task_reranking/) · [WWW (TheWebConf) 2024 (3)](../../2024/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../2026/multi_task_reranking/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/multi_task_reranking/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/multi_task_reranking/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/multi_task_reranking/) · [CIKM 2022 (2)](../../../CIKM/2022/multi_task_reranking/) · [CIKM 2023 (5)](../../../CIKM/2023/multi_task_reranking/) · [CIKM 2024 (1)](../../../CIKM/2024/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [ECIR 2024 (1)](../../../ECIR/2024/multi_task_reranking/) · [ICDE 2023 (1)](../../../ICDE/2023/multi_task_reranking/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/multi_task_reranking/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/multi_task_reranking/) · [KDD 2022 (1)](../../../KDD/2022/multi_task_reranking/) · [KDD 2023 (3)](../../../KDD/2023/multi_task_reranking/) · [KDD 2024 (3)](../../../KDD/2024/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2023 (1)](../../../RecSys/2023/multi_task_reranking/) · [RecSys 2024 (5)](../../../RecSys/2024/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2022 (1)](../../../SIGIR/2022/multi_task_reranking/) · [SIGIR 2023 (1)](../../../SIGIR/2023/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/multi_task_reranking/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2023 (1)](../../../WSDM/2023/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/)

🔥 **高频主题：** WWW 2022 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多反馈用户建模&quot;,&quot;强到弱注意力蒸馏&quot;,&quot;参与度多任务学习&quot;]"><strong><a href="feedrec_news_feed_recommendation_with_various_user_feedbacks/">FeedRec: News Feed Recommendation with Various User Feedbacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用异构/同构 Transformer 与强到弱注意力统一建模点击、跳过、分享、点踩、读完、秒关六类反馈，并以点击-读完-时长多任务训练，AUC 相对最强基线提升约 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;双边匹配市场&quot;,&quot;双随机排序&quot;,&quot;PBM&quot;]"><strong><a href="optimizing_rankings_for_recommendation_in_matching_markets/">Optimizing Rankings for Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双边匹配市场列表级联合排序，双随机策略优化社会福利，离线仿真匹配数显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Contextualized Transformer&quot;,&quot;Merged Cross-Attention&quot;,&quot;列表级多任务学习&quot;]"><strong><a href="pear_personalized_re_ranking_with_contextualized_transformer_for_recommenda/">PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以特征级 MLP + 列表/历史双向交叉注意力 Transformer 重排 top-30 列表，并加列表级多任务监督，MicroVideo 与华为新闻日志上 gAUC@30 分别提升 1.07%/1.71%</p>
</dd>
</dl>
