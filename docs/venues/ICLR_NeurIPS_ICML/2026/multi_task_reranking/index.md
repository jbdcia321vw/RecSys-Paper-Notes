# ⚖️ 多任务学习与列表级重排

**ICLR / NeurIPS / ICML (RecSys) 2026** · **2** 篇论文解读

> **方向说明**：MMoE/PLE 多任务多目标精排、帕累托最优、上下文列表重排 (Re-ranking) 与粗精排级联一致性

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/multi_task_reranking/) · [CIKM 2025 (4)](../../../CIKM/2025/multi_task_reranking/) · [CIKM 2026 (5)](../../../CIKM/2026/multi_task_reranking/) · [KDD 2025 (4)](../../../KDD/2025/multi_task_reranking/) · [KDD 2026 (3)](../../../KDD/2026/multi_task_reranking/) · [RecSys 2025 (5)](../../../RecSys/2025/multi_task_reranking/) · [RecSys 2026 (7)](../../../RecSys/2026/multi_task_reranking/) · [SIGIR 2025 (4)](../../../SIGIR/2025/multi_task_reranking/) · [SIGIR 2026 (4)](../../../SIGIR/2026/multi_task_reranking/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/multi_task_reranking/) · [WSDM 2025 (1)](../../../WSDM/2025/multi_task_reranking/) · [WSDM 2026 (1)](../../../WSDM/2026/multi_task_reranking/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/multi_task_reranking/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/multi_task_reranking/)

🔥 **高频主题：** 多任务学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标推荐" aria-pressed="false">多目标推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式列表排序" aria-pressed="false">生成式列表排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标条件监督学习" aria-pressed="false">目标条件监督学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组相对优化" aria-pressed="false">组相对优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放定律" aria-pressed="false">缩放定律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;目标条件监督学习&quot;,&quot;多目标推荐&quot;]"><strong><a href="goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式列表排序&quot;,&quot;组相对优化&quot;,&quot;缩放定律&quot;]"><strong><a href="goalrank_group_relative_optimization_for_a_large_ranking_model/">GoalRank: Group-Relative Optimization for a Large Ranking Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>证明单生成器排序器近似误差严格优于多生成器-评估器，提出组相对优化训练大排序模型，离线 H@6 最高 +25.4%，线上全量部署。</p>
</dd>
</dl>
