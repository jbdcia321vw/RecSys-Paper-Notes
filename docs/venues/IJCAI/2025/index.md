---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2025 推荐系统论文解读

4 篇 IJCAI (RecSys Track) 2025 论文解读，覆盖 3 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2025 ×4

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Flow Matching" aria-pressed="false">Flow Matching<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN 鲁棒性" aria-pressed="false">LightGCN 鲁棒性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Test-Time Training" aria-pressed="false">Test-Time Training<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单节点触发器" aria-pressed="false">单节点触发器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数校正" aria-pressed="false">参数校正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图后门攻击" aria-pressed="false">图后门攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标奖励" aria-pressed="false">多目标奖励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习策略梯度" aria-pressed="false">强化学习策略梯度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户行为采样" aria-pressed="false">用户行为采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="确定性ODE采样" aria-pressed="false">确定性ODE采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端云协同" aria-pressed="false">端云协同<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;Test-Time Training&quot;,&quot;端云协同&quot;,&quot;参数校正&quot;]"><strong><a href="sequential_rec/device_cloud_collaborative_correction_for_on_device_recommendation/">Device-Cloud Collaborative Correction for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端侧 TTT 自校正网络 + 云端全局校正网络，以 RNN 级开销逼近 Transformer 精度，UAUC 最高提升 4.56%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;用户行为采样&quot;,&quot;强化学习策略梯度&quot;,&quot;多目标奖励&quot;]"><strong><a href="sequential_rec/towards_automatic_sampling_of_user_behaviors_for_sequential_recommender_sys/">Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 AutoSAM，用轻量采样器+强化学习多目标奖励自适应丢弃低信息历史行为，在四个真实数据集全库排序下 NDCG@10 平均提升约 8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;生成式推荐&quot;,&quot;Flow Matching&quot;,&quot;确定性ODE采样&quot;,&quot;序列推荐&quot;]"><strong><a href="generative_rec/flow_matching_based_sequential_recommender_model/">Flow Matching based Sequential Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用直线流轨迹+无噪预测目标+ODE确定性采样替代扩散式序列推荐，四数据集全库排序平均提升6.53%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;图后门攻击&quot;,&quot;单节点触发器&quot;,&quot;LightGCN 鲁棒性&quot;]"><strong><a href="debias_fairness_industrial/single_node_trigger_backdoor_attacks_in_graph_based_recommendation_systems/">Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出单节点触发器后门攻击 SNT-BA，仅注入 1 个虚假用户即可让目标物品在 99% 目标用户 Top-10 中曝光，同时 HR@20 仅降约 5%。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
