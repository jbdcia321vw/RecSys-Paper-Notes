---
hide:
  - toc
---

# 🏛️ IJCAI (RecSys Track) 2024 推荐系统论文解读

5 篇 IJCAI (RecSys Track) 2024 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** IJCAI 2024 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;截断SVD&quot;,&quot;闭式解线性自编码器&quot;,&quot;噪声鲁棒&quot;]"><strong><a href="collaborative_filtering/svd_ae_simple_autoencoders_for_collaborative_filtering/">SVD-AE: Simple Autoencoders for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出基于截断SVD的线性自编码器闭式解，直接重建用户-物品矩阵，以低秩归纳偏置兼顾全库排序精度、噪声鲁棒性与免训练效率。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;过平滑 (Over-smoothing)&quot;,&quot;模型无关即插即用框架&quot;,&quot;协同邻域校正&quot;]"><strong><a href="graph_rec/dgr_a_general_graph_desmoothing_framework_for_recommendation_via_global_and/">DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以全局向量扰动 (GMP) 与局部协同校正损失 (LEC) 构成模型无关的过平滑抑制框架，即插即用于 5 种 GCN 骨干，在 5 个基准上 Full-Ranking 平均 Recall@20 提升 2%~11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/序列建模&quot;,&quot;四维张量降维&quot;,&quot;多级跨域融合与跨域桥&quot;,&quot;辅助学习技术&quot;]"><strong><a href="cross_domain_multi_scenario/a_survey_on_cross_domain_sequential_recommendation/">A Survey on Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以四维数据张量统一刻画 CDSR，宏观抽象三级融合结构与三类跨域桥、微观梳理基础/辅助技术，并汇总 Amazon Foods/Kitchens 全库排序结果（HR@10 较 SOCCF 最优提升 42.2%）</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/端到端联邦个性化&quot;,&quot;联邦推荐&quot;,&quot;低秩适配器(LoRA 式)&quot;,&quot;自适应门控融合&quot;]"><strong><a href="debias_fairness_industrial/federated_adaptation_for_foundation_model_based_recommendations/">Federated Adaptation for Foundation Model-based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 FedPA：以预训练双塔推荐模型为底座，客户端仅训练低秩适配器（用户级私有 + 用户组级共享）并由自适应门控融合公共知识与个性化，联邦优化中冻结物品/预测模块，AUC 最高相对提升 3.4%，可训练参数减少 40%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;黑盒隐私评测&quot;,&quot;成员推断攻击&quot;,&quot;无影子攻击&quot;]"><strong><a href="debias_fairness_industrial/shadow_free_membership_inference_attacks_recommender_systems_are_more_vulne/">Shadow-Free Membership Inference Attacks: Recommender Systems Are More Vulnerable Than You Thought</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出无影子成员推断攻击：空画像热门锚点+自适应用户距离比较，黑盒免训练，3.7秒取得高TPR低FPR。</p>
</dd>
</dl>
</section>
</div>
