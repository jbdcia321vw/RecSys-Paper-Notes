---
hide:
  - toc
---

# 📖 ACM TKDD 2023 推荐系统论文解读

2 篇 ACM TKDD 2023 论文解读，覆盖 2 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TKDD ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;解耦表征学习&quot;,&quot;通道感知图聚合&quot;,&quot;变分自编码器&quot;]"><strong><a href="sequential_rec/edge_enhanced_global_disentangled_graph_neural_network_for_sequential_recom/">Edge-Enhanced Global Disentangled Graph Neural Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用通道感知解耦把物品转移边拆成 K 个潜在因子通道，全局链路图 + 局部 SA-VAE 滑窗双层级学习用户意图，ML-1M N@5 相对提升 8.21%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;社交网络混杂&quot;,&quot;逆倾向加权&quot;]"><strong><a href="debias_fairness_industrial/be_causal_de_biasing_social_network_confounding_in_recommendation/">Be Causal: De-biasing Social Network Confounding in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 MNAR 选择偏差归因于社交网络混杂，用 node2vec 混杂表征 + 倾向暴露模型 + Wasserstein 平衡表征去混杂，在 Epinions/Ciao/ML-1M 上 MAE 相对最强基线降 27.7%/11.4%/19.1%。</p>
</dd>
</dl>
</section>
</div>
