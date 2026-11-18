# 🤝 协同过滤与双塔召回

**ICLR / NeurIPS / ICML (RecSys) 2025** · **2** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/collaborative_filtering/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协变量调制先验" aria-pressed="false">协变量调制先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对比较学习" aria-pressed="false">成对比较学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵 Bernstein 不等式" aria-pressed="false">矩阵 Bernstein 不等式<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵分解" aria-pressed="false">矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="经验贝叶斯" aria-pressed="false">经验贝叶斯<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非凸矩阵分解收敛性" aria-pressed="false">非凸矩阵分解收敛性<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;经验贝叶斯&quot;,&quot;协变量调制先验&quot;,&quot;矩阵分解&quot;]"><strong><a href="covariate_moderated_empirical_bayes_matrix_factorization/">Covariate-moderated Empirical Bayes Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用协变量调制的经验贝叶斯先验统一矩阵分解，MovieLens 100K 上 RMSE 全面优于 EBMF/MFAI/VAE 系基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;成对比较学习&quot;,&quot;非凸矩阵分解收敛性&quot;,&quot;矩阵 Bernstein 不等式&quot;]"><strong><a href="recommendations_with_sparse_comparison_data_provably_fast_convergence_for_n/">Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将成对比较反馈建模为低秩矩阵分解下的离散选择似然，证明稀疏数据下投影梯度下降在热启动邻域内指数收敛，样本复杂度近最优。</p>
</dd>
</dl>
