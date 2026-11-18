# 🔄 自监督对比学习与交互去噪

**CIKM 2025** · **3** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../../WWW/2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/self_supervised_denoising/)

🔥 **高频主题：** 对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM2Vec" aria-pressed="false">LLM2Vec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列去噪" aria-pressed="false">序列去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品互补矩阵" aria-pressed="false">物品互补矩阵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式数据增强" aria-pressed="false">生成式数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态兴趣对齐" aria-pressed="false">跨模态兴趣对齐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨模态兴趣对齐&quot;,&quot;序列去噪&quot;,&quot;LLM2Vec&quot;]"><strong><a href="empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;生成式数据增强&quot;,&quot;物品互补矩阵&quot;]"><strong><a href="generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
</dl>
