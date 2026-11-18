# 🕸️ 图神经网络与知识图谱推荐

**SIGIR 2025** · **2** 篇论文解读

> **方向说明**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

📌 **同领域跨会议浏览：** [SIGIR 2026 (3)](../../2026/graph_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/graph_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/graph_rec/) · [CIKM 2025 (3)](../../../CIKM/2025/graph_rec/) · [CIKM 2026 (3)](../../../CIKM/2026/graph_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/graph_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/graph_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/graph_rec/) · [KDD 2025 (3)](../../../KDD/2025/graph_rec/) · [KDD 2026 (1)](../../../KDD/2026/graph_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/graph_rec/) · [ACM TOIS 2025 (2)](../../../TOIS/2025/graph_rec/) · [ACM TOIS 2026 (2)](../../../TOIS/2026/graph_rec/) · [WSDM 2025 (3)](../../../WSDM/2025/graph_rec/) · [WWW (TheWebConf) 2025 (1)](../../../WWW/2025/graph_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索增强表征" aria-pressed="false">检索增强表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型血缘追踪" aria-pressed="false">模型血缘追踪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱增强推荐" aria-pressed="false">知识图谱增强推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱构建" aria-pressed="false">知识图谱构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨类型资源推荐" aria-pressed="false">跨类型资源推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱构建&quot;,&quot;跨类型资源推荐&quot;,&quot;模型血缘追踪&quot;]"><strong><a href="benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱增强推荐&quot;,&quot;LLM 语义嵌入&quot;,&quot;检索增强表征&quot;]"><strong><a href="comprehending_knowledge_graphs_with_large_language_models_for_recommender_s/">Comprehending Knowledge Graphs with Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 理解 KG 局部子图并构建语义物品图做全局检索增强，离线预计算语义嵌入后接入 LightGCN，全库排序下全面超越 12 个基线。</p>
</dd>
</dl>
