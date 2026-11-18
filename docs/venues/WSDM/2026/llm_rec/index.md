# 🧠 大语言模型推荐 (LLM4Rec)

**WSDM 2026** · **3** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [WSDM 2025 (1)](../../2025/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩分解" aria-pressed="false">低秩分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复杂查询推理" aria-pressed="false">复杂查询推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="幻觉度量" aria-pressed="false">幻觉度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐助手基准" aria-pressed="false">推荐助手基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方面级语义控制" aria-pressed="false">方面级语义控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型压缩" aria-pressed="false">模型压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧推理" aria-pressed="false">端侧推理<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;方面级语义控制&quot;,&quot;幻觉度量&quot;]"><strong><a href="hadsf_aspect_aware_semantic_control_for_explainable_recommendation/">HADSF: Aspect Aware Semantic Control for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>两阶段受控方面词表+动态个性化抽取，提出ADR/OFR幻觉度量，揭示幻觉-性能非单调关系，小模型可媲美大模型</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端侧推理&quot;,&quot;模型压缩&quot;,&quot;低秩分解&quot;,&quot;序列推荐&quot;]"><strong><a href="on_device_large_language_models_for_sequential_recommendation/">On-Device Large Language Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向端侧序列推荐的 LLM 任务自适应压缩框架 OD-LLM，融合 Token 协方差归一化、SVD 低秩分解与逐层渐进对齐，50% 压缩率下推荐效果无损且推理提速 3.4 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;推荐助手基准&quot;,&quot;复杂查询推理&quot;]"><strong><a href="towards_next_generation_recommender_systems_a_benchmark_for_personalized_re/">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建 RecBench+ 约3万条复杂用户查询基准，覆盖显式/隐式/误导条件与画像偏好，系统评测7个LLM并揭示SFT+RFT两阶段微调最优。</p>
</dd>
</dl>
