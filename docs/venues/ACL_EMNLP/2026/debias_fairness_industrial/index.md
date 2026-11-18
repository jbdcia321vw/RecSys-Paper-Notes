# 🛡️ 去偏公平、可信隐私与工业架构

**ACL / EMNLP (RecSys & CRS) 2026** · **3** 篇论文解读

> **方向说明**：曝光/流行度/位置因果去偏 (Causal Rec)、双边公平性、联邦防投毒推荐、工业级全链路架构与评测基准

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2025 (4)](../../../AAAI/2025/debias_fairness_industrial/) · [AAAI (RecSys Track) 2026 (7)](../../../AAAI/2026/debias_fairness_industrial/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/debias_fairness_industrial/) · [ACM MM 2026 (1)](../../../ACM_MM/2026/debias_fairness_industrial/) · [CIKM 2025 (9)](../../../CIKM/2025/debias_fairness_industrial/) · [CIKM 2026 (4)](../../../CIKM/2026/debias_fairness_industrial/) · [DASFAA 2025 (1)](../../../DASFAA/2025/debias_fairness_industrial/) · [ECIR 2025 (6)](../../../ECIR/2025/debias_fairness_industrial/) · [ECIR 2026 (2)](../../../ECIR/2026/debias_fairness_industrial/) · [ICDM 2025 (1)](../../../ICDM/2025/debias_fairness_industrial/) · [ICDM 2026 (2)](../../../ICDM/2026/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (7)](../../../ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/debias_fairness_industrial/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/debias_fairness_industrial/) · [KDD 2025 (10)](../../../KDD/2025/debias_fairness_industrial/) · [KDD 2026 (7)](../../../KDD/2026/debias_fairness_industrial/) · [RecSys 2025 (9)](../../../RecSys/2025/debias_fairness_industrial/) · [RecSys 2026 (14)](../../../RecSys/2026/debias_fairness_industrial/) · [SIGIR 2025 (13)](../../../SIGIR/2025/debias_fairness_industrial/) · [SIGIR 2026 (13)](../../../SIGIR/2026/debias_fairness_industrial/) · [SIGIR-AP 2025 (2)](../../../SIGIR-AP/2025/debias_fairness_industrial/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/debias_fairness_industrial/) · [IEEE TKDE 2025 (6)](../../../TKDE/2025/debias_fairness_industrial/) · [ACM TOIS 2025 (4)](../../../TOIS/2025/debias_fairness_industrial/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/debias_fairness_industrial/) · [ACM TORS 2025 (7)](../../../TORS/2025/debias_fairness_industrial/) · [ACM TORS 2026 (3)](../../../TORS/2026/debias_fairness_industrial/) · [WSDM 2025 (1)](../../../WSDM/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2025 (9)](../../../WWW/2025/debias_fairness_industrial/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/debias_fairness_industrial/)

🔥 **高频主题：** 去偏公平与可信推荐 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec 公平性" aria-pressed="false">LLM4Rec 公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="二维分类法" aria-pressed="false">二维分类法<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息茧房" aria-pressed="false">信息茧房<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全链路（输入/数据/模型/输出）" aria-pressed="false">全链路（输入/数据/模型/输出）<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实扰动" aria-pressed="false">反事实扰动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="权威偏见审计" aria-pressed="false">权威偏见审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模糊测试" aria-pressed="false">模糊测试<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="立场检测" aria-pressed="false">立场检测<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;权威偏见审计&quot;,&quot;反事实扰动&quot;]"><strong><a href="authority_bias_in_conversational_search_engines_for_academic_paper_recommen/">Authority Bias in Conversational Search Engines for Academic Paper Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>内容受控反事实审计揭示 LLM 学术推荐存在显著权威偏见，并提出 say-do gap 与前沿模型反向失效两大诊断现象。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026 Findings&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;信息茧房&quot;,&quot;模糊测试&quot;,&quot;立场检测&quot;]"><strong><a href="content_fuzzing_for_escaping_information_cocoons_on_digital_social_media/">Content Fuzzing for Escaping Information Cocoons on Digital Social Media</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以立场检测置信度为灰盒反馈，用 LLM 迭代改写帖子翻转机器立场标签，助创作者突破信息茧房，跨 3 数据集 4 模型平均逃逸率 40%–91%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026 Findings&quot;,&quot;去偏公平与可信推荐&quot;,&quot;全链路（输入/数据/模型/输出）&quot;,&quot;LLM4Rec 公平性&quot;,&quot;二维分类法&quot;,&quot;评估协议&quot;]"><strong><a href="rethinking_fairness_in_llm_based_recommender_systems_a_survey/">Rethinking Fairness in LLM-Based Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个 LLM4Rec 公平性综述，提出「偏置机制×公平目标」二维分类法，系统梳理数据集、评估协议、缓解策略与可信交叉议题。</p>
</dd>
</dl>
