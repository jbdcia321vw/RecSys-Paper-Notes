---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**WWW (TheWebConf) 2024** · **8** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2025 (19)](../../2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (9)](../../../ACL_EMNLP/2024/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2024 (1)](../../../ICDM/2024/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/llm_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×8 · WWW 2024 ×5 · WWW ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">8</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;序列推荐打分（召回/精排）&quot;,&quot;CoT 知识蒸馏&quot;,&quot;Rationale 编码&quot;]"><strong><a href="can_small_language_models_be_good_reasoners_for_sequential_recommendation/">Can Small Language Models be Good Reasoners for Sequential Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>WWW 2024][大语言模型推荐] 用教师 CoT 三步 rationale 蒸馏 LLaMA2-7B，编码增强 ID/ID-agnostic 序列推荐；Amazon 三数据集 101 候选排序，Home Hit@10 最高提升 28.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;词表扩展ID Token&quot;,&quot;互正则化预训练&quot;,&quot;Soft+Hard Prompting&quot;]"><strong><a href="collaborative_large_language_model_for_recommender_systems/">Collaborative Large Language Model for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>扩展LLM词表注入用户/物品ID token，软硬提示+互正则预训练，多项分布头单步全库打分，全库排序Recall@20相对提升7–11%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;主题级表征&quot;,&quot;用户兴趣对齐微调&quot;]"><strong><a href="generative_news_recommendation/">Generative News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GNR 用 LLM 生成主题级双层表征做召回，再融合多篇相关新闻生成个性化叙事；在 MIND 政治子集上 NDCG@5 提升 3.08%，UIFT 使叙事胜率达 74.54%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;提示增强/查询建议&quot;,&quot;实体中心知识库&quot;,&quot;In-Context Personalization&quot;]"><strong><a href="knowledge_augmented_large_language_models_for_personalized_contextual_query/">Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用实体中心个人知识库检索增强 GPT-4 提示，在 Bing 日志与人工评测上使上下文查询建议的相关性/有用性较 LLM 基线提升约 17%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;用户画像摘要&quot;,&quot;长文本摘要&quot;]"><strong><a href="modeling_user_viewing_flow_using_large_language_models_for_article_recommen/">Modeling User Viewing Flow Using Large Language Models for Article Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 做用户档案摘要、BERT 编码文章并瞬时注意力读历史，ATA 线上 UVCTR 相对 DCN 提升 2.4%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Note Compression Prompt&quot;,&quot;生成式对比学习 GCL&quot;,&quot;多任务指令微调 CSFT&quot;]"><strong><a href="notellm_a_retrievable_large_language_model_for_note_recommendation/">NoteLLM: A Retrievable Large Language Model for Note Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Note Compression Prompt 把笔记压缩为单个 [EMB] 隐状态做 I2I 召回，并以话题标签/类目生成做多任务增强；257K 全库 Full-Ranking 下 Avg Recall 94.66%，线上 CTR +16.20%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;对话/端到端生成式&quot;,&quot;LLM Agent 工具调用&quot;,&quot;推荐导向语言模型微调&quot;]"><strong><a href="recai_leveraging_large_language_models_for_next_generation_recommender_syst/">RecAI: Leveraging Large Language Models for Next-Generation Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>开源工具包 RecAI 以五大支柱（Agent/RecLM/知识插件/解释器/评测器）打通 LLM 与推荐系统，含 RecLlama 工具规划超越 GPT-3.5、7B 微调排序超越 GPT-4 等定性结论</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回/表征学习&quot;,&quot;跨视图对齐&quot;,&quot;互信息最大化&quot;]"><strong><a href="representation_learning_with_large_language_models_for_recommendation/">Representation Learning with Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RLMRec 用互信息最大化把 LLM 语义表征与协同表征对齐，模型无关地增强 6 个 CF 骨干，全库排序下 Recall@20 最高提升 7.6%</p>
</dd>
</dl>
