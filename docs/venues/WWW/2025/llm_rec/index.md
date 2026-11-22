---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**WWW (TheWebConf) 2025** · **19** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2024 (8)](../../2024/llm_rec/) · [WWW (TheWebConf) 2026 (10)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (4)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (9)](../../../ACL_EMNLP/2024/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (4)](../../../DASFAA/2026/llm_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2024 (1)](../../../ICDM/2024/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (11)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/llm_rec/) · [IJCAI (RecSys Track) 2026 (4)](../../../IJCAI/2026/llm_rec/) · [KDD 2022 (1)](../../../KDD/2022/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (10)](../../../KDD/2026/llm_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/llm_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2021 (1)](../../../SIGIR/2021/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (24)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (4)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/)

🔥 **高频主题：** WWW 2025 ×19 · 大语言模型推荐 ×18 · 生成式推荐 ×8 · 对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;CF-LLM 融合&quot;,&quot;Soft Prompt&quot;,&quot;叙事生成&quot;]"><strong><a href="beyond_retrieval_generating_narratives_in_conversational_recommender_system/">Beyond Retrieval: Generating Narratives in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 REGEN 叙事数据集并提出 CF+内容嵌入经 Adapter 注入 LLM 的融合架构，在 Office/Clothing 两域上 BLEU/ROUGE/语义相似度较单模态提升 4–12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;多维评测框架&quot;,&quot;候选位置偏差&quot;,&quot;幻觉&quot;]"><strong><a href="beyond_utility_evaluating_llm_as_recommender/">Beyond Utility: Evaluating LLM as Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出含历史长度敏感性、候选位置偏差、画像生成与幻觉四维的 LLM 推荐多维评测框架，揭示 LLM 在重排与冷启动占优但位置偏差严重。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;三元知识库&quot;,&quot;数据为中心&quot;,&quot;知识检索增强&quot;]"><strong><a href="d2k_turning_historical_data_into_retrievable_knowledge_for_recommender_syst/">D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将历史日志转为用户-物品-上下文三元知识库，Transformer 编码 + 个性化适配注入任意 CTR 模型，AUC 相对提升 1.35%–2.90%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;双源物品索引&quot;,&quot;非侵入对齐重建&quot;,&quot;退火适配器&quot;]"><strong><a href="eager_llm_enhancing_large_language_models_as_recommenders_through_exogenous/">EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以双源知识富集物品索引+非侵入多尺度对齐重建+退火适配器，将外源行为/语义信号注入 LLM 生成式推荐，Amazon 三数据集 Full-Ranking 下 Recall@10 最高提升 26.16%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;分层提示&quot;,&quot;新闻推荐&quot;,&quot;数据增强&quot;]"><strong><a href="enhancing_news_recommendation_with_hierarchical_llm_prompting/">Enhancing News Recommendation with Hierarchical LLM Prompting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层提示让 LLM 生成更丰富的新闻标题与实体，作为模型无关的数据增强模块，在 MIND 上以 AUC 68.88 超越 GLoCIM 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;外部知识蒸馏&quot;,&quot;流式分布漂移&quot;,&quot;基础模型&quot;]"><strong><a href="external_large_foundation_model_how_to_efficiently_serve_trillions_of_param/">External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以外部蒸馏+数据增强服务解耦教师训练，用辅助头与学生适配器弥合流式分布漂移，实现万亿参数基础模型零延迟增益服务多业务精排模型。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;不适推荐过滤&quot;,&quot;偏好画像&quot;,&quot;多智能体&quot;]"><strong><a href="filtering_discomforting_recommendations_with_large_language_models/">Filtering Discomforting Recommendations with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可编辑偏好画像+对话式规则生成，以插件方式过滤主观不适推荐；离线代理任务与24人一周用户研究验证有效性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;GraphRAG&quot;,&quot;可解释推荐&quot;,&quot;知识剪枝&quot;]"><strong><a href="g_refer_graph_retrieval_augmented_large_language_model_for_explainable_reco/">G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用路径级+节点级混合图检索把协同过滤信号显式翻译为文本，经知识剪枝与 RAFT 微调驱动 LLM 生成可解释推荐，BERT-F1 最高提升 2.81%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;叙事驱动推荐&quot;,&quot;提示策略评测&quot;]"><strong><a href="large_language_models_as_narrative_driven_recommenders/">Large Language Models as Narrative-Driven Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统评测 38 个 LLM 在 Reddit 叙事式电影推荐上的表现，发现零样本提示即可显著超越 doc2vec 基线 70%+，中型开源模型接近大型闭源模型。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回+精排&quot;,&quot;端到端联合训练&quot;,&quot;内容编码器&quot;,&quot;推理缓存&quot;]"><strong><a href="legommenders_a_comprehensive_content_based_recommendation_library_with_llm_/">Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个支持内容编码器与行为/交互模块端到端联合训练的推荐库，内置 LLM 编码与数据增强，覆盖 1000+ 模型组合与 15 个数据集，推理缓存带来最高 50x 加速。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;主动推荐&quot;,&quot;影响路径规划&quot;,&quot;用户模拟&quot;]"><strong><a href="leveraging_llms_for_influence_path_planning_in_proactive_recommendation/">Leveraging LLMs for Influence Path Planning in Proactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用提示工程驱动 LLM 零样本规划主动推荐影响路径，配合 LLM 用户模拟器与 SASRec 模拟器双轨评测，显著提升路径连贯性与用户接受度。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;记忆检索&quot;,&quot;困惑度标注&quot;,&quot;长期兴趣建模&quot;]"><strong><a href="leveraging_memory_retrieval_to_enhance_llm_based_generative_recommendation/">Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PPL 降幅自动标注记忆价值并训练检索器，为 LLM 生成式推荐注入长期兴趣，全库排序下 Recall@1 稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;LLM推荐&quot;,&quot;生成式推荐&quot;,&quot;个性化标题生成&quot;,&quot;风格-内容解耦&quot;,&quot;对比学习&quot;]"><strong><a href="panoramic_interests_stylistic_content_aware_personalized_headline_generatio/">Panoramic Interests: Stylistic-Content Aware Personalized Headline Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 协同抽取标题内容与风格双维特征，层次门控融合长短期全景兴趣，指导轻量生成器产出个性化新闻标题，PENS 上 ROUGE-1 达 34.26。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;表征对齐&quot;,&quot;对比学习&quot;]"><strong><a href="rallrec_improving_retrieval_augmented_large_language_model_recommendation_w/">RALLRec: Improving Retrieval Augmented Large Language Model Recommendation with Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成物品详述并做文本-协同表征对比对齐，配合时间感知重排器提升 RAG 式 LLM 推荐的检索质量。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;数据增强/冷启动&quot;,&quot;表格数据合成&quot;,&quot;特征归因重要性采样&quot;]"><strong><a href="samplellm_optimizing_tabular_data_synthesis_in_recommendations/">SampleLLM: Optimizing Tabular Data Synthesis in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>两阶段 LLM 表格数据合成框架：聚类采样示例 + 特征归因重要性采样对齐分布，冷启动场景线上 RPM +2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;自对弈偏好优化&quot;,&quot;流行度去偏&quot;]"><strong><a href="sprec_self_play_to_debias_llm_based_recommendation/">SPRec: Self-Play to Debias LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 DPO 反向 KL 的 mode-seeking 本质放大流行度偏差，提出自对弈 SPRec 以模型自身输出作负样本自适应去偏，兼顾准确率与公平性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;数据压缩/训练加速&quot;,&quot;数据集浓缩&quot;,&quot;提示演化&quot;,&quot;聚类用户合成&quot;]"><strong><a href="tf_dcon_leveraging_large_language_models_llms_to_empower_training_free_data/">TF-DCon: Leveraging Large Language Models (LLMs) to Empower Training-Free Dataset Condensation for Content-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 ChatGPT 前向式浓缩文本内容与聚类合成假用户，免迭代优化压缩 CBR 数据集 95% 仍保 97% 性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;Token Alignment&quot;,&quot;Logits Matching&quot;,&quot;LightGCN&quot;]"><strong><a href="training_large_recommendation_models_via_graph_language_token_alignment/">Training Large Recommendation Models via Graph-Language Token Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 LightGCN 用户/物品节点经投影器对齐进 LLaMA-2 词表空间，用 GLLM logits 匹配层做端到端物品预测，Full-Ranking 下 NDCG@10 最高提升 185%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;不确定性量化&quot;,&quot;不确定性分解&quot;]"><strong><a href="uncertainty_quantification_and_decomposition_for_llm_based_recommendation/">Uncertainty Quantification and Decomposition for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Plackett-Luce 单次推理估计列表排序预测熵，并分解为推荐/提示不确定性，指导不确定性感知提示提升 N@20 约 17%。</p>
</dd>
</dl>
