# 🧠 大语言模型推荐 (LLM4Rec)

**ACL / EMNLP (RecSys & CRS) 2025** · **19** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×19 · ACL 2025 ×8 · 生成式推荐 ×6 · ACL ×4 · EMNLP 2025 ×3 · ACL/EMNLP 2025 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contrastive Summarization" aria-pressed="false">Contrastive Summarization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Debate Prompting" aria-pressed="false">Debate Prompting<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GCN 检索器" aria-pressed="false">GCN 检索器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KG-RAG" aria-pressed="false">KG-RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Knowledge Graph RAG" aria-pressed="false">Knowledge Graph RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixture of LoRA Experts" aria-pressed="false">Mixture of LoRA Experts<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Multi-LoRA" aria-pressed="false">Multi-LoRA<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Query-Driven Recommendation" aria-pressed="false">Query-Driven Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化估值" aria-pressed="false">个性化估值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化奖励模型" aria-pressed="false">个性化奖励模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化推荐" aria-pressed="false">个性化推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化查询生成" aria-pressed="false">个性化查询生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层规划" aria-pressed="false">分层规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级重排" aria-pressed="false">列表级重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤用户表征" aria-pressed="false">协同过滤用户表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型-多样性均衡" aria-pressed="false">原型-多样性均衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="咨询-行为交互" aria-pressed="false">咨询-行为交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="咨询价值评估" aria-pressed="false">咨询价值评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="哈希嵌入压缩" aria-pressed="false">哈希嵌入压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="售前咨询动机建模" aria-pressed="false">售前咨询动机建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图软提示" aria-pressed="false">图软提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头早退" aria-pressed="false">多头早退<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态指令推荐" aria-pressed="false">多模态指令推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比对齐" aria-pressed="false">对比对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并行架构" aria-pressed="false">并行架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习画像精炼" aria-pressed="false">强化学习画像精炼<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="情感认知状态偏好" aria-pressed="false">情感认知状态偏好<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拍卖闭环评估" aria-pressed="false">拍卖闭环评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指令微调" aria-pressed="false">指令微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理加速" aria-pressed="false">推理加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理时扩展" aria-pressed="false">推理时扩展<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本序列推荐" aria-pressed="false">文本序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本画像对齐" aria-pressed="false">文本画像对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="智能体推荐" aria-pressed="false">智能体推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="核心集选择" aria-pressed="false">核心集选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模板增强推理" aria-pressed="false">模板增强推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="泛化能力评估" aria-pressed="false">泛化能力评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="深思式提示" aria-pressed="false">深思式提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合注意力专家池化" aria-pressed="false">混合注意力专家池化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在空间解码" aria-pressed="false">潜在空间解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="熵拐点分析" aria-pressed="false">熵拐点分析<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户反馈对齐" aria-pressed="false">用户反馈对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像/离线建模" aria-pressed="false">用户画像/离线建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束解码" aria-pressed="false">约束解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应重排" aria-pressed="false">自适应重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="词表扩展" aria-pressed="false">词表扩展<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="选择性检索" aria-pressed="false">选择性检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排/解释生成" aria-pressed="false">重排/解释生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式最高级查询" aria-pressed="false">隐式最高级查询<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐状态聚合" aria-pressed="false">隐状态聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本推荐" aria-pressed="false">零样本推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;情感认知状态偏好&quot;,&quot;文本序列推荐&quot;]"><strong><a href="a_text_based_recommender_system_that_leverages_explicit_affective_state_pre/">A Text-Based Recommender System that Leverages Explicit Affective State Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出情感-认知状态显式偏好推荐任务，用 GPT-4o 从书评挖掘 AC 语句构建数据集，Transformer 融合长短期偏好与 AC 描述，全库排序 HR@10 达 15.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同过滤用户表征&quot;,&quot;Mixture of LoRA Experts&quot;,&quot;个性化奖励模型&quot;]"><strong><a href="copl_collaborative_preference_learning_for_personalizing_llms/">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用用户-回复二部图协同过滤学习用户表征，结合 MoLE 路由个性化奖励模型，并支持无优化未见用户适配，在 UltraFeedback-P 等基准上显著超越个性化奖励模型基线。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;词表扩展&quot;,&quot;哈希嵌入压缩&quot;]"><strong><a href="cove_compressed_vocabulary_expansion_makes_better_llm_based_recommender_sys/">CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将物品映射为扩展词表唯一 token，用 logits 直接打分替代文本生成，并以哈希压缩嵌入表，实现最高 62% 精度提升与约 100× 推理加速。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;潜在空间解码&quot;,&quot;隐状态聚合&quot;,&quot;推理加速&quot;]"><strong><a href="decoding_in_latent_spaces_for_efficient_inference_in_llm_based_recommendati/">Decoding in Latent Spaces for Efficient Inference in LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用训练样本末层隐状态聚合出物品潜在表征，以 L2 距离匹配替代自回归语言空间解码，推理提速 10 倍以上且性能不降反升。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;文本画像对齐&quot;,&quot;零样本推荐&quot;]"><strong><a href="easyrec_simple_yet_effective_language_models_for_recommendation/">EasyRec: Simple yet Effective Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成用户/物品画像并以对比学习对齐协同信号，训练轻量文本编码器实现零样本推荐，Full-Ranking 下 Recall@20 最高提升 32%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;隐式最高级查询&quot;,&quot;列表级重排&quot;,&quot;深思式提示&quot;]"><strong><a href="generative_product_recommendations_for_implicit_superlative_queries/">Generative Product Recommendations for Implicit Superlative Queries</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 SUPERB 四级标注体系与深思式属性生成，用 LLM 列表级重排处理隐式最高级商品查询，nDCG@10 相对 BM25 提升约 18%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;Knowledge Graph RAG&quot;,&quot;Multi-LoRA&quot;]"><strong><a href="kerl_knowledge_enhanced_personalized_recipe_recommendation_using_large_lang/">KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 FoodKG 子图作上下文 + 多 LoRA 适配器，统一完成约束食谱推荐、烹饪步骤生成与微量营养素估计，KGQA 基准 F1 达 0.973。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;KG-RAG&quot;,&quot;图软提示&quot;,&quot;选择性检索&quot;]"><strong><a href="knowledge_graph_retrieval_augmented_generation_for_llm_based_recommendation/">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用跳场知识子图索引+流行度选择性检索+GNN软提示，将KG结构知识注入冻结LLM，MovieLens-1M ACC 相对提升58.6%且推理仅慢0.14s</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;个性化估值&quot;,&quot;拍卖闭环评估&quot;]"><strong><a href="lamp_val_large_language_models_empower_personalized_valuation_in_auction/">LaMP-Val: Large Language Models Empower Personalized Valuation in Auction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 从用户评论语义中做个性化估值，构建 23k 估值数据集 + 多样化指令 SFT，并提出 PU/PV 闭环拍卖指标，PU 较 GPT-4 提升约 6.5 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;售前咨询动机建模&quot;,&quot;混合注意力专家池化&quot;,&quot;对比对齐&quot;]"><strong><a href="maps_motivation_aware_personalized_search_via_llm_driven_consultation_align/">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM+MoAE 将售前咨询动机与查询/商品/历史双向对齐，Commercial 与 Amazon 上排序 HR@10 提升约 20%/35%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;多模态指令推荐&quot;,&quot;约束解码&quot;,&quot;模板增强推理&quot;]"><strong><a href="mira_empowering_one_touch_ai_services_on_smartphones_with_mllm_based_instru/">MIRA: Empowering One-Touch AI Services on Smartphones with MLLM-based Instruction Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构化推理+模板检索增强+前缀树约束解码，把长按图文触发对象映射到预定义手机AI指令，F1较SFT提升约60%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL Findings 2025&quot;,&quot;大语言模型推荐&quot;,&quot;用户画像/离线建模&quot;,&quot;核心集选择&quot;,&quot;原型-多样性均衡&quot;,&quot;智能体推荐&quot;]"><strong><a href="personax_a_recommendation_agent_oriented_user_modeling_framework_for_long_b/">PersonaX: A Recommendation Agent Oriented User Modeling Framework for Long Behavior Sequence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向推荐智能体的离线多画像用户建模框架，用聚类+原型-多样性均衡采样压缩长行为序列，30-50%数据即超越全量基线并大幅降低在线延迟。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排/解释生成&quot;,&quot;Debate Prompting&quot;,&quot;Contrastive Summarization&quot;,&quot;Query-Driven Recommendation&quot;]"><strong><a href="q_strum_debate_query_driven_contrastive_summarization_for_recommendation_co/">Q-STRUM Debate: Query-Driven Contrastive Summarization for Recommendation Comparison</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双人格辩论式提示替代单次摘要，为查询驱动的推荐对比摘要注入对比性与多样性，在酒店/餐厅/旅游三域 Win Rate 最高达 89%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;指令微调&quot;,&quot;强化学习画像精炼&quot;,&quot;冷启动&quot;]"><strong><a href="reclm_recommendation_instruction_tuning/">RecLM: Recommendation Instruction Tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以两轮协同指令微调+RL画像精炼为LLM注入协同信号，作为即插即用画像增强器在零样本冷启动下带来最高 200%+ 的 Recall 提升。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;并行架构&quot;,&quot;个性化查询生成&quot;,&quot;自适应重排&quot;]"><strong><a href="rethinking_llm_based_recommendations_a_personalized_query_driven_parallel_i/">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 QUEREC 并行框架，用 LLM 零样本生成个性化用户查询与特征增强物品描述，与 CF 模型并行全库检索后自适应重排，Full-Ranking 下最高提升 57%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;咨询价值评估&quot;,&quot;咨询-行为交互&quot;,&quot;个性化搜索&quot;]"><strong><a href="similarity_value_consultation_value_assessment_and_alignment_for_personaliz/">Similarity = Value? Consultation Value Assessment and Alignment for Personalized Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>指出语义相似度无法代理咨询价值，提出场景范围/后验行为/时间衰减三维价值评估与 CAI 对齐的 VAPS，检索排序双任务全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;RAG&quot;,&quot;多头早退&quot;,&quot;GCN 检索器&quot;]"><strong><a href="the_efficiency_vs_accuracy_trade_off_optimizing_rag_enhanced_llm_recommende/">The Efficiency vs. Accuracy Trade-off: Optimizing RAG-Enhanced LLM Recommender Systems Using Multi-Head Early Exit</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 GCN 检索器替代 LLM embedder 加速 RAG，并以多头早退动态终止 LLM 推理，在 5 个 CTR 数据集上兼顾 AUC 与吞吐。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;评测基准&quot;,&quot;泛化能力评估&quot;,&quot;个性化推荐&quot;,&quot;熵拐点分析&quot;]"><strong><a href="user_behavior_prediction_as_a_generic_robust_scalable_and_low_cost_evaluati/">User Behavior Prediction as a Generic, Robust, Scalable, and Low-Cost Evaluation Strategy for Estimating Generalization in LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以用户行为预测（个性化）作为抗数据污染的 LLM 泛化评测范式，用熵-交叉熵拐点框架在电影/音乐推荐上量化 GPT-4o、GPT-4o-mini、Llama 的泛化边界。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;推理时扩展&quot;,&quot;分层规划&quot;,&quot;用户反馈对齐&quot;]"><strong><a href="user_feedback_alignment_for_llm_powered_exploration_in_large_scale_recommen/">User Feedback Alignment for LLM-powered Exploration in Large-scale Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>解耦新颖性与用户对齐双 LLM，用集体隐式反馈训练对齐模型做 best-of-n 选择，线上同时提升新颖曝光与满意度。</p>
</dd>
</dl>
