---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2023 推荐系统论文解读

7 篇 ACL / EMNLP (RecSys & CRS) 2023 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** ACL ×2 · ACL 2023 ×2 · EMNLP ×2 · 端到端生成式 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;基础设施与评测&quot;,&quot;模块化设计&quot;,&quot;可复现研究&quot;]"><strong><a href="sequential_rec/newsreclib_a_pytorch_lightning_library_for_neural_news_recommendation/">NewsRecLib: A PyTorch-Lightning Library for Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>EMNLP 2023 序列与会话推荐，提出高度模块化的神经新闻推荐开源库，统一评估协议与组件消融，解决研究不可复现与不公平比较痛点。</p>
</dd>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;召回与精排&quot;,&quot;预训练&quot;,&quot;新闻推荐&quot;]"><strong><a href="sequential_rec/punr_pre_training_with_user_behavior_modeling_for_news_recommendation/">PUNR: Pre-training with User Behavior Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Findings of EMNLP 2023 序列与会话推荐：提出用户行为掩码与生成双预训练任务，结合孪生编码器优化新闻推荐用户表征。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2023&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;行为序列语言建模&quot;,&quot;跨业务迁移学习&quot;]"><strong><a href="llm_rec/pivotal_role_of_language_modeling_in_recommender_systems_enriching_task_spe/">Pivotal Role of Language Modeling in Recommender Systems: Enriching Task-specific and Task-agnostic Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>LMRec 以共享 Transformer 联合优化行为序列语言建模与推荐点积目标，并用 [EOS] 隐向量线性探针迁移，采样排序下 Recall@10 提升 3.0%–8.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;检索增强生成&quot;,&quot;方面学习&quot;,&quot;可解释推荐&quot;]"><strong><a href="agent_conversational_rec/explainable_recommendation_with_personalized_review_retrieval_and_aspect_le/">Explainable Recommendation with Personalized Review Retrieval and Aspect Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>ERRA 以训练集评论为检索语料、用 Sentence-BERT+Faiss 检索个性化句子并结合 top-n 方面掩码注入 Transformer，联合打分与解释生成；8:1:1 划分下 TripAdvisor RMSE 降 3.3%、BLEU-4 升 15.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;主观偏好消歧&quot;,&quot;多模态推荐基准&quot;]"><strong><a href="agent_conversational_rec/multimodal_recommendation_dialog_with_subjective_preference_a_new_challenge/">Multimodal Recommendation Dialog with Subjective Preference: A New Challenge and Benchmark</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>构建 12K 主观偏好购物对话基准 SURE，提出偏好消歧/区域理解/多模态推荐三任务与 MRA 基线，揭示现有模型远未解决。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回/生成式端到端&quot;,&quot;树结构推理&quot;,&quot;知识图谱&quot;]"><strong><a href="agent_conversational_rec/trea_tree_structure_reasoning_schema_for_conversational_recommendation/">TREA: Tree-Structure Reasoning Schema for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多层级推理树显式建模实体因果链，配合分支隔离/语义对齐损失，ReDial/TG-ReDial 全实体 R@50 提升 1.7%/41.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;多模态推荐&quot;,&quot;端到端生成式&quot;,&quot;多模态个性化提示&quot;,&quot;参数高效 Adapter 微调&quot;]"><strong><a href="multimodal_rec/vip5_towards_multimodal_foundation_models_for_recommendation/">VIP5: Towards Multimodal Foundation Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>冻结 P5 主干、插入 Adapter 微调，将图像 token 拼入多模态个性化提示，统一序列/直接推荐与解释生成三任务，全榜超越 P5。</p>
</dd>
</dl>
</section>
</div>
