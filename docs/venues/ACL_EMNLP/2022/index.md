---
hide:
  - toc
---

# 🏛️ ACL / EMNLP (RecSys & CRS) 2022 推荐系统论文解读

6 篇 ACL / EMNLP (RecSys & CRS) 2022 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** EMNLP ×2 · 新闻推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔语义匹配&quot;,&quot;自监督语言对齐&quot;,&quot;档案感知多头注意力&quot;]"><strong><a href="collaborative_filtering/doctor_recommendation_in_online_health_forums_via_expertise_learning/">Doctor Recommendation in Online Health Forums via Expertise Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 MC-BERT 双塔编码匿名患者查询与医生档案/历史问诊对话，以档案-对话自监督对齐语言鸿沟、档案感知多头注意力聚合专家画像，春雨医生 119K 对话上 P@1 达 0.616。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2022 Findings&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排（印象内新闻排序）&quot;,&quot;语义增强图 SAG&quot;,&quot;双图交互注意力&quot;]"><strong><a href="graph_rec/digat_modeling_news_recommendation_with_dual_graph_interaction/">DIGAT: Modeling News Recommendation with Dual-Graph Interaction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用语义增强图 SAG 补足单条候选新闻语义，并以双图交互注意力迭代融合新闻图与用户图，MIND-large AUC 70.08、nDCG@5 提升 3.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;领域自适应后训练&quot;,&quot;新闻推荐&quot;]"><strong><a href="llm_rec/tiny_newsrec_effective_and_efficient_plm_based_news_recommendation/">Tiny-NewsRec: Effective and Efficient PLM-based News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以“标题-正文”对比匹配做领域后训练，再用两阶段多教师蒸馏压缩 PLM 新闻推荐；MIND/Feeds impression 级排序下 4 层学生超越 12 层教师，参数减半、编码提速 2.6 倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端生成式&quot;,&quot;知识图谱路径推理&quot;,&quot;双向模仿学习&quot;]"><strong><a href="agent_conversational_rec/aligning_recommendation_and_conversation_via_dual_imitation/">Aligning Recommendation and Conversation via Dual Imitation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 KG 推荐路径与对话兴趣迁移路径的双向模仿学习紧耦合推荐与对话，OpenDialKG 上 Recall@1 相对最强基线提升 14.1%–62.3%，并显著提升解释连贯性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;NAACL&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;广告诉求抽取&quot;,&quot;文本辅助特征&quot;]"><strong><a href="ctr_feature_interaction/aspect_based_analysis_of_advertising_appeals_for_search_engine_advertising/">Aspect-based Analysis of Advertising Appeals for Search Engine Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建 21 类广告诉求 (A³) 标注体系，用 BERT 检测诉求并作为辅助特征注入 CTR 回归，AUC 相对提升 4.4%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ACL/EMNLP 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;位置偏差&quot;,&quot;对抗学习&quot;,&quot;新闻推荐&quot;]"><strong><a href="debias_fairness_industrial/debiasgan_eliminating_position_bias_in_news_recommendation_with_adversarial/">DebiasGAN: Eliminating Position Bias in News Recommendation with Adversarial Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>通过位置感知/位置随机双分支与对抗学习消除新闻推荐位置偏差，在 News/Uniform 上 AUC 相对提升 1.1–1.7%。</p>
</dd>
</dl>
</section>
</div>
