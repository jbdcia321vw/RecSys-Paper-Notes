---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐

**RecSys 2025** · **9** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [RecSys 2021 (1)](../../2021/cold_start_rec/) · [RecSys 2023 (1)](../../2023/cold_start_rec/) · [RecSys 2024 (3)](../../2024/cold_start_rec/) · [RecSys 2026 (4)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/cold_start_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/cold_start_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2021 (1)](../../../ACM_MM/2021/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/cold_start_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/cold_start_rec/) · [CIKM 2023 (3)](../../../CIKM/2023/cold_start_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2022 (1)](../../../ICDE/2022/cold_start_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICDM 2021 (1)](../../../ICDM/2021/cold_start_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/cold_start_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/cold_start_rec/) · [KDD 2021 (2)](../../../KDD/2021/cold_start_rec/) · [KDD 2023 (2)](../../../KDD/2023/cold_start_rec/) · [KDD 2024 (3)](../../../KDD/2024/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/cold_start_rec/) · [SIGIR 2023 (3)](../../../SIGIR/2023/cold_start_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/cold_start_rec/) · [IEEE TKDE 2023 (2)](../../../TKDE/2023/cold_start_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/cold_start_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/cold_start_rec/) · [WSDM 2022 (1)](../../../WSDM/2022/cold_start_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/cold_start_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/cold_start_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/cold_start_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/cold_start_rec/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cold_start_rec/)

🔥 **高频主题：** 冷启动 ×9

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;通用文本嵌入&quot;,&quot;零样本迁移&quot;,&quot;有效维度&quot;]"><strong><a href="do_we_really_need_specialization_evaluating_generalist_text_embeddings_for_/">Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>实证通用文本嵌入(GTE)零样本即超越领域微调模型，并以PCA有效维度揭示空间利用率与压缩收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;评测与元评估&quot;,&quot;LLM-as-Evaluator&quot;,&quot;Serendipity&quot;,&quot;用户模拟&quot;]"><strong><a href="exploring_the_potential_of_llms_for_serendipity_evaluation_in_recommender_s/">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 模拟用户对推荐结果做惊喜度（serendipity）打分，在淘宝与 MovieLens 两个真实用户调研数据集上做元评估，最优配置 Pearson 相关达 21.5%，显著超越传统代理指标。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;物品中心探索&quot;,&quot;Beta 后验过滤&quot;]"><strong><a href="item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;内容初始化&quot;,&quot;有界 delta 校正&quot;,&quot;序列推荐&quot;]"><strong><a href="let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;零样本跨域迁移&quot;,&quot;域语义偏置&quot;,&quot;序列模式聚类&quot;]"><strong><a href="llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;流行度偏置&quot;,&quot;向量模长重标定&quot;,&quot;物品公平性&quot;]"><strong><a href="on_inherited_popularity_bias_in_cold_start_item_recommendation/">On Inherited Popularity Bias in Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示生成式冷启动模型继承热模型流行度偏置，提出基于向量模长重标定的免训练后处理去偏方法，在三个多媒体数据集上显著提升物品公平性与曝光多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;主题偏置&quot;,&quot;公平性评估&quot;]"><strong><a href="reading_between_the_lines_a_study_of_thematic_bias_in_book_recommender_syst/">Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 BERTopic 为 Book-Crossing 打主题标签，构建数据-推荐-用户三阶段主题偏置评估框架，揭示长尾小众读者被主流主题同质化的现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;冷热阈值&quot;,&quot;p-core 过滤&quot;,&quot;评估协议&quot;]"><strong><a href="recommendation_is_a_dish_better_served_warm/">Recommendation Is a Dish Better Served Warm</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统量化冷启动阈值：物品侧训练交互扫描+用户侧推理历史截断，揭示 6–15 条交互的冷热分界并批判 p-core 过滤的任意性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;社交图种子扩散&quot;,&quot;KNN 邻居扩展&quot;]"><strong><a href="socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
</dl>
