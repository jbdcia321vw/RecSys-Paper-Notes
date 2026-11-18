# 🧠 大语言模型推荐 (LLM4Rec)

**CIKM 2025** · **6** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [CIKM 2026 (9)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×6

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 知识抽取" aria-pressed="false">LLM 知识抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务自适应预训练" aria-pressed="false">任务自适应预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多语言相关性判别" aria-pressed="false">多语言相关性判别<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示迭代" aria-pressed="false">提示迭代<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本协同过滤" aria-pressed="false">文本协同过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="桥接向量" aria-pressed="false">桥接向量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征选择" aria-pressed="false">特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像推断" aria-pressed="false">用户画像推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放定律" aria-pressed="false">缩放定律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="翻译增强" aria-pressed="false">翻译增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义-行为对齐" aria-pressed="false">语义-行为对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软提示调优" aria-pressed="false">软提示调优<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用物品表征" aria-pressed="false">通用物品表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾推荐" aria-pressed="false">长尾推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="闭式解知识蒸馏" aria-pressed="false">闭式解知识蒸馏<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;多语言相关性判别&quot;,&quot;任务自适应预训练&quot;,&quot;翻译增强&quot;]"><strong><a href="alibaba_international_e_commerce_product_search_competition_dcuragons_team_/">Alibaba International E-commerce Product Search Competition DcuRAGONs Team Technical Report</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 Gemma-3-12B 为骨干，用翻译增强+任务自适应预训练+类别感知查询分组划分，在阿里国际电商多语言相关性竞赛夺冠（F1_avg≈89.1）。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;文本协同过滤&quot;,&quot;缩放定律&quot;,&quot;通用物品表征&quot;]"><strong><a href="exploring_the_upper_limits_of_text_based_collaborative_filtering_using_larg/">Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 TCF 物品编码器从 125M 一路放大到 175B，系统检验缩放极限、通用表征、对 IDCF 的竞争力与零样本迁移四项命题。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;线性自编码器&quot;,&quot;闭式解知识蒸馏&quot;,&quot;长尾推荐&quot;]"><strong><a href="llm_enhanced_linear_autoencoders_for_recommendation/">LLM-Enhanced Linear Autoencoders for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义矩阵构造 item-item 相关阵并以知识蒸馏正则注入 EASE 闭式解，全库排序下 R@20 平均提升 27.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;软提示调优&quot;,&quot;用户画像推断&quot;,&quot;向量量化&quot;]"><strong><a href="prompt_tuning_as_user_inherent_profile_inference_machine/">Prompt Tuning as User Inherent Profile Inference Machine</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户潜在画像建模为可训练软提示，用因果掩码+EM 推断、码本量化成协同 ID，AUC 最高相对提升 4.77%，已上线华为应用市场。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;特征选择&quot;,&quot;桥接向量&quot;,&quot;提示迭代&quot;]"><strong><a href="self_surrogate_light_feature_selection_with_large_language_models_in_deep_r/">SELF: Surrogate-light Feature Selection with Large Language Models in Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多 LLM 迭代提示生成语义特征排序，再以可学习桥接向量对齐任务空间，实现轻量特征选择，AUC 全面领先且线上 CTR +3.01%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;LLM 知识抽取&quot;,&quot;语义-行为对齐&quot;]"><strong><a href="trawl_external_knowledge_enhanced_recommendation_with_llm_assistance/">TRAWL: External Knowledge-Enhanced Recommendation with LLM Assistance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 从原始外部文本抽取推荐知识，再以对比学习适配器对齐语义与行为空间，ML-1M 上 AUC 稳定提升且微信线上 A/B 三项指标正向。</p>
</dd>
</dl>
