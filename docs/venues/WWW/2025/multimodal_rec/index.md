# 🧩 多模态推荐与内容理解

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (3)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/)

🔥 **高频主题：** WWW 2025 ×7 · 多模态推荐 ×6

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化图像生成" aria-pressed="false">个性化图像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪查询-视频匹配" aria-pressed="false">伪查询-视频匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好对齐" aria-pressed="false">偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="停留时长预测" aria-pressed="false">停留时长预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实去偏" aria-pressed="false">反事实去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图文对齐" aria-pressed="false">图文对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务辅助监督" aria-pressed="false">多任务辅助监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态双塔" aria-pressed="false">多模态双塔<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐与内容理解" aria-pressed="false">多模态推荐与内容理解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层次Softmax" aria-pressed="false">层次Softmax<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型数据补全" aria-pressed="false">扩散模型数据补全<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="效用评估" aria-pressed="false">效用评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="显式时间建模" aria-pressed="false">显式时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询-短视频相关性" aria-pressed="false">查询-短视频相关性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稠密检索" aria-pressed="false">稠密检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视觉语言模型微调" aria-pressed="false">视觉语言模型微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论排序辅助任务" aria-pressed="false">评论排序辅助任务<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="贴纸个性化检索" aria-pressed="false">贴纸个性化检索<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排&quot;,&quot;停留时长预测&quot;,&quot;LLM 语义嵌入&quot;,&quot;评论排序辅助任务&quot;]"><strong><a href="comment_staytime_prediction_with_llm_enhanced_comment_understanding/">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个评论区停留时长数据集 KuaiComt 与两阶段 LCU 框架：LLM 领域微调 + 评论排序辅助任务，离线全指标提升且线上停留时长 +1.27%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;扩散模型数据补全&quot;,&quot;反事实去偏&quot;]"><strong><a href="generating_with_fairness_a_modality_diffused_counterfactual_framework_for_i/">Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模态感知扩散补全缺失模态，并以反事实推断剥离可见性偏差，在三个数据集上同时提升准确率与公平性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;伪查询-视频匹配&quot;,&quot;层次Softmax&quot;,&quot;查询-短视频相关性&quot;]"><strong><a href="hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;显式时间建模&quot;,&quot;多任务辅助监督&quot;]"><strong><a href="hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态双塔&quot;,&quot;稠密检索&quot;,&quot;图文对齐&quot;]"><strong><a href="multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;生成式推荐&quot;,&quot;个性化图像生成&quot;,&quot;偏好对齐&quot;]"><strong><a href="personalized_image_generation_with_large_multimodal_models/">Personalized Image Generation with Large Multimodal Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pigeon 用 LMM+掩码生成器从噪声历史图中抽取视觉偏好，两阶段偏好对齐（掩码重建+DPO）实现贴纸/海报个性化生成，个性化指标提升 7%~31%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回+重排&quot;,&quot;视觉语言模型微调&quot;,&quot;贴纸个性化检索&quot;,&quot;效用评估&quot;]"><strong><a href="persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
</dl>
