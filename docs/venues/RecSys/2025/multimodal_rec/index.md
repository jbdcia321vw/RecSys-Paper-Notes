---
hide:
  - toc
---

# 🧩 多模态推荐与内容理解

**RecSys 2025** · **9** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [RecSys 2026 (2)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [KDD 2025 (3)](../../../KDD/2025/multimodal_rec/) · [KDD 2026 (3)](../../../KDD/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×9

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ACT-R 认知记忆建模" aria-pressed="false">ACT-R 认知记忆建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Algorithm Performance Space" aria-pressed="false">Algorithm Performance Space<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ANN 向量检索" aria-pressed="false">ANN 向量检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CLIP 对比微调" aria-pressed="false">CLIP 对比微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CLIP 嵌入空间个性化" aria-pressed="false">CLIP 嵌入空间个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 文本增强" aria-pressed="false">LLM 文本增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Annotator" aria-pressed="false">LLM-as-Annotator<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PCA 可视化" aria-pressed="false">PCA 可视化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RecSys GenAIECommerce" aria-pressed="false">RecSys GenAIECommerce<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SigLIP 微调" aria-pressed="false">SigLIP 微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TransE 平移建模" aria-pressed="false">TransE 平移建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Visual Grounding" aria-pressed="false">Visual Grounding<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化受限召回" aria-pressed="false">个性化受限召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间切分" aria-pressed="false">全局时间切分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回前置意图判别" aria-pressed="false">召回前置意图判别<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态交叉注意力" aria-pressed="false">多模态交叉注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业数据集" aria-pressed="false">工业数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="弱监督代理标签" aria-pressed="false">弱监督代理标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集元特征" aria-pressed="false">数据集元特征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有机/算法交互解耦" aria-pressed="false">有机/算法交互解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散先验" aria-pressed="false">条件扩散先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自然语言查询推荐" aria-pressed="false">自然语言查询推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视觉语言模型" aria-pressed="false">视觉语言模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准与数据集选择" aria-pressed="false">评测基准与数据集选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨设备搜索" aria-pressed="false">跨设备搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="音频表征" aria-pressed="false">音频表征<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;评测基准与数据集选择&quot;,&quot;Algorithm Performance Space&quot;,&quot;PCA 可视化&quot;,&quot;数据集元特征&quot;]"><strong><a href="aps_explorer_navigating_algorithm_performance_spaces_for_informed_dataset_s/">APS Explorer: Navigating Algorithm Performance Spaces for Informed Dataset Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 29 算法 × 96 数据集的性能向量投影为可交互 PCA 性能空间，用可视化工具支撑可辩护的数据集选择。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;ACT-R 认知记忆建模&quot;,&quot;音频表征&quot;,&quot;序列与会话推荐&quot;]"><strong><a href="beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys GenAIECommerce&quot;,&quot;多模态推荐&quot;,&quot;召回前置意图判别&quot;,&quot;跨设备搜索&quot;,&quot;弱监督代理标签&quot;]"><strong><a href="image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉语言模型&quot;,&quot;SigLIP 微调&quot;,&quot;ANN 向量检索&quot;]"><strong><a href="improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;TransE 平移建模&quot;,&quot;多模态交叉注意力&quot;,&quot;自然语言查询推荐&quot;]"><strong><a href="just_ask_for_music_jam_multimodal_and_personalized_natural_language_music_r/">Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将查询建模为用户到物品的向量平移，用交叉注意力聚合音频/歌词/CF 多模态，在 JAMSessions 全库排序上 NDCG@10 相对 TalkRec 提升约 105%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;LLM-as-Annotator&quot;,&quot;知识蒸馏&quot;,&quot;个性化受限召回&quot;]"><strong><a href="llm_powered_nuanced_video_attribute_annotation_for_enhanced_recommendations/">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Gemini 多模态 LLM 做短视频“氛围”属性标注，经金/银集评估与蒸馏扩量，接入个性化受限召回，线上参与度 +0.49%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;生成式推荐&quot;,&quot;条件扩散先验&quot;,&quot;CLIP 嵌入空间个性化&quot;]"><strong><a href="personalized_image_generation_for_recommendations_beyond_catalogs/">Personalized Image Generation for Recommendations Beyond Catalogs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用轻量条件扩散先验从隐式反馈采样用户个性化 CLIP 嵌入，冻结 SD+IP-Adapter 解码，免微调实现可扩展个性化图像生成。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;Visual Grounding&quot;,&quot;LLM 文本增强&quot;,&quot;CLIP 对比微调&quot;]"><strong><a href="vl_clip_enhancing_multimodal_recommendations_via_visual_grounding_and_llm_a/">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Grounding DINO 裁剪商品主体 + LLM 自反思改写商品描述，再微调 CLIP 双塔，Walmart 线上 CTR +18.6%、GMV +4.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;有机/算法交互解耦&quot;]"><strong><a href="yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
</dl>
  </div>
</div>
