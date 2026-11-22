---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**RecSys 2023** · **3** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [RecSys 2024 (5)](../../2024/llm_rec/) · [RecSys 2025 (4)](../../2025/llm_rec/) · [RecSys 2026 (11)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (4)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (9)](../../../ACL_EMNLP/2024/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (4)](../../../DASFAA/2026/llm_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (3)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICDM 2024 (1)](../../../ICDM/2024/llm_rec/) · [ICDM 2025 (2)](../../../ICDM/2025/llm_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (11)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/llm_rec/) · [IJCAI (RecSys Track) 2026 (4)](../../../IJCAI/2026/llm_rec/) · [KDD 2022 (1)](../../../KDD/2022/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (10)](../../../KDD/2026/llm_rec/) · [SIGIR 2021 (1)](../../../SIGIR/2021/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (24)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (4)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/llm_rec/)

🔥 **高频主题：** 大语言模型推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Genre Spectrum&quot;,&quot;内容元数据嵌入&quot;]"><strong><a href="beyond_labels_leveraging_deep_learning_and_llms_for_content_metadata/">Beyond Labels: Leveraging Deep Learning and LLMs for Content Metadata</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 Genre Spectrum，用多标签监督把离散类型标签映射到连续隐空间，Mixup 增广后于 1.1M 影视数据 top-100 类型相似度低热度桶提升约 12 个百分点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;大语言模型推荐&quot;,&quot;召回/特征增强&quot;,&quot;异构行为融合&quot;,&quot;指令微调 LoRA&quot;,&quot;自然语言用户画像&quot;]"><strong><a href="heterogeneous_knowledge_fusion_a_novel_approach_for_personalized_recommenda/">Heterogeneous Knowledge Fusion: A Novel Approach for Personalized Recommendation via LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ChatGPT 将美团外卖多维异构行为改写为自然语言知识，再以 LoRA 指令微调 ChatGLM-6B 输出类别/POI 偏好，离线 HR@5 提升 52.5%，冷启动在线 CTR +2.45%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;LLM4Rec&quot;,&quot;召回&quot;,&quot;合成查询数据增强&quot;,&quot;叙事驱动推荐&quot;,&quot;双塔检索&quot;]"><strong><a href="large_language_model_augmented_narrative_driven_recommendations/">Large Language Model Augmented Narrative Driven Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 175B LLM 从 Yelp 评论合成叙事查询，训练 110M 双塔/交叉编码器，Pointrec 上 NDCG@10 相对提升 7%+。</p>
</dd>
</dl>
