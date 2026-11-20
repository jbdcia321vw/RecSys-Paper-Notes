---
hide:
  - toc
---

# 🧠 大语言模型推荐 (LLM4Rec)

**ICDM 2025** · **2** 篇论文解读

> **方向说明**：LLM 指令微调推荐、In-Context 列表重排、语义表征增强、大模型向轻量推荐模型知识蒸馏

📌 **同领域跨会议浏览：** [ICDM 2024 (1)](../../2024/llm_rec/) · [ICDM 2026 (3)](../../2026/llm_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/llm_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/llm_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (1)](../../../ACL_EMNLP/2023/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (9)](../../../ACL_EMNLP/2024/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (19)](../../../ACL_EMNLP/2025/llm_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (10)](../../../ACL_EMNLP/2026/llm_rec/) · [CIKM 2024 (4)](../../../CIKM/2024/llm_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/llm_rec/) · [CIKM 2026 (9)](../../../CIKM/2026/llm_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/llm_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/llm_rec/) · [ECIR 2024 (2)](../../../ECIR/2024/llm_rec/) · [ECIR 2025 (3)](../../../ECIR/2025/llm_rec/) · [ECIR 2026 (2)](../../../ECIR/2026/llm_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/llm_rec/) · [ICDE 2025 (3)](../../../ICDE/2025/llm_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (10)](../../../ICLR_NeurIPS_ICML/2025/llm_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/llm_rec/) · [KDD 2024 (2)](../../../KDD/2024/llm_rec/) · [KDD 2025 (10)](../../../KDD/2025/llm_rec/) · [KDD 2026 (7)](../../../KDD/2026/llm_rec/) · [RecSys 2023 (3)](../../../RecSys/2023/llm_rec/) · [RecSys 2024 (5)](../../../RecSys/2024/llm_rec/) · [RecSys 2025 (4)](../../../RecSys/2025/llm_rec/) · [RecSys 2026 (11)](../../../RecSys/2026/llm_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/llm_rec/) · [SIGIR 2024 (3)](../../../SIGIR/2024/llm_rec/) · [SIGIR 2025 (14)](../../../SIGIR/2025/llm_rec/) · [SIGIR 2026 (22)](../../../SIGIR/2026/llm_rec/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/llm_rec/) · [IEEE TKDE 2025 (4)](../../../TKDE/2025/llm_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/llm_rec/) · [ACM TOIS 2025 (9)](../../../TOIS/2025/llm_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/llm_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/llm_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/llm_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/llm_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/llm_rec/) · [WWW (TheWebConf) 2024 (8)](../../../WWW/2024/llm_rec/) · [WWW (TheWebConf) 2025 (19)](../../../WWW/2025/llm_rec/) · [WWW (TheWebConf) 2026 (8)](../../../WWW/2026/llm_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;时序用户画像&quot;,&quot;注意力融合&quot;]"><strong><a href="effectiveness_of_llms_in_temporal_user_profiling_for_recommendation/">Effectiveness of LLMs in Temporal User Profiling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 分别生成短期/长期自然语言画像，经 SBERT 编码与注意力融合做内容推荐，Movies&amp;TV 上 Recall@10 相对提升 17%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;小语言模型蒸馏&quot;,&quot;倒置强化学习&quot;,&quot;多模态提示生成&quot;]"><strong><a href="scaling_multimodal_search_and_recommendation_with_small_language_models_via/">Scaling Multimodal Search and Recommendation with Small Language Models via Upside-Down Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Llama-3 合成 5200 万意图-提示对蒸馏进 104M nanoGPT，并以倒置强化学习实现长度/模态可控的多任务提示生成，性能逼近 8B 模型而显存仅 500MB。</p>
</dd>
</dl>
