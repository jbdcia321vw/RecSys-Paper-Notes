---
hide:
  - toc
---

# 👥 社交与群组推荐

**RecSys 2026** · **2** 篇论文解读

> **方向说明**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

📌 **同领域跨会议浏览：** [RecSys 2025 (3)](../../2025/social_group_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/social_group_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/social_group_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/social_group_rec/) · [CIKM 2023 (1)](../../../CIKM/2023/social_group_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/social_group_rec/) · [CIKM 2025 (2)](../../../CIKM/2025/social_group_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/social_group_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/social_group_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/social_group_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/social_group_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (1)](../../../ICLR_NeurIPS_ICML/2023/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/social_group_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/social_group_rec/) · [KDD 2024 (1)](../../../KDD/2024/social_group_rec/) · [SIGIR 2022 (1)](../../../SIGIR/2022/social_group_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/social_group_rec/) · [IEEE TKDE 2022 (5)](../../../TKDE/2022/social_group_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/social_group_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/social_group_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/social_group_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/social_group_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/social_group_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/social_group_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/social_group_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/social_group_rec/) · [WWW (TheWebConf) 2026 (2)](../../../WWW/2026/social_group_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;社交与群组推荐&quot;,&quot;评测协议&quot;,&quot;并列感知评估&quot;,&quot;分数压缩&quot;,&quot;BPR 温度缩放&quot;]"><strong><a href="are_we_really_making_progress_in_group_recommendation_unmasking_the_tie_bre/">Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示群组推荐中训练期 sigmoid 分数压缩与评估期确定性破平交互导致的系统性评估偏差，提出并列感知 HR/NDCG 精确期望协议，使多个 SOTA 方法报告增益大幅缩水。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;LLM-as-Judge&quot;,&quot;社会选择聚合&quot;]"><strong><a href="consensus_vs_dissent_dynamic_llm_modeling_of_subjective_preferences_in_grou/">Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用人类问卷+DeepSeek-V3.1 蒸馏 CoT 微调 Llama/OLMo 作实时判断模型，动态挑选社会选择聚合策略，用户研究(n=284)中公平与共识得分最高。</p>
</dd>
</dl>
