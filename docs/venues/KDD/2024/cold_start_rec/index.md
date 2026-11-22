---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐

**KDD 2024** · **3** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [KDD 2021 (2)](../../2021/cold_start_rec/) · [KDD 2023 (2)](../../2023/cold_start_rec/) · [KDD 2025 (1)](../../2025/cold_start_rec/) · [KDD 2026 (2)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/cold_start_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/cold_start_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2021 (1)](../../../ACM_MM/2021/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/cold_start_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/cold_start_rec/) · [CIKM 2023 (3)](../../../CIKM/2023/cold_start_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2022 (1)](../../../ICDE/2022/cold_start_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICDM 2021 (1)](../../../ICDM/2021/cold_start_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/cold_start_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/cold_start_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/cold_start_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/cold_start_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/cold_start_rec/) · [SIGIR 2023 (3)](../../../SIGIR/2023/cold_start_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/cold_start_rec/) · [IEEE TKDE 2023 (2)](../../../TKDE/2023/cold_start_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/cold_start_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/cold_start_rec/) · [WSDM 2022 (1)](../../../WSDM/2022/cold_start_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/cold_start_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/cold_start_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/cold_start_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/cold_start_rec/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cold_start_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态对比学习&quot;,&quot;关系感知预训练&quot;]"><strong><a href="larp_language_audio_relational_pre_training_for_cold_start_playlist_continu/">LARP: Language Audio Relational Pre-training for Cold-Start Playlist Continuation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出语言-音频三阶段对比预训练 LARP，在无重叠冷启动歌单续写的全库排序中显著超越多模态基线。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;MAML 元学习&quot;,&quot;Meta Embedding&quot;,&quot;工业级个性化&quot;]"><strong><a href="limaml_personalization_of_deep_recommender_models_via_meta_learning/">LiMAML: Personalization of Deep Recommender Models via Meta Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将推荐网络解耦为 Meta/Global 双块，仅对 Meta Block 做 MAML 元学习并离线蒸馏成定长 meta embedding，实现百亿参数模型的少样本个性化在线服务，线上 CTR 提升 2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;超网络生成特征图&quot;,&quot;元学习暖启动&quot;]"><strong><a href="warming_up_cold_start_ctr_prediction_by_learning_item_specific_feature_inte/">Warming Up Cold-Start CTR Prediction by Learning Item-Specific Feature Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用超网络为每个新物品生成专属特征交互图，配合可证任意阶交互的定制 GNN 与元学习暖启动，在 MovieLens/Taobao 冷启动与三阶段暖启动 AUC 上全面领先。</p>
</dd>
</dl>
