---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**WWW (TheWebConf) 2026** · **7** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (9)](../../2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/)

🔥 **高频主题：** WWW 2026 ×5 · 长序列建模 ×3 · WWW ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长序列建模&quot;]"><strong><a href="blossomrec_block_level_fused_sparse_attention_mechanism_for_sequential_reco/">BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双路块级稀疏注意力（LTIS 块选择 + STIS 幂律掩码）经门控融合，长序列训练提速 3.2×、显存降至 1/11，精度持平或超越 SASRec。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;LLM 语义嵌入&quot;,&quot;行为校准&quot;]"><strong><a href="generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Legendre 正交多项式&quot;,&quot;Hyena 长卷积&quot;,&quot;次二次复杂度&quot;]"><strong><a href="hyena_operator_for_fast_sequential_recommendation/">Hyena Operator for Fast Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Legendre 正交多项式参数化 Hyena 长卷积核，线性复杂度建模超长用户序列，XLong 上 NDCG@10 提升 17.65% 且训练最高 6× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;交叉注意力&quot;,&quot;长度外推&quot;]"><strong><a href="make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;长序列建模&quot;]"><strong><a href="mixture_of_sequence_theme_aware_mixture_of_experts_for_long_sequence_recomm/">Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示长序列“会话跳跃”现象，用主题码本路由+多尺度专家融合做子序列抽取，AUC 平均提升 0.68% 且 FLOPs 更低</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时序子图&quot;,&quot;兴趣演化建模&quot;]"><strong><a href="modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;冷启动基准&quot;]"><strong><a href="vk_lsvd_a_large_scale_industrial_dataset_for_short_video_recommendation/">VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>发布 407 亿交互、千万用户、两千万短视频的工业级开放数据集，配全局时间切分与冷启动竞赛基准。</p>
</dd>
</dl>
