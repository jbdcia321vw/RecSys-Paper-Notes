---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**ACM MM 2023** · **3** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [ACM MM 2024 (1)](../../2024/sequential_rec/) · [ACM MM 2025 (1)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;知识图谱提示&quot;,&quot;知识树掩码&quot;]"><strong><a href="knowledge_prompt_tuning_for_sequential_recommendation/">Knowledge Prompt-tuning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将知识图谱经关系模板转成知识提示，并用知识树掩码去噪，与 T5 掩码个性化提示联合调优；Leave-One-Out + Full-Ranking 下 books NDCG@5 较 P5 提升 40.65%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态特征融合&quot;,&quot;在线知识蒸馏&quot;,&quot;去偏 In-batch 损失&quot;]"><strong><a href="online_distillation_enhanced_multi_modal_transformer_for_sequential_recomme/">Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ID-aware 多模态 Transformer 与在线蒸馏解耦 ID/模态特征的优化冲突，四数据集 Recall@10 相对最优基线提升 7.4%–13.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布表征&quot;]"><strong><a href="personalized_behavior_aware_transformer_for_multi_behavior_sequential_recom/">Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布刻画物品/行为/位置并借 Wasserstein 距离重构自注意力，把个性化多行为模式与行为-时间协同注入 Transformer；Leave-One-Out + HR/NDCG 上 Taobao HR@5 相对最优基线提升 6.37%。</p>
</dd>
</dl>
