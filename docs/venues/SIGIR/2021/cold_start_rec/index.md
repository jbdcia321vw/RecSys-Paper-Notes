---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐

**SIGIR 2021** · **4** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [SIGIR 2023 (3)](../../2023/cold_start_rec/) · [SIGIR 2024 (2)](../../2024/cold_start_rec/) · [SIGIR 2026 (2)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2021 (2)](../../../AAAI/2021/cold_start_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/cold_start_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2021 (1)](../../../ACM_MM/2021/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/cold_start_rec/) · [CIKM 2022 (1)](../../../CIKM/2022/cold_start_rec/) · [CIKM 2023 (3)](../../../CIKM/2023/cold_start_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2022 (1)](../../../ICDE/2022/cold_start_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICDM 2021 (1)](../../../ICDM/2021/cold_start_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/cold_start_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/cold_start_rec/) · [KDD 2021 (2)](../../../KDD/2021/cold_start_rec/) · [KDD 2023 (2)](../../../KDD/2023/cold_start_rec/) · [KDD 2024 (3)](../../../KDD/2024/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2021 (1)](../../../RecSys/2021/cold_start_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/cold_start_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/cold_start_rec/) · [IEEE TKDE 2023 (2)](../../../TKDE/2023/cold_start_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/cold_start_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/cold_start_rec/) · [WSDM 2022 (1)](../../../WSDM/2022/cold_start_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/cold_start_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/cold_start_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/cold_start_rec/) · [WWW (TheWebConf) 2023 (7)](../../../WWW/2023/cold_start_rec/) · [WWW (TheWebConf) 2024 (3)](../../../WWW/2024/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cold_start_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;图注意力网络&quot;,&quot;元学习嵌入生成&quot;]"><strong><a href="learning_graph_meta_embeddings_for_cold_start_ads_in_click_through_rate_pre/">Learning Graph Meta Embeddings for Cold-Start Ads in Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用广告属性图 + GAT + MAML 式元学习为新广告 ID 生成初始嵌入，在三个真实数据集上对五种 CTR 主干模型实现冷启动与预热双阶段 AUC 提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/精排通用模块&quot;,&quot;元学习&quot;,&quot;冷 ID 嵌入热身&quot;,&quot;特征变换去噪&quot;]"><strong><a href="learning_to_warm_up_cold_item_embeddings_for_cold_start_recommendation_with/">Learning to Warm Up Cold Item Embeddings for Cold-start Recommendation with Meta Scaling and Shifting Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用元缩放/平移网络将冷物品 ID 嵌入变换到热特征空间并以交互用户均值去噪，三点数据集八基线 + 六基础模型上 AUC 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;异质图卷积&quot;]"><strong><a href="privileged_graph_distillation_for_cold_start_recommendation/">Privileged Graph Distillation for Cold Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将属性作为节点构建异质图教师模型，用二阶间接链接构造无 CF 边的学生图，通过特权图蒸馏实现新用户/新物品/双冷启动统一推荐，Yelp/XING/Amazon 全库排序 HR/NDCG 平均最高提升 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 MAML&quot;,&quot;翻译式转移模式&quot;]"><strong><a href="sequential_recommendation_for_cold_start_users_with_meta_transitional_learn/">Sequential Recommendation for Cold-start Users with Meta Transitional Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将冷启动序列推荐重构为少样本任务，用翻译式转移编码器 + MAML 双层优化实现仅 K=2~4 次交互即可快速适配，时间戳冷启动划分下 Hit@1/MRR 显著超越 SASRec/BERT4Rec/MetaCF。</p>
</dd>
</dl>
