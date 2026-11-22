---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回

**WWW (TheWebConf) 2021** · **6** 篇论文解读

> **方向说明**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2022 (6)](../../2022/collaborative_filtering/) · [WWW (TheWebConf) 2023 (6)](../../2023/collaborative_filtering/) · [WWW (TheWebConf) 2024 (2)](../../2024/collaborative_filtering/) · [WWW (TheWebConf) 2025 (7)](../../2025/collaborative_filtering/) · [WWW (TheWebConf) 2026 (6)](../../2026/collaborative_filtering/) · [AAAI (RecSys Track) 2021 (3)](../../../AAAI/2021/collaborative_filtering/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/collaborative_filtering/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/collaborative_filtering/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/collaborative_filtering/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2022 (1)](../../../ACL_EMNLP/2022/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/collaborative_filtering/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/collaborative_filtering/) · [CIKM 2021 (4)](../../../CIKM/2021/collaborative_filtering/) · [CIKM 2022 (5)](../../../CIKM/2022/collaborative_filtering/) · [CIKM 2023 (3)](../../../CIKM/2023/collaborative_filtering/) · [CIKM 2024 (4)](../../../CIKM/2024/collaborative_filtering/) · [CIKM 2025 (2)](../../../CIKM/2025/collaborative_filtering/) · [CIKM 2026 (2)](../../../CIKM/2026/collaborative_filtering/) · [DASFAA 2025 (2)](../../../DASFAA/2025/collaborative_filtering/) · [ECIR 2021 (1)](../../../ECIR/2021/collaborative_filtering/) · [ECIR 2022 (1)](../../../ECIR/2022/collaborative_filtering/) · [ECIR 2026 (1)](../../../ECIR/2026/collaborative_filtering/) · [ICDE 2021 (2)](../../../ICDE/2021/collaborative_filtering/) · [ICDE 2023 (1)](../../../ICDE/2023/collaborative_filtering/) · [ICDE 2024 (2)](../../../ICDE/2024/collaborative_filtering/) · [ICDM 2021 (2)](../../../ICDM/2021/collaborative_filtering/) · [ICDM 2023 (1)](../../../ICDM/2023/collaborative_filtering/) · [ICDM 2024 (1)](../../../ICDM/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (2)](../../../ICLR_NeurIPS_ICML/2021/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2022 (3)](../../../ICLR_NeurIPS_ICML/2022/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/collaborative_filtering/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/collaborative_filtering/) · [IJCAI (RecSys Track) 2022 (1)](../../../IJCAI/2022/collaborative_filtering/) · [IJCAI (RecSys Track) 2024 (1)](../../../IJCAI/2024/collaborative_filtering/) · [KDD 2021 (6)](../../../KDD/2021/collaborative_filtering/) · [KDD 2022 (3)](../../../KDD/2022/collaborative_filtering/) · [KDD 2023 (1)](../../../KDD/2023/collaborative_filtering/) · [KDD 2024 (1)](../../../KDD/2024/collaborative_filtering/) · [KDD 2025 (7)](../../../KDD/2025/collaborative_filtering/) · [KDD 2026 (3)](../../../KDD/2026/collaborative_filtering/) · [RecSys 2021 (4)](../../../RecSys/2021/collaborative_filtering/) · [RecSys 2022 (2)](../../../RecSys/2022/collaborative_filtering/) · [RecSys 2023 (7)](../../../RecSys/2023/collaborative_filtering/) · [RecSys 2024 (4)](../../../RecSys/2024/collaborative_filtering/) · [RecSys 2025 (17)](../../../RecSys/2025/collaborative_filtering/) · [RecSys 2026 (8)](../../../RecSys/2026/collaborative_filtering/) · [SIGIR 2021 (7)](../../../SIGIR/2021/collaborative_filtering/) · [SIGIR 2022 (4)](../../../SIGIR/2022/collaborative_filtering/) · [SIGIR 2023 (6)](../../../SIGIR/2023/collaborative_filtering/) · [SIGIR 2024 (4)](../../../SIGIR/2024/collaborative_filtering/) · [SIGIR 2025 (3)](../../../SIGIR/2025/collaborative_filtering/) · [SIGIR 2026 (6)](../../../SIGIR/2026/collaborative_filtering/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/collaborative_filtering/) · [ACM TKDD 2021 (1)](../../../TKDD/2021/collaborative_filtering/) · [IEEE TKDE 2021 (2)](../../../TKDE/2021/collaborative_filtering/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/collaborative_filtering/) · [IEEE TKDE 2023 (4)](../../../TKDE/2023/collaborative_filtering/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/collaborative_filtering/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/collaborative_filtering/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/collaborative_filtering/) · [ACM TOIS 2021 (1)](../../../TOIS/2021/collaborative_filtering/) · [ACM TOIS 2022 (2)](../../../TOIS/2022/collaborative_filtering/) · [ACM TOIS 2023 (2)](../../../TOIS/2023/collaborative_filtering/) · [ACM TOIS 2024 (1)](../../../TOIS/2024/collaborative_filtering/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/collaborative_filtering/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/collaborative_filtering/) · [ACM TORS 2023 (1)](../../../TORS/2023/collaborative_filtering/) · [ACM TORS 2025 (2)](../../../TORS/2025/collaborative_filtering/) · [ACM TWEB 2022 (1)](../../../TWEB/2022/collaborative_filtering/) · [WSDM 2021 (1)](../../../WSDM/2021/collaborative_filtering/) · [WSDM 2022 (1)](../../../WSDM/2022/collaborative_filtering/) · [WSDM 2023 (1)](../../../WSDM/2023/collaborative_filtering/) · [WSDM 2024 (3)](../../../WSDM/2024/collaborative_filtering/) · [WSDM 2025 (1)](../../../WSDM/2025/collaborative_filtering/) · [WSDM 2026 (1)](../../../WSDM/2026/collaborative_filtering/)

🔥 **高频主题：** WWW 2021 ×5 · 协同过滤 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;秩差异感知采样&quot;,&quot;双向互学习&quot;]"><strong><a href="bidirectional_distillation_for_top_k_recommender_system/">Bidirectional Distillation for Top-K Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>打破“教师恒优”单向蒸馏假设，用方向差异化的秩差异感知采样实现师生互教，Full-Ranking 下学生 H@50 相对 CD/RD 提升 3.0%–13.9%，教师自身亦涨 3.8%–20.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/用户建模&quot;,&quot;EEG/ERP&quot;,&quot;NCF&quot;]"><strong><a href="collaborative_filtering_with_preferences_inferred_from_brain_signals/">Collaborative Filtering with Preferences Inferred from Brain Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用EEG/ERP直接解码面孔偏好并输入NCF，在31人×240项、10%缺失下RMSE 0.29，较SVD/kNN与随机显著更优。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;解释反馈配对学习&quot;,&quot;随机游走重启 (RWR)&quot;]"><strong><a href="elixir_learning_from_user_feedback_on_explanations_to_improve_recommender_m/">ELIXIR: Learning from User Feedback on Explanations to Improve Recommender Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>让解释反馈可行动：对⟨推荐,解释⟩配对做二值相似度判断，经伪物品标签传播稠密化后学习用户偏好向量，注入 RWR 图推荐，Movie/Book 真实用户研究 P@5 提升 60–79%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;行程感知注意力&quot;,&quot;双目标联合训练&quot;]"><strong><a href="itinerary_aware_personalized_deep_matching_at_fliggy/">Itinerary-aware Personalized Deep Matching at Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以「未消费订单列表=行程」为用户侧输入，用三重行程感知注意力 + 意图/点击双目标训练的双塔深度召回，在飞猪生产数据与线上 A/B 中 CTR 较 MIND 提升 1.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;评论属性注意力&quot;,&quot;信息采纳理论&quot;,&quot;属性感知负采样&quot;]"><strong><a href="leveraging_review_properties_for_effective_recommendation/">Leveraging Review Properties for Effective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 BERT 语义 + 六维评论属性注意力刻画评论有用性，配合属性一致性损失与属性感知负采样，Yelp/Amazon 时间切分下 MAP 较最优基线提升 19%–36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;投影汉明不相似度&quot;,&quot;二值哈希码&quot;,&quot;变分哈希&quot;]"><strong><a href="projected_hamming_dissimilarity_for_bit_level_importance_coding_in_collabor/">Projected Hamming Dissimilarity for Bit-Level Importance Coding in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出投影汉明不相似度与变分哈希模型，在4个数据集上以同等存储和时延实现哈希CF最高+7% NDCG、+14% MRR。</p>
</dd>
</dl>
