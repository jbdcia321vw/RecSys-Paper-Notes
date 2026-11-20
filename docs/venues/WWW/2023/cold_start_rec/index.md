---
hide:
  - toc
---

# ❄️ 冷启动与长尾推荐

**WWW (TheWebConf) 2023** · **7** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2024 (3)](../../2024/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (3)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2023 (3)](../../../CIKM/2023/cold_start_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [DASFAA 2023 (1)](../../../DASFAA/2023/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [KDD 2023 (2)](../../../KDD/2023/cold_start_rec/) · [KDD 2024 (3)](../../../KDD/2024/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2023 (1)](../../../RecSys/2023/cold_start_rec/) · [RecSys 2024 (3)](../../../RecSys/2024/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2023 (3)](../../../SIGIR/2023/cold_start_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [IEEE TKDE 2023 (2)](../../../TKDE/2023/cold_start_rec/) · [ACM TOIS 2023 (3)](../../../TOIS/2023/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2023 (1)](../../../WSDM/2023/cold_start_rec/) · [WSDM 2024 (1)](../../../WSDM/2024/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/)

🔥 **高频主题：** WWW 2023 ×6 · 对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;特征级数据增广&quot;,&quot;人群定向&quot;]"><strong><a href="bootstrapping_contrastive_learning_enhanced_music_cold_start_matching/">Bootstrapping Contrastive Learning Enhanced Music Cold-Start Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用动态自举的特征相关性矩阵驱动特征级对比增广，学习鲁棒歌曲表征，并结合聚类人群定向完成冷启歌曲快速加热，全库排序 Recall@50 提升 4.74%，线上有效播放 +38.47%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;嵌入压缩&quot;,&quot;分簇学习&quot;,&quot;GPCA&quot;]"><strong><a href="clustered_embedding_learning_for_recommender_systems/">Clustered Embedding Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出自上而下分簇嵌入学习 CEL，以梯度主成分分裂簇并动态重分配，在压缩嵌入表的同时显著提升冷物品与线上 AUC。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;神经架构搜索&quot;,&quot;超网络调制&quot;,&quot;FiLM&quot;]"><strong><a href="coldnas_search_to_modulate_for_user_cold_start_recommendation/">ColdNAS: Search to Modulate for User Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>把冷启动用户的 FiLM 调制函数与调制位置建模为 NAS 问题，经可证等价的空间变换将 $6^{CL}$搜索空间压缩到$2^{4L}$，一次性可微搜索后在三个基准上稳定超越 MAML 系与超网络系基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;模糊协同嵌入矫正&quot;,&quot;内容-共现双视图&quot;]"><strong><a href="contrastive_collaborative_filtering_for_cold_start_item_recommendation/">Contrastive Collaborative Filtering for Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用内容 CF 与共现 CF 双模块的对比学习把共现协同信号隐式&quot;背&quot;进内容编码器，矫正冷启动物品的模糊协同嵌入，全库用户排序下 HR@5 提升 6.1%/29.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;零样本属性推理&quot;,&quot;会话图神经网络&quot;,&quot;双意图建模&quot;]"><strong><a href="dual_intent_enhanced_graph_neural_network_for_session_based_new_item_recomm/">Dual Intent Enhanced Graph Neural Network for Session-based New Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 GSNIR 新设定与 NirGNN：用注意力+β分布双意图模拟用户决策，并以零样本属性推理把新物品映射进 GNN 空间，Amazon G&amp;GF 上 P@20 相对提升 19.33%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/候选打分&quot;,&quot;跨语言迁移&quot;,&quot;少样本用户建模&quot;,&quot;域对齐&quot;]"><strong><a href="few_shot_news_recommendation_via_cross_lingual_transfer/">Few-shot News Recommendation via Cross-lingual Transfer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>无重叠用户/新闻的跨语言少样本新闻推荐：跨域新闻检索增广 + 文档级随机掩码 + 新闻对齐 MSE 消除域偏移，MIND→Adressa 4-shot AUC 相对提升 7.69%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;类别潜空间翻译&quot;,&quot;CycleGAN 半监督&quot;]"><strong><a href="semi_supervised_adversarial_learning_for_complementary_item_recommendation/">Semi-supervised Adversarial Learning for Complementary Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为冷启动商品维护类别专属潜空间并用 CycleGAN 式半监督对抗学习训练跨类别翻译器，类别感知场景 Clothing NDCG 提升 33%。</p>
</dd>
</dl>
