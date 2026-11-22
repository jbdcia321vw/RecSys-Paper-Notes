---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**SIGIR 2022** · **12** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [SIGIR 2021 (4)](../../2021/sequential_rec/) · [SIGIR 2023 (9)](../../2023/sequential_rec/) · [SIGIR 2024 (12)](../../2024/sequential_rec/) · [SIGIR 2025 (12)](../../2025/sequential_rec/) · [SIGIR 2026 (15)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2021 (4)](../../../AAAI/2021/sequential_rec/) · [AAAI (RecSys Track) 2022 (2)](../../../AAAI/2022/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [ACM MM 2026 (2)](../../../ACM_MM/2026/sequential_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/sequential_rec/) · [CIKM 2022 (9)](../../../CIKM/2022/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (10)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/sequential_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/sequential_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/sequential_rec/) · [ICDE 2022 (7)](../../../ICDE/2022/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2021 (2)](../../../ICDM/2021/sequential_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/sequential_rec/) · [IJCAI (RecSys Track) 2022 (2)](../../../IJCAI/2022/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/sequential_rec/) · [KDD 2022 (4)](../../../KDD/2022/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (11)](../../../KDD/2026/sequential_rec/) · [RecSys 2021 (8)](../../../RecSys/2021/sequential_rec/) · [RecSys 2022 (10)](../../../RecSys/2022/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (4)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TORS 2026 (3)](../../../TORS/2026/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/sequential_rec/) · [WSDM 2022 (4)](../../../WSDM/2022/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (4)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/sequential_rec/) · [WWW (TheWebConf) 2022 (8)](../../../WWW/2022/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;数据分布自适应&quot;,&quot;模型补丁参数调制&quot;]"><strong><a href="ada_ranker_a_data_distribution_adaptive_ranking_paradigm_for_sequential_rec/">Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 Neural Process 提取候选集分布 z，经 FiLM 输入调制与参数池补丁调制，把全局精排器即时改写为局部模型，NDCG 平均提升约 2.5%，新分布下增益达 15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图序列粗到细建模&quot;,&quot;学习式局部注意力&quot;,&quot;联合概率解码&quot;]"><strong><a href="coarse_to_fine_sparse_sequential_recommendation/">Coarse-to-Fine Sparse Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>显式建模类目/卖家意图序列作为稠密先验，与物品序列双编码器融合并按联合概率解码，稀疏数据上 NDCG@5 相对提升 44.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;侧信息融合&quot;,&quot;解耦注意力&quot;,&quot;辅助属性预测&quot;]"><strong><a href="decoupled_side_information_fusion_for_sequential_recommendation/">Decoupled Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将侧信息从输入层移至注意力层解耦计算，配合辅助属性预测器，在四个 Amazon/Yelp 数据集全库排序上稳定超越 SASRec/NOVA/ICAI-SR。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;CTR预估&quot;,&quot;时间周期建模&quot;,&quot;行为去噪&quot;,&quot;时间嵌入&quot;]"><strong><a href="denoising_time_cycle_modeling_for_recommendation/">Denoising Time Cycle Modeling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>DiCycle 用绝对/相对时间核编码 + 门控去噪分别刻画 ATC/RTC 两种周期模式，CTR 预测在 LastFM/ML-1M/Books/工业集上 AUC 稳定超越时序基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Determinantal Point Process&quot;,&quot;集合似然损失&quot;,&quot;质量-多样性权衡&quot;]"><strong><a href="determinantal_point_process_likelihoods_for_sequential_recommendation/">Determinantal Point Process Likelihoods for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 DPP 集合似然替换逐点/成对损失，把「下 T 项」建模为集合抽取，即插即用于 Caser/MARank/SASRec，Beauty Recall@3 最高 +37%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;判别式训练&quot;,&quot;ELECTRA 式替换检测&quot;,&quot;采样难负例&quot;]"><strong><a href="elecrec_training_sequential_recommenders_as_discriminators/">ELECRec: Training Sequential Recommenders as Discriminators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>借鉴 ELECTRA 思想，将序列推荐从下一项生成改为真假判别训练，辅助生成器采样难负例，四数据集 Full-Ranking 下 HR@5 提升 38%–137%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Session-aware Recommendation&quot;,&quot;Temporal Self-Attention&quot;]"><strong><a href="exploiting_session_information_in_bert_based_session_aware_sequential_recom/">Exploiting Session Information in BERT-based Session-aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>在 BERT4Rec 输入层以会话标记、会话段嵌入与时间感知自注意力三种轻量插件注入会话结构，仅增 0.1% 参数即在全库排序下取得最高 55% NDCG 提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Fastformer&quot;,&quot;细粒度用户建模&quot;,&quot;新闻推荐&quot;]"><strong><a href="fum_fine_grained_and_fast_user_modeling_for_news_recommendation/">FUM: Fine-grained and Fast User Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户点击新闻拼接成一条长文档，用 Fastformer 线性复杂度捕获词级跨新闻交互，并融合新闻级粗粒度通路；MIND/Feeds 上 AUC 相对提升 2.2%–2.5%，推理仅 2.75s/1k 样本。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间多样性&quot;,&quot;顺序无关Transformer&quot;]"><strong><a href="is_news_recommendation_a_sequential_recommendation_task/">Is News Recommendation a Sequential Recommendation Task?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>论证新闻推荐非标准序列任务，提出双序无关 Transformer 与时间多样性惩罚 TempRec，在 MIND/News 曝光排序稳定提升 AUC/nDCG。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;候选感知用户建模&quot;,&quot;单塔交互式匹配&quot;]"><strong><a href="news_recommendation_with_candidate_aware_user_modeling/">News Recommendation with Candidate-aware User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用候选新闻反向引导用户建模，Candi-SelfAtt/Candi-CNN/Candi-Att 三路聚合，MIND 上 nDCG@5 相对提升 4.2%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理/召回&quot;,&quot;张量链分解压缩&quot;,&quot;自监督知识蒸馏&quot;,&quot;长尾物品扶持&quot;]"><strong><a href="on_device_next_item_recommendation_with_self_supervised_knowledge_distillat/">On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用半张量积松弛 TT 分解的维度一致性实现 30 倍嵌入压缩，并以热冷嵌入重组驱动的自监督蒸馏补回容量，全库排序下几乎无损甚至超越教师。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异质超图&quot;,&quot;价格偏好建模&quot;,&quot;协同引导学习&quot;]"><strong><a href="price_does_matter_modeling_price_and_interest_preferences_in_session_based_/">Price DOES Matter! Modeling Price and Interest Preferences in Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将价格离散为等概率价格等级并构建异质超图，用双通道聚合+协同引导门控联合建模价格与兴趣偏好，全库排序下 Prec@20 最高提升 6.86%。</p>
</dd>
</dl>
