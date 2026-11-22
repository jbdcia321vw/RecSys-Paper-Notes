---
hide:
  - toc
---

# 👥 社交与群组推荐

**CIKM 2021** · **3** 篇论文解读

> **方向说明**：社交信任网络传播、社交同质性去噪、群组偏好共识聚合 (Group Rec) 与影响力建模

📌 **同领域跨会议浏览：** [CIKM 2022 (1)](../../2022/social_group_rec/) · [CIKM 2023 (1)](../../2023/social_group_rec/) · [CIKM 2024 (1)](../../2024/social_group_rec/) · [CIKM 2025 (2)](../../2025/social_group_rec/) · [CIKM 2026 (1)](../../2026/social_group_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/social_group_rec/) · [AAAI (RecSys Track) 2024 (1)](../../../AAAI/2024/social_group_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/social_group_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/social_group_rec/) · [DASFAA 2024 (1)](../../../DASFAA/2024/social_group_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/social_group_rec/) · [ICDE 2023 (1)](../../../ICDE/2023/social_group_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (1)](../../../ICLR_NeurIPS_ICML/2023/social_group_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (1)](../../../ICLR_NeurIPS_ICML/2024/social_group_rec/) · [IJCAI (RecSys Track) 2023 (1)](../../../IJCAI/2023/social_group_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/social_group_rec/) · [KDD 2024 (1)](../../../KDD/2024/social_group_rec/) · [RecSys 2025 (3)](../../../RecSys/2025/social_group_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/social_group_rec/) · [SIGIR 2021 (2)](../../../SIGIR/2021/social_group_rec/) · [SIGIR 2022 (1)](../../../SIGIR/2022/social_group_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/social_group_rec/) · [IEEE TKDE 2022 (5)](../../../TKDE/2022/social_group_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/social_group_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/social_group_rec/) · [IEEE TKDE 2026 (1)](../../../TKDE/2026/social_group_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/social_group_rec/) · [ACM TORS 2023 (1)](../../../TORS/2023/social_group_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/social_group_rec/) · [WWW (TheWebConf) 2023 (2)](../../../WWW/2023/social_group_rec/) · [WWW (TheWebConf) 2024 (2)](../../../WWW/2024/social_group_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/social_group_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/social_group_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;社交与群组推荐&quot;,&quot;端到端组决策预测/排序&quot;,&quot;群组隐式反馈&quot;,&quot;反向社会选择&quot;,&quot;隐私泄露&quot;]"><strong><a href="deepgroup_representation_learning_for_group_recommendation_with_implicit_fe/">DeepGroup: Representation Learning for Group Recommendation with Implicit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>仅用群组隐式反馈学习置换不变群表征，预测新组决策并反推个人偏好，在合成群组数据上显著优于AGREE等基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;双尺度自监督对比学习&quot;,&quot;群组偏好聚合&quot;]"><strong><a href="double_scale_self_supervised_hypergraph_learning_for_group_recommendation/">Double-Scale Self-Supervised Hypergraph Learning for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用用户级/群组级双层超图刻画群内与跨群组高阶交互，并以粗/细粒度节点丢弃构造双尺度自监督对比信号，缓解群组交互稀疏；在三个基准上 NDCG@20 相对最强基线最高提升约 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;元路径异质图神经网络&quot;,&quot;图互信息自监督学习&quot;]"><strong><a href="social_recommendation_with_self_supervised_metagraph_informax_network/">Social Recommendation with Self-Supervised Metagraph Informax Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出元路径异质图编码 + 跨元路径注意力 + 子结构感知互信息自监督的 SMIN，在 Ciao/Epinions/Yelp 的 1:99 采样排序上较 NGCF+ 提升约 2.4% HR@10。</p>
</dd>
</dl>
