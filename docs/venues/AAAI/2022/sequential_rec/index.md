---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**AAAI (RecSys Track) 2022** · **2** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2021 (4)](../../2021/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (6)](../../2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [ACM MM 2026 (2)](../../../ACM_MM/2026/sequential_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/sequential_rec/) · [CIKM 2022 (9)](../../../CIKM/2022/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (10)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/sequential_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/sequential_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/sequential_rec/) · [ICDE 2022 (7)](../../../ICDE/2022/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2021 (2)](../../../ICDM/2021/sequential_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/sequential_rec/) · [IJCAI (RecSys Track) 2022 (2)](../../../IJCAI/2022/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/sequential_rec/) · [KDD 2022 (4)](../../../KDD/2022/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (11)](../../../KDD/2026/sequential_rec/) · [RecSys 2021 (8)](../../../RecSys/2021/sequential_rec/) · [RecSys 2022 (10)](../../../RecSys/2022/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/sequential_rec/) · [SIGIR 2022 (12)](../../../SIGIR/2022/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (4)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TOIS 2022 (1)](../../../TOIS/2022/sequential_rec/) · [ACM TOIS 2023 (5)](../../../TOIS/2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TOIS 2026 (4)](../../../TOIS/2026/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TORS 2026 (3)](../../../TORS/2026/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/sequential_rec/) · [WSDM 2022 (4)](../../../WSDM/2022/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (4)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/sequential_rec/) · [WWW (TheWebConf) 2022 (8)](../../../WWW/2022/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** AAAI 2022 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;序列建模/记忆压缩&quot;,&quot;可微数据选择&quot;,&quot;双层优化&quot;,&quot;梯度近似&quot;]"><strong><a href="dips_differentiable_policy_for_sketching_in_recommender_systems/">DiPS: Differentiable Policy for Sketching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以双层优化端到端学习可微草图（sketching）策略，并用队列式近似梯度替代全历史重算，全库排序下同等精度省 50% 草图槽位。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;轨迹式元学习&quot;,&quot;时序动态建模&quot;]"><strong><a href="leaping_through_time_with_gradient_based_adaptation_for_recommendation/">Leaping Through Time with Gradient-based Adaptation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用轨迹式元学习 Leap 替代 RNN 建模时序动态：GTL 抓跨时段共性、OTL 抓短期趋势，时间切分+1:99 采样排序下 HR@1 最高相对提升 18.59%。</p>
</dd>
</dl>
