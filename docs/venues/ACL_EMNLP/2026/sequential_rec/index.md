# ⏱️ 序列与会话推荐

**ACL / EMNLP (RecSys & CRS) 2026** · **1** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示出现频率最高的 10 个标签；默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态空间模型" aria-pressed="false">状态空间模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称评论结转" aria-pressed="false">非对称评论结转<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非随机缺失 (MNAR)" aria-pressed="false">非随机缺失 (MNAR)<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2026 Findings&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;状态空间模型&quot;,&quot;非随机缺失 (MNAR)&quot;,&quot;非对称评论结转&quot;]"><strong><a href="two_sided_state_space_models_for_sequential_recommendation_with_non_random_/">Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将评论视为双侧状态空间中的内生事件，用 MNAR 观测模式、非对称评论结转与局部图传播联合演化用户/物品状态，Full-Ranking 下 R@20 较 BSARec 提升 14.8%–18.8%。</p>
</dd>
</dl>
