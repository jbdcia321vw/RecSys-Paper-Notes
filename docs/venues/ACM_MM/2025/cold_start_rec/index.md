# ❄️ 冷启动与长尾推荐

**ACM MM 2025** · **1** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/) · [WWW (TheWebConf) 2025 (2)](../../../WWW/2025/cold_start_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/cold_start_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型聚类" aria-pressed="false">原型聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数编辑" aria-pressed="false">参数编辑<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端云协同" aria-pressed="false">端云协同<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧推理" aria-pressed="false">端侧推理<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;冷启动&quot;,&quot;端侧推理&quot;,&quot;参数编辑&quot;,&quot;原型聚类&quot;,&quot;端云协同&quot;]"><strong><a href="tackling_device_data_distribution_real_time_shift_via_prototype_based_param/">Tackling Device Data Distribution Real-time Shift via Prototype-based Parameter Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端神经适配器免反传生成参数编辑矩阵，多原型聚类+跨层知识迁移实现端侧实时分布偏移适配，1s 内完成个性化且 NDCG@5 显著超越微调。</p>
</dd>
</dl>
