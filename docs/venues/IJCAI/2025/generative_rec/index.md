# ✨ 生成式推荐与离散语义 ID

**IJCAI (RecSys Track) 2025** · **1** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [IJCAI (RecSys Track) 2026 (1)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="Flow Matching" aria-pressed="false">Flow Matching<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="确定性ODE采样" aria-pressed="false">确定性ODE采样<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;生成式推荐&quot;,&quot;Flow Matching&quot;,&quot;确定性ODE采样&quot;,&quot;序列推荐&quot;]"><strong><a href="flow_matching_based_sequential_recommender_model/">Flow Matching based Sequential Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用直线流轨迹+无噪预测目标+ODE确定性采样替代扩散式序列推荐，四数据集全库排序平均提升6.53%</p>
</dd>
</dl>
