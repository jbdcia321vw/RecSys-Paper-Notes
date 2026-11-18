# 📍 时空轨迹与 POI 本地生活推荐

**KDD 2026** · **2** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [KDD 2025 (1)](../../2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../../SIGIR/2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../../SIGIR/2026/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

🔥 **高频主题：** 时空轨迹与POI推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推理" aria-pressed="false">LLM推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="优势校准" aria-pressed="false">优势校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动POI预测" aria-pressed="false">冷启动POI预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空图神经网络" aria-pressed="false">时空图神经网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;冷启动POI预测&quot;,&quot;因果推断&quot;,&quot;时空图神经网络&quot;]"><strong><a href="causalpoi_spatio_temporal_graph_based_causal_modeling_for_cold_start_poi_ch/">CausalPOI: Spatio-Temporal Graph-Based Causal Modeling for Cold-Start POI Check-in Forecasting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用功能交互图+结构对齐的干预/对照图做冷启动POI签到量预测与因果提升估计，四区域RMSE最高降57.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;生成式推荐&quot;,&quot;LLM推理&quot;,&quot;优势校准&quot;,&quot;语义ID&quot;]"><strong><a href="think2go_generative_next_poi_recommendation_with_llm_reasoning/">Think2Go: Generative Next POI Recommendation with LLM Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一SFT与RL推理的生成式Next-POI框架，用KDE时空认知不确定性与难度感知奖励差校准优势函数，Acc@1较GNPR提升5.8–7.5%。</p>
</dd>
</dl>
