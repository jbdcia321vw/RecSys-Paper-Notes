# ❄️ 冷启动与长尾推荐

**WWW (TheWebConf) 2025** · **2** 篇论文解读

> **方向说明**：Strict Zero-shot 新用户/新物品冷启动、元学习 (Meta-Learning)、少样本适应与长尾物品扶持

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (3)](../../2026/cold_start_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/cold_start_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cold_start_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/cold_start_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cold_start_rec/) · [CIKM 2025 (4)](../../../CIKM/2025/cold_start_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/cold_start_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/cold_start_rec/) · [ICDE 2025 (1)](../../../ICDE/2025/cold_start_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/cold_start_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/cold_start_rec/) · [KDD 2025 (1)](../../../KDD/2025/cold_start_rec/) · [KDD 2026 (2)](../../../KDD/2026/cold_start_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/cold_start_rec/) · [RecSys 2026 (4)](../../../RecSys/2026/cold_start_rec/) · [SIGIR 2026 (2)](../../../SIGIR/2026/cold_start_rec/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/cold_start_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/cold_start_rec/) · [WSDM 2025 (1)](../../../WSDM/2025/cold_start_rec/) · [WSDM 2026 (1)](../../../WSDM/2026/cold_start_rec/)

🔥 **高频主题：** WWW 2025 ×2 · 冷启动 ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散生成推荐" aria-pressed="false">扩散生成推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="标题-内容一致性" aria-pressed="false">标题-内容一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;意图解耦&quot;,&quot;标题-内容一致性&quot;,&quot;新闻推荐&quot;]"><strong><a href="crown_a_novel_approach_to_comprehending_users_preferences_for_accurate_pers/">CROWN: A Novel Approach to Comprehending Users&#x27; Preferences for Accurate Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别引导意图解耦、标题-内容一致性加权与 GNN 混合用户表征，统一解决新闻推荐的多意图、读后偏好差异与冷启动三大难题。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;扩散生成推荐&quot;,&quot;捆绑推荐&quot;]"><strong><a href="personalized_diffusion_model_reshapes_cold_start_bundle_recommendation/">Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用个性化条件扩散在分布空间生成捆绑包，配合解耦自注意力与KL正则，冷启动捆绑推荐全库排序Recall@20最高提升1155%。</p>
</dd>
</dl>
