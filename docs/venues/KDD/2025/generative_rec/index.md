# ✨ 生成式推荐与离散语义 ID

**KDD 2025** · **4** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [KDD 2026 (13)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Flow Matching" aria-pressed="false">Flow Matching<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Trie 约束解码" aria-pressed="false">Trie 约束解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图约束解码" aria-pressed="false">图约束解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性控制" aria-pressed="false">多样性控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并行解码" aria-pressed="false">并行解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询生成" aria-pressed="false">查询生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离散流" aria-pressed="false">离散流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="类别偏好解耦" aria-pressed="false">类别偏好解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为引导先验" aria-pressed="false">行为引导先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;多样性控制&quot;,&quot;类别偏好解耦&quot;]"><strong><a href="controlling_diversity_at_inference_guiding_diffusion_recommender_models_wit/">Controlling Diversity at Inference: Guiding Diffusion Recommender Models with Targeted Category Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出解耦扩散模型 D3Rec，通过类别偏好引导的反向去噪过程，在推理阶段灵活控制推荐准确率-多样性权衡。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Flow Matching&quot;,&quot;行为引导先验&quot;,&quot;离散流&quot;]"><strong><a href="flow_matching_for_collaborative_filtering/">Flow Matching for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为引导先验与离散流匹配重构生成式CF，全库排序下精度SOTA且仅需2步推理。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;并行解码&quot;,&quot;图约束解码&quot;]"><strong><a href="generating_long_semantic_ids_in_parallel_for_recommendation/">Generating Long Semantic IDs in Parallel for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 RPG 用多 token 并行预测 + 图约束解码，将语义 ID 长度扩至 64，NDCG@10 平均超生成式基线 12.6% 且推理更省。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;生成式推荐&quot;,&quot;Trie 约束解码&quot;,&quot;查询生成&quot;]"><strong><a href="great_guiding_query_generation_with_a_trie_for_recommending_related_search_/">GREAT: Guiding Query Generation with a Trie for Recommending Related Search about Video at Kuaishou</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用高质量查询构建 Trie 约束 LLM 训练与解码，实现短视频到查询的 I2Q 推荐，离线 Edit@1 最优、线上 CTR 与相关性双升。</p>
</dd>
</dl>
