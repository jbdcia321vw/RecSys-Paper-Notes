# 🧩 多模态推荐与内容理解

**KDD 2025** · **3** 篇论文解读

> **方向说明**：视觉-文本-音频特征融合与解耦、多模态图对齐、短视频/微视频推荐与图文电商匹配

📌 **同领域跨会议浏览：** [KDD 2026 (3)](../../2026/multimodal_rec/) · [AAAI (RecSys Track) 2025 (5)](../../../AAAI/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (2)](../../../ACL_EMNLP/2025/multimodal_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/multimodal_rec/) · [ACM MM 2025 (4)](../../../ACM_MM/2025/multimodal_rec/) · [ACM MM 2026 (3)](../../../ACM_MM/2026/multimodal_rec/) · [CIKM 2025 (9)](../../../CIKM/2025/multimodal_rec/) · [CIKM 2026 (4)](../../../CIKM/2026/multimodal_rec/) · [DASFAA 2025 (1)](../../../DASFAA/2025/multimodal_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/multimodal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/multimodal_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (2)](../../../ICLR_NeurIPS_ICML/2025/multimodal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/multimodal_rec/) · [RecSys 2025 (9)](../../../RecSys/2025/multimodal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/multimodal_rec/) · [SIGIR 2025 (6)](../../../SIGIR/2025/multimodal_rec/) · [SIGIR 2026 (8)](../../../SIGIR/2026/multimodal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/multimodal_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/multimodal_rec/) · [IEEE TMM 2025 (1)](../../../TMM/2025/multimodal_rec/) · [IEEE TMM 2026 (1)](../../../TMM/2026/multimodal_rec/) · [ACM TORS 2025 (1)](../../../TORS/2025/multimodal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/multimodal_rec/) · [WSDM 2025 (2)](../../../WSDM/2025/multimodal_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/multimodal_rec/) · [WWW (TheWebConf) 2026 (3)](../../../WWW/2026/multimodal_rec/)

🔥 **高频主题：** 多模态推荐 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mamba 序列建模" aria-pressed="false">Mamba 序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Stein 核分布对齐" aria-pressed="false">Stein 核分布对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="后期融合" aria-pressed="false">后期融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态上下文学习" aria-pressed="false">多模态上下文学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态大表征模型" aria-pressed="false">多模态大表征模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据稀疏" aria-pressed="false">数据稀疏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="虚拟三元组" aria-pressed="false">虚拟三元组<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;Stein 核分布对齐&quot;,&quot;跨模态专家路由&quot;,&quot;Mamba 序列建模&quot;]"><strong><a href="findrec_stein_guided_entropic_flow_for_multi_modal_sequential_recommendatio/">FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Stein 核对齐图文与 ID 流分布、MoE 路由动态筛选跨模态信号，Mamba 建模长序列，三数据集 NDCG@5 相对提升 1.4–3.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;虚拟三元组&quot;,&quot;预热阈值策略&quot;,&quot;数据稀疏&quot;]"><strong><a href="mdvt_enhancing_multimodal_recommendation_with_model_agnostic_multimodal_dri/">MDVT: Enhancing Multimodal Recommendation with Model-Agnostic Multimodal-Driven Virtual Triplets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多模态融合表征动态构造虚拟三元组作为额外监督信号，配合三种预热阈值策略与增强 pairwise 损失，即插即用缓解数据稀疏，稀疏用户组 NDCG@10 最高提升 12.85%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态大表征模型&quot;,&quot;后期融合&quot;,&quot;多模态上下文学习&quot;]"><strong><a href="notellm_2_multimodal_large_representation_models_for_recommendation/">NoteLLM-2: Multimodal Large Representation Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端到端微调 LLM+视觉编码器构建高效 MLRM，用 mICL 与门控后期融合缓解视觉忽视，I2I 全库召回 R@100 显著提升并落地线上 A/B。</p>
</dd>
</dl>
