# ✨ 生成式推荐与离散语义 ID

**AAAI (RecSys Track) 2026** · **5** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** AAAI 2026 ×5 · 生成式推荐 ×5 · 语义ID ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE 码本量化" aria-pressed="false">MoE 码本量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="互信息校准" aria-pressed="false">互信息校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多方位对齐" aria-pressed="false">多方位对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="归纳式推荐" aria-pressed="false">归纳式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="投机解码" aria-pressed="false">投机解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差扩散" aria-pressed="false">残差扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一物品分词" aria-pressed="false">统一物品分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为词表" aria-pressed="false">行为词表<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态量化" aria-pressed="false">跨模态量化<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;行为词表&quot;,&quot;可解释推荐&quot;]"><strong><a href="behavior_tokens_speak_louder_disentangled_explainable_recommendation_with_b/">Behavior Tokens Speak Louder: Disentangled Explainable Recommendation with Behavior Vocabulary</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用VQ-VAE将用户/物品行为离散化为宏观兴趣+微观意图的行为词表，经多级文本监督与语义对齐正则注入冻结LLM，零样本解释推荐SOTA</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;投机解码&quot;,&quot;归纳式推荐&quot;,&quot;语义ID&quot;]"><strong><a href="inductive_generative_recommendation_via_retrieval_based_speculation/">Inductive Generative Recommendation via Retrieval-based Speculation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将投机解码的草稿-验证范式迁移至生成式推荐，用归纳式检索器起草候选、GR 模型做目标感知验证，使语义 ID 生成模型具备新物品推荐能力。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;残差扩散&quot;,&quot;捆绑推荐&quot;]"><strong><a href="modeling_item_level_dynamic_variability_with_residual_diffusion_for_bundle_/">Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差扩散在潜空间重建捆绑包item级表征，模型无关地适配B-I归属动态变化，Recall/NDCG最高提升约23%，训练开销仅增4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;跨模态量化&quot;,&quot;语义ID&quot;,&quot;多方位对齐&quot;]"><strong><a href="multi_aspect_cross_modal_quantization_for_generative_recommendation/">Multi-Aspect Cross-modal Quantization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 RQ-VAE 量化各层注入跨模态对比与重建对齐，并在 GR 训练中加隐式/显式对齐，Full-Ranking 下 HR@10 最高提升约 9.4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;统一物品分词&quot;,&quot;MoE 码本量化&quot;,&quot;互信息校准&quot;]"><strong><a href="tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi_domain/">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TokenMoE+码本+互信息校准的统一物品分词框架 UniTok，单模型跨 10 域训练，NDCG@10 最高提升 51.89%，参数量降 9.63×。</p>
</dd>
</dl>
