# ✨ 生成式推荐与离散语义 ID

**ACL / EMNLP (RecSys & CRS) 2025** · **5** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×5 · EMNLP 2025 Findings ×2 · 零样本跨域推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本跨域推荐" aria-pressed="false">零样本跨域推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化用户表征" aria-pressed="false">个性化用户表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多粒度后期融合" aria-pressed="false">多粒度后期融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知解码" aria-pressed="false">意图感知解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知提示" aria-pressed="false">时间感知提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有限标量量化(FSQ)" aria-pressed="false">有限标量量化(FSQ)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式检索" aria-pressed="false">生成式检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一物品分词器" aria-pressed="false">统一物品分词器<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP 2025 Findings&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;生成式检索&quot;,&quot;个性化用户表征&quot;,&quot;意图感知解码&quot;]"><strong><a href="a_generative_framework_for_personalized_sticker_retrieval/">A Generative Framework for Personalized Sticker Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>PEARL 用用户组嵌入+LLM意图排序加权解码，实现微信表情包个性化生成式检索，离线 MRR@10 提升 15%、线上 CTR +7.12%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025 Findings&quot;,&quot;生成式推荐&quot;,&quot;时间感知提示&quot;,&quot;趋势感知推理&quot;]"><strong><a href="enhancing_time_awareness_in_generative_recommendation/">Enhancing Time Awareness in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将绝对时间戳与目标相对间隔verbalize进提示，并叠加全局转移图与免训练趋势重排，Full-Ranking下R@5最高提升15.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;生成式推荐&quot;,&quot;语义到词法翻译&quot;,&quot;多粒度后期融合&quot;]"><strong><a href="gram_generative_recommendation_via_semantic_aware_multi_granular_late_fusio/">GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将层次/协同语义翻译为 LLM 词表内文本 ID，并用多粒度提示+后期融合保留长文本信息，Full-Ranking 下 R@5 提升 11.5–16.0%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;生成式推荐&quot;,&quot;统一物品分词器&quot;,&quot;课程学习码本&quot;,&quot;零样本跨域推荐&quot;]"><strong><a href="recbase_generative_foundation_model_pretraining_for_zero_shot_recommendatio/">RecBase: Generative Foundation Model Pretraining for Zero-Shot Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用课程学习增强的 RQ-VAE 将跨域物品统一离散为层级概念 ID，从零预训练 1.5B 自回归推荐基座，在 8 个未见数据集零样本 AUC 全面超越 7B LLM。</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2025&quot;,&quot;生成式推荐&quot;,&quot;有限标量量化(FSQ)&quot;,&quot;零样本跨域推荐&quot;]"><strong><a href="recgpt_a_foundation_model_for_sequential_recommendation/">RecGPT: A Foundation Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>纯文本驱动+FSQ统一分词+混合双向因果注意力的序列推荐基础模型，零样本跨域显著超越10%数据微调的ID基线。</p>
</dd>
</dl>
