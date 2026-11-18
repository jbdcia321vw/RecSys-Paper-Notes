# ✨ 生成式推荐与离散语义 ID

**ICLR / NeurIPS / ICML (RecSys) 2025** · **6** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (19)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×6 · ICLR ×3 · NeurIPS ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="BPE" aria-pressed="false">BPE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Top-K 对齐" aria-pressed="false">Top-K 对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="一步生成" aria-pressed="false">一步生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文感知分词" aria-pressed="false">上下文感知分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化排序损失" aria-pressed="false">个性化排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好对齐" aria-pressed="false">偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级排序" aria-pressed="false">列表级排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分推断" aria-pressed="false">变分推断<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICML 2025&quot;,&quot;生成式推荐&quot;,&quot;上下文感知分词&quot;,&quot;集合置换正则&quot;,&quot;BPE&quot;]"><strong><a href="actionpiece_contextually_tokenizing_action_sequences_for_generative_recomme/">ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个上下文感知的动作序列分词器，将动作建模为无序特征集，用加权共现合并+集合置换正则实现同一动作在不同上下文下差异化分词，NDCG@10 相对提升 6.0%–12.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;生成式推荐&quot;,&quot;投机解码&quot;,&quot;Top-K 对齐&quot;,&quot;推理加速&quot;]"><strong><a href="efficient_inference_for_large_language_model_based_generative_recommendatio/">Efficient Inference for Large Language Model-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将投机解码引入 LLM 生成式推荐，提出 N-to-K 验证下的 AtSpeed-S/R 对齐目标与松弛采样验证，实现近 2×–2.5× 无损/近似无损加速。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;生成式推荐&quot;,&quot;扩散模型&quot;,&quot;列表级排序&quot;,&quot;轨迹预测&quot;]"><strong><a href="listwise_preference_diffusion_optimization_for_user_behavior_trajectories_p/">Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Plackett-Luce 列表级排序信号注入扩散 ELBO，实现多步用户行为轨迹的连贯生成，并提出 SeqMatch 轨迹级指标，Full-Ranking 下 SeqMatch@50 最高提升 36.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;生成式推荐&quot;,&quot;量化语言&quot;,&quot;RQ-VAE&quot;,&quot;多模态迁移&quot;]"><strong><a href="multimodal_quantitative_language_for_generative_recommendation/">Multimodal Quantitative Language for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 RQ-VAE 将多域多模态物品翻译为统一量化语言，配合多任务预训练微调实现知识迁移，全库排序下 NDCG@10 最高提升 14.82%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;一步生成&quot;,&quot;偏好对齐&quot;]"><strong><a href="on_efficiency_effectiveness_trade_off_of_diffusion_based_recommenders/">On Efficiency-Effectiveness Trade-off of Diffusion-based Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间一致性正则平滑去噪函数实现一步生成，再用自适应偏好对齐修正轨迹偏差，兼顾扩散推荐效率与效果。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;生成式推荐&quot;,&quot;扩散模型推荐&quot;,&quot;个性化排序损失&quot;,&quot;变分推断&quot;]"><strong><a href="preference_diffusion_for_recommendation/">Preference Diffusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 BPR 改写为对数似然排序目标并推导变分上界，为扩散推荐器注入多负样本偏好信号，Full-Ranking 下 R@5 最高提升 19.35%。</p>
</dd>
</dl>
