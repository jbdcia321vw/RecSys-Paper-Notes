---
hide:
  - toc
---

# 🏛️ ICDM 2024 推荐系统论文解读

5 篇 ICDM 2024 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">仅显示 ≥2 篇论文的标签；默认 AND</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;双层用户建模&quot;,&quot;双极组嵌入&quot;,&quot;正交解耦&quot;]"><strong><a href="collaborative_filtering/gprec_bi_level_user_modeling_for_deep_recommenders/">GPRec: Bi-level User Modeling for Deep Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可学习分组器 + 双极组嵌入 + 正交个人表征，将组级与个体级用户建模统一为即插即用模块，三数据集 CTR/AUC 全面刷新 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba&quot;,&quot;选择性状态空间模型&quot;,&quot;表格特征分词&quot;]"><strong><a href="sequential_rec/mamba_for_scalable_and_efficient_personalized_recommendations/">Mamba for Scalable and Efficient Personalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Mamba 选择性状态空间层替换 FT-Transformer 中的注意力层，在双塔架构下处理表格型用户/内容特征，以 O(L) 线性复杂度与约 40% 参数量取得持平或更优的 P/R/MRR/HR。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端会话推荐排序&quot;,&quot;Neural Decision Forest&quot;,&quot;James-Stein Shrinkage&quot;]"><strong><a href="sequential_rec/sr_predictao_session_based_recommendation_with_high_capability_predictor_ad/">SR-PredictAO: Session-based Recommendation with High-Capability Predictor Add-On</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在编码器-预测器范式上外挂 James-Stein 去噪与神经决策森林高能力预测器；全库排序下 HR@20 最高提升 2.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;LLM 用户画像重建&quot;,&quot;模型无关知识迁移&quot;,&quot;RecBole 实验框架&quot;]"><strong><a href="llm_rec/llm_kt_a_versatile_framework_for_knowledge_transfer_from_large_language_mod/">LLM-KT: A Versatile Framework for Knowledge Transfer from Large Language Models to Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 LLM 生成的用户画像注入任意 CF 模型的中间层并以重建为辅助任务做知识迁移，模型无关、基于 RecBole 的插件式框架，ML-1M/Amazon CDs 上 NDCG@10 最高 +21.57%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;生成式端到端&quot;,&quot;条件扩散模型&quot;,&quot;反事实公平&quot;,&quot;群体公平&quot;]"><strong><a href="debias_fairness_industrial/diffairec_generative_fair_recommender_with_conditional_diffusion_model/">DifFaiRec: Generative Fair Recommender with Conditional Diffusion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用条件扩散模型重建评分向量，并以注意力反事实模块降低敏感属性敏感性，把公平与精度压缩为单一去噪损失，在 Full-List 排序下 Recall/NDCG 与群体公平指标同升。</p>
</dd>
</dl>
</section>
</div>
