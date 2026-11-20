---
hide:
  - toc
---

# 🏛️ ECIR 2022 推荐系统论文解读

6 篇 ECIR 2022 论文解读，覆盖 5 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/排序&quot;,&quot;GAN&quot;,&quot;可复现性评估&quot;]"><strong><a href="collaborative_filtering/an_evaluation_study_of_generative_adversarial_networks_for_collaborative_fi/">An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>ECIR 2022 复现评估 CFGAN：指出真实画像条件+无噪声使生成器退化为自编码器；在 Ciao/ML100K/ML1M 全排序 top-N 下，CFGAN 稳定但不及调优 EASE R，训练慢两个数量级。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长程依赖记忆参数&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="sequential_rec/recommender_systems_when_memory_matters/">Recommender Systems: When Memory Matters</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以平稳性与长程依赖记忆参数 d 筛除&#x27;行为飘忽&#x27;用户，配合块级正负反馈在线学习，在大规模稀疏数据上 MAP 提升 7–9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2022&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;零样本提示&quot;,&quot;语言模型似然&quot;,&quot;冷启动&quot;]"><strong><a href="llm_rec/zero_shot_recommendation_as_language_modeling/">Zero-Shot Recommendation as Language Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>冻结 GPT-2，用 “Movies like ⟨m1⟩,…,⟨mi⟩” 提示的序列似然对物品零样本打分；MovieLens-1M 上 MAP@1 在训练用户少于 50/100 时反超 BPR。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;多模态推荐&quot;,&quot;生成式端到端&quot;,&quot;搭配兼容性&quot;,&quot;风格条件生成&quot;]"><strong><a href="multimodal_rec/recommendation_of_compatible_outfits_conditioned_on_style/">Recommendation of Compatible Outfits Conditioned on Style</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Set Transformer 学搭配风格高斯隐空间，再以风格向量调制子空间注意力，实现风格可控的搭配生成与补全。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;消费者公平&quot;,&quot;可复现性基准&quot;,&quot;Demographic Parity&quot;]"><strong><a href="debias_fairness_industrial/consumer_fairness_in_recommender_systems_contextualizing_definitions_and_mi/">Consumer Fairness in Recommender Systems: Contextualizing Definitions and Mitigations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复现 8 个消费者公平缓解方法并在统一协议下对照，揭示去偏对效用冲击微弱、DP 与 KS 两类公平度量互不一致、且差异影响并非总伤及少数群体。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回评测诊断&quot;,&quot;人口统计公平性&quot;,&quot;流行度偏置&quot;,&quot;Explainable Boosting Machine&quot;]"><strong><a href="debias_fairness_industrial/revisiting_popularity_and_demographic_biases_in_recommender_evaluation_and_/">Revisiting Popularity and Demographic Biases in Recommender Evaluation and Effectiveness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>复现并扩展 Ekstrand 等的公平性研究：用三种年龄分桶策略 + pop-index + EBM 归因，揭示效用差异主要由使用量与内容流行度驱动，而非纯人口属性。</p>
</dd>
</dl>
</section>
</div>
