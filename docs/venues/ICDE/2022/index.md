---
hide:
  - toc
---

# 🏛️ ICDE 2022 推荐系统论文解读

12 篇 ICDE 2022 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** CIKM 2024 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/精排" aria-pressed="false">召回/精排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;可学习序列增强&quot;,&quot;质量感知对比学习&quot;]"><strong><a href="self_supervised_denoising/quality_aware_collaborative_multi_positive_contrastive_learning_for_sequent/">Quality-Aware Collaborative Multi-Positive Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以可学习协同增强模块生成同目标/相似双视图，用增强置信度加权多正样本 InfoNCE，Beauty 上 HR@10 较 CL4SRec 提升 11.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端训练（训练期增广）&quot;,&quot;扩散模型数据增广&quot;,&quot;上下文条件生成&quot;]"><strong><a href="sequential_rec/diffusion_based_contrastive_learning_for_sequential_recommendation/">Context Matters: Enhancing Sequential Recommendation with Context-aware Diffusion-based Contrastive Learning (CaDiRec)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用双向上下文条件扩散模型为序列生成语义合理的替代物品构造对比正视图，5 数据集全库排序 HR@10 最高提升 11.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;序列数据增强&quot;]"><strong><a href="sequential_rec/contrastive_learning_for_sequential_recommendation/">Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>CL4SRec 用裁剪/掩码/重排三种序列增强构造视图，以 InfoNCE 对比损失联合下一项预测，在四个数据集 Full-Ranking 上 HR@5 平均提升 11.02%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频率感知对比学习&quot;,&quot;自适应数据增广&quot;]"><strong><a href="sequential_rec/frequency_aware_adaptive_contrastive_learning_for_sequential_recommendation/">Frequency-aware Adaptive Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用频率自适应扰动与序列级重加权缓解低频物品和稀疏用户在对比增广中的过扰动，全库排序 HR@10 最高提升约 3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;微行为建模&quot;,&quot;多重图神经网络&quot;]"><strong><a href="sequential_rec/micro_behavior_encoding_for_session_based_recommendation/">Micro-Behavior Encoding for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将会话转为带星节点的有向多重图，用GRU边特征与操作感知自注意力联合编码微行为序列/二元关系，全库排序H@10最高提升14.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;多意图解耦&quot;,&quot;意图感知对比学习&quot;,&quot;K-means 原型聚类&quot;]"><strong><a href="sequential_rec/multi_intent_aware_contrastive_learning_for_sequential_recommendation/">Multi-intent Aware Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MCLRec 用 K-means 意图原型、Top-R 过滤与距离衰减项构建多意图感知对比学习，在四个 5-core 数据集全库排序下 HR/NDCG 最高提升 15.64%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;软硬双层序列去噪&quot;,&quot;跨信号对比学习&quot;,&quot;S形课程学习&quot;]"><strong><a href="sequential_rec/multi_level_sequence_denoising_with_cross_signal_contrastive_learning_for_s/">Multi-Level Sequence Denoising with Cross-Signal Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MSDCCL 用软/硬双层去噪与跨信号对比学习互补消除序列噪声，并引入 S 形课程学习；在 5 个数据集全库排序下 HR@20 较最强去噪基线最高提升 118.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;相对对比学习&quot;,&quot;相似正样本选择&quot;]"><strong><a href="sequential_rec/relative_contrastive_learning_for_sequential_recommendation_with_similarity/">Relative Contrastive Learning for Sequential Recommendation with Similarity-based Positive Pair Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将同目标序列设为强正、相似序列设为弱正，用带 max 边界的加权相对对比损失训练 SASRec/FMLP，全库排序下 6 数据集平均提升 4.88%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 (MAML)&quot;,&quot;跨域偏好增广&quot;,&quot;条件变分自编码器&quot;]"><strong><a href="cold_start_rec/diverse_preference_augmentation_with_multiple_domains_for_cold_start_recomm/">Diverse Preference Augmentation with Multiple Domains for Cold-start Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 k 个 Dual-CVAE 做多源域适配，借 MDI/ME 约束在目标域内容上生成多样化评分以增广 MAML 元训练任务，缓解稀疏交互下的元过拟合；Amazon Books/CDs 上 1:99 采样排序评估，C-I 的 NDCG@10 提升 23.6%/40.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;信息瓶颈&quot;,&quot;变分二分图编码器&quot;,&quot;冷启动用户&quot;]"><strong><a href="cross_domain_multi_scenario/cross_domain_recommendation_to_cold_start_users_via_variational_information/">Cross-Domain Recommendation to Cold-Start Users via Variational Information Bottleneck</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用变分信息瓶颈（IB）同时约束跨域/域内交互，使表征只保留域共享信息，四组 Amazon 冷启动场景下 MRR 平均提升 34.38%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;兴趣级自监督对比学习&quot;,&quot;多兴趣建模&quot;]"><strong><a href="ctr_feature_interaction/miss_multi_interest_self_supervised_learning_framework_for_click_through_ra/">MISS: Multi-Interest Self-Supervised Learning Framework for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 CNN 抽取多兴趣视图并施加兴趣级 InfoNCE 对比，作为即插即用模块提升 CTR，AUC 最高相对提升 13.55%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练（联邦推荐）&quot;,&quot;联邦推荐投毒攻击&quot;,&quot;模型投毒与梯度操纵&quot;,&quot;隐私保护与鲁棒聚合&quot;]"><strong><a href="debias_fairness_industrial/fedrecattack_model_poisoning_attack_to_federated_recommendation/">FedRecAttack: Model Poisoning Attack to Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用仅1%公开交互反解全局用户隐向量，配合指数尾部代理损失生成限幅分摊的投毒梯度，3%恶意用户即可把目标物品ER@5推至0.94+且HR@10仅降2.5%。</p>
</dd>
</dl>
</section>
</div>
