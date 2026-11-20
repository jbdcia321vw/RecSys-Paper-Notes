---
hide:
  - toc
---

# 🏛️ ICDM 2023 推荐系统论文解读

11 篇 ICDM 2023 论文解读，覆盖 8 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 零样本推荐 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本推荐" aria-pressed="false">零样本推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;组合嵌入 (Compositional Embedding)&quot;,&quot;可学习软阈值剪枝&quot;,&quot;嵌入表内存压缩&quot;]"><strong><a href="collaborative_filtering/learning_compact_compositional_embeddings_via_regularized_pruning_for_recom/">Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CERP 用双码本平衡哈希组合嵌入 + 软阈值联合剪枝 + 互补性正则，在 90%/95%/99% 稀疏预算下把嵌入层压到 18 维仍超越 PEP/QR。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023 Workshop&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;多任务学习&quot;,&quot;行为数据失衡&quot;]"><strong><a href="graph_rec/alleviating_behavior_data_imbalance_for_multi_behavior_graph_collaborative_/">Alleviating Behavior Data Imbalance for Multi-Behavior Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多任务 LightGCN + 富行为表征注入稀疏行为 + 梯度路径隔离缓解点击/加购/购买数据失衡，Beibei/Taobao 上 Recall@20 最高提升 35.84%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;LightGCN&quot;,&quot;连续图传播&quot;]"><strong><a href="graph_rec/graph_neural_ordinary_differential_equations_based_method_for_collaborative/">Graph Neural Ordinary Differential Equations-based method for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 1–2 层 GCN 参数化导数函数、以 Neural ODE 积分替代多层堆叠与层组合，Euler 求解器在四个 Amazon 数据集上取得最优 Recall/NDCG 且训练最快。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;词图协同预训练&quot;,&quot;跨场景可迁移物品表征&quot;,&quot;零样本推荐&quot;]"><strong><a href="sequential_rec/collaborative_word_based_pre_trained_item_representation_for_transferable_r/">Collaborative Word-based Pre-trained Item Representation for Transferable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用共点击词图做词级对比预训练，将 CF 信号注入 PLM 物品表征，在 6 个跨场景数据集上微调与零样本设定全面超越 UniSRec/MoRec。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式&quot;,&quot;个性化标题生成&quot;,&quot;事实一致性对比学习&quot;]"><strong><a href="generative_rec/fact_preserved_personalized_news_headline_generation/">Fact-Preserved Personalized News Headline Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>FPG 用候选新闻-历史点击相似度做事实感知用户表征并配对比学习训练，在 PENS 上 ROUGE-1 27.33、FactCC 89.26 兼顾个性化与事实一致性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;零样本推荐&quot;,&quot;多模态语义嵌入&quot;]"><strong><a href="cold_start_rec/zero_shot_recommendations_with_pre_trained_large_language_models_for_multim/">Zero-Shot Recommendations with Pre-Trained Large Language Models for Multimodal Nudging</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多模态内容文本化并用LLM语义嵌入做零样本匹配，合成屏幕时间干预场景Top-5推荐中83%合适、8%不合适。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;分布隐式匹配&quot;,&quot;变分信息瓶颈&quot;]"><strong><a href="cross_domain_multi_scenario/distributional_domain_invariant_preference_matching_for_cross_domain_recomm/">Distributional Domain-Invariant Preference Matching for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以分层变分后验刻画域级偏好分布，用 JS 散度对齐两个非重叠域的域不变偏好分布，Amazon 八类冷启动全库排序 MRR 最高提升 34.9%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Thompson Sampling&quot;,&quot;自反馈闭环去偏&quot;,&quot;序贯训练范式&quot;]"><strong><a href="rl_bandit_rec/epsilon_non_greedy_a_bandit_approach_for_unbiased_recommendation_via_unifor/">Epsilon non-Greedy: A Bandit Approach for Unbiased Recommendation via Uniform Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用小编量均匀数据蒸馏无偏教师、Dropout 近似 Thompson 采样做非贪婪探索，并提出序贯训练范式模拟自反馈闭环，AUC 相对提升 4.5%–15.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDMW&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦学习&quot;,&quot;异质图注意力&quot;,&quot;跨校选课推荐&quot;]"><strong><a href="debias_fairness_industrial/heterogeneity_aware_cross_school_electives_recommendation_a_hybrid_federate/">Heterogeneity-aware Cross-school Electives Recommendation: a Hybrid Federated Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为跨校选课构建校内异质图 + 边感知注意力 + 约束矩阵分解联邦训练，在 1:99 采样排序下 HR@10 相对 FedGNN 提升 15.9%~23.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;序列差分隐私&quot;,&quot;跨域序列推荐&quot;,&quot;随机响应机制&quot;]"><strong><a href="debias_fairness_industrial/privacy_preserving_cross_domain_sequential_recommendation/">Privacy-Preserving Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>定义兼顾物品 ID 与顺序信息的序列差分隐私 SDP，设计扰动机制 M 在辅助域一次性加噪，使任意跨域序列模型可免改造地合法迁移知识。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练期数据投毒（数据层）&quot;,&quot;Transformer-GAN 伪造画像生成&quot;,&quot;评论文本增强与提示微调&quot;,&quot;推荐系统鲁棒性&quot;]"><strong><a href="debias_fairness_industrial/review_incorporated_model_agnostic_profile_injection_attacks_on_recommender/">Review-Incorporated Model-Agnostic Profile Injection Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>R-Trojan 首次将商品评论文本引入投毒画像生成，用 Transformer-GAN 三模块（生成/代理/检测）联合优化迁移性与隐蔽性，在 3 个黑盒推荐器上 HR@10 最高提升 53.8%。</p>
</dd>
</dl>
</section>
</div>
