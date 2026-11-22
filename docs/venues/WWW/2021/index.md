---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2021 推荐系统论文解读

39 篇 WWW (TheWebConf) 2021 论文解读，覆盖 11 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** WWW 2021 ×28 · WWW ×11 · CTR预估与特征交叉 ×2 · Thompson Sampling ×2 · 协同过滤 ×2 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Thompson Sampling" aria-pressed="false">Thompson Sampling<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/粗排" aria-pressed="false">召回/粗排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;秩差异感知采样&quot;,&quot;双向互学习&quot;]"><strong><a href="collaborative_filtering/bidirectional_distillation_for_top_k_recommender_system/">Bidirectional Distillation for Top-K Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>打破“教师恒优”单向蒸馏假设，用方向差异化的秩差异感知采样实现师生互教，Full-Ranking 下学生 H@50 相对 CD/RD 提升 3.0%–13.9%，教师自身亦涨 3.8%–20.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/用户建模&quot;,&quot;EEG/ERP&quot;,&quot;NCF&quot;]"><strong><a href="collaborative_filtering/collaborative_filtering_with_preferences_inferred_from_brain_signals/">Collaborative Filtering with Preferences Inferred from Brain Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用EEG/ERP直接解码面孔偏好并输入NCF，在31人×240项、10%缺失下RMSE 0.29，较SVD/kNN与随机显著更优。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;解释反馈配对学习&quot;,&quot;随机游走重启 (RWR)&quot;]"><strong><a href="collaborative_filtering/elixir_learning_from_user_feedback_on_explanations_to_improve_recommender_m/">ELIXIR: Learning from User Feedback on Explanations to Improve Recommender Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>让解释反馈可行动：对⟨推荐,解释⟩配对做二值相似度判断，经伪物品标签传播稠密化后学习用户偏好向量，注入 RWR 图推荐，Movie/Book 真实用户研究 P@5 提升 60–79%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;行程感知注意力&quot;,&quot;双目标联合训练&quot;]"><strong><a href="collaborative_filtering/itinerary_aware_personalized_deep_matching_at_fliggy/">Itinerary-aware Personalized Deep Matching at Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以「未消费订单列表=行程」为用户侧输入，用三重行程感知注意力 + 意图/点击双目标训练的双塔深度召回，在飞猪生产数据与线上 A/B 中 CTR 较 MIND 提升 1.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;评论属性注意力&quot;,&quot;信息采纳理论&quot;,&quot;属性感知负采样&quot;]"><strong><a href="collaborative_filtering/leveraging_review_properties_for_effective_recommendation/">Leveraging Review Properties for Effective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 BERT 语义 + 六维评论属性注意力刻画评论有用性，配合属性一致性损失与属性感知负采样，Yelp/Amazon 时间切分下 MAP 较最优基线提升 19%–36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;投影汉明不相似度&quot;,&quot;二值哈希码&quot;,&quot;变分哈希&quot;]"><strong><a href="collaborative_filtering/projected_hamming_dissimilarity_for_bit_level_importance_coding_in_collabor/">Projected Hamming Dissimilarity for Bit-Level Importance Coding in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出投影汉明不相似度与变分哈希模型，在4个数据集上以同等存储和时延实现哈希CF最高+7% NDCG、+14% MRR。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;类别多样性&quot;,&quot;对抗学习&quot;]"><strong><a href="graph_rec/dgcn_diversified_recommendation_with_graph_convolutional_networks/">DGCN: Diversified Recommendation with Graph Convolutional Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将多样性优化前移至 GCN 召回层，用再平衡邻居采样、类别增强负采样与对抗学习，在 FAISS 全库排序下使 Top-300 覆盖分提升、精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;套餐/集合推荐&quot;,&quot;药物相互作用图&quot;,&quot;患者条件掩码&quot;]"><strong><a href="graph_rec/drug_package_recommendation_via_interaction_aware_graph_induction/">Drug Package Recommendation via Interaction-aware Graph Induction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将药物套餐建模为带边属性的交互图，用患者掩码调制边权/边向量并做图归纳打分，在真实 EMR 数据上 F1 相对最优基线提升 3.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;过平滑 (Over-smoothing)&quot;,&quot;兴趣感知子图消息传递&quot;,&quot;无监督用户分组&quot;]"><strong><a href="graph_rec/interest_aware_message_passing_gcn_for_recommendation/">Interest-aware Message-Passing GCN for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 IMP-GCN：先无监督聚类相似兴趣用户构造子图，再将高阶图卷积限制在子图内传播，使模型堆叠至 7 层仍持续增益，NDCG@20 较 LightGCN 相对提升 3.66%–7.85%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;用户意图建模&quot;,&quot;关系路径聚合&quot;,&quot;知识图谱推荐&quot;]"><strong><a href="graph_rec/learning_intents_behind_interactions_with_knowledge_graph_for_recommendatio/">Learning Intents behind Interactions with Knowledge Graph for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户-物品交互细粒度拆解为多意图并与KG关系耦合，配合关系路径感知聚合，在Full-Ranking下ndcg@20相对KGAT最高提升14.51%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序（下一物品预测）&quot;,&quot;变分自编码器&quot;,&quot;对抗变分贝叶斯&quot;,&quot;对比学习&quot;]"><strong><a href="sequential_rec/adversarial_and_contrastive_variational_autoencoder_for_sequential_recommen/">Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将AVB对抗训练与对比学习引入序列VAE，用RNN-CNN编码提升隐变量质量，在四个数据集Top-k全排序上显著超越SVAE。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;未来行为建模&quot;,&quot;胶囊动态路由&quot;,&quot;多趋势表征&quot;]"><strong><a href="sequential_rec/future_aware_diverse_trends_framework_for_recommendation/">Future-Aware Diverse Trends Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用相似用户的相对未来行为构造未来序列，经胶囊动态路由聚成多趋势向量并以时间感知注意力融合，在 Amazon/Steam/ML-1M 上 Full-Ranking 下 Recall@50 最高提升 9.4% 且显著更diverse。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多粒度周期建模&quot;,&quot;渐进注意力&quot;,&quot;时间感知注意力&quot;]"><strong><a href="sequential_rec/learning_heterogeneous_temporal_patterns_of_user_preference_for_timely_reco/">Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>TimelyRec 用个性化多粒度周期编码与时间感知历史编码，联合刻画周期性与演化性两种异构时间模式，并在物品-时机双任务上 HR@10 最高提升 48.26%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列推荐&quot;,&quot;召回/精排架构&quot;,&quot;高效自注意力&quot;,&quot;向量量化码本&quot;,&quot;码字直方图&quot;]"><strong><a href="sequential_rec/linear_time_self_attention_with_codeword_histogram_for_efficient_recommenda/">Linear-Time Self Attention with Codeword Histogram for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>借助向量量化码本与码字直方图，将自注意力计算从 O(L²D) 降至 O(LBWD)，以 Full-Context 注意力替代稀疏注意力，四项真实数据集上 HR@10 最高提升 8.78%，推理速度最高快 57×、内存效率提升 78×。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回（端到端排序）&quot;,&quot;归纳式推荐&quot;,&quot;关系注意力图神经网络&quot;]"><strong><a href="sequential_rec/retagnn_relational_temporal_attentive_graph_neural_networks_for_holistic_se/">RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在用户-物品-属性三分图的 h-hop 包围子图上做关系级注意力 GNN 与因果序列自注意力，一套参数同时支撑常规/归纳/跨域序列推荐，MovieLens P@10 较 HGAN 提升 5.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;闭式解线性模型&quot;,&quot;物品-物品相似度&quot;,&quot;会话时效性&quot;]"><strong><a href="sequential_rec/session_aware_linear_item_item_models_for_session_based_recommendation/">Session-aware Linear Item-Item Models for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以闭式解线性物品-物品模型统一建模会话一致性、序列依赖、重复消费与时效性的 SLIST，全库排序下 HR@20 最高提升 9.76%，训练比 SR-GNN 快 768×</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;重排/列表级生成&quot;,&quot;条件变分自编码器&quot;,&quot;石板推荐 (Slate Recommendation)&quot;,&quot;多样性控制&quot;]"><strong><a href="generative_rec/variation_control_and_evaluation_for_generative_slaterecommendations/">Variation Control and Evaluation for Generative Slate Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>揭示 CVAE 列表推荐的「重构-集中」困境，提出两阶段 Pivot-CVAE 预生成扰动与列表方差评估协议，物品覆盖率提升数十倍而精度仅微降。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;多关系图神经网络&quot;,&quot;上下文调制&quot;,&quot;物品吸引力建模&quot;]"><strong><a href="social_group_rec/dual_side_deep_context_aware_modulation_for_social_recommendation/">Dual Side Deep Context-aware Modulation for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>DICER 用多关系 RGNN 得到图增强的用户/物品表征作为“深度上下文”，双向调制历史交互以提取用户兴趣与物品吸引力，Ciao/Epinion 上 Recall@10 相对 DiffNet++ 提升 9.6%/3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;自监督互信息最大化&quot;]"><strong><a href="social_group_rec/self_supervised_multi_channel_hypergraph_convolutional_network_for_social_r/">Self-Supervised Multi-Channel Hypergraph Convolutional Network for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用三角形模体诱导多通道超图卷积建模高阶社交关系，并以层级互信息最大化自监督补偿聚合损失，全库排序下 NDCG@10 最高提升 21.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习&quot;,&quot;课程学习&quot;,&quot;长尾分布&quot;]"><strong><a href="cold_start_rec/a_model_of_two_tales_dual_transfer_learning_framework_for_improved_long_tai/">A Model of Two Tales: Dual Transfer Learning Framework for Improved Long-tail Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>双迁移 MIRec：元映射学习少样本→多样本模型参数演化，配合长尾感知课程迁移打通头部与尾部物品，MovieLens1M 尾部 HR@10 相对提升 14.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;神经过程&quot;,&quot;元学习任务自适应&quot;]"><strong><a href="cold_start_rec/task_adaptive_neural_process_for_user_cold_start_recommendation/">Task-adaptive Neural Process for User Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用神经过程替代 MAML 梯度初始化，以任务软聚类嵌入调制解码器参数，三个数据集 P@5 稳定提升 3.6–4.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多智能体强化学习&quot;,&quot;集中式注意力 Critic&quot;,&quot;多目标动态梯度重加权&quot;]"><strong><a href="poi_spatiotemporal_rec/intelligent_electric_vehicle_charging_recommendation_based_on_multi_agent_r/">Intelligent Electric Vehicle Charging Recommendation Based on Multi-Agent Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>把每个充电站视为一个智能体，用集中式注意力 Critic、延迟访问未来竞争信息与多目标动态梯度重加权，在真实京沪数据上同时降低等待时长、充电价格与失败率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;自注意力&quot;,&quot;时空关系矩阵&quot;]"><strong><a href="poi_spatiotemporal_rec/stan_spatio_temporal_attention_network_for_next_location_recommendation/">STAN: Spatio-Temporal Attention Network for Next Location Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以全轨迹时空关系矩阵和双层注意力聚合非连续签到，平衡采样训练；四数据集全库Recall@5/10较SOTA提升9–17%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;域感知特征交叉&quot;,&quot;矩阵分解机&quot;,&quot;软剪枝&quot;,&quot;推理加速&quot;]"><strong><a href="ctr_feature_interaction/fm_2_field_matrixed_factorization_machines_for_recommender_systems/">$FM^2$: Field-matrixed Factorization Machines for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用域对矩阵替代 FwFM 的标量权重，统一 FM/FwFM/FvFM 为自由度阶梯，并给出域自适应维度软剪枝与中间向量缓存，AUC 超越 FFM 且 FLOPs 降至 1/3。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排/在线创意优选&quot;,&quot;AutoML交互搜索&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="ctr_feature_interaction/automated_creative_optimization_for_e_commerce_advertising/">Automated Creative Optimization for E-Commerce Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用one-shot算子搜索建模广告创意元素交互，并以变分Thompson采样平衡探索利用，线上CTR提升7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回/粗排&quot;,&quot;双塔文本编码器&quot;,&quot;点击图邻居聚合&quot;,&quot;知识蒸馏&quot;]"><strong><a href="ctr_feature_interaction/textgnn_improving_text_encoder_via_graph_neural_network_in_sponsored_search/">TextGNN: Improving Text Encoder via Graph Neural Network in Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户点击图的 top-3 邻居经共享文本编码器与 GAT 聚合后残差注入 TwinBERT 双塔，离线 AUC 0.8459→0.8471、线上 RPM +2.03%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;分层抽样配比 (Stratified Composition)&quot;,&quot;列表顺序编排 (Sequencing)&quot;,&quot;使用与满足理论 (U&amp;G)&quot;]"><strong><a href="multi_task_reranking/toward_the_next_generation_of_news_recommender_systems/">Toward the Next Generation of News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 table d&#x27;hôte 新闻推荐范式：按 U&amp;G 理论对文章分层（surveillance/serendipity），再以配额配比与顺序编排生成“套餐式”列表；为纯 vision 论文，无数据集与实验。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/在线创意选择&quot;,&quot;Visual-aware Ranking&quot;,&quot;Hybrid Bandit&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="rl_bandit_rec/a_hybrid_bandit_model_with_visual_priors_for_creative_ranking_in_display_ad/">A Hybrid Bandit Model with Visual Priors for Creative Ranking in Display Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>视觉list-wise先验热启动商品/创意双线性混合Bandit，在CreativeRanking回放协议下sCTR达3.320%、Regret降至78.11%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;深度上下文老虎机&quot;,&quot;引导式 Bootstrap&quot;,&quot;不确定性探索&quot;]"><strong><a href="rl_bandit_rec/guideboot_guided_bootstrap_for_deep_contextual_bandits/">GuideBoot: Guided Bootstrap for Deep Contextual Bandits in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>WWW 2021｜强化学习与探索推荐｜以预测不确定性反比生成伪样本引导深度 bootstrap 探索，五大模型袋随机决策，广告仿真平均奖励最高提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/粗排&quot;,&quot;Policy Gradient&quot;,&quot;下游自适应剪枝&quot;,&quot;奖励塑形&quot;]"><strong><a href="rl_bandit_rec/optimizing_ad_pruning_of_sponsored_search_with_reinforcement_learning/">Optimizing AD Pruning of Sponsored Search with Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将下游精排/拍卖视为黑盒环境，用策略梯度的序列式选品智能体替代 eCPM 规则剪枝，百度在线 A/B 取得 CPM +1.95%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;Pareto 优化&quot;,&quot;多目标确定性策略梯度&quot;]"><strong><a href="rl_bandit_rec/reinforcement_recommendation_with_user_multi_aspect_preference/">Reinforcement Recommendation with User Multi-aspect Preference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将单值 reward 扩展为多维度方面级 reward 向量，用 Pareto-DPG 多 critic 单 actor 学得帕累托有效策略，并理论分析梯度偏差上界，三个数据集 F1/NDCG 相对最优基线提升 4.4%–9.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;在线竞价与预算分配&quot;,&quot;Bandits with Knapsacks&quot;,&quot;跨平台预算优化&quot;]"><strong><a href="rl_bandit_rec/stochastic_bandits_for_multi_platform_budget_optimization_in_online_adverti/">Stochastic Bandits for Multi-platform Budget Optimization in Online Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将跨平台预算分配建模为带背包的随机赌博机，设计乘性权重原始-对偶算法实现 $O(OPT\sqrt{mn/B})$ 级 regret 并验证预算均匀耗尽。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;对抗物品推广&quot;,&quot;视觉冷启动&quot;,&quot;可信推荐系统&quot;]"><strong><a href="debias_fairness_industrial/adversarial_item_promotion_vulnerabilities_at_the_core_of_top_n_recommender/">Adversarial Item Promotion: Vulnerabilities at the Core of Top-N Recommenders that Use Images to Address Cold Start</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出面向视觉冷启动排序器的对抗物品推广攻击，用嵌入扰动或语义合成提升冷物品排名，并揭示现有防御不足。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;因果嵌入解耦&quot;,&quot;流行度去偏&quot;,&quot;非IID鲁棒性&quot;]"><strong><a href="debias_fairness_industrial/disentangling_user_interest_and_conformity_for_recommendation_with_causal_e/">Disentangling User Interest and Conformity for Recommendation with Causal Embedding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用对撞效应构造因果特定数据，把用户兴趣与从众拆成两套嵌入，在干预非IID测试集上 Recall@20 相对提升逾 20%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;差分隐私&quot;,&quot;属性推断攻击&quot;,&quot;图卷积推荐&quot;]"><strong><a href="debias_fairness_industrial/graph_embedding_for_recommendation_against_attribute_inference_attacks/">Graph Embedding for Recommendation against Attribute Inference Attacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GERAI 以输入级 LDP 特征扰动 + 损失级功能机制双重加噪构建差分隐私 GCN 推荐，在 ML-100K 上攻击者 F1 较 GCN 降 11%~14.4% 且 Hit@5 仅损失 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;图对抗去偏&quot;,&quot;公平表征学习&quot;,&quot;ego-centric 子图&quot;]"><strong><a href="debias_fairness_industrial/learning_fair_representations_for_recommendation_a_graph_based_perspective/">Learning Fair Representations for Recommendation: A Graph-based Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出模型无关的 FairGo 滤波器组，在用户 ego-centric 子图上做双通道对抗去偏，把性别 AUC 从 0.7041 压到 0.5042 且 RMSE 仅损 5.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排/策略层&quot;,&quot;异质因果效应估计&quot;,&quot;约束优化&quot;,&quot;A/B 实验个性化&quot;]"><strong><a href="debias_fairness_industrial/personalized_treatment_selection_using_causal_heterogeneity/">Personalized Treatment Selection using Causal Heterogeneity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 Causal Tree/Forest 估计异质因果效应，再通过带约束的随机/确定性优化为每个队列选择最优 treatment，在 LinkedIn 通知配额场景取得 +1.39% Sessions 提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;随机游走擦除&quot;,&quot;意识形态估计&quot;,&quot;信息茧房&quot;]"><strong><a href="debias_fairness_industrial/random_walks_with_erasure_diversifying_personalized_recommendations_on_soci/">Random Walks with Erasure: Diversifying Personalized Recommendations on Social and Information Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出带擦除的随机游走 RWE 与联合意识形态估计，在 Twitter 政治数据与 ML-1M/Yelp 上实现更均衡多样且不损精度的推荐。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;用户群体公平&quot;,&quot;约束重排&quot;,&quot;0-1 整数规划&quot;]"><strong><a href="debias_fairness_industrial/user_oriented_fairness_in_recommendation/">User-oriented Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>按活跃度划分优势/劣势用户群，将用户公平约束注入重排阶段，用 0-1 整数规划求解 top-K 列表，在降 75% 群体差距的同时反超整体精度。</p>
</dd>
</dl>
</section>
</div>
