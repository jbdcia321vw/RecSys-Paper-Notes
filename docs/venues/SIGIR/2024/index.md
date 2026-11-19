---
hide:
  - toc
---

# 🏛️ SIGIR 2024 推荐系统论文解读

57 篇 SIGIR 2024 论文解读，覆盖 14 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 端到端生成式 ×4 · 大语言模型推荐 ×3 · CTR预估与特征交叉 ×2 · 会话推荐 ×2 · 多行为推荐 ×2 · 扩散模型 ×2 · 生成式端到端 ×2 · 结构位置编码 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">20</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式端到端" aria-pressed="false">生成式端到端<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构位置编码" aria-pressed="false">结构位置编码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;行为模式挖掘&quot;,&quot;贝叶斯似然比&quot;]"><strong><a href="collaborative_filtering/behavior_pattern_mining_based_multi_behavior_recommendation/">Behavior Pattern Mining-based Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以行为模式路径计数为特征、贝叶斯似然比闭式打分替代GNN，在三个电商数据集Full-Ranking下Recall@10最高提升393.98%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;物品级偏好建模&quot;,&quot;图卷积预训练&quot;]"><strong><a href="collaborative_filtering/behavior_contextualized_item_preference_modeling_for_multi_behavior_recomme/">Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 BCIPM：用「行为上下文化物品偏好网络」只从目标行为聚合物品级偏好做推荐，辅助行为仅参与参数训练以规避噪声；四个数据集 Leave-One-Out + Full-Ranking 下 HR@10 相对提升 12.15%–26.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Meal Recommendation Dataset&quot;,&quot;Bundle Recommendation&quot;,&quot;Healthiness&quot;]"><strong><a href="collaborative_filtering/mealrec_a_meal_recommendation_dataset_with_meal_course_affiliation_for_pers/">MealRec$^+$: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>SIGIR 2024｜协同过滤与双塔召回｜构建含餐食-菜品关联与FSA/WHO健康分的MealRec⁺数据集，全库排序下CrossCBR Recall@20提升72.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/用户表征&quot;,&quot;Neural Hawkes Process&quot;,&quot;用户开屏动机预测&quot;]"><strong><a href="collaborative_filtering/to_search_or_to_recommend_predicting_open_app_motivation_with_neural_hawkes/">To Search or to Recommend: Predicting Open-App Motivation with Neural Hawkes Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用神经 Hawkes 过程+层级 Transformer 建模用户开屏动机，OAMD 上 F0.5 提升 2.9%、AUC 达 0.9326</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;特征过相关&quot;,&quot;过平滑&quot;,&quot;自适应正则&quot;]"><strong><a href="graph_rec/afdgcf_adaptive_feature_de_correlation_graph_collaborative_filtering_for_re/">AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 GNN-CF 中被忽视的“特征过相关”问题并证明其与过平滑成正比，提出层级自适应的列相关惩罚正则，模型无关地提升 4 类骨干模型最多 14.98%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;持续图学习&quot;,&quot;图提示微调&quot;,&quot;流式推荐&quot;]"><strong><a href="graph_rec/gpt4rec_graph_prompt_tuning_for_streaming_recommendation/">GPT4Rec: Graph Prompt Tuning for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将演化交互图解耦为多视图，用节点/结构/视图三级轻量图提示替代回放与模型扩张，实现流式推荐持续学习，R@20 最高相对提升 4.16%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;参数高效嵌入&quot;,&quot;元嵌入组合&quot;,&quot;矩阵伪逆闭式更新&quot;]"><strong><a href="graph_rec/lightweight_embeddings_for_graph_collaborative_filtering/">Lightweight Embeddings for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可学习稀疏分配矩阵替代固定哈希映射，在扩展图上联合优化元嵌入码本，实现 44× 参数压缩下接近全量嵌入的精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;线性化核注意力&quot;,&quot;结构位置编码&quot;,&quot;度数重加权掩码&quot;]"><strong><a href="graph_rec/masked_graph_transformer_for_large_scale_recommendation/">Masked Graph Transformer for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>纯 Transformer 全节点注意力 + SimRFs 线性核 + 可学习相对度数掩码，Full-Ranking 下 Recall@20 最高提升 8.62%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;负反馈感知超图卷积&quot;,&quot;Transformer Hawkes Process&quot;]"><strong><a href="graph_rec/nfarec_a_negative_feedback_aware_recommender_model/">NFARec: A Negative Feedback-Aware Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以反馈极性引导超图卷积消息通路并用 Transformer Hawkes 过程辅助预测下一交互情感极性，五数据集全库排序 R@20 最高提升 6.6%、N@20 提升 14.2%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表示学习&quot;,&quot;符号图&quot;,&quot;图Transformer&quot;]"><strong><a href="graph_rec/sigformer_sign_aware_graph_transformer_for_recommendation/">SIGformer: Sign-aware Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用Transformer直接建模整张符号图，以符号谱/路径编码融合正负反馈，五数据集Recall@20最高提升15.61%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;注意力采样&quot;,&quot;结构位置编码&quot;]"><strong><a href="graph_rec/transgnn_harnessing_the_collaborative_power_of_transformers_and_graph_neura/">TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>交替堆叠 Transformer 与 GNN 层，用注意力采样+三种结构位置编码互相扩感受野与注入拓扑，全库排序下 Recall@20 最高提升 21%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多项式图滤波&quot;,&quot;免训练协同过滤&quot;,&quot;低通滤波&quot;]"><strong><a href="graph_rec/turbo_cf_matrix_decomposition_free_graph_filtering_for_fast_recommendation/">Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式图滤波器替代特征分解实现免训练免分解的协同过滤，全库排序下精度持平SOTA而耗时降至亚秒级。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散模型去噪&quot;,&quot;隐式反馈噪声&quot;,&quot;模型无关插件&quot;]"><strong><a href="self_supervised_denoising/denoising_diffusion_recommender_model/">Denoising Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用扩散前向-逆向过程对任意后端模型的用户/物品嵌入做多步去噪，以协同信息为引导、以历史偏好均值为推理起点，三数据集全排序 R@10 最高提升 25.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端篮子生成&quot;,&quot;Next Basket Recommendation&quot;,&quot;重复-探索解耦&quot;,&quot;评测范式批判&quot;]"><strong><a href="sequential_rec/are_we_really_achieving_better_beyond_accuracy_performance_in_next_basket_r/">Are We Really Achieving Better Beyond-Accuracy Performance in Next Basket Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出TREx解耦重复/探索商品，用简单重复模块保精度、探索模块提升公平/多样性，并批判现有beyond-accuracy评测可被“走捷径”。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果推断解耦&quot;,&quot;会话推荐&quot;,&quot;多模态对齐&quot;]"><strong><a href="sequential_rec/disentangling_id_and_modality_effects_for_session_based_recommendation/">Disentangling ID and Modality Effects for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DIMO 物品级把共现矩阵显式注入 ID 表征并将图文统一对齐到文本空间，会话级用代理机制与反事实推断无监督解耦 ID/模态成因，Cellphones Prec@20 较最优基线提升 55.12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排（序列编码器骨干）&quot;,&quot;复数向量注意力&quot;,&quot;旋转位置编码&quot;,&quot;相位对比学习&quot;]"><strong><a href="sequential_rec/eulerformer_sequential_user_behavior_modeling_with_complex_vector_attention/">EulerFormer: Sequential User Behavior Modeling with Complex Vector Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用欧拉变换把语义差与位置差统一为复向量旋转角，配合自适应差分旋转与相位对比学习，SASRec 骨干 NDCG@10 最高提升 10.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知特征&quot;,&quot;高斯核平滑&quot;,&quot;流式训练&quot;]"><strong><a href="sequential_rec/interest_clock_time_perception_in_real_time_streaming_recommendation_system/">Interest Clock: Time Perception in Real-Time Streaming Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户 24 小时分桶兴趣经高斯核按请求时间平滑聚合成兴趣时钟嵌入，抖音音乐线上 A/B 活跃天数 +0.509%、使用时长 +0.758%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;提示自动优化&quot;,&quot;LLM 意图推断&quot;]"><strong><a href="sequential_rec/large_language_models_for_intent_driven_session_recommendations/">Large Language Models for Intent-Driven Session Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>PO4ISR 用“提示初始化→自反思优化→跨域选择”三段式自动提示工程驱动 LLM 在语义层识别会话内变动数量的意图，20 候选重排下 HR/NDCG 平均提升 57.37%/61.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;混合提示 (Hybrid Prompting)&quot;,&quot;课程学习模态对齐&quot;,&quot;指令微调&quot;]"><strong><a href="sequential_rec/llara_large_language_recommendation_assistant/">LLaRA: Large Language-Recommendation Assistant</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将传统序列推荐器的 ID 行为嵌入经投影器对齐进 LLM 词元空间，用混合提示+课程学习实现行为模态注入，采样排序 HitRatio@1 最高提升 21.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;对比学习序列增广&quot;,&quot;多兴趣表征&quot;]"><strong><a href="sequential_rec/modeling_user_fatigue_for_sequential_recommendation/">Modeling User Fatigue for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以兴趣感知相似度矩阵刻画目标-历史细粒度重复度，经 Cross 网络与疲劳门控 GRU 注入长短期兴趣，并用序列增广构造疲劳对比监督；Taobao AUC 提升 0.026，线上 EVTR 与类目多样性同升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多意图表征&quot;,&quot;稀疏意图筛选&quot;,&quot;频次嵌入&quot;]"><strong><a href="sequential_rec/multi_intent_aware_session_based_recommendation/">Multi-intent-aware Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MiaSRec 用位置+频次嵌入、highway 多意图表征与 α-entmax 动态意图筛选，配合 max/avg 双池化解码聚合，在六个数据集上全场刷新 SOTA，长会话 R@20 最高 +24.56%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Transformer&quot;,&quot;Scaling Law&quot;,&quot;特征编码&quot;]"><strong><a href="sequential_rec/scaling_sequential_recommendation_models_with_transformers/">Scaling Sequential Recommendation Models with Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用特征编码替代物品ID嵌入，使SRT参数与目录规模解耦，在Amazon全量数据上验证Transformer序列推荐的缩放律与预训练微调收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化自增强学习&quot;,&quot;短期协作图&quot;,&quot;序列去噪&quot;]"><strong><a href="sequential_rec/selfgnn_self_supervised_graph_neural_networks_for_sequential_recommendation/">SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按时间切片构建短期协作图并用个性化自增强学习以长期兴趣矫正短期噪声，1:999 采样排序下四个数据集全面超越 17 个基线。</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部12篇「序列与会话推荐」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端&quot;,&quot;扩散模型&quot;,&quot;套装生成&quot;,&quot;Classifier-Free Guidance&quot;]"><strong><a href="generative_rec/diffusion_models_for_generative_outfit_recommendation/">Diffusion Models for Generative Outfit Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 GOR 新任务与 DiFashion，以类别提示、互条件、历史条件三路 CFG 引导套装多图并行扩散生成，iFashion FID 降低 19.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;扩散模型&quot;]"><strong><a href="generative_rec/graph_signal_diffusion_model_for_collaborative_filtering/">Graph Signal Diffusion Model for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品相似图上的热方程平滑替代高斯加噪作为前向扩散，反向用确定性去噪器做精修-锐化重建，三数据集全库排序 SOTA，Recall@20 最高 +5.44%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;大语言模型推荐&quot;,&quot;微调阶段&quot;,&quot;数据剪枝&quot;,&quot;影响函数&quot;,&quot;少样本微调&quot;]"><strong><a href="llm_rec/data_efficient_fine_tuning_for_llm_based_recommendation/">Data-efficient Fine-tuning for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 DEALRec，用影响函数打分选取高影响力样本，以 2% 数据超越全量微调并降低 97% 训练耗时。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;大语言模型推荐&quot;,&quot;重排/端到端决策&quot;,&quot;Tool Learning&quot;,&quot;Surrogate User&quot;]"><strong><a href="llm_rec/let_me_do_it_for_you_towards_llm_empowered_recommendation_via_tool_learning/">Let Me Do It For You: Towards LLM Empowered Recommendation via Tool Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 ToolRec，让 LLM 作为代理用户经属性检索/排序工具多轮探索物品池，并用记忆校验抑制幻觉；ML-1M/Amazon-Book 的 Recall@10 相对提升 3.36%/14.28%，Yelp 因本地知识不足退化。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;T5/Llama-2&quot;,&quot;物品索引&quot;]"><strong><a href="llm_rec/openp5_an_open_source_platform_for_developing_training_and_evaluating_llm_b/">OpenP5: An Open-Source Platform for Developing, Training, and Evaluating LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>开源P5式生成推荐平台，覆盖T5/Llama-2、10数据集、3种物品索引与序列/直推任务，T5+协同索引多数领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端智能体决策&quot;,&quot;多智能体协作&quot;,&quot;LLM Agent 工作流&quot;]"><strong><a href="agent_conversational_rec/macrec_a_multi_agent_collaboration_framework_for_recommendation/">MACRec: a Multi-Agent Collaboration Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 Manager 为核心、Reflector 迭代反思、Analyst/Searcher 供工具的多智能体免训练协作框架，覆盖评分预测/序列推荐/对话推荐/解释生成四任务，仅给案例与演示 UI，无基准量化实验</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;冷启动引导&quot;,&quot;User Simulation&quot;,&quot;Preference Elicitation&quot;,&quot;Off-Policy Evaluation&quot;]"><strong><a href="agent_conversational_rec/minimizing_live_experiments_in_recommender_systems_user_simulation_to_evalu/">Minimizing Live Experiments in Recommender Systems: User Simulation to Evaluate Preference Elicitation Policies</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用反事实鲁棒的用户状态/会话生成器与生产级模拟器，在 YouTube Music 新用户引导场景中替代小样本在线实验，准确预测策略上线指标并正确排序候选策略。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端仿真评测&quot;,&quot;LLM Generative Agents&quot;,&quot;用户行为模拟&quot;]"><strong><a href="agent_conversational_rec/on_generative_agents_in_recommendation/">On Generative Agents in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 1000 个 LLM 生成式智能体构建推荐用户仿真器 Agent4Rec，以画像/记忆/动作三模块模拟真实用户，并复现信息茧房与因果结构。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;排序&quot;,&quot;多模态用户交互&quot;,&quot;缺失模态&quot;,&quot;会话推荐&quot;]"><strong><a href="multimodal_rec/dataset_and_models_for_item_recommendation_using_multi_modal_user_interacti/">Dataset and Models for Item Recommendation Using Multi-Modal User Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>发布保险场景多模态用户交互数据集，提出将对话与网页会话映射到共同特征空间以应对自然缺失模态，全库排序 MAP@3 较强基线提升 3.46%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排/候选排序&quot;,&quot;多语言新闻推荐&quot;,&quot;跨语言迁移&quot;]"><strong><a href="cold_start_rec/mind_your_language_a_multilingual_dataset_for_cross_lingual_news_recommenda/">MIND Your Language: A Multilingual Dataset for Cross-lingual News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>构建14语言平行新闻推荐数据集xMIND，系统评测零样本/少样本跨语言迁移，揭示mPLM仍显著掉点。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;多模态缺失&quot;,&quot;不变学习&quot;,&quot;新物品推荐&quot;]"><strong><a href="cold_start_rec/multimodality_invariant_learning_for_multimedia_based_new_item_recommendati/">Multimodality Invariant Learning for Multimedia-Based New Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>面向多模态新物品推荐，用跨模态对齐与循环 Mixup 构造缺失模态环境，IRM 学习不变用户偏好；新物品全库排序下 NDCG@20 最高提升约 17.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景学习&quot;,&quot;全局标签空间&quot;,&quot;对比解耦&quot;]"><strong><a href="cross_domain_multi_scenario/a_unified_search_and_recommendation_framework_based_on_multi_scenario_learn/">A Unified Search and Recommendation Framework Based on Multi-Scenario Learning for Ranking in E-commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用场景触发器驱动 S&amp;R 双视角兴趣/特征抽取与对比解耦，并以全局标签条件概率做多任务，AUC 提升 0.08~0.62 点并落地 7Fresh。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/排序&quot;,&quot;负迁移过滤&quot;,&quot;用户表征变换&quot;,&quot;监督对比学习&quot;]"><strong><a href="cross_domain_multi_scenario/aiming_at_the_target_filter_collaborative_information_for_cross_domain_reco/">Aiming at the Target: Filter Collaborative Information for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以目标域用户相似度为显式约束，结合用户变换层与监督对比损失过滤源域协同信息，缓解跨域负迁移。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;Mixture-of-Experts&quot;,&quot;多任务学习&quot;,&quot;AutoML 双层优化&quot;]"><strong><a href="cross_domain_multi_scenario/m3oe_multi_domain_multi_task_mixture_of_experts_recommendation_framework/">M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出共享/域/任务三视角 MoE 与两级融合，并用 AutoML 双层优化自适应学习融合权重，解决多域多任务联合建模的 MDMT 跷跷板问题，MovieLens/KuaiRand 平均 AUC 相对提升 2.54%/5.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;负迁移量化&quot;,&quot;序列混合专家&quot;,&quot;互信息最大化&quot;]"><strong><a href="cross_domain_multi_scenario/pacer_and_runner_cooperative_learning_framework_between_single_and_cross_do/">Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用停梯度非对称MoE解耦SDSR/CDSR以精确估计负迁移差，动态加权跨域损失并加互信息对齐，10个域上全面抑制负迁移，线上CTR+17.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;行为转移建模&quot;,&quot;对比学习对齐&quot;,&quot;MMoE 多任务&quot;]"><strong><a href="cross_domain_multi_scenario/unisar_modeling_user_transition_behaviors_between_search_and_recommendation/">UniSAR: Modeling User Transition Behaviors between Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用掩码 Transformer 抽取 r2r/s2s/r2s/s2r 四类细粒度转移，对比对齐 + 交叉注意力融合，MMoE 联合训练统一搜索与推荐，双任务全面超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端生成式&quot;,&quot;行程推荐&quot;,&quot;重复抑制&quot;,&quot;解码采样策略&quot;]"><strong><a href="poi_spatiotemporal_rec/analyzing_and_mitigating_repetitions_in_trip_recommendation/">Analyzing and Mitigating Repetitions in Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次剖析行程推荐中POI重复生成的两大成因，提出循环感知预测器（引导矩阵+漂移惩罚+置信度自适应采样）在Flickr四城上兼顾精度与去重。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;异构图基准数据集&quot;,&quot;选址推荐&quot;,&quot;全库排序评测&quot;]"><strong><a href="poi_spatiotemporal_rec/opensiterec_an_open_dataset_for_site_recommendation/">OpenSiteRec: An Open Dataset for Site Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个公开选址推荐数据集，用 5 类实体 9 类关系的异构图刻画 4 座国际都市，并在 5-core + 全库排序协议下基准测试 16 个推荐模型。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;生成式端到端&quot;,&quot;语义ID量化&quot;,&quot;强化微调GRPO&quot;,&quot;熵坍缩&quot;]"><strong><a href="poi_spatiotemporal_rec/refine_poi_reinforcement_fine_tuned_large_language_models_for_next_point_of/">Refine-POI: Reinforcement Fine-Tuned Large Language Models for Next Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用层次自组织映射生成拓扑保序语义ID，并以GRPO列表级奖励替代SFT标签模仿，配合KL-Cov熵正则恢复有据推理，Acc@5最高提升12.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM4Rec&quot;,&quot;行为级分层编码&quot;,&quot;序列长文本效率&quot;]"><strong><a href="ctr_feature_interaction/breaking_the_length_barrier_llm_enhanced_ctr_prediction_in_long_textual_use/">Breaking the Length Barrier: LLM-Enhanced CTR Prediction in Long Textual User Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 LLM 分层解耦为离线原子行为编码表 + 在线行为级交互，长序列 CTR 训练开销降 5 倍且 AUC 反升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户行为模式&quot;,&quot;自监督去噪&quot;,&quot;Target Pattern Attention&quot;]"><strong><a href="ctr_feature_interaction/deep_pattern_network_for_click_through_rate_prediction/">Deep Pattern Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 Target Attention 从 item 级拓宽到 pattern 级，用 target-aware 检索 + 自监督去噪精炼 + Target Pattern Attention 挖掘用户行为模式，三数据集 AUC 相对 CAN 提升 1.53%–2.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;纵向联邦学习&quot;,&quot;表征迁移与知识蒸馏&quot;,&quot;未对齐数据利用&quot;]"><strong><a href="ctr_feature_interaction/fedud_exploiting_unaligned_data_for_cross_platform_federated_click_through_/">FedUD: Exploiting Unaligned Data for Cross-Platform Federated Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用知识蒸馏把对齐样本上学到的 guest 表征迁移到 host 未对齐样本，让纵向联邦 CTR 的未对齐流量也能受益，AUC 提升约 0.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;神经点击模型&quot;,&quot;用户响应仿真&quot;]"><strong><a href="ctr_feature_interaction/neural_click_models_for_recommender_systems/">Neural Click Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将搜索侧神经点击模型迁移到 slate 推荐，以 readout 条件 GRU、对抗辅助损失与仅点击注意力建模曝光级用户响应，ContentWise AUC 相对提升 6.3%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;序列推荐（多轮课程路径规划）&quot;,&quot;LLM技能抽取&quot;,&quot;就业市场对齐&quot;,&quot;MDP/PPO&quot;]"><strong><a href="rl_bandit_rec/course_recommender_systems_need_to_consider_the_job_market/">Course Recommender Systems Need to Consider the Job Market</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 少样本技能抽取对齐 ESCO 岗位需求，将课程序列推荐建模为以「可申请岗位数」为奖励的 MDP，PPO 在长序列上以约 1/5 时延逼近贪心效果。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;长期留存优化&quot;,&quot;离线仿真评测基准&quot;]"><strong><a href="rl_bandit_rec/easyrl4rec_an_easy_to_use_library_for_reinforcement_learning_based_recommen/">EasyRL4Rec: An Easy-to-use Library for Reinforcement Learning Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>构建五个真实数据集上的轻量 RL 推荐环境与统一长期收益评测管线（FreeB/NX_0_/NX_10_），系统复现 15+ 策略并揭示离散动作、On-Policy、偏好高估等关键结论。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;LLM Planner&quot;,&quot;长期收益规划&quot;,&quot;记忆式上下文学习&quot;]"><strong><a href="rl_bandit_rec/large_language_models_are_learnable_planners_for_long_term_recommendation/">Large Language Models are Learnable Planners for Long-Term Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用四个 LLM 实例构成宏/微双层可学习规划器，以外部记忆代替梯度更新，在模拟交互环境中显著提升轨迹长度与累计收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Uplift建模&quot;,&quot;因果去偏&quot;]"><strong><a href="rl_bandit_rec/treatment_effect_estimation_for_user_interest_exploration_on_recommender_sy/">Treatment Effect Estimation for User Interest Exploration on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将Top-N类目曝光比视为多元连续处理，用IPW去偏ADRF/MTEF估计与动态规划优化兴趣探索，RCT评测R@10最高相对提升17.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;流行度去偏&quot;,&quot;位置编码解耦&quot;]"><strong><a href="debias_fairness_industrial/cadrec_contextualized_and_debiased_recommender_model/">CaDRec: Contextualized and Debiased Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以超图卷积融合序列自注意力抑制过平滑，并用位置编码流行度与个体偏差双路解耦及梯度加权重整实现去偏，全库排序下 R@20 相对提升 4.2%–4.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;离线评估与度量基准&quot;,&quot;联合公平-相关性度量&quot;,&quot;Kendall τ 一致性&quot;,&quot;排序敏感度探针&quot;]"><strong><a href="debias_fairness_industrial/can_we_trust_recommender_system_fairness_evaluation_the_role_of_fairness_an/">Can We Trust Recommender System Fairness Evaluation? The Role of Fairness and Relevance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次系统体检 9 个「相关性+公平」联合度量，用 4 数据集 × 4 推荐器 × 4 类探针揭示其互相矛盾、位置不敏感与分数压缩三大病灶。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;长尾罕见病公平性&quot;,&quot;自监督预训练&quot;]"><strong><a href="debias_fairness_industrial/leave_no_patient_behind_enhancing_medication_recommendation_for_rare_diseas/">Leave No Patient Behind: Enhancing Medication Recommendation for Rare Disease Patients</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RAREMed 用统一疾病-手术序列 Transformer + 序列匹配/自重建双自监督预训练，显著抬升罕见病患者的用药推荐精度并压缩群体间公平性方差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;嵌入演化监控&quot;,&quot;流行度偏置&quot;,&quot;实时训练&quot;]"><strong><a href="debias_fairness_industrial/monitoring_the_evolution_of_behavioural_embeddings_in_social_media_recommen/">Monitoring the Evolution of Behavioural Embeddings in Social Media Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在 1.8 亿用户短视频生产系统上对比批量与实时物品嵌入的演化，用成熟度、学习量与 ℓ2 范数指标量化实时更新收敛快 5 倍且显著缓解流行度偏置。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练&quot;,&quot;模型投毒攻击&quot;,&quot;用户级梯度裁剪&quot;]"><strong><a href="debias_fairness_industrial/poisoning_decentralized_collaborative_recommender_system_and_its_countermea/">Poisoning Decentralized Collaborative Recommender System and Its Countermeasures</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出面向去中心化协同推荐的 PAMN 自适应投毒与 UCSU 用户级裁剪防御，在 ML/Amazon 上大幅提升目标物品曝光且保持 HR。</p>
</dd>
</dl>
</section>
</div>
