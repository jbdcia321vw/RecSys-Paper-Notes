---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**SIGIR 2024** · **12** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [SIGIR 2025 (12)](../../2025/sequential_rec/) · [SIGIR 2026 (15)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2024 (1)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2024 (9)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2024 (4)](../../../TOIS/2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端篮子生成&quot;,&quot;Next Basket Recommendation&quot;,&quot;重复-探索解耦&quot;,&quot;评测范式批判&quot;]"><strong><a href="are_we_really_achieving_better_beyond_accuracy_performance_in_next_basket_r/">Are We Really Achieving Better Beyond-Accuracy Performance in Next Basket Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出TREx解耦重复/探索商品，用简单重复模块保精度、探索模块提升公平/多样性，并批判现有beyond-accuracy评测可被“走捷径”。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果推断解耦&quot;,&quot;会话推荐&quot;,&quot;多模态对齐&quot;]"><strong><a href="disentangling_id_and_modality_effects_for_session_based_recommendation/">Disentangling ID and Modality Effects for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DIMO 物品级把共现矩阵显式注入 ID 表征并将图文统一对齐到文本空间，会话级用代理机制与反事实推断无监督解耦 ID/模态成因，Cellphones Prec@20 较最优基线提升 55.12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排（序列编码器骨干）&quot;,&quot;复数向量注意力&quot;,&quot;旋转位置编码&quot;,&quot;相位对比学习&quot;]"><strong><a href="eulerformer_sequential_user_behavior_modeling_with_complex_vector_attention/">EulerFormer: Sequential User Behavior Modeling with Complex Vector Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用欧拉变换把语义差与位置差统一为复向量旋转角，配合自适应差分旋转与相位对比学习，SASRec 骨干 NDCG@10 最高提升 10.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知特征&quot;,&quot;高斯核平滑&quot;,&quot;流式训练&quot;]"><strong><a href="interest_clock_time_perception_in_real_time_streaming_recommendation_system/">Interest Clock: Time Perception in Real-Time Streaming Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户 24 小时分桶兴趣经高斯核按请求时间平滑聚合成兴趣时钟嵌入，抖音音乐线上 A/B 活跃天数 +0.509%、使用时长 +0.758%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;提示自动优化&quot;,&quot;LLM 意图推断&quot;]"><strong><a href="large_language_models_for_intent_driven_session_recommendations/">Large Language Models for Intent-Driven Session Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>PO4ISR 用“提示初始化→自反思优化→跨域选择”三段式自动提示工程驱动 LLM 在语义层识别会话内变动数量的意图，20 候选重排下 HR/NDCG 平均提升 57.37%/61.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;混合提示 (Hybrid Prompting)&quot;,&quot;课程学习模态对齐&quot;,&quot;指令微调&quot;]"><strong><a href="llara_large_language_recommendation_assistant/">LLaRA: Large Language-Recommendation Assistant</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将传统序列推荐器的 ID 行为嵌入经投影器对齐进 LLM 词元空间，用混合提示+课程学习实现行为模态注入，采样排序 HitRatio@1 最高提升 21.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;对比学习序列增广&quot;,&quot;多兴趣表征&quot;]"><strong><a href="modeling_user_fatigue_for_sequential_recommendation/">Modeling User Fatigue for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以兴趣感知相似度矩阵刻画目标-历史细粒度重复度，经 Cross 网络与疲劳门控 GRU 注入长短期兴趣，并用序列增广构造疲劳对比监督；Taobao AUC 提升 0.026，线上 EVTR 与类目多样性同升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多意图表征&quot;,&quot;稀疏意图筛选&quot;,&quot;频次嵌入&quot;]"><strong><a href="multi_intent_aware_session_based_recommendation/">Multi-intent-aware Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MiaSRec 用位置+频次嵌入、highway 多意图表征与 α-entmax 动态意图筛选，配合 max/avg 双池化解码聚合，在六个数据集上全场刷新 SOTA，长会话 R@20 最高 +24.56%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Transformer&quot;,&quot;Scaling Law&quot;,&quot;特征编码&quot;]"><strong><a href="scaling_sequential_recommendation_models_with_transformers/">Scaling Sequential Recommendation Models with Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用特征编码替代物品ID嵌入，使SRT参数与目录规模解耦，在Amazon全量数据上验证Transformer序列推荐的缩放律与预训练微调收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化自增强学习&quot;,&quot;短期协作图&quot;,&quot;序列去噪&quot;]"><strong><a href="selfgnn_self_supervised_graph_neural_networks_for_sequential_recommendation/">SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按时间切片构建短期协作图并用个性化自增强学习以长期兴趣矫正短期噪声，1:999 采样排序下四个数据集全面超越 17 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;Decision Transformer&quot;,&quot;多任务多奖励 RTG 平衡&quot;]"><strong><a href="sequential_recommendation_for_optimizing_both_immediate_feedback_and_long_t/">Sequential Recommendation for Optimizing Both Immediate Feedback and Long-term Retention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DT4IER 用用户特征自适应重加权双奖励 RTG，将 Decision Transformer 扩展为 CTR+留存的长期-短期多任务序列推荐，三数据集 NDCG 提升 11%–16%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;潜在关系发现&quot;,&quot;离散变分自编码器&quot;]"><strong><a href="sequential_recommendation_with_latent_relations_based_on_large_language_mod/">Sequential Recommendation with Latent Relations based on Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 语义表征 + DVAE 自监督挖掘物品间潜在关系，与关系感知序列推荐联合优化，Office 上 HR@5 相对提升 8.38%。</p>
</dd>
</dl>
