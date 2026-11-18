# ⏱️ 序列与会话推荐

**SIGIR 2026** · **15** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [SIGIR 2025 (12)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Hawkes 过程" aria-pressed="false">Hawkes 过程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="In-Context Learning" aria-pressed="false">In-Context Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 上下文增强" aria-pressed="false">LLM 上下文增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 意图画像" aria-pressed="false">LLM 意图画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prior-Fitted Network" aria-pressed="false">Prior-Fitted Network<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sub-Sequence Splitting" aria-pressed="false">Sub-Sequence Splitting<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一购物篮推荐" aria-pressed="false">下一购物篮推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性采样" aria-pressed="false">不确定性采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层对齐" aria-pressed="false">双层对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成数据先验" aria-pressed="false">合成数据先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="地理-时间嵌入" aria-pressed="false">地理-时间嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复购周期自监督" aria-pressed="false">复购周期自监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复购节奏建模" aria-pressed="false">复购节奏建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多序列构造" aria-pressed="false">多序列构造<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多粒度时间建模" aria-pressed="false">多粒度时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子带图传播" aria-pressed="false">子带图传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小波包变换" aria-pressed="false">小波包变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="引导式互学习" aria-pressed="false">引导式互学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图终止建模" aria-pressed="false">意图终止建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据预处理/评估协议" aria-pressed="false">数据预处理/评估协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="旋转位置编码" aria-pressed="false">旋转位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时频多分辨率建模" aria-pressed="false">时频多分辨率建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块化集成" aria-pressed="false">模块化集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性注意力" aria-pressed="false">线性注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自训练伪标签" aria-pressed="false">自训练伪标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应门控融合" aria-pressed="false">自适应门控融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾物品" aria-pressed="false">长尾物品<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负例挖掘" aria-pressed="false">难负例挖掘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="集合注意力" aria-pressed="false">集合注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波" aria-pressed="false">频域滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频谱熵" aria-pressed="false">频谱熵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="饱食门控" aria-pressed="false">饱食门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高效Transformer" aria-pressed="false">高效Transformer<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;长序列建模&quot;,&quot;高效Transformer&quot;]"><strong><a href="beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用深度截断 Bi-gram 语义 ID 替代正交 Video ID，并以无参时间折叠压缩 Transformer 将序列扩至 2000，线上满意观看时长 +1.42%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;复购节奏建模&quot;,&quot;集合注意力&quot;]"><strong><a href="case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以日历时间二值信号+共享多尺度时序卷积建模物品复购节奏，再用 ISAB 建模跨物品依赖，在四个杂货数据集与千万级生产系统上取得显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多序列构造&quot;,&quot;线性注意力&quot;]"><strong><a href="cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出可学习多序列构造与线性注意力，缓解单序列上下文污染，Meta 四场景 NE 降 0.3%–0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;知识蒸馏&quot;,&quot;难负例挖掘&quot;]"><strong><a href="divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;目标注意力&quot;,&quot;频域滤波&quot;,&quot;频谱熵&quot;]"><strong><a href="fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以目标条件化频谱熵观察为动机，用目标感知频谱滤波+时域双分支做CTR预估，三数据集GAUC/AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;模块化集成&quot;,&quot;引导式互学习&quot;]"><strong><a href="flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;LLM 上下文增强&quot;,&quot;地理-时间嵌入&quot;]"><strong><a href="from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，以特征融合或辅助对齐损失注入 SASRec，并提出轻量信息量诊断测试。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自适应门控融合&quot;,&quot;双层对齐&quot;,&quot;长尾物品&quot;]"><strong><a href="fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习消除空间结构冲突，全库排序下尾部长尾 H@10 最高提升 427%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;意图终止建模&quot;,&quot;饱食门控&quot;,&quot;复购周期自监督&quot;]"><strong><a href="learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱食门控+TTNP自监督学习商品复购周期，PPRR降超60%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;数据预处理/评估协议&quot;,&quot;Sub-Sequence Splitting&quot;,&quot;可复现性研究&quot;,&quot;数据增强&quot;]"><strong><a href="pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致模型评估失真，系统分析SSS生效条件与数据分布重塑机理。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;Prior-Fitted Network&quot;,&quot;In-Context Learning&quot;,&quot;合成数据先验&quot;]"><strong><a href="recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用结构因果先验合成点击流预训练 PFN，单次前向即可零样本适配新域序列推荐，8 个基准 Full-Ranking 上零样本 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;旋转位置编码&quot;,&quot;多粒度时间建模&quot;]"><strong><a href="rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;LLM 意图画像&quot;,&quot;不确定性采样&quot;,&quot;自训练伪标签&quot;]"><strong><a href="sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局意图池约束 LLM 生成、以预测-纠错回路校验意图，并蒸馏为轻量意图预测器，实现训练选择性调用 LLM、推理零 LLM 依赖的会话推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;下一购物篮推荐&quot;,&quot;意图解耦&quot;,&quot;Hawkes 过程&quot;]"><strong><a href="time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TIDE 用 Hawkes 增强傅里叶时间编码刻画物品级非单调补货周期，并以双专家解耦习惯与探索意图，Leave-One-Out 全库排序下 R@10 最高提升 18.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;小波包变换&quot;,&quot;子带图传播&quot;,&quot;时频多分辨率建模&quot;]"><strong><a href="wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，在子带内做 Chebyshev 图传播并以能量-谱平坦度门控融合，Full-Ranking 下稀疏数据集 HR@20 最高 +15.4%。</p>
</dd>
</dl>
