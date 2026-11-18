---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**SIGIR 2025** · **8** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [SIGIR 2026 (9)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

<div class="pn-filter-layout">
  <aside class="pn-filter-rail">
<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文；标签较多可滚动查看</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSTU" aria-pressed="false">HSTU<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM增强CTR" aria-pressed="false">LLM增强CTR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪孪生网络" aria-pressed="false">伪孪生网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩矩阵分解" aria-pressed="false">低秩矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低阶信号保持" aria-pressed="false">低阶信号保持<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局归一化注意力" aria-pressed="false">全局归一化注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣生命周期" aria-pressed="false">兴趣生命周期<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化聚类" aria-pressed="false">向量量化聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标偏好对齐" aria-pressed="false">多目标偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多码本向量量化" aria-pressed="false">多码本向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑广告" aria-pressed="false">捆绑广告<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="显隐式双流融合" aria-pressed="false">显隐式双流融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询改写" aria-pressed="false">查询改写<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合拍卖" aria-pressed="false">混合拍卖<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵吸收" aria-pressed="false">矩阵吸收<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="竞价词生成" aria-pressed="false">竞价词生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动机制设计" aria-pressed="false">自动机制设计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆物品频率IIF" aria-pressed="false">逆物品频率IIF<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐层注意力融合" aria-pressed="false">逐层注意力融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列目标注意力" aria-pressed="false">长序列目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期行为序列建模" aria-pressed="false">长期行为序列建模<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长期行为序列建模&quot;,&quot;多码本向量量化&quot;,&quot;HSTU&quot;]"><strong><a href="deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;显隐式双流融合&quot;,&quot;低阶信号保持&quot;,&quot;逐层注意力融合&quot;]"><strong><a href="dlf_enhancing_explicit_implicit_interaction_via_dynamic_low_order_aware_fus/">DLF: Enhancing Explicit-Implicit Interaction via Dynamic Low-Order-Aware Fusion for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以残差感知低阶交互网络与逐层注意力融合，动态平衡显隐式特征交叉，四数据集AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;LLM增强CTR&quot;,&quot;伪孪生网络&quot;]"><strong><a href="elec_efficient_large_language_model_empowered_click_through_rate_prediction/">ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>伪孪生增益/朴素双网络+分数级CLID与表示级MSE双蒸馏，在线仅跑表格DCNv2，AUC较FLIP提升0.0020/0.0034。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动机制设计&quot;,&quot;混合拍卖&quot;,&quot;捆绑广告&quot;]"><strong><a href="hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;兴趣生命周期&quot;,&quot;向量量化聚类&quot;,&quot;多任务学习&quot;]"><strong><a href="interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩矩阵分解&quot;,&quot;长序列目标注意力&quot;,&quot;矩阵吸收&quot;]"><strong><a href="lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;查询改写&quot;,&quot;多目标偏好对齐&quot;,&quot;竞价词生成&quot;]"><strong><a href="multi_objective_aligned_bidword_generation_model_for_e_commerce_search_adve/">Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用判别器提供相关性/真实性/CPM三维奖励，扩展MODPO做多目标偏好对齐，驱动LLM生成高商业价值竞价词，线上广告收入+2.13%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;逆物品频率IIF&quot;,&quot;全局归一化注意力&quot;,&quot;个性化搜索&quot;]"><strong><a href="nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
</dl>
  </div>
</div>
