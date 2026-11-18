# 🎯 CTR/CVR 预估与特征交叉

**CIKM 2025** · **5** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [CIKM 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../../WWW/2026/ctr_feature_interaction/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向信息流" aria-pressed="false">双向信息流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头子空间匹配" aria-pressed="false">多头子空间匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多归因学习" aria-pressed="false">多归因学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构信息交互" aria-pressed="false">异构信息交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有序回归" aria-pressed="false">有序回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="树结构学习" aria-pressed="false">树结构学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式表征对齐" aria-pressed="false">生成式表征对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏MoE" aria-pressed="false">稀疏MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放定律" aria-pressed="false">缩放定律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观看时长预测" aria-pressed="false">观看时长预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="辅助训练" aria-pressed="false">辅助训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="选择性聚合" aria-pressed="false">选择性聚合<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;生成式表征对齐&quot;,&quot;多头子空间匹配&quot;]"><strong><a href="hit_model_a_hierarchical_interaction_enhanced_two_tower_model_for_pre_ranki/">HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双生成器预生成跨塔粗粒度向量+多头子空间细粒度匹配，在保持双塔预缓存效率下逼近全交互模型精度，腾讯线上GMV+1.66%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构信息交互&quot;,&quot;双向信息流&quot;,&quot;选择性聚合&quot;]"><strong><a href="interformer_effective_heterogeneous_interaction_learning_for_click_through_/">InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出交错式双向信息流的 InterFormer，用独立 Cross Arch 做选择性聚合，在三个公开集与 Meta 工业集上取得 SOTA，线上 NE 提升 0.15% 且 QPS 提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;有序回归&quot;,&quot;树结构学习&quot;]"><strong><a href="personalized_tree_based_progressive_regression_model_for_watch_time_predict/">Personalized Tree-Based Progressive Regression Model for Watch-Time Prediction in Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将TPM固定满二叉树改为样本级可学习剪枝树，并用IPS校正条件建模选择偏差，离线MAE/XAUC全面领先，线上观看时长+0.428%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉&quot;,&quot;稀疏MoE&quot;,&quot;缩放定律&quot;]"><strong><a href="rankmixer_scaling_up_ranking_models_in_industrial_recommenders/">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用无参多头Token Mixing替代自注意力、Per-token FFN隔离特征子空间，配合ReLU路由稀疏MoE，在抖音万亿级数据上将MFU从4.5%提至45%，参数扩100倍而延迟不增。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多归因学习&quot;,&quot;多任务学习&quot;,&quot;辅助训练&quot;]"><strong><a href="see_beyond_a_single_view_multi_attribution_learning_leads_to_better_convers/">See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多归因标签作为辅助任务联合学习，用笛卡尔积辅助训练增强监督信号，离线GAUC+0.51%、线上ROI+2.6%</p>
</dd>
</dl>
