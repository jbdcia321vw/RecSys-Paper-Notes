---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**WWW (TheWebConf) 2024** · **8** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2023 (5)](../../2023/ctr_feature_interaction/) · [WWW (TheWebConf) 2025 (7)](../../2025/ctr_feature_interaction/) · [WWW (TheWebConf) 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2023 (4)](../../../AAAI/2023/ctr_feature_interaction/) · [AAAI (RecSys Track) 2024 (3)](../../../AAAI/2024/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/ctr_feature_interaction/) · [ACM MM 2023 (1)](../../../ACM_MM/2023/ctr_feature_interaction/) · [CIKM 2023 (13)](../../../CIKM/2023/ctr_feature_interaction/) · [CIKM 2024 (2)](../../../CIKM/2024/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [DASFAA 2023 (1)](../../../DASFAA/2023/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (2)](../../../ICLR_NeurIPS_ICML/2024/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/ctr_feature_interaction/) · [KDD 2023 (3)](../../../KDD/2023/ctr_feature_interaction/) · [KDD 2024 (7)](../../../KDD/2024/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2023 (3)](../../../RecSys/2023/ctr_feature_interaction/) · [RecSys 2024 (7)](../../../RecSys/2024/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2023 (3)](../../../SIGIR/2023/ctr_feature_interaction/) · [SIGIR 2024 (4)](../../../SIGIR/2024/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/ctr_feature_interaction/) · [IEEE TKDE 2024 (1)](../../../TKDE/2024/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2024 (1)](../../../TORS/2024/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2023 (1)](../../../WSDM/2023/ctr_feature_interaction/) · [WSDM 2024 (1)](../../../WSDM/2024/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/)

🔥 **高频主题：** WWW 2024 ×7 · CTR预估与特征交叉 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排/创意生成&quot;,&quot;Stable Diffusion Inpainting&quot;,&quot;LoRA 自循环训练&quot;,&quot;多模态 Reward Model&quot;]"><strong><a href="a_new_creative_generation_pipeline_for_click_through_rate_with_stable_diffu/">A New Creative Generation Pipeline for Click-Through Rate with Stable Diffusion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 SD-Inpainting 只改背景保商品，Prompt 模型注入用户分组、多模态 Reward 模型闭环自循环生成高 CTR 创意，线上 CTR +10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;语义 ID&quot;,&quot;残差量化&quot;]"><strong><a href="discrete_semantic_tokenization_for_deep_ctr_prediction/">Discrete Semantic Tokenization for Deep CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 RQ-VAE 把用户/物品稠密语义压成 4 个离散 token，配分层混合推理 HMI 打分，在 MIND 上以约 200 倍内存压缩保住 98% AUC 且推理仅 3ms。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨域知识迁移&quot;,&quot;特征翻译&quot;,&quot;显式特征增强&quot;]"><strong><a href="enhancing_cross_domain_click_through_rate_prediction_via_explicit_feature_a/">Enhancing Cross-domain Click-Through Rate Prediction via Explicit Feature Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用跨监督特征翻译器把目标域隐特征映射到源域并拼回原特征做显式增强，免超域依赖、上下文感知，淘宝线上 CTR +0.11pt、GMV +1.26%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;宏推荐图&quot;,&quot;图神经网络&quot;,&quot;十亿级工业部署&quot;]"><strong><a href="macro_graph_neural_networks_for_online_billion_scale_recommender_systems/">Macro Graph Neural Networks for Online Billion-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将十亿级微观用户-物品图按行为模式聚类为百级宏节点图，用宏边求和+温度softmax权重与分离式注意力聚合，CTR任务AUC平均+1.00%，线上PCTR+3.13%且响应时间比SIM快20.97%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;有限库存商品推荐&quot;,&quot;序列建模&quot;]"><strong><a href="metasplit_meta_split_network_for_limited_stock_product_recommendation/">MetaSplit: Meta-Split Network for Limited-Stock Product Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按库存切分用户序列并引入元缩放/位移网络与辅助损失，解决 C2C 孤品冷启与注意力偏置，线上 CTR +3.56%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多模态预训练&quot;,&quot;端到端联合训练&quot;,&quot;冷启动&quot;]"><strong><a href="ppm_a_pre_trained_plug_in_model_for_click_through_rate_prediction/">PPM: A Pre-trained Plug-in Model for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多模态特征做 ID-free CTR 预训练，以“缓存+部分参数可训”的插件形式接入 IDRec 联合训练，端到端不提延迟，京东线上 UCTR +1.09%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;检索增强&quot;,&quot;跨样本特征交互&quot;,&quot;级联注意力&quot;]"><strong><a href="rat_retrieval_augmented_transformer_for_click_through_rate_prediction/">RAT: Retrieval-Augmented Transformer for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 BM25 检索 K 个历史样本作为参考上下文，以级联 intra/cross attention 建模样本内与跨样本特征交互，AUC 平均较 xDeepFM 提升 1.13%，长尾用户提升更显著。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨阶段数据复用&quot;,&quot;Set-to-Set 共交互建模&quot;,&quot;SimHash 近邻选择&quot;]"><strong><a href="recall_augmented_ranking_enhancing_click_through_rate_prediction_accuracy_w/">Recall-Augmented Ranking: Enhancing Click-Through Rate Prediction Accuracy with Cross-Stage Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RAR 用 SimHash 从跨阶段召回池选出相似用户集与召回物品集，经 set-to-set 匹配矩阵共交互聚合增强排序表征，即插即用；在 KKBox/MovieLens/CandiCTR-Pub 上 AUC 相对提升最高 4.7%。</p>
</dd>
</dl>
