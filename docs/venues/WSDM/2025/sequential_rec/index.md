# ⏱️ 序列与会话推荐

**WSDM 2025** · **5** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [WSDM 2026 (3)](../../2026/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (6)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (6)](../../../KDD/2026/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (2)](../../../TKDE/2026/sequential_rec/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (7)](../../../WWW/2026/sequential_rec/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Oracle 引导对齐" aria-pressed="false">Oracle 引导对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两阶段训练" aria-pressed="false">两阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果自注意力" aria-pressed="false">因果自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头子嵌入预测" aria-pressed="false">多头子嵌入预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="套装兼容性建模" aria-pressed="false">套装兼容性建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔加权" aria-pressed="false">时间间隔加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力矩阵谱分析" aria-pressed="false">注意力矩阵谱分析<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合专家网络" aria-pressed="false">混合专家网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="监督对比学习" aria-pressed="false">监督对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性 item-item 模型" aria-pressed="false">线性 item-item 模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应间隔损失" aria-pressed="false">自适应间隔损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="趋势感知归一化" aria-pressed="false">趋势感知归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域低通去噪" aria-pressed="false">频域低通去噪<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多头子嵌入预测&quot;,&quot;混合专家网络&quot;]"><strong><a href="facet_aware_multi_head_mixture_of_experts_model_for_sequential_recommendati/">Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多头子嵌入独立预测+头内 MoE 解耦用户偏好，在 4 个 Amazon/ML-20m 数据集 Full-Ranking 下 NDCG@20 最高提升 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Oracle 引导对齐&quot;,&quot;频域低通去噪&quot;,&quot;两阶段训练&quot;]"><strong><a href="oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;套装兼容性建模&quot;,&quot;监督对比学习&quot;,&quot;自适应间隔损失&quot;]"><strong><a href="personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性 item-item 模型&quot;,&quot;时间间隔加权&quot;,&quot;趋势感知归一化&quot;]"><strong><a href="temporal_linear_item_item_model_for_sequential_recommendation/">Temporal Linear Item-Item Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在闭式解线性 item-item 模型上引入时间间隔加权与趋势感知归一化，Full-Ranking 下平均 NDCG@10 提升 9.07%，训练快 6.9–57 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果自注意力&quot;,&quot;注意力矩阵谱分析&quot;]"><strong><a href="your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
</dl>
