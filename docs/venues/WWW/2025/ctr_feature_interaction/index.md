---
hide:
  - toc
---

# 🎯 CTR/CVR 预估与特征交叉

**WWW (TheWebConf) 2025** · **7** 篇论文解读

> **方向说明**：显式/隐式高阶特征交叉网络 (DCN/EulerNet)、转化率延迟反馈 (Delayed Feedback) 与概率校准

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2026 (5)](../../2026/ctr_feature_interaction/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/ctr_feature_interaction/) · [CIKM 2025 (5)](../../../CIKM/2025/ctr_feature_interaction/) · [CIKM 2026 (5)](../../../CIKM/2026/ctr_feature_interaction/) · [ICDE 2025 (1)](../../../ICDE/2025/ctr_feature_interaction/) · [ICDM 2025 (1)](../../../ICDM/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/ctr_feature_interaction/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (2)](../../../ICLR_NeurIPS_ICML/2026/ctr_feature_interaction/) · [KDD 2025 (8)](../../../KDD/2025/ctr_feature_interaction/) · [KDD 2026 (5)](../../../KDD/2026/ctr_feature_interaction/) · [RecSys 2025 (7)](../../../RecSys/2025/ctr_feature_interaction/) · [RecSys 2026 (3)](../../../RecSys/2026/ctr_feature_interaction/) · [SIGIR 2025 (8)](../../../SIGIR/2025/ctr_feature_interaction/) · [SIGIR 2026 (9)](../../../SIGIR/2026/ctr_feature_interaction/) · [ACM TKDD 2025 (1)](../../../TKDD/2025/ctr_feature_interaction/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/ctr_feature_interaction/) · [IEEE TKDE 2025 (2)](../../../TKDE/2025/ctr_feature_interaction/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/ctr_feature_interaction/) · [ACM TORS 2025 (1)](../../../TORS/2025/ctr_feature_interaction/) · [WSDM 2025 (2)](../../../WSDM/2025/ctr_feature_interaction/)

🔥 **高频主题：** WWW 2025 ×7

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
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-Encoder" aria-pressed="false">Cross-Encoder<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DHEN 特征交叉集成" aria-pressed="false">DHEN 特征交叉集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Middleman Bias" aria-pressed="false">Middleman Bias<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两步决策与记忆检索" aria-pressed="false">两步决策与记忆检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化相关性激励" aria-pressed="false">个性化相关性激励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好优化" aria-pressed="false">偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="关键词相关性" aria-pressed="false">关键词相关性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原生广告分配" aria-pressed="false">原生广告分配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态奖励模型" aria-pressed="false">多模态奖励模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实时竞价出价优化" aria-pressed="false">实时竞价出价优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="广告图像生成" aria-pressed="false">广告图像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拍卖基准" aria-pressed="false">拍卖基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件概率融合" aria-pressed="false">条件概率融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力衰减" aria-pressed="false">注意力衰减<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督未来行为预测" aria-pressed="false">自监督未来行为预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义一致性正则" aria-pressed="false">语义一致性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="转化广告 CVR" aria-pressed="false">转化广告 CVR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似算法" aria-pressed="false">近似算法<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预算平滑" aria-pressed="false">预算平滑<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动出价&quot;,&quot;预算平滑&quot;,&quot;拍卖基准&quot;]"><strong><a href="bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;生成式推荐&quot;,&quot;多模态奖励模型&quot;,&quot;偏好优化&quot;,&quot;广告图像生成&quot;]"><strong><a href="ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;原生广告分配&quot;,&quot;近似算法&quot;,&quot;注意力衰减&quot;]"><strong><a href="efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;Middleman Bias&quot;,&quot;Cross-Encoder&quot;,&quot;关键词相关性&quot;]"><strong><a href="middleman_bias_in_advertising_aligning_relevance_of_keyphrase_recommendatio/">Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示广告关键词相关性过滤中的“中间人偏差”，用Search相关性判定重构训练标签，并以轻量Cross-Encoder在eBay线上取得CVR+11.6%的收益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;DHEN 特征交叉集成&quot;,&quot;自监督未来行为预测&quot;,&quot;转化广告 CVR&quot;]"><strong><a href="on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;条件概率融合&quot;,&quot;个性化相关性激励&quot;,&quot;语义一致性正则&quot;]"><strong><a href="prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM Agent&quot;,&quot;实时竞价出价优化&quot;,&quot;两步决策与记忆检索&quot;]"><strong><a href="rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
</dl>
  </div>
</div>
