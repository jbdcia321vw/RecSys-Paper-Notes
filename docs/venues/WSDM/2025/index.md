---
hide:
  - toc
---

# 🏛️ WSDM 2025 推荐系统论文解读

26 篇 WSDM 2025 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 生成式推荐 ×3 · 多任务学习 ×2 · 多模态推荐 ×2 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Advantage Actor-Critic" aria-pressed="false">Advantage Actor-Critic<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Elo 评级" aria-pressed="false">Elo 评级<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 偏好蒸馏" aria-pressed="false">LLM 偏好蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 行为模拟" aria-pressed="false">LLM 行为模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM解释生成" aria-pressed="false">LLM解释生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA三阶段迭代" aria-pressed="false">LoRA三阶段迭代<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Oracle 引导对齐" aria-pressed="false">Oracle 引导对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PPO双奖励对齐" aria-pressed="false">PPO双奖励对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两阶段训练" aria-pressed="false">两阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="严格冷启动" aria-pressed="false">严格冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务分解" aria-pressed="false">任务分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="众包对战评测" aria-pressed="false">众包对战评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布对齐" aria-pressed="false">分布对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分数生成模型" aria-pressed="false">分数生成模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单 token 重索引" aria-pressed="false">单 token 重索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数解耦" aria-pressed="false">参数解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲嵌入" aria-pressed="false">双曲嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可控多样性推荐" aria-pressed="false">可控多样性推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="各向异性噪声" aria-pressed="false">各向异性噪声<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果自注意力" aria-pressed="false">因果自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图剪枝" aria-pressed="false">图剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图谱域扩散" aria-pressed="false">图谱域扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域解耦" aria-pressed="false">域解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头子嵌入预测" aria-pressed="false">多头子嵌入预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性正则" aria-pressed="false">多样性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态去噪" aria-pressed="false">多模态去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态图学习" aria-pressed="false">多模态图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标强化学习" aria-pressed="false">多目标强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为对比学习" aria-pressed="false">多行为对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="套装兼容性建模" aria-pressed="false">套装兼容性建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="安全聚合" aria-pressed="false">安全聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列多样性" aria-pressed="false">序列多样性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异质超图" aria-pressed="false">异质超图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增广" aria-pressed="false">数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方面级情感聚合" aria-pressed="false">方面级情感聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔加权" aria-pressed="false">时间间隔加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="最优传输" aria-pressed="false">最优传输<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有序哈密顿路径" aria-pressed="false">有序哈密顿路径<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件去噪网络" aria-pressed="false">条件去噪网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="正交聚合" aria-pressed="false">正交聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力矩阵谱分析" aria-pressed="false">注意力矩阵谱分析<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合专家网络" aria-pressed="false">混合专家网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户-物品-评论三方对齐" aria-pressed="false">用户-物品-评论三方对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="监督对比学习" aria-pressed="false">监督对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="真实用户反馈" aria-pressed="false">真实用户反馈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经架构搜索" aria-pressed="false">神经架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性 item-item 模型" aria-pressed="false">线性 item-item 模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组公平" aria-pressed="false">组公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="耦合漏斗过滤" aria-pressed="false">耦合漏斗过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐聚合" aria-pressed="false">联邦推荐聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应策略融合" aria-pressed="false">自适应策略融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应间隔损失" aria-pressed="false">自适应间隔损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="融合学习" aria-pressed="false">融合学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论增广对比学习" aria-pressed="false">评论增广对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论驱动推荐" aria-pressed="false">评论驱动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样 Q-learning" aria-pressed="false">负采样 Q-learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图神经网络" aria-pressed="false">超图神经网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="趋势感知归一化" aria-pressed="false">趋势感知归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过平滑" aria-pressed="false">过平滑<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似算法" aria-pressed="false">近似算法<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负采样" aria-pressed="false">难负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域低通去噪" aria-pressed="false">频域低通去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波去噪" aria-pressed="false">频域滤波去噪<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图谱域扩散&quot;,&quot;各向异性噪声&quot;,&quot;条件去噪网络&quot;]"><strong><a href="collaborative_filtering/s_diff_an_anisotropic_diffusion_model_for_collaborative_filtering_in_spectr/">S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将扩散前向过程搬到物品图拉普拉斯谱域，用特征值参数化各向异性噪声保低频，配合 FiLM 条件去噪，全库排序下 Recall@20 最高提升约 5.5%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图神经网络&quot;,&quot;方面级情感聚合&quot;,&quot;评论驱动推荐&quot;]"><strong><a href="graph_rec/an_aspect_performance_aware_hypergraph_neural_network_for_review_based_reco/">An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用四元组超图建模用户-物品-方面-情感极性，以用户偏好加权聚合冲突情感推断物品方面表现，六数据集 MSE 平均降 2.30%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图剪枝&quot;,&quot;知识蒸馏&quot;,&quot;过平滑&quot;]"><strong><a href="graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质超图&quot;,&quot;意图解耦&quot;,&quot;多行为对比学习&quot;]"><strong><a href="graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;评论增广对比学习&quot;,&quot;用户-物品-评论三方对齐&quot;]"><strong><a href="self_supervised_denoising/a_contrastive_framework_with_user_item_and_review_alignment_for_recommendat/">A Contrastive Framework with User, Item and Review Alignment for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论从“特征”升级为“对比信号”，用评论增广+三方对齐双对比任务缓解稀疏与评论缺失，Full-Ranking 下 NDCG@5 最高提升 7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;分数生成模型&quot;,&quot;难负采样&quot;]"><strong><a href="self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多头子嵌入预测&quot;,&quot;混合专家网络&quot;]"><strong><a href="sequential_rec/facet_aware_multi_head_mixture_of_experts_model_for_sequential_recommendati/">Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多头子嵌入独立预测+头内 MoE 解耦用户偏好，在 4 个 Amazon/ML-20m 数据集 Full-Ranking 下 NDCG@20 最高提升 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Oracle 引导对齐&quot;,&quot;频域低通去噪&quot;,&quot;两阶段训练&quot;]"><strong><a href="sequential_rec/oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;套装兼容性建模&quot;,&quot;监督对比学习&quot;,&quot;自适应间隔损失&quot;]"><strong><a href="sequential_rec/personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性 item-item 模型&quot;,&quot;时间间隔加权&quot;,&quot;趋势感知归一化&quot;]"><strong><a href="sequential_rec/temporal_linear_item_item_model_for_sequential_recommendation/">Temporal Linear Item-Item Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在闭式解线性 item-item 模型上引入时间间隔加权与趋势感知归一化，Full-Ranking 下平均 NDCG@10 提升 9.07%，训练快 6.9–57 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果自注意力&quot;,&quot;注意力矩阵谱分析&quot;]"><strong><a href="sequential_rec/your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;可控多样性推荐&quot;,&quot;任务分解&quot;,&quot;数据增广&quot;]"><strong><a href="llm_rec/dlcrec_a_novel_approach_for_managing_diversity_in_llm_based_recommender_sys/">DLCRec: A Novel Approach for Managing Diversity in LLM-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多样性可控推荐拆解为类型预测/填充/物品预测三子任务级联，配合两类数据增广，在 Movie/Steam 上 MAE_Cov@10 大幅领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;众包对战评测&quot;,&quot;Elo 评级&quot;,&quot;真实用户反馈&quot;]"><strong><a href="agent_conversational_rec/crs_arena_crowdsourced_benchmarking_of_conversational_recommender_systems/">CRS Arena: Crowdsourced Benchmarking of Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出基于成对对战与真实用户反馈的众包 CRS 评测平台，发布 474 段对话数据集与 Elo 榜单，揭示 R@10 与用户满意度负相关。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;单 token 重索引&quot;,&quot;分布对齐&quot;]"><strong><a href="agent_conversational_rec/reindex_then_adapt_improving_large_language_models_for_conversational_recom/">Reindex-Then-Adapt: Improving Large Language Models for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多 token 电影标题重索引为单 token，再用偏置/门控适配目标平台分布，Llama2-7b 上 H@10 相对提升 59.37%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;频域滤波去噪&quot;,&quot;多模态图学习&quot;]"><strong><a href="multimodal_rec/spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;多模态去噪&quot;,&quot;知识蒸馏&quot;,&quot;最优传输&quot;]"><strong><a href="multimodal_rec/teach_me_how_to_denoise_a_universal_framework_for_denoising_multi_modal_rec/">Teach Me How to Denoise: A Universal Framework for Denoising Multi-modal Recommender Systems via Guided Calibration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 ID 空间去噪假设在多模态空间失效，提出 AMSC 重校准 + DBPR + OT 蒸馏的即插即用去噪框架，全库排序下 Recall@20 最高提升约 11%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;LLM 行为模拟&quot;,&quot;耦合漏斗过滤&quot;,&quot;严格冷启动&quot;]"><strong><a href="cold_start_rec/large_language_model_simulator_for_cold_start_recommendation/">Large Language Model Simulator for Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 模拟冷物品的用户交互行为，并以耦合漏斗（FAISS 过滤 + LLM 精炼）把十亿级候选压到数百，离线冷召回 NDCG 提升 37%+，线上冷启 GMV +16.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;双曲嵌入&quot;,&quot;域解耦&quot;]"><strong><a href="cross_domain_multi_scenario/review_based_hyperbolic_cross_domain_recommendation/">Review-Based Hyperbolic Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论文本嵌入双曲空间做跨域推荐，用度数归一化与尺度对齐避免域对齐引发的层级坍塌，NDCG@10 较 SER 提升约 10.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;联邦学习&quot;,&quot;多任务学习&quot;,&quot;参数解耦&quot;]"><strong><a href="cross_domain_multi_scenario/towards_personalized_federated_multi_scenario_multi_task_recommendation/">Towards Personalized Federated Multi-Scenario Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个联邦多场景多任务推荐框架 PF-MSMTrec，通过专家参数解耦、联邦批归一化与冲突协调缓解数据/模型/任务三级冲突，AliExpress 与 Tenrec 上 CTR/CTCVR AUC 全面超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM解释生成&quot;,&quot;PPO双奖励对齐&quot;,&quot;LoRA三阶段迭代&quot;]"><strong><a href="ctr_feature_interaction/explainable_ctr_prediction_via_llm_reasoning/">Explainable CTR Prediction via LLM Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PPO 双奖励（LC 用户意图对齐 + IC 与 ID-CTR 模型对齐）把 LLM 解释生成嵌入 CTR 流程，免解释数据集，AUC 最高提升 18.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;融合学习&quot;,&quot;神经架构搜索&quot;,&quot;特征交叉&quot;]"><strong><a href="ctr_feature_interaction/fusion_matters_learning_fusion_in_deep_click_through_rate_prediction_models/">Fusion Matters: Learning Fusion in Deep Click-through Rate Prediction Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将CTR模型的融合连接与融合算子统一建模为可微搜索空间，用one-shot联合选择替代手工stacked/parallel，三数据集AUC稳定超越EDCN与NASRec。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;序列多样性&quot;,&quot;近似算法&quot;,&quot;有序哈密顿路径&quot;]"><strong><a href="multi_task_reranking/sequential_diversification_with_provable_guarantees/">Sequential Diversification with Provable Guarantees</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多样性从集合度量升级为序列期望度量，证明 MaxSSD 的 NP 难性并给出 BτI/SBτI 等常数近似算法，在 7 个数据集上稳定超越 EXPLORE/MMR/DPP。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;多目标强化学习&quot;,&quot;多样性正则&quot;,&quot;序列与会话推荐&quot;]"><strong><a href="rl_bandit_rec/choosing_the_best_of_both_worlds_diverse_and_novel_recommendations_through_/">Choosing the Best of Both Worlds: Diverse and Novel Recommendations through Multi-Objective Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用标量化多目标 RL 头正则化序列推荐主干，同时优化准确率/多样性/新颖性，聚合覆盖度提升约 20% 且准确率不降。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;LLM 偏好蒸馏&quot;,&quot;自适应策略融合&quot;]"><strong><a href="rl_bandit_rec/large_language_model_driven_policy_exploration_for_recommender_systems/">Large Language Model driven Policy Exploration for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 蒸馏用户偏好离线预训练 A2C 推荐策略，并提出冻结+可学习双策略自适应在线适配，显著提升冷启动回报与长期收益。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;负采样 Q-learning&quot;,&quot;Advantage Actor-Critic&quot;]"><strong><a href="rl_bandit_rec/supervised_advantage_actor_critic_for_recommender_systems/">Supervised Advantage Actor-Critic for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>为 RL 头引入负采样构造 SNQN，并用采样优势 A2C 重加权监督头，在 RC15/RetailRocket 上显著超越 SQN/SAC。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;联邦推荐聚合&quot;,&quot;组公平&quot;,&quot;正交聚合&quot;,&quot;安全聚合&quot;]"><strong><a href="debias_fairness_industrial/privacy_preserving_orthogonal_aggregation_for_guaranteeing_gender_fairness_/">Privacy-Preserving Orthogonal Aggregation for Guaranteeing Gender Fairness in Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用正交映射+安全聚合让各性别组独立聚合，兼顾隐私与组公平，ML-100K 整体 NDCG@10 提升 7.30%。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
