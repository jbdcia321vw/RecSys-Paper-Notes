---
hide:
  - toc
---

# 🏛️ AAAI (RecSys Track) 2025 推荐系统论文解读

31 篇 AAAI (RecSys Track) 2025 论文解读，覆盖 12 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** AAAI 2025 ×31 · 多模态推荐 ×6 · 大语言模型推荐 ×6 · 去偏公平与可信推荐 ×4 · 生成式推荐 ×4 · 对比学习 ×3 · 可解释推荐 ×2 · 知识蒸馏 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="K-12 AI 素养教育" aria-pressed="false">K-12 AI 素养教育<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 关键词蒸馏" aria-pressed="false">LLM 关键词蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 日志语义增强" aria-pressed="false">LLM 日志语义增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mamba/SSM" aria-pressed="false">Mamba/SSM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixup" aria-pressed="false">Mixup<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP替代GCN" aria-pressed="false">MLP替代GCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PPO 检索智能体" aria-pressed="false">PPO 检索智能体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Test-Time Training" aria-pressed="false">Test-Time Training<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Thompson采样" aria-pressed="false">Thompson采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TTT-Linear" aria-pressed="false">TTT-Linear<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文组合老虎机" aria-pressed="false">上下文组合老虎机<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化软提示" aria-pressed="false">个性化软提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化门控" aria-pressed="false">个性化门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主动学习实例选择" aria-pressed="false">主动学习实例选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交互式推荐" aria-pressed="false">交互式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪排序构造" aria-pressed="false">伪排序构造<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="共享账户推荐" aria-pressed="false">共享账户推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层专家网络" aria-pressed="false">分层专家网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层排序损失" aria-pressed="false">分层排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单调超臂网络" aria-pressed="false">单调超臂网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向序列建模" aria-pressed="false">双向序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层图卷积" aria-pressed="false">双层图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实数据增强" aria-pressed="false">反事实数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反向 BPR" aria-pressed="false">反向 BPR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分自编码器" aria-pressed="false">变分自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可持续推荐" aria-pressed="false">可持续推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="困难负采样" aria-pressed="false">困难负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积推荐" aria-pressed="false">图卷积推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图掩码自编码器" aria-pressed="false">图掩码自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经架构搜索" aria-pressed="false">图神经架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图胶囊网络" aria-pressed="false">图胶囊网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态序列推荐" aria-pressed="false">多模态序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标损失" aria-pressed="false">多目标损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多面向用户建模" aria-pressed="false">多面向用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子空间对齐" aria-pressed="false">子空间对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="循环偏好推断" aria-pressed="false">循环偏好推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑剪枝" aria-pressed="false">拓扑剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序损失" aria-pressed="false">排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐透明度与数据能动性" aria-pressed="false">推荐透明度与数据能动性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码剪枝" aria-pressed="false">掩码剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空知识图谱" aria-pressed="false">时空知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="智能空间行为预测" aria-pressed="false">智能空间行为预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="样本效率" aria-pressed="false">样本效率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="梯度手术" aria-pressed="false">梯度手术<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="梯度置信度" aria-pressed="false">梯度置信度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型遗忘/隐私合规" aria-pressed="false">模型遗忘/隐私合规<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态遗忘" aria-pressed="false">模态遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品语义图" aria-pressed="false">物品语义图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户侧知识图谱" aria-pressed="false">用户侧知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="监督微调" aria-pressed="false">监督微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交超图" aria-pressed="false">社交超图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经UCB聚类" aria-pressed="false">神经UCB聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="空间位置编码" aria-pressed="false">空间位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束嵌入更新" aria-pressed="false">约束嵌入更新<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性注意力" aria-pressed="false">线性注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组门控路由" aria-pressed="false">组门控路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="经验贝叶斯先验" aria-pressed="false">经验贝叶斯先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦协同过滤" aria-pressed="false">联邦协同过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦基础模型" aria-pressed="false">联邦基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自反思提示知识库" aria-pressed="false">自反思提示知识库<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习" aria-pressed="false">自监督对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为引导特征适配" aria-pressed="false">行为引导特征适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为重要性加权" aria-pressed="false">行为重要性加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征级融合" aria-pressed="false">表征级融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解耦表征学习" aria-pressed="false">解耦表征学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评分-解释一致性" aria-pressed="false">评分-解释一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域对比学习" aria-pressed="false">跨域对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态对齐" aria-pressed="false">跨模态对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跳过行为建模" aria-pressed="false">跳过行为建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软标签" aria-pressed="false">软标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐步图卷积" aria-pressed="false">逐步图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻域对齐损失" aria-pressed="false">邻域对齐损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾用户" aria-pressed="false">长尾用户<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控双编码器" aria-pressed="false">门控双编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="集成奖励建模" aria-pressed="false">集成奖励建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非平稳在线学习" aria-pressed="false">非平稳在线学习<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;伪排序构造&quot;,&quot;排序损失&quot;,&quot;梯度置信度&quot;]"><strong><a href="collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;Mixup&quot;,&quot;表征级融合&quot;]"><strong><a href="sequential_rec/augmenting_sequential_recommendation_with_balanced_relevance_and_diversity/">Augmenting Sequential Recommendation with Balanced Relevance and Diversity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出即插即用平衡增强插件 BASRec，以单序列随机算子+表征级 mixup 与跨序列非线性 mixup 兼顾相关性与多样性，全库排序下 SASRec 平均提升 33.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;软标签&quot;,&quot;困难负采样&quot;]"><strong><a href="sequential_rec/future_sight_and_tough_fights_revolutionizing_sequential_recommendation_wit/">Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间相关软标签挖掘未来交互、用锚-负混合构造持久困难负样本，四数据集全库排序平均提升 6.16%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;共享账户推荐&quot;,&quot;图胶囊网络&quot;,&quot;子空间对齐&quot;]"><strong><a href="sequential_rec/lightweight_yet_fine_grained_a_graph_capsule_convolutional_network_with_sub/">Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用线性注意力胶囊图+低秩子空间对齐，在共享账户混合序列上做细粒度潜在用户解耦，精度大幅领先且参数/耗时更低。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;联邦基础模型&quot;,&quot;组门控路由&quot;,&quot;多面向用户建模&quot;]"><strong><a href="sequential_rec/multifaceted_user_modeling_in_recommendation_a_federated_foundation_models_/">Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>联邦场景下用组门控网络将用户分层路由到组级 FFN，配合用户私有 FFN 实现多面向个性化，CTR 任务 AUC 最高提升 1.16%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;Test-Time Training&quot;,&quot;TTT-Linear&quot;,&quot;线性注意力&quot;]"><strong><a href="sequential_rec/preliminary_evaluation_of_the_test_time_training_layers_in_recommendation_s/">Preliminary Evaluation of the Test-Time Training Layers in Recommendation System (Student Abstract)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 TTT-Linear 作为序列推荐特征提取层，逐 token 梯度更新隐状态，在 Beauty/Electronics/ML-1M 上以 1 epoch 追平 SASRec/ComiRec 数十轮训练效果。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba/SSM&quot;,&quot;双向序列建模&quot;,&quot;长尾用户&quot;]"><strong><a href="sequential_rec/sigma_selective_gated_mamba_for_sequential_recommendation/">SIGMA: Selective Gated Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用部分翻转双向 Mamba + 输入敏感稠密门控 + FE-GRU 解决 Mamba 单向性与短序列状态不稳问题，五数据集 Full-Ranking 上 HR@10 最高提升 8.82%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;主动学习实例选择&quot;,&quot;知识蒸馏&quot;,&quot;会话推荐&quot;]"><strong><a href="llm_rec/active_large_language_model_based_knowledge_distillation_for_session_based_/">Active Large Language Model-based Knowledge Distillation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 max-min 主动学习从全量会话中挑出少量高增益实例喂给 GPT-4 教师蒸馏，500 条样本即超越全量 KD 与 10 倍参数教师。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户侧知识图谱&quot;,&quot;图掩码自编码器&quot;,&quot;跨域对比学习&quot;]"><strong><a href="llm_rec/bridging_the_user_side_knowledge_gap_in_knowledge_aware_recommendations_wit/">Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 推断用户兴趣并结构化为 CIKG 异构图，配合兴趣重建与跨域对比学习去噪，在三个数据集全库排序下 R@100 最高提升 9.31%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;可解释推荐&quot;,&quot;评分-解释一致性&quot;,&quot;课程学习&quot;]"><strong><a href="llm_rec/coherency_improved_explainable_recommendation_via_large_language_model/">Coherency Improved Explainable Recommendation via Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLaMA2 先预测评分再以软评分嵌入引导解释生成，配合课程学习与无标注一致性自动评测，解释一致性提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;多模态序列推荐&quot;,&quot;循环偏好推断&quot;,&quot;监督微调&quot;]"><strong><a href="llm_rec/harnessing_multimodal_large_language_models_for_multimodal_sequential_recom/">Harnessing Multimodal Large Language Models for Multimodal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 将商品图转为文本摘要，再以循环式提示推断动态用户偏好，SFT 微调 LLaVA 做多模态序列推荐，AUC 最高提升约 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;LLM 日志语义增强&quot;,&quot;个性化软提示&quot;,&quot;智能空间行为预测&quot;]"><strong><a href="llm_rec/predicting_user_behavior_in_smart_spaces_with_llm_enhanced_logs_and_persona/">Predicting User Behavior in Smart Spaces with LLM-Enhanced Logs and Personalized Prompts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义增强智能空间事件日志 + 个性化图软提示，在智能车/智能家居四数据集上显著提升下一行为预测，稀有事件 H@1 提升超 40%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;自反思提示知识库&quot;,&quot;PPO 检索智能体&quot;]"><strong><a href="llm_rec/re2llm_reflective_reinforcement_large_language_model_for_session_based_reco/">Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 自反思生成纠错提示构建知识库，再以 PPO 训练轻量检索智能体按会话选提示，冻结主干即超越全量训练基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;LLM 关键词蒸馏&quot;,&quot;集成奖励建模&quot;]"><strong><a href="agent_conversational_rec/llm_powered_user_simulator_for_recommender_system/">LLM-Powered User Simulator for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 离线蒸馏物品优缺点关键词，构建关键词匹配+语义相似+序列统计三模型集成用户模拟器，为 RL 推荐提供可解释高保真环境。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;行为引导特征适配&quot;,&quot;参数高效微调&quot;]"><strong><a href="multimodal_rec/befa_a_general_behavior_driven_feature_adapter_for_multimedia_recommendatio/">BeFA: A General Behavior-driven Feature Adapter for Multimedia Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为特征引导的轻量适配器解耦-过滤-重构预训练内容特征，修复信息漂移与遗漏，全库排序下平均提升约 9-11%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;拓扑剪枝&quot;,&quot;邻域对齐损失&quot;,&quot;MLP替代GCN&quot;]"><strong><a href="multimodal_rec/beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跳过行为建模&quot;,&quot;双层图卷积&quot;,&quot;分层排序损失&quot;]"><strong><a href="multimodal_rec/exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跨模态对齐&quot;,&quot;自监督对比学习&quot;,&quot;物品语义图&quot;]"><strong><a href="multimodal_rec/mentor_multi_level_self_supervised_learning_for_multimodal_recommendation/">MENTOR: Multi-level Self-supervised Learning for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 ID 嵌入为锚做四级跨模态分布对齐，叠加特征掩码与图扰动自监督，Amazon 三集 Recall@20 最高提升 74.73%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;逐步图卷积&quot;,&quot;模态遗忘&quot;,&quot;约束嵌入更新&quot;]"><strong><a href="multimodal_rec/stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;解耦表征学习&quot;,&quot;社交超图&quot;,&quot;对比学习&quot;]"><strong><a href="social_group_rec/disentangled_modeling_of_preferences_and_social_influence_for_group_recomme/">Disentangled Modeling of Preferences and Social Influence for Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>解耦用户偏好与社会影响的双通道(超)图传播，配合社交重要性引导的对比学习，缓解群组推荐的偏好偏差与数据稀疏。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;经验贝叶斯先验&quot;,&quot;Thompson采样&quot;,&quot;非平稳在线学习&quot;]"><strong><a href="cold_start_rec/bayescns_a_unified_bayesian_approach_to_address_cold_start_and_non_stationa/">BayesCNS: A Unified Bayesian Approach to Address Cold Start and Non-Stationarity in Search Systems at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用经验贝叶斯先验网络+Thompson采样在线更新，统一解决物品冷启动与非平稳漂移，线上A/B新物品曝光+10.60%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;图神经架构搜索&quot;,&quot;行为重要性加权&quot;,&quot;双层优化&quot;]"><strong><a href="cross_domain_multi_scenario/behavior_importance_aware_graph_neural_architecture_search_for_cross_domain/">Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用一次性免重训超网为各域定制 GNN 架构，并以双层隐式梯度学习源域行为重要性权重，Amazon 与微信广告数据 AUC 最高提升 10.57%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;分层专家网络&quot;,&quot;掩码剪枝&quot;,&quot;反事实数据增强&quot;]"><strong><a href="cross_domain_multi_scenario/one_for_dozens_adaptive_recommendation_for_all_domains_with_counterfactual_/">One for Dozens: Adaptive REcommendation for All Domains with Counterfactual Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层专家+彩票假设掩码剪枝为数十域学专属子网，并以流行度反事实增强少数域，DomainAUC 提升 2‰~8.4‰。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;时空知识图谱&quot;,&quot;空间位置编码&quot;]"><strong><a href="poi_spatiotemporal_rec/spatial_temporal_knowledge_distillation_for_takeaway_recommendation/">Spatial-Temporal Knowledge Distillation for Takeaway Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 STKG 教师模型蒸馏时空知识给 ST-Transformer 学生模型，融合地理距离与功能区序列建模外卖动态偏好，三城 HR@10 提升约 1.5–2.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;梯度手术&quot;,&quot;个性化门控&quot;]"><strong><a href="multi_task_reranking/direct_routing_gradient_drgrad_a_personalized_information_surgery_for_multi/">Direct Routing Gradient (DRGrad): A Personalized Information Surgery for Multi-Task Learning (MTL) Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用梯度余弦相似度做路由+Split-MMoE+PPNet个性化门控，在150亿样本工业数据上Click AUC提升0.0027且不增推理延迟。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;对比学习&quot;,&quot;样本效率&quot;,&quot;交互式推荐&quot;]"><strong><a href="rl_bandit_rec/contrastive_representation_for_interactive_recommendation/">Contrastive Representation for Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用偏好排序对比学习(PRCL)为 DRL 交互推荐提炼状态表征，冷启动下样本效率显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;上下文组合老虎机&quot;,&quot;神经UCB聚类&quot;,&quot;单调超臂网络&quot;]"><strong><a href="rl_bandit_rec/neural_combinatorial_clustered_bandits_for_recommendation_systems/">Neural Combinatorial Clustered Bandits for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双神经网络+上下文聚类替代组合老虎机的精确优化 oracle，实现 $\tilde{O}(\tilde{d}\sqrt{T})$ 遗憾并显著降低推荐累积遗憾。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;可解释推荐&quot;,&quot;推荐透明度与数据能动性&quot;,&quot;K-12 AI 素养教育&quot;]"><strong><a href="debias_fairness_industrial/an_xai_social_media_platform_for_teaching_k_12_students_ai_driven_profiling/">An XAI Social Media Platform for Teaching K-12 Students AI-Driven Profiling, Clustering, and Engagement-Based Recommending</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向 K-12 的 XAI 社交媒体教学平台，实时可视化数据采集、画像聚类与推荐机制，209 名学生实测揭示三类使用画像。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;可持续推荐&quot;,&quot;多目标损失&quot;]"><strong><a href="debias_fairness_industrial/bites_of_tomorrow_personalized_recommendations_for_a_healthier_and_greener_/">Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出绿色食品推荐任务 GRAPE，用双注意力建模用户对可持续指标的动态偏好，并以两种 Green Loss 在保精度前提下提升推荐列表绿色度。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型遗忘/隐私合规&quot;,&quot;多模态推荐&quot;,&quot;反向 BPR&quot;,&quot;图卷积推荐&quot;]"><strong><a href="debias_fairness_industrial/multi_modal_recommendation_unlearning_for_legal_licensing_and_modality_cons/">Multi-Modal Recommendation Unlearning for Legal, Licensing, and Modality Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个多模态推荐遗忘框架 MMRecUn，用反向 BPR 衰减遗忘集、正向 BPR 保真保留集，Recall 最高提升 49.85% 且比从头重训快 1.3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦协同过滤&quot;,&quot;变分自编码器&quot;,&quot;门控双编码器&quot;]"><strong><a href="debias_fairness_industrial/personalized_federated_collaborative_filtering_a_variational_autoencoder_ap/">Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 FedCF 重构为 VAE 任务，用门控双编码器分离共享/个性化知识，HR@20 较 FedRAP 提升约 97%。</p>
</dd>
</dl>
</section>
</div>
