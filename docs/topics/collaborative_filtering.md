---
hide:
  - toc
---

# 🤝 协同过滤与双塔召回 (Collaborative Filtering & Two-Tower Retrieval)

> **研究范畴**：矩阵分解、双塔向量内积检索、ANN 索引量化、解耦表征与负采样理论

当前方向共收录 **356** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 去偏公平与可信推荐 ×28 · 大语言模型推荐 ×16 · 流行度去偏 ×12 · 对比学习 ×8 · 知识蒸馏 ×7 · 生成式推荐 ×6 · 线性自编码器 ×6 · 因果去偏 ×5

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">238</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">163</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">40</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">33</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">31</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">28</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤" aria-pressed="false">协同过滤<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去偏" aria-pressed="false">因果去偏<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分自编码器" aria-pressed="false">变分自编码器<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣表征" aria-pressed="false">多兴趣表征<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵分解" aria-pressed="false">矩阵分解<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解耦表征" aria-pressed="false">解耦表征<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负样本挖掘" aria-pressed="false">难负样本挖掘<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Machine Unlearning" aria-pressed="false">Machine Unlearning<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推荐" aria-pressed="false">免训练推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔检索" aria-pressed="false">双塔检索<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲几何" aria-pressed="false">双曲几何<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信号处理" aria-pressed="false">图信号处理<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤" aria-pressed="false">图协同过滤<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣建模" aria-pressed="false">多兴趣建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗训练" aria-pressed="false">对抗训练<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱滤波" aria-pressed="false">谱滤波<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样偏差校正" aria-pressed="false">负采样偏差校正<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过平滑" aria-pressed="false">过平滑<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式反馈去噪" aria-pressed="false">隐式反馈去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="EASE 闭式解" aria-pressed="false">EASE 闭式解<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Embedding 尺寸搜索" aria-pressed="false">Embedding 尺寸搜索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="iALS" aria-pressed="false">iALS<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Knowledge Distillation" aria-pressed="false">Knowledge Distillation<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Neural ODE" aria-pressed="false">Neural ODE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Softmax Loss" aria-pressed="false">Softmax Loss<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Two-Tower" aria-pressed="false">Two-Tower<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练协同过滤" aria-pressed="false">免训练协同过滤<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布鲁棒优化" aria-pressed="false">分布鲁棒优化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双稳健估计" aria-pressed="false">双稳健估计<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/全库排序" aria-pressed="false">召回/全库排序<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/粗排" aria-pressed="false">召回/粗排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性" aria-pressed="false">可复现性<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="后处理/模型修正" aria-pressed="false">后处理/模型修正<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络" aria-pressed="false">图神经网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线学习" aria-pressed="false">在线学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多准则推荐" aria-pressed="false">多准则推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多项式图滤波" aria-pressed="false">多项式图滤波<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗学习" aria-pressed="false">对抗学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐与均匀性" aria-pressed="false">对齐与均匀性<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性遗忘 (Attribute Unlearning)" aria-pressed="false">属性遗忘 (Attribute Unlearning)<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入坍缩" aria-pressed="false">嵌入坍缩<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入表压缩" aria-pressed="false">嵌入表压缩<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="差分隐私" aria-pressed="false">差分隐私<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="截断SVD" aria-pressed="false">截断SVD<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="投毒攻击" aria-pressed="false">投毒攻击<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="损失函数设计" aria-pressed="false">损失函数设计<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="未观测混杂" aria-pressed="false">未观测混杂<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="本地差分隐私" aria-pressed="false">本地差分隐私<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度偏置" aria-pressed="false">流行度偏置<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义 ID" aria-pressed="false">生成式推荐与离散语义 ID<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏自编码器" aria-pressed="false">稀疏自编码器<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐安全" aria-pressed="false">联邦推荐安全<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱图滤波" aria-pressed="false">谱图滤波<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆倾向加权" aria-pressed="false">逆倾向加权<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆倾向加权 (IPS)" aria-pressed="false">逆倾向加权 (IPS)<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重要性采样" aria-pressed="false">重要性采样<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控融合" aria-pressed="false">门控融合<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负样本采样" aria-pressed="false">难负样本采样<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排/在线增量更新&quot;,&quot;Meta-Learning&quot;,&quot;参数级学习率&quot;,&quot;在线学习&quot;]"><strong><a href="../../venues/AAAI/2022/collaborative_filtering/meta_learning_for_online_update_of_recommender_systems/">Meta-Learning for Online Update of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MeLON 用元学习为每个「交互-参数」对生成二维自适应学习率，突破重要性重加权与元优化的单向灵活性；在三个在线数据集上 HR@5 最高提升 29.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;用户-物品交互图&quot;,&quot;BM25 检索增强&quot;,&quot;门控融合&quot;]"><strong><a href="../../venues/AAAI/2023/agent_conversational_rec/cola_improving_conversational_recommender_systems_by_collaborative_augmenta/">COLA: Improving Conversational Recommender Systems by Collaborative Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用用户-物品交互图注入流行度、BM25 检索相似会话做门控增强，全库排序下 REDIAL R@50 相对提升 7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;离线策略学习&quot;,&quot;重要性采样&quot;,&quot;最大内积检索&quot;]"><strong><a href="../../venues/AAAI/2023/rl_bandit_rec/fast_offline_policy_optimization_for_large_scale_recommendation/">Fast Offline Policy Optimization for Large Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用协方差梯度+自归一化重要性采样+训练期 MIPS 索引，将大规模目录离线策略梯度训练从 O(P) 降至 O(log P)，提速 5–25× 且策略质量不降。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;多重稳健估计 (MR)&quot;,&quot;逆倾向加权 (IPS)&quot;,&quot;误差插补&quot;,&quot;MNAR 去偏&quot;]"><strong><a href="../../venues/AAAI/2023/debias_fairness_industrial/multiple_robust_learning_for_recommendation/">Multiple Robust Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多个倾向/误差插补模型构成基向量并岭回归求解组合系数，把 DR 升级为多重稳健估计器，Coat/Yahoo 全库评测 MSE 与 nDCG@10 显著优于 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多轮自适应检索&quot;,&quot;表征适配器&quot;,&quot;FFT 可学习滤波&quot;]"><strong><a href="../../venues/AAAI/2024/sequential_rec/ada_retrieval_an_adaptive_multi_round_retrieval_paradigm_for_sequential_rec/">Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>模型无关的多轮自适应召回：物品/用户表征适配器逐轮修正用户向量并补召回，Full-Ranking 下 NDCG@50 最高 +9.07%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="../../venues/AAAI/2024/collaborative_filtering/adaptive_hardness_negative_sampling_for_collaborative_filtering/">Adaptive Hardness Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示固定难度负采样引发 FPP/FNP，提出三准则自适应难度范式 AHNS 及实例 AHNS_{p&lt;0}，4 数据集 R@20 最高 +8.2%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱域图卷积&quot;,&quot;互补物品推荐&quot;,&quot;两阶段注意力&quot;]"><strong><a href="../../venues/AAAI/2024/graph_rec/spectral_based_graph_neural_networks_for_complementary_item_recommendation/">Spectral-Based Graph Neural Networks for Complementary Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>谱域解耦互补关系为低频相关性与中频差异性，用低通/中通 GCN 滤波器加两阶段注意力建模，四个 Amazon 数据集 HR@10 提升 4.4%–14.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;伪排序构造&quot;,&quot;排序损失&quot;,&quot;梯度置信度&quot;]"><strong><a href="../../venues/AAAI/2025/collaborative_filtering/from_pairwise_to_ranking_climbing_the_ladder_to_ideal_collaborative_filteri/">From Pairwise to Ranking: Climbing the Ladder to Ideal Collaborative Filtering with Pseudo-Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用噪声注入构造伪排序并以多序分类排序损失替代成对损失，在四个数据集 Full-Ranking 下 NDCG@10 最高提升 38.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦协同过滤&quot;,&quot;变分自编码器&quot;,&quot;门控双编码器&quot;]"><strong><a href="../../venues/AAAI/2025/debias_fairness_industrial/personalized_federated_collaborative_filtering_a_variational_autoencoder_ap/">Personalized Federated Collaborative Filtering: A Variational AutoEncoder Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 FedCF 重构为 VAE 任务，用门控双编码器分离共享/个性化知识，HR@20 较 FedRAP 提升约 97%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;逐步图卷积&quot;,&quot;模态遗忘&quot;,&quot;约束嵌入更新&quot;]"><strong><a href="../../venues/AAAI/2025/multimodal_rec/stair_manipulating_collaborative_and_multimodal_information_for_e_commerce_/">STAIR: Manipulating Collaborative and Multimodal Information for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示电商场景模态信息弱于协同信号，提出逐步图卷积沿维度分配层权重实现协同-模态共存，并以反向逐步卷积约束更新缓解模态遗忘，三数据集全排序提升2-6%且开销近LightGCN。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正样本扩充&quot;,&quot;分层对比学习&quot;]"><strong><a href="../../venues/AAAI/2026/collaborative_filtering/crops_improving_dense_retrieval_with_cross_perspective_positive_samples_in_/">CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用查询改写/推荐流/LLM 三源扩充正样本，配分层标签与 H-InfoNCE，快手搜索线上 CTR +0.87%、改写率 -0.65%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/事后干预&quot;,&quot;稀疏自编码器&quot;,&quot;单义神经元&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/extracting_interaction_aware_monosemantic_concepts_in_recommender_systems/">Extracting Interaction-Aware Monosemantic Concepts in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预测感知 SAE 从双塔嵌入中提取单义神经元，在 ML1M/Last.FM 上实现高纯度概念发现与事后可控干预。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;路径积分归因&quot;,&quot;反事实保真度&quot;]"><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/fidelity_aware_recommendation_explanations_via_stochastic_path_integration/">Fidelity-Aware Recommendation Explanations via Stochastic Path Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将路径积分归因引入推荐解释，用随机基线采样适配稀疏二值交互，在三种模型三数据集上刷新反事实保真度基准。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;半监督矩阵完成&quot;,&quot;低秩子空间共享&quot;,&quot;泛化界&quot;]"><strong><a href="../../venues/AAAI/2026/collaborative_filtering/generalization_bounds_for_semi_supervised_matrix_completion_with_distributi/">Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出隐式反馈采样分布与评分矩阵共享低秩子空间的半监督矩阵完成范式，证明误差可解耦为两项独立界，并在 ML-100K/Douban/Yelp 上验证。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔语义匹配&quot;,&quot;自监督语言对齐&quot;,&quot;档案感知多头注意力&quot;]"><strong><a href="../../venues/ACL_EMNLP/2022/collaborative_filtering/doctor_recommendation_in_online_health_forums_via_expertise_learning/">Doctor Recommendation in Online Health Forums via Expertise Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 MC-BERT 双塔编码匿名患者查询与医生档案/历史问诊对话，以档案-对话自监督对齐语言鸿沟、档案感知多头注意力聚合专家画像，春雨医生 119K 对话上 P@1 达 0.616。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;协同过滤与双塔召回&quot;,&quot;排序&quot;,&quot;模块化新闻推荐&quot;,&quot;监督对比学习&quot;,&quot;方面定制&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/collaborative_filtering/train_once_use_flexibly_a_modular_framework_for_multi_aspect_neural_news_re/">Train Once, Use Flexibly: A Modular Framework for Multi-Aspect Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用监督对比学习训练内容/方面专用新闻编码器，推理时线性组合相似度实现按需个性化与多样化，MIND上nDCG@10相对提升2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同过滤用户表征&quot;,&quot;Mixture of LoRA Experts&quot;,&quot;个性化奖励模型&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/copl_collaborative_preference_learning_for_personalizing_llms/">CoPL: Collaborative Preference Learning for Personalizing LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用用户-回复二部图协同过滤学习用户表征，结合 MoLE 路由个性化奖励模型，并支持无优化未见用户适配，在 UltraFeedback-P 等基准上显著超越个性化奖励模型基线。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;并行架构&quot;,&quot;个性化查询生成&quot;,&quot;自适应重排&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/rethinking_llm_based_recommendations_a_personalized_query_driven_parallel_i/">Rethinking LLM-Based Recommendations: A Personalized Query-Driven Parallel Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 QUEREC 并行框架，用 LLM 零样本生成个性化用户查询与特征增强物品描述，与 CF 模型并行全库检索后自适应重排，Full-Ranking 下最高提升 57%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;用户聚类&quot;,&quot;个性化 RAG&quot;,&quot;HDBSCAN&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/collaborative_filtering/clusterrag_cluster_based_collaborative_filtering_for_personalized_retrieval/">ClusterRAG: Cluster-Based Collaborative Filtering for Personalized Retrieval-Augmented Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 HDBSCAN 将用户按画像嵌入聚类，簇内 ColBERTv2 排序选近邻，簇-文档两级检索融合协同画像做个性化 RAG，LaMP 全任务最优。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;后处理/模型修正&quot;,&quot;属性遗忘 (Attribute Unlearning)&quot;,&quot;属性推断攻击 (AIA)&quot;,&quot;MMD 分布对齐&quot;]"><strong><a href="../../venues/ACM_MM/2023/debias_fairness_industrial/making_users_indistinguishable_attribute_wise_unlearning_in_recommender_sys/">Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出训练后属性遗忘 PoT-AU，用可分性损失+正则损失仅微调用户嵌入，D2D-MMD 变体把攻击 AUC 压至随机水平且推荐指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回（训练后嵌入级遗忘后处理）&quot;,&quot;属性遗忘 (Attribute Unlearning)&quot;,&quot;用户嵌入分布对齐&quot;,&quot;可复现性&quot;]"><strong><a href="../../venues/ACM_MM/2023/debias_fairness_industrial/reproducibility_companion_paper_making_users_indistinguishable_attribute_wi/">Reproducibility Companion Paper: Making Users Indistinguishable: Attribute-wise Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>复现属性遗忘 U2U-R/D2D-R 两法：在 NMF/LightGCN 上验证 ML-100K/1M/LFM-2B 的推荐保真度、抗攻击性与百倍级效率优势。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;后处理/嵌入层&quot;,&quot;机器遗忘 (Machine Unlearning)&quot;,&quot;互信息最小化&quot;,&quot;属性推断攻击&quot;]"><strong><a href="../../venues/ACM_MM/2025/debias_fairness_industrial/lego_a_lightweight_and_efficient_multiple_attribute_unlearning_framework_fo/">LEGO: A Lightweight and Efficient Multiple-Attribute Unlearning Framework for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多属性遗忘拆为并行嵌入校准与权重组合两步，以互信息最小化保证同时遗忘，动态需求下近零额外开销。</p>
</dd>
<dt data-paper-tags="[&quot;ACM Computing Surveys&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回与表征学习&quot;,&quot;对比学习&quot;,&quot;重建学习&quot;,&quot;对抗学习&quot;]"><strong><a href="../../venues/CIKM/2022/self_supervised_denoising/a_comprehensive_survey_on_self_supervised_learning_for_recommendation/">A Comprehensive Survey on Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统综述 170+ 篇 SSL4Rec 工作，提出对比/重建/对抗三范式与九场景视图中心分类，汇总评估协议并维护开源库 SSLRec。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;Gumbel 分布&quot;,&quot;变分编码器&quot;,&quot;评分长尾失衡&quot;]"><strong><a href="../../venues/CIKM/2022/collaborative_filtering/a_gumbel_based_rating_prediction_framework_for_imbalanced_recommendation/">A Gumbel-based Rating Prediction Framework for Imbalanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 Gumbel 变分编码器把评分频率映射到非正态空间，多尺度卷积融合评论交互特征并加残差个性化，5 数据集上 MAE 最高相对降 57.82%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;三元组重要性加权&quot;,&quot;双层优化&quot;,&quot;端到端聚类&quot;]"><strong><a href="../../venues/CIKM/2022/collaborative_filtering/adapting_triplet_importance_of_implicit_feedback_for_personalized_recommend/">Adapting Triplet Importance of Implicit Feedback for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以双层优化学习 (u,i,j) 三元组重要性权重，结合单/多兴趣状态设计与端到端物品聚类，Recall@20 较最优基线提升 3–21%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;多域双塔检索&quot;,&quot;领域自适应&quot;]"><strong><a href="../../venues/CIKM/2022/cross_domain_multi_scenario/adaptive_domain_interest_network_for_multi_domain_recommendation/">Adaptive Domain Interest Network for Multi-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>ADI 用共享/领域专属双路网络 + DSBN + SE 域注意力 + 跨域自训练伪标签，在阿里展示广告三场景召回上全场景 Recall 提升，线上 RPM +1.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Recall@N 直接优化&quot;,&quot;损失函数空间统一&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="../../venues/CIKM/2022/collaborative_filtering/croloss_towards_a_customizable_loss_for_retrieval_models_in_recommender_sys/">CROLoss: Towards a Customizable Loss for Retrieval Models in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将 Recall@N 重写为可微的成对比较核函数加权形式，用幂律权重 $w_\alpha$ 定制检索规模 N 并派生 Lambda 双核梯度法；Amazon/Taobao 召回指标相对交叉熵提升 5–12%，线上 A/B 收入 +4.75%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;跨通道对比学习&quot;,&quot;多路召回融合&quot;]"><strong><a href="../../venues/CIKM/2022/collaborative_filtering/mic_model_agnostic_integrated_cross_channel_recommenders/">MIC: Model-agnostic Integrated Cross-channel Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨通道对比学习把 U2I/U2U/I2I 三路召回统一进单模型，作为即插即用插件同时提升召回精度与多样性，并在腾讯新闻线上生效。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;原型对比学习&quot;]"><strong><a href="../../venues/CIKM/2022/collaborative_filtering/prototypical_contrastive_learning_and_adaptive_interest_selection_for_candi/">Prototypical Contrastive Learning and Adaptive Interest Selection for Candidate Generation in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微掩码按用户活跃度自适应选择兴趣向量数目，并以聚类原型对比学习缓解负采样类冲突，1:100 采样排序下 HR@10 相对提升 5–9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低秩分解&quot;,&quot;图卷积简化&quot;]"><strong><a href="../../venues/CIKM/2022/graph_rec/svd_gcn_a_simplified_graph_convolution_paradigm_for_recommendation/">SVD-GCN: A Simplified Graph Convolution Paradigm for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用截断 SVD 替代 GCN 邻域聚合，仅取 K 个奇异向量并重归一化，全库排序精度提升且百倍加速。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自编码器协同过滤&quot;,&quot;增量推理&quot;,&quot;物品转移矩阵&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/autoseqrec_autoencoder_for_efficient_sequential_recommendation/">AutoSeqRec: Autoencoder for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用单个共享编码器同时重建用户-物品矩阵与转移矩阵行/列，增量推理只更新输入矩阵不更新参数，ML-100K 上 Recall@10 相对提升 35%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负样本采样&quot;,&quot;Mixup 数据增广&quot;]"><strong><a href="../../venues/CIKM/2023/collaborative_filtering/dimension_independent_mixup_for_hard_negative_sample_in_collaborative_filte/">Dimension Independent Mixup for Hard Negative Sample in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>从采样区域视角重塑隐式反馈负采样，提出首个 Area-wise 方法 DINS（边界定义+逐维独立 Mixup+多层池化），在 Full-Ranking 协议下 MF backbone 上 Recall@20 最高提升 36.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;嵌入维度搜索&quot;,&quot;流式推荐&quot;]"><strong><a href="../../venues/CIKM/2023/rl_bandit_rec/dynamic_embedding_size_search_with_minimum_regret_for_streaming_recommender/">Dynamic Embedding Size Search with Minimum Regret for Streaming Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将流式推荐中的动态嵌入维度搜索建模为非平稳 LinUCB 赌博机，用频率+兴趣多样性指标做上下文，取得亚线性遗憾与更低内存/时延。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;拓扑接近度 Topological Closeness&quot;,&quot;GNN 表达性理论&quot;,&quot;免学习推荐算法&quot;]"><strong><a href="../../venues/CIKM/2023/graph_rec/how_expressive_are_graph_neural_networks_in_recommendation/">How Expressive are Graph Neural Networks in Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出三层 GNN 表达性度量框架，用链路级「拓扑接近度」替代图同构度量，并给出免学习 GTE 算法与六数据集全库排序验证。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022 (EvalRS DataChallenge)&quot;,&quot;去偏公平&quot;,&quot;召回 + 列表级配额重排&quot;,&quot;Item-based VAE&quot;,&quot;流行度偏差 MRED&quot;]"><strong><a href="../../venues/CIKM/2023/debias_fairness_industrial/item_based_variational_auto_encoder_for_fair_music_recommendation/">Item-based Variational Auto-encoder for Fair Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将 VAE-CF 转置为物品视角并按流行度分组训练、加组间重建公平正则, 配合配额式列表拼装与 BPRMF 兜底, 用 MRED 与行为测试换取长尾曝光公平。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数统一分析&quot;,&quot;对齐与均匀性&quot;,&quot;流行度去偏边缘&quot;]"><strong><a href="../../venues/CIKM/2023/collaborative_filtering/toward_a_better_understanding_of_loss_functions_for_collaborative_filtering/">Toward a Better Understanding of Loss Functions for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用统一数学框架把 7 种 CF 损失还原为 SSM 的极限特例，并提出边缘感知对齐+加权均匀性损失 MAWU，MF/LightGCN 全库排序下稳定超越 DirectAU 与 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;分组公平&quot;,&quot;item2vec&quot;,&quot;MR-ITF 指标&quot;]"><strong><a href="../../venues/CIKM/2023/debias_fairness_industrial/track2vec_fairness_music_recommendation_with_a_gpu_free_customizable_driven/">Track2Vec: fairness music recommendation with a GPU-free customizable-driven framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用特征分桶 + item2vec 分组召回 + 投票集成构建无 GPU 的公平音乐推荐，并提出 MR-ITF 逆频率公平指标，榜单超官方基线约 198%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Triplet Loss&quot;,&quot;矩阵分解&quot;,&quot;公平性评测&quot;]"><strong><a href="../../venues/CIKM/2023/collaborative_filtering/triplet_losses_based_matrix_factorization_for_robust_recommendations/">Triplet Losses-based Matrix Factorization for Robust Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多重 triplet 损失 + 稀有实体加权矩阵分解，在 LFM-1b 多面公平评测下 MRED 类指标较基线改善 55–89%，总分第 3。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;混排与供给调控（策略层）&quot;,&quot;双稳健学习 (DRL)&quot;,&quot;增益建模 Uplift / CATE&quot;,&quot;广告供给个性化&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/ads_supply_personalization_via_doubly_robust_learning/">Ads Supply Personalization via Doubly Robust Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 RCT 已知倾向分构造双稳健伪结局，轻量随机森林即可在长周期广告供给密度调控上把 CATE 估计与线上变现效率同时拉升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端召回&quot;,&quot;生成式标识 GID&quot;,&quot;内容-协同对齐&quot;]"><strong><a href="../../venues/CIKM/2024/generative_rec/content_based_collaborative_generation_for_recommender_systems/">Content-Based Collaborative Generation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ColaRec 用 LightGCN 层次聚类构造协同 GID，T5 序列到序列直接生成物品标识，并以物品索引任务+对比损失对齐内容与协同空间；四数据集 8:1:1 全库排序 Recall@20 提升 3.5%–6.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;云端协同训练/召回表征&quot;,&quot;联邦学习&quot;,&quot;正则化经验风险最小化&quot;,&quot;局部差分隐私&quot;]"><strong><a href="../../venues/CIKM/2024/debias_fairness_industrial/efficient_and_robust_regularized_federated_recommendation/">Efficient and Robust Regularized Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将联邦推荐重构为凸 RERM 问题，提出 RFRec/RFRecF，通信轮次降至 O(log 1/ε)、兼具鲁棒性与差分隐私。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤&quot;,&quot;反馈分图对比学习&quot;,&quot;宏观聚类建模&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/feedback_reciprocal_graph_collaborative_filtering/">Feedback Reciprocal Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按用户反馈将交互图拆为 I&amp;F/I&amp;U 双图分别做 GCF，用冲动性互惠对比与宏观聚类建模交换信息，全库排序下 Recall@20 平均提升 6.28%，淘宝 A/B GMV +1.61%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样偏差校正&quot;,&quot;排名损失 (WARP/LambdaRank)&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/improved_estimation_of_ranks_for_learning_item_recommenders_with_negative_s/">Improved Estimation of Ranks for Learning Item Recommenders with Negative Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将采样指标的二项校正迁移进 WARP/LambdaRank 训练，把串行 F-Rank 排名估计换成可批量并行校正估计，MSD 上 WARP Recall@20 相对提升 16.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Triplet Shapley&quot;,&quot;负采样与三元组重加权&quot;]"><strong><a href="../../venues/CIKM/2024/collaborative_filtering/interpretable_triplet_importance_for_personalized_ranking/">Interpretable Triplet Importance for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用三元组 Shapley 值可解释地度量 BPR 三元组重要性，蒙特卡洛+控制变量稳定估计并指导重采样，六数据集 R@20 平均提升约 9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;近似最大内积搜索&quot;,&quot;交叉熵损失近似&quot;,&quot;硬负采样&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/rece_reduced_cross_entropy_loss_for_large_catalogue_sequential_recommenders/">RECE: Reduced Cross-Entropy Loss for Large-Catalogue Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 GPU 友好的 LSH 分桶+MIPS 近似筛选高梯度负类，将全库 cross-entropy 的 logits 张量切片计算，峰值显存最多降 12 倍且质量持平或超越。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤量化&quot;,&quot;量化感知训练&quot;,&quot;动态步长&quot;]"><strong><a href="../../venues/CIKM/2025/collaborative_filtering/a_node_aware_dynamic_quantization_approach_for_graph_collaborative_filterin/">A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出节点感知动态量化 GNAQ，用参数共享学习每节点量化步长并以图消息传递更新，2-bit 下 Recall@10 平均提升 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Fourier KAN&quot;,&quot;特征变换&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/enhancing_graph_collaborative_filtering_with_fourierkan_feature_transformat/">Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Fourier KAN 替代 NGCF 中被误删的交互特征变换 W2，在 7:1:2 时间切分全库排序下 R@20 最高提升 7.8%，且可作即插即用骨干。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;负采样公平性&quot;,&quot;分组超参优化&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/evaluating_and_addressing_fairness_across_user_groups_in_negative_sampling_/">Evaluating and Addressing Fairness Across User Groups in Negative Sampling for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>实证揭示负采样比 K 增大使活跃用户增益、非活跃用户受损，提出分组特定负采样比并用带先验 Hyperband 优化，全库排序下 NDCG 提升 2–10%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;线性自编码器&quot;,&quot;闭式解知识蒸馏&quot;,&quot;长尾推荐&quot;]"><strong><a href="../../venues/CIKM/2025/llm_rec/llm_enhanced_linear_autoencoders_for_recommendation/">LLM-Enhanced Linear Autoencoders for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义矩阵构造 item-item 相关阵并以知识蒸馏正则注入 EASE 闭式解，全库排序下 R@20 平均提升 27.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标召回&quot;,&quot;目标条件化 Transformer&quot;,&quot;动态配额分配&quot;]"><strong><a href="../../venues/CIKM/2025/multi_task_reranking/mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;非负矩阵分解&quot;,&quot;患者画像聚类&quot;,&quot;多源数据融合&quot;]"><strong><a href="../../venues/CIKM/2025/collaborative_filtering/patient_clustering_via_integrated_profiling_of_clinical_and_digital_data/">Patient Clustering via Integrated Profiling of Clinical and Digital Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用掩码约束非负低秩分解统一融合诊断码与门户浏览/搜索行为，同时产出可解释软聚类与患者嵌入，聚类与推荐双任务全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;EASE 闭式解&quot;,&quot;侧信息增强&quot;,&quot;冷物品先验&quot;]"><strong><a href="../../venues/CIKM/2025/cold_start_rec/shallow_autoencoding_recommender_with_cold_start_handling_via_side_features/">Shallow AutoEncoding Recommender with Cold Start Handling via Side Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户/物品侧特征拼入 EASE 输入矩阵，并用内容相似度矩阵作冷物品权重先验，闭式解统一处理双冷启动。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;显式负反馈&quot;]"><strong><a href="../../venues/CIKM/2026/collaborative_filtering/dualspectralcf_training_free_sign_aware_spectral_collaborative_filtering/">DualSpectralCF: Training-Free Sign-Aware Spectral Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在任意谱滤波骨干上以带符号输入信号与带符号物品算子注入显式负反馈，免训练即插即用，全库排序下 R@20 最高提升 32.6%，比 SIGformer 快 7.7–155.3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;LLM推荐&quot;,&quot;召回&quot;,&quot;意图推理&quot;,&quot;个性化稠密检索&quot;,&quot;GRPO&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/think_to_personalize_unifying_reasoning_and_retrieval_for_user_centric_pers/">Think-to-Personalize: Unifying Reasoning and Retrieval for User-Centric Personalized Dense Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 SFT+GRPO 两阶段让 LLM 显式推理用户历史生成意图增强查询并统一编码，Full-Ranking 下 Broad/LongTail 召回显著提升，线上订单量 +0.46%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;免训练推荐&quot;,&quot;协同信号注入&quot;,&quot;图传播表征细化&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/training_free_llm_based_recommendation_with_post_llm_item_refinement_using_/">Training-Free LLM-Based Recommendation with Post-LLM Item Refinement Using Collaborative Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 post-LLM 范式 CoRRe，用共购图方向细化与流行度幅度校准改造 LLM 物品表征，免训练即在三域 Amazon 上超越全部免训练基线并多数超越训练式方法。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Transformer Scaling Law&quot;,&quot;特征异构归一化&quot;,&quot;多域统一建模&quot;]"><strong><a href="../../venues/CIKM/2026/collaborative_filtering/transretrieval_scaling_up_transformer_based_retrieval_for_industrial_recomm/">TransRetrieval: Scaling Up Transformer-Based Retrieval for Industrial Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用加权平均聚合+目标Token压缩+位置式域嵌入，让Transformer在召回阶段实现对数线性Scaling，线上收入+2.53%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;双专家门控混合&quot;,&quot;动态知识蒸馏&quot;,&quot;用户行为偏置&quot;]"><strong><a href="../../venues/DASFAA/2023/cold_start_rec/cold_warm_net_addressing_cold_start_users_in_recommender_systems/">Cold &amp; Warm Net: Addressing Cold-Start Users in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>召回阶段用冷启/升温双专家+门控+动态知识蒸馏+行为偏置网建模用户冷启动，冷启用户 HR@100 提升 9.5%–23.9%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;物品侧对比学习&quot;,&quot;多粒度数据增广&quot;]"><strong><a href="../../venues/DASFAA/2023/self_supervised_denoising/multi_granularity_item_based_contrastive_recommendation/">Multi-granularity Item-based Contrastive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>为双塔召回设计特征/语义/会话三级物品侧对比任务，工业级全库 HIT@500 相对 SOTA 提升 2.8%，线上 ACC +3.29%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;情境感知推荐&quot;,&quot;条件神经网络&quot;,&quot;可插拔增强器&quot;]"><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/a_situation_aware_enhancer_for_personalized_recommendation/">A Situation-aware Enhancer for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将情境视为交互前置条件，用可插拔双塔增强器 SARE 建模个性化情境感知与偏好影响，在 MIND/KuaiRand 上显著超越骨干与 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;软标签解耦&quot;,&quot;标签传播&quot;]"><strong><a href="../../venues/DASFAA/2025/collaborative_filtering/learning_recommender_systems_with_soft_target_a_decoupled_perspective/">Learning Recommender Systems with Soft Target: A Decoupled Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Softmax 软标签目标解耦为目标置信度与未观测项兴趣分布两项，配合 K-means 邻居标签传播生成软标签，在四个骨干模型三个数据集上稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;变分推断解耦&quot;,&quot;未观测混杂&quot;]"><strong><a href="../../venues/DASFAA/2025/debias_fairness_industrial/separating_and_learning_latent_confounders_to_enhancing_user_preferences_mo/">Separating and Learning Latent Confounders to Enhancing User Preferences Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将前序推荐系统作为未观测混杂代理，用变分推断在隐空间解耦用户偏好与混杂，实现通用去偏，五数据集 Recall@10 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/排序&quot;,&quot;GAN&quot;,&quot;可复现性评估&quot;]"><strong><a href="../../venues/ECIR/2022/collaborative_filtering/an_evaluation_study_of_generative_adversarial_networks_for_collaborative_fi/">An Evaluation Study of Generative Adversarial Networks for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>ECIR 2022 复现评估 CFGAN：指出真实画像条件+无噪声使生成器退化为自编码器；在 Ciao/ML100K/ML1M 全排序 top-N 下，CFGAN 稳定但不及调优 EASE R，训练慢两个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;Filter Bubble&quot;,&quot;动态仿真&quot;,&quot;校准推荐&quot;]"><strong><a href="../../venues/ECIR/2023/debias_fairness_industrial/evolution_of_filter_bubbles_and_polarization_in_news_recommendation/">Evolution of Filter Bubbles and Polarization in News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 4 万篇立场标注新闻与 500 个 Pew 仿真用户构建长程推荐闭环，揭示即使无说服效应，MF 反馈回路也会自发极化，且 SOTA 校准干预仅减缓不能阻断。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;市场嵌入 (Market Embedding)&quot;,&quot;跨市场推荐 CMR&quot;,&quot;元学习替代方案&quot;]"><strong><a href="../../venues/ECIR/2023/cross_domain_multi_scenario/market_aware_models_for_efficient_cross_market_recommendation/">Market-Aware Models for Efficient Cross-Market Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用市场嵌入 o_l 逐元素调制物品表征替代元学习，在 XMarket 上以 15% 训练时间取得或超越 MAML/FOREC 的 nDCG@10，全局建模下 6/7 市场夺冠。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（pairwise 训练期采样）&quot;,&quot;提供者组曝光公平&quot;,&quot;代价敏感元学习&quot;,&quot;重加权负采样&quot;]"><strong><a href="../../venues/ECIR/2024/debias_fairness_industrial/a_cost_sensitive_meta_learning_strategy_for_fair_provider_exposure_in_recom/">A Cost-Sensitive Meta-Learning Strategy for Fair Provider Exposure in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用类别代价重加权三元组负采样，使提供者组曝光对齐目录占比，且 NDCG 无损（ML-1M Exp@10 由 1.4% 升至 6.0%）。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;表示解耦&quot;,&quot;可解释性量化&quot;,&quot;可复现性研究&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/are_representation_disentanglement_and_interpretability_linked_in_recommend/">Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现5个推荐模型×4数据集，量化解耦度与可解释性，发现解耦与效果无关但与表示可解释性强正相关。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;原型矩阵分解&quot;,&quot;流行度去偏&quot;,&quot;文化多样性&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/embedding_cultural_diversity_in_prototype_based_recommender_systems/">Embedding Cultural Diversity in Prototype-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 ProtoMF 原型矩阵分解上引入 K-近邻原型过滤与原型正交分布正则，以物品国别为文化代理缓解流行度偏差，全库排序下长尾平均排名降 27%、HR@10 反升 2%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Next Basket Recommendation&quot;,&quot;频率感知模块&quot;,&quot;Transformer 序列建模&quot;]"><strong><a href="../../venues/ECIR/2025/sequential_rec/saferec_self_attention_and_frequency_enriched_model_for_next_basket_recomme/">SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 SASRec 上并联频率感知打分模块，用物品出现频次与历史指示向量补足重复购买建模，Leave-One-Basket 全库排序下 Recall@10 最高提升 8%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图划分相似度&quot;,&quot;可复现性评测&quot;,&quot;长尾分析&quot;]"><strong><a href="../../venues/ECIR/2026/collaborative_filtering/a_reproducible_and_fair_evaluation_of_partition_aware_collaborative_filteri/">A Reproducible and Fair Evaluation of Partition-aware Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在统一用户留出划分与 TPE 调参下复现 FPSR/FPSR+，发现 BISM 在 Amazon-CDs 反超，FPSR 家族在长尾显著占优。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练（联邦推荐）&quot;,&quot;联邦推荐投毒攻击&quot;,&quot;模型投毒与梯度操纵&quot;,&quot;隐私保护与鲁棒聚合&quot;]"><strong><a href="../../venues/ICDE/2022/debias_fairness_industrial/fedrecattack_model_poisoning_attack_to_federated_recommendation/">FedRecAttack: Model Poisoning Attack to Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用仅1%公开交互反解全局用户隐向量，配合指数尾部代理损失生成限幅分摊的投毒梯度，3%恶意用户即可把目标物品ER@5推至0.94+且HR@10仅降2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（训练期负采样）&quot;,&quot;假阴性辨识&quot;,&quot;贝叶斯后验风险最小化&quot;,&quot;负采样&quot;]"><strong><a href="../../venues/ICDE/2023/collaborative_filtering/bayesian_negative_sampling_for_recommendation/">Bayesian Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>由排序关系导出真/假阴性类条件密度，构造模型无关贝叶斯后验负信号与风险最小化采样准则；MF/LightGCN 双主干 Top-20 NDCG 最高 +5.1%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/排序&quot;,&quot;Softmax Loss&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Bilateral Softmax Loss&quot;]"><strong><a href="../../venues/ICDE/2024/collaborative_filtering/bsl_understanding_and_improving_softmax_loss_for_recommendation/">BSL: Understanding and Improving Softmax Loss for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用KL-DRO解释Softmax Loss的负样本鲁棒性与方差公平性，提出正负双侧Log-Expectation-Exp的BSL，一行代码在四数据集三骨干上提升Recall/NDCG。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;k-DPP&quot;,&quot;集合级排序准则&quot;,&quot;多样性-相关性权衡&quot;]"><strong><a href="../../venues/ICDE/2024/collaborative_filtering/learning_k_determinantal_point_processes_for_personalized_ranking/">Learning k-Determinantal Point Processes for Personalized Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 k-DPP 集合概率比较替代 BPR 的点对比较，把相关性排序与多样性排序统一进同一优化准则，全库排序下 Recall@5 相对最优基线提升 8–11%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;LightGCN&quot;,&quot;连续图传播&quot;]"><strong><a href="../../venues/ICDM/2023/graph_rec/graph_neural_ordinary_differential_equations_based_method_for_collaborative/">Graph Neural Ordinary Differential Equations-based method for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 1–2 层 GCN 参数化导数函数、以 Neural ODE 积分替代多层堆叠与层组合，Euler 求解器在四个 Amazon 数据集上取得最优 Recall/NDCG 且训练最快。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;组合嵌入 (Compositional Embedding)&quot;,&quot;可学习软阈值剪枝&quot;,&quot;嵌入表内存压缩&quot;]"><strong><a href="../../venues/ICDM/2023/collaborative_filtering/learning_compact_compositional_embeddings_via_regularized_pruning_for_recom/">Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CERP 用双码本平衡哈希组合嵌入 + 软阈值联合剪枝 + 互补性正则，在 90%/95%/99% 稀疏预算下把嵌入层压到 18 维仍超越 PEP/QR。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;双层用户建模&quot;,&quot;双极组嵌入&quot;,&quot;正交解耦&quot;]"><strong><a href="../../venues/ICDM/2024/collaborative_filtering/gprec_bi_level_user_modeling_for_deep_recommenders/">GPRec: Bi-level User Modeling for Deep Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可学习分组器 + 双极组嵌入 + 正交个人表征，将组级与个体级用户建模统一为即插即用模块，三数据集 CTR/AUC 全面刷新 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;LLM 用户画像重建&quot;,&quot;模型无关知识迁移&quot;,&quot;RecBole 实验框架&quot;]"><strong><a href="../../venues/ICDM/2024/llm_rec/llm_kt_a_versatile_framework_for_knowledge_transfer_from_large_language_mod/">LLM-KT: A Versatile Framework for Knowledge Transfer from Large Language Models to Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 LLM 生成的用户画像注入任意 CF 模型的中间层并以重建为辅助任务做知识迁移，模型无关、基于 RecBole 的插件式框架，ML-1M/Amazon CDs 上 NDCG@10 最高 +21.57%。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;重要性重采样&quot;,&quot;负采样偏差校正&quot;,&quot;难负样本缓存&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2022/collaborative_filtering/cache_augmented_inbatch_importance_resampling_for_training_recommender_retr/">Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用查询相关的批内重要性重采样加难负样本缓存，将 inbatch 采样偏差压向真实 softmax，五数据集 NDCG@10 最高相对提升 17.1%</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;角度 Margin 对比损失&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2022/debias_fairness_industrial/incorporating_bias_aware_margins_into_contrastive_loss_for_collaborative_fi/">Incorporating Bias-aware Margins into Contrastive Loss for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用流行度偏置提取器为每个交互定制角度 margin 并注入 softmax 对比损失，在 8 个数据集的平衡/非平衡/时间切分评测上同时提升头尾性能 12.1%–58.4%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/全库排序&quot;,&quot;NTK&quot;,&quot;数据蒸馏&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2022/collaborative_filtering/infinite_recommendation_networks_a_data_centric_approach/">Infinite Recommendation Networks: A Data-Centric Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 NTK 将推荐自编码器推到无限宽瓶颈，闭式 KRR 求解，并以 Gumbel 数据蒸馏合成 0.1% 用户摘要保持 96–105% 性能。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2022&quot;,&quot;协同过滤（按指定归档）&quot;,&quot;非推荐链路/神经形态推理&quot;,&quot;ANN-SNN Conversion&quot;,&quot;Quantization Clip-Floor-Shift&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2022/collaborative_filtering/optimal_ann_snn_conversion_for_high_accuracy_and_ultra_low_latency_spiking_/">Optimal ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>按指定归档；论文实为 ANN-SNN 转换，提出量化 clip-floor-shift 激活，证明期望转换误差为零，CIFAR-10 仅 2 步达 91.18%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/精排&quot;,&quot;对抗负采样&quot;,&quot;难例挖掘&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/self_supervised_denoising/empowering_collaborative_filtering_with_principled_adversarial_contrastive_/">Empowering Collaborative Filtering with Principled Adversarial Contrastive Loss</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 AdvInfoNCE：用可学习难度 δ_j 对抗重加权负样本，证明等价 KL-DRO，在四个 Top-K 基准上显著提升 OOD 泛化。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;双重稳健估计 (DR)&quot;,&quot;靶向学习 (Targeted Learning)&quot;,&quot;逆倾向加权 (IPS)&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/debias_fairness_industrial/tdr_cl_targeted_doubly_robust_collaborative_learning_for_debiased_recommend/">TDR-CL: Targeted Doubly Robust Collaborative Learning for Debiased Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用靶向学习把 DR 的校正项经验均值归零，并将插补误差拆为参数化+非参数化协同更新，Coat/Music 上 AUC、NDCG 全面超越 DR 家族且方差更小。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练与模型维护（隐私卸载）&quot;,&quot;Machine Unlearning&quot;,&quot;Recommendation Fairness&quot;,&quot;Benchmark&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/cure4rec_a_benchmark_for_recommendation_unlearning_with_deeper_influence/">CURE4Rec: A Benchmark for Recommendation Unlearning with Deeper Influence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个推荐卸载综合基准，以完整性、效用、效率、公平四维及 core/edge/random 三类卸载集系统评测 5 种方法。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/排序预测&quot;,&quot;因果去偏&quot;,&quot;倾向得分&quot;,&quot;RKHS核平衡&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/debiased_collaborative_filtering_with_kernel_based_causal_balancing/">Debiased Collaborative Filtering with Kernel-Based Causal Balancing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用RKHS核自适应平衡倾向得分，提出KBIPS/KBDR，在COAT/MUSIC/PRODUCT提升AUC/NDCG/F1。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;差分隐私&quot;,&quot;加性个性化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/federated_recommendation_with_additive_personalization/">Federated Recommendation with Additive Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>FedRAP 以稀疏全局物品嵌入加本地个性化偏置实现联邦推荐，在六数据集 Leave-One-Out 采样排序下显著优于 FL 基线。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表征增强&quot;,&quot;测试时增强&quot;,&quot;消息传递机理&quot;,&quot;轻量图协同过滤&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/graph_rec/how_does_message_passing_improve_collaborative_filtering/">How Does Message Passing Improve Collaborative Filtering?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示消息传递主要靠前向邻居表征而非梯度，提出仅推理时聚合的TAG-CF，全库排序下冷/低度用户最高提升39.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;LLE 流形学习&quot;,&quot;EASE 闭式解&quot;,&quot;嵌入正则化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/collaborative_filtering/implicitslim_and_how_it_improves_embedding_based_collaborative_filtering/">ImplicitSLIM and How it Improves Embedding-based Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 EASE 闭式解嵌入 LLE 流形学习框架，用 Woodbury 恒等式在 $L\times L$空间直接抽取嵌入，作为任意 CF 模型的初始化与正则，免去$I\times I$ 相似度矩阵存储。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;损失函数设计&quot;,&quot;假阴性鲁棒&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/collaborative_filtering/psl_rethinking_and_improving_softmax_loss_from_pairwise_perspective_for_rec/">PSL: Rethinking and Improving Softmax Loss from Pairwise Perspective for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 Softmax Loss 改写为成对形式并把 exp 换成 tanh/atan/ReLU 有界激活，兼具更紧 DCG 上界、可控样本权重与 DRO 化 BPR 三重性质，IID/OOD/噪声三场景 NDCG@20 稳定提升 1.1%–5.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/训练&quot;,&quot;CVaR&quot;,&quot;iALS&quot;,&quot;卷积平滑&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/collaborative_filtering/safe_collaborative_filtering/">Safe Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以卷积平滑CVaR嵌入iALS，在保持平均Recall的同时显著改善尾部用户排序质量。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（CF 表征训练）&quot;,&quot;对抗协同过滤&quot;,&quot;投毒攻击鲁棒性&quot;,&quot;个性化扰动&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/debias_fairness_industrial/understanding_and_improving_adversarial_collaborative_filtering_for_robust_/">Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论证明 ACF 降误差并据此按用户嵌入尺度分配扰动幅度，PamaCF 在 Gowalla/Yelp/MIND 上兼顾推荐与抗投毒。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;成对表征&quot;,&quot;混合双塔架构&quot;,&quot;时序异构图&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/graph_rec/contextgnn_beyond_two_tower_recommendation_systems/">ContextGNN: Beyond Two-Tower Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出局部成对表征+全局双塔回退的混合GNN架构，在RelBench十个任务上平均MAP较最强成对基线提升20%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;经验贝叶斯&quot;,&quot;协变量调制先验&quot;,&quot;矩阵分解&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/collaborative_filtering/covariate_moderated_empirical_bayes_matrix_factorization/">Covariate-moderated Empirical Bayes Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用协变量调制的经验贝叶斯先验统一矩阵分解，MovieLens 100K 上 RMSE 全面优于 EBMF/MFAI/VAE 系基线。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;向量量化&quot;,&quot;对比对齐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/face_a_general_framework_for_mapping_collaborative_filtering_embeddings_int/">FACE: A General Framework for Mapping Collaborative Filtering Embeddings into LLM Tokens</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用解耦投影+RQ-VAE 将 CF 嵌入映射为冻结 LLM 词表中的离散 token，配合对比对齐实现免微调的语义可解释与性能增益。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语言表征同构&quot;,&quot;线性映射探针&quot;,&quot;零样本推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/language_representations_can_be_what_recommenders_need_findings_and_potenti/">Language Representations Can be What Recommenders Need: Findings and Potentials</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示语言空间与推荐行为空间同构，提出纯语言表征的 AlphaRec，在 Amazon 三数据集 Full-Ranking 上超越 ID-based CF 6.8%–9.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;全流程数据集&quot;,&quot;分布偏移&quot;,&quot;级联一致性&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/debias_fairness_industrial/recflow_an_industrial_full_flow_recommendation_dataset/">RecFlow: An Industrial Full Flow Recommendation Dataset</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个覆盖工业六阶段漏斗全流程的推荐数据集，含1.9B阶段样本，揭示训练-服务分布偏移并验证阶段样本作为难负例与级联蒸馏的显著收益。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;成对比较学习&quot;,&quot;非凸矩阵分解收敛性&quot;,&quot;矩阵 Bernstein 不等式&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/collaborative_filtering/recommendations_with_sparse_comparison_data_provably_fast_convergence_for_n/">Recommendations with Sparse Comparison Data: Provably Fast Convergence for Nonconvex Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将成对比较反馈建模为低秩矩阵分解下的离散选择似然，证明稀疏数据下投影梯度下降在热启动邻域内指数收敛，样本复杂度近最优。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;KV Cache 压缩&quot;,&quot;跨用户协同共享&quot;,&quot;长序列推理加速&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;线性自编码器&quot;,&quot;闭式解&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/collaborative_filtering/generalizing_linear_autoencoder_recommenders_with_decoupled_expected_quadra/">Generalizing Linear Autoencoder Recommenders with Decoupled Expected Quadratic Loss</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 EDLAE 目标推广为 DEQL，首次给出 b&gt;0 闭式解并用 Miller 矩阵逆定理把复杂度从 O(n^4) 降到 O(n^3)，在 9 个基准上稳定超越 b=0 基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型更新/遗忘&quot;,&quot;Machine Unlearning&quot;,&quot;Low-Rank Adapter&quot;,&quot;Knowledge Distillation&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/obliviate_efficient_unlearning_in_recommender_systems/">Obliviate: Efficient Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>两阶段低秩适配器+局部校准的推荐系统机器遗忘框架，20%删除率下兼顾遗忘完整性与效用，比全量重训快约 3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;交叉熵损失&quot;,&quot;排序模仿&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/rejuvenating_cross_entropy_loss_in_knowledge_distillation_for_recommender_s/">Rejuvenating Cross-Entropy Loss in Knowledge Distillation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示CE蒸馏损失与NDCG的闭包假设矛盾，提出师生协同子集拆分与自适应加权的RCE-KD，全库排序下NDCG@20最高提升11.76%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;两级 Softmax 采样&quot;,&quot;负采样偏差校正&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/collaborative_filtering/two_level_softmax_sampling_done_right_correcting_bias_from_size_imbalance_a/">Two-Level Softmax Sampling Done Right: Correcting Bias from Size Imbalance and Dispersion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示两级 softmax 采样因簇大小失衡与簇内相似度离散导致的系统性偏差，提出 S-2LS/SD-2LS 校正，KL 保真度提升 1–4 个数量级且开销近乎为零。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;逆倾向加权 IPS&quot;,&quot;对抗域适应去偏&quot;]"><strong><a href="../../venues/IJCAI/2022/debias_fairness_industrial/towards_resolving_propensity_contradiction_in_offline_recommender_learning/">Towards Resolving Propensity Contradiction in Offline Recommender Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出倾向无关泛化误差上界与对抗式矩阵分解 DAMF，无需 MCAR 数据与倾向估计，Yahoo! R3 上 MSE 相对降 26.3%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;负采样与去偏&quot;]"><strong><a href="../../venues/IJCAI/2022/collaborative_filtering/trading_hard_negatives_and_true_negatives_a_debiased_contrastive_collaborat/">Trading Hard Negatives and True Negatives: A Debiased Contrastive Collaborative Filtering Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用集合级对比目标自适应挖掘硬负例，并以频次感知批内采样 + 去偏项消除假负例偏差，MovieLens/Yelp/Gowalla 上 NDCG@10 提升 4.9%–9.1%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/端侧推理&quot;,&quot;联邦推荐&quot;,&quot;双重个性化&quot;]"><strong><a href="../../venues/IJCAI/2023/debias_fairness_industrial/dual_personalization_on_federated_recommendation/">Dual Personalization on Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>PFedRec 以端侧轻量模型、局部评分函数与物品嵌入后微调实现双重个性化，在 ML-100K/1M 上 HR@10 较 Fed 最优提升约 8–10%，并给出 LDP 隐私-效用分析。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;神经符号学习&quot;,&quot;Beta 分布嵌入&quot;]"><strong><a href="../../venues/IJCAI/2023/sequential_rec/sequential_recommendation_with_probabilistic_logical_reasoning/">Sequential Recommendation with Probabilistic Logical Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布做概率逻辑嵌入，将可闭运算的 AND/NOT 逻辑推理网络与 DNN 序列模型双路解耦融合，全库排序下各骨干稳定涨点。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;过平滑 (Over-smoothing)&quot;,&quot;模型无关即插即用框架&quot;,&quot;协同邻域校正&quot;]"><strong><a href="../../venues/IJCAI/2024/graph_rec/dgr_a_general_graph_desmoothing_framework_for_recommendation_via_global_and/">DGR: A General Graph Desmoothing Framework for Recommendation via Global and Local Perspectives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以全局向量扰动 (GMP) 与局部协同校正损失 (LEC) 构成模型无关的过平滑抑制框架，即插即用于 5 种 GCN 骨干，在 5 个基准上 Full-Ranking 平均 Recall@20 提升 2%~11%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;截断SVD&quot;,&quot;闭式解线性自编码器&quot;,&quot;噪声鲁棒&quot;]"><strong><a href="../../venues/IJCAI/2024/collaborative_filtering/svd_ae_simple_autoencoders_for_collaborative_filtering/">SVD-AE: Simple Autoencoders for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出基于截断SVD的线性自编码器闭式解，直接重建用户-物品矩阵，以低秩归纳偏置兼顾全库排序精度、噪声鲁棒性与免训练效率。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;生成式推荐&quot;,&quot;Draft-and-Verify&quot;,&quot;混合稀疏-稠密表征&quot;,&quot;并行验证推理&quot;]"><strong><a href="../../venues/IJCAI/2026/generative_rec/dav_gen_end_to_end_generative_retrieval_via_draft_and_verify/">DaV-Gen: End-to-End Generative Retrieval via Draft-and-Verify</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成式检索重构为“起草-验证”双阶段单模型范式，以混合稀疏-稠密表征与复合损失统一召回与排序，工业搜索延迟降至70ms。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;隐私攻击评测（黑盒 top-N 输出）&quot;,&quot;成员推断攻击&quot;,&quot;解耦表征（Gaussian/vMF）&quot;,&quot;样本重加权与交替训练&quot;]"><strong><a href="../../venues/KDD/2022/debias_fairness_industrial/debiasing_learning_for_membership_inference_attacks_against_recommender_sys/">Debiasing Learning for Membership Inference Attacks Against Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Gaussian+vMF 双先验 VAE 解耦差异向量、以 truth-level score 交替重加权，同时消除影子/目标推荐器落差与隐状态估计偏差，黑盒 AUC 由 &lt;0.7 升至 0.98+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练（联邦训练阶段攻击）&quot;,&quot;联邦推荐安全&quot;,&quot;硬负采样投毒&quot;]"><strong><a href="../../venues/KDD/2022/debias_fairness_industrial/fedattack_effective_and_covert_poisoning_attack_on_federated_recommendation/">FedAttack: Effective and Covert Poisoning Attack on Federated Recommendation via Hard Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用全局最难正负样本污染联邦推荐的本地训练数据，1% 恶意客户端即可使 HR@5 降 8%，并绕过鲁棒聚合与恶意梯度检测。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;洛伦兹模型&quot;,&quot;间隔排序损失&quot;,&quot;信息负采样&quot;]"><strong><a href="../../venues/KDD/2022/collaborative_filtering/hicf_hyperbolic_informative_collaborative_filtering/">HICF: Hyperbolic Informative Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>拆解双曲与欧氏 CF 在头/尾物品上的行为差异，用几何感知自适应 margin 与双曲信息负采样改造间隔排序损失，全库指标最高提升 12.92%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;二值量化&quot;,&quot;图协同过滤&quot;]"><strong><a href="../../venues/KDD/2022/collaborative_filtering/learning_binarized_graph_representations_with_multi_faceted_quantization_re/">Learning Binarized Graph Representations with Multi-faceted Quantization Reinforcement for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>BiGeaR 以逐层二值化、Top-R 推理蒸馏和 Dirac 梯度估计，在5个基准上接近全精度 LightGCN 且约8倍加速。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双塔 Transformer&quot;,&quot;因果掩码序列建模&quot;]"><strong><a href="../../venues/KDD/2022/sequential_rec/nxtpost_user_to_post_recommendations_in_facebook_groups/">NxtPost: User to Post Recommendations in Facebook Groups</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用预训练帖子向量替代词表嵌入的双塔 Transformer 因果序列召回，短/长期双损失在 Facebook Groups 上线，离线 Hits@1 绝对提升 49%、线上 +0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;对齐-均匀性&quot;,&quot;免负采样损失&quot;]"><strong><a href="../../venues/KDD/2022/collaborative_filtering/towards_representation_alignment_and_uniformity_in_collaborative_filtering/">Towards Representation Alignment and Uniformity in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>从超球面均匀性与对齐性视角重构 CF 损失，提出无需负采样的 DirectAU，简单 MF 编码器在三个数据集全库排序下超 SOTA 最高 14.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;LightGCN&quot;,&quot;多准则推荐&quot;,&quot;PairNorm&quot;]"><strong><a href="../../venues/KDD/2023/graph_rec/criteria_tell_you_more_than_ratings_criteria_preference_aware_light_graph_c/">Criteria Tell You More than Ratings: Criteria Preference-Aware Light Graph Convolution for Effective Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多准则评分展开为准则-物品扩展二部图，用轻量图卷积 + 用户准则偏好/物品准则双嵌入建模，全库排序下 P@5 最高提升 141.2%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;近端因果推断&quot;,&quot;潜在混淆因子识别&quot;]"><strong><a href="../../venues/KDD/2023/debias_fairness_industrial/debiasing_recommendation_by_learning_identifiable_latent_confounders/">Debiasing Recommendation by Learning Identifiable Latent Confounders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用代理变量做近端因果推断，学习可识别的潜在混淆因子，为反事实反馈提供可识别性保证；三数据集全库排序 NDCG@5 最高提升 3.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/模型设计&quot;,&quot;联合搜索&quot;,&quot;AutoML&quot;]"><strong><a href="../../venues/KDD/2023/collaborative_filtering/efficient_and_joint_hyperparameter_and_architecture_search_for_collaborativ/">Efficient and Joint Hyperparameter and Architecture Search for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出CF超参与架构联合搜索：超参筛减+频率子图两阶段BORE/RF搜索，全库Recall@20最高提升13.02%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;双层优化&quot;,&quot;梯度匹配&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="../../venues/KDD/2023/self_supervised_denoising/efficient_bi_level_optimization_for_recommendation_denoising/">Efficient Bi-Level Optimization for Recommendation Denoising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将隐式反馈去噪建模为双层优化，用权重生成器+一步梯度匹配高效求解，LightGCN 上 Recall@20 最高提升 10.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;有效电导&quot;,&quot;难负样本采样&quot;,&quot;数据子采样&quot;]"><strong><a href="../../venues/KDD/2023/graph_rec/graph_based_model_agnostic_data_subsampling_for_recommendation_systems/">Graph-Based Model-Agnostic Data Subsampling for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用用户-物品二部图的有效电导度量负样本难度并做边传播平滑，实现免 pilot 模型的数据子采样，KuaiRec/MIND 上 AUC 稳定超越均匀采样且可与模型法互补融合。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;路径特定反事实公平&quot;,&quot;潜中介分析&quot;,&quot;弱监督变分推断&quot;]"><strong><a href="../../venues/KDD/2023/debias_fairness_industrial/path_specific_counterfactual_fairness_for_recommender_systems/">Path-Specific Counterfactual Fairness for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双潜中介分解敏感特征的公平/不公平因果路径，以最小改动原则构造零路径特定偏差的公平世界，PSF-VAE 弱监督变分推断在保持多样性的同时显著降低不公平命中率。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排/召回模型压缩&quot;,&quot;持续学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/KDD/2024/collaborative_filtering/continual_collaborative_distillation_for_recommender_system/">Continual Collaborative Distillation for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 CCD，在非平稳数据流中协同更新教师与学生：双代理回放防遗忘、新实体初始化与教师吸收学生侧知识；Yelp 学生 H-mean 较最佳基线提升 10.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;双稳健估计&quot;,&quot;噪声反馈&quot;]"><strong><a href="../../venues/KDD/2024/debias_fairness_industrial/debiased_recommendation_with_noisy_feedback/">Debiased Recommendation with Noisy Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次联合建模 MNAR 与结果测量误差 (OME)，提出 OME-EIB/IPS/DR 估计量、弱可分误差参数辨识与交替去噪训练，在 Coat/Music/KuaiRec 上 AUC 相对提升约 1.5–2.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱图滤波&quot;,&quot;广义图归一化&quot;,&quot;免训练协同过滤&quot;]"><strong><a href="../../venues/KDD/2024/graph_rec/how_powerful_is_graph_filtering_for_recommendation/">How Powerful is Graph Filtering for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以谱锐度视角提出广义图归一化 G²N 与个性化图滤波 IGF，免训练闭式 SGFCF 在四个密度差异数据集上 nDCG@10 最高提升 14.3%，训练耗时降低三个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;偏序推荐图&quot;,&quot;多行为协同过滤&quot;,&quot;偏序 BPR 采样&quot;]"><strong><a href="../../venues/KDD/2024/graph_rec/multi_behavior_collaborative_filtering_with_partial_order_graph_convolution/">Multi-Behavior Collaborative Filtering with Partial Order Graph Convolutional Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以分级偏序秩函数融合多行为图为单张加权 POG，POGCN 图卷积 + 偏序 BPR 采样，全库排序下三基准 Mean NDCG 提升 19.67%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图卷积等价&quot;,&quot;简单对比协同过滤&quot;]"><strong><a href="../../venues/KDD/2024/self_supervised_denoising/unifying_graph_convolution_and_contrastive_learning_in_collaborative_filter/">Unifying Graph Convolution and Contrastive Learning in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论证明对比损失等价双图卷积，提出无GCN的SCCF，在4数据集全库排序最高提升6.4%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Box Embedding&quot;,&quot;标签感知推荐&quot;]"><strong><a href="../../venues/KDD/2024/graph_rec/when_box_meets_graph_neural_network_in_tag_aware_recommendation/">When Box Meets Graph Neural Network in Tag-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户/物品/标签统一建模为超矩形盒，用交并逻辑算子替代 GNN 均值聚合，并以 Gumbel 平滑盒体积作为匹配分，全库排序下全面超越 LFGCF。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;NDCG@K 代理损失&quot;,&quot;分位数估计&quot;,&quot;Softmax Loss&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/breaking_the_top_k_barrier_advancing_top_k_ranking_metrics_optimization_in_/">Breaking the Top-$K$Barrier: Advancing Top-$K$ Ranking Metrics Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用量化分位数把 Top-K 截断转成阈值比较，推导 NDCG@K 的紧致平滑上界 SL@K，四数据集三骨干平均提升 6.03%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;序数逻辑回归&quot;,&quot;多反馈统一嵌入&quot;,&quot;嵌套优化&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/embed_progressive_implicit_preference_in_unified_space_for_deep_collaborati/">Embed Progressive Implicit Preference in Unified Space for Deep Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多类隐式反馈按稀疏度映射为有序类别，用嵌套编码+嵌套OLR把渐进偏好压入统一空间，实现单次NNS一阶段排序。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;图信号处理&quot;,&quot;频域重加权&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/exploring_feature_based_knowledge_distillation_for_recommender_system_a_fre/">Exploring Feature-based Knowledge Distillation for Recommender System: A Frequency Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从频域视角揭示特征蒸馏等权缺陷，提出轻量图滤波重加权 FreqD，全库排序下 Recall@20 最高提升 18.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;联邦训练通信&quot;,&quot;梯度聚类压缩&quot;,&quot;自适应动作共享&quot;,&quot;联邦推荐&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/gradients_as_an_action_towards_communication_efficient_federated_recommende/">Gradients as an Action: Towards Communication-Efficient Federated Recommender Systems via Adaptive Action Sharing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将物品梯度聚类为少量“动作”共享传输，配合自适应聚类与带宽感知压缩，在 ML-100K/1M/Lastfm-2K 上以最高 96.88% 压缩率不损性能。</p>
</dd>
<dt data-paper-tags="[&quot;KDD TSMO 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;极端多分类&quot;,&quot;S2 地理单元&quot;,&quot;位置召回&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/high_precision_audience_expansion_via_extreme_classification_in_a_two_sided/">High Precision Audience Expansion via Extreme Classification in a Two-Sided Marketplace</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Airbnb 位置召回从矩形边界重构为 25M 级 S2 单元极端多分类，离线精度 +11.01%、线上未取消预订者 +0.17%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;兴趣单元聚类&quot;,&quot;分层序列建模&quot;,&quot;限库存推荐&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/iu4rec_interest_unit_based_product_organization_and_recommendation_for_e_co/">IU4Rec: Interest Unit-Based Product Organization and Recommendation for E-Commerce Platform</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>闲鱼C2C限库存场景下以兴趣单元(IU)重构商品组织与两阶段推荐，IU-Boosted网络离线AUC+1.91%、线上CTR+2.90%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;权重衰减&quot;,&quot;流行度偏置&quot;,&quot;嵌入模长初始化&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/on_the_role_of_weight_decay_in_collaborative_filtering_a_popularity_perspec/">On the Role of Weight Decay in Collaborative Filtering: A Popularity Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示权重衰减本质是将流行度编码进嵌入模长，提出 PRISM 初始化策略替代权重衰减，NDCG 最高提升 4.77% 且训练加速 38.48%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;I2I Retrieval&quot;,&quot;Multi-Task Learning&quot;,&quot;Knowledge Distillation&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/optimizing_recall_or_relevance_a_multi_task_multi_head_approach_for_item_to/">Optimizing Recall or Relevance? A Multi-Task Multi-Head Approach for Item-to-Item Retrieval in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多任务蒸馏损失+多头I2I架构联合优化共互动召回与语义相关性，离线召回@500提升14.4%、L2主题相关性提升56.5%，并完成线上A/B验证。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;向量量化&quot;,&quot;实时索引&quot;,&quot;索引均衡&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/real_time_indexing_for_large_scale_recommendation_by_streaming_vector_quant/">Real-time Indexing for Large-scale Recommendation by Streaming Vector Quantization Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用流式 VQ 码本替代 HNSW/DR 静态索引，实现实时挂载、索引均衡与可修复，抖音全量上线显著提升用户参与。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;可微聚类&quot;,&quot;条件召回&quot;]"><strong><a href="../../venues/KDD/2025/collaborative_filtering/synergizing_implicit_and_explicit_user_interests_a_multi_embedding_retrieva/">Synergizing Implicit and Explicit User Interests: A Multi-Embedding Retrieval Framework at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pinterest 工业级多嵌入召回框架，用可微聚类建模隐式兴趣、条件召回建模显式兴趣，线上 HF Repins +1.09%、A-Pincepts +0.81%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;语义因子路由&quot;,&quot;对齐与均匀性&quot;]"><strong><a href="../../venues/KDD/2026/collaborative_filtering/beyond_instance_level_alignment_and_uniformity_semantic_factor_learning_for/">Beyond Instance-Level Alignment and Uniformity: Semantic Factor Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用批次内语义因子路由+匹配构造语义潜在正对，替代GCN邻域聚合，在四个稀疏数据集全库排序下NDCG@20最高提升6.24%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲图神经网络&quot;,&quot;层次聚类与探索利用&quot;,&quot;语义-协同对齐&quot;]"><strong><a href="../../venues/KDD/2026/collaborative_filtering/breaking_information_cocoons_a_hyperbolic_framework_for_balancing_explorati/">Breaking Information Cocoons: A Hyperbolic Framework for Balancing Exploration and Exploitation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>洛伦兹双曲空间对齐 LLM 语义与协同信号，并用 Dasgupta 代价驱动的无超参层次聚类实现用户可调探索-利用权衡，效用最高 +5.49%、多样性 +11.39%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;端到端推荐&quot;,&quot;协同记忆库&quot;,&quot;潜在上下文压缩&quot;,&quot;目标感知检索&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/collaborative_memory_augmentation_for_generative_recommendation/">Collaborative Memory Augmentation for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>为生成式推荐引入跨用户协同记忆库，用可学习查询token压缩序列、目标感知检索与门控交叉注意力融合，Full-Ranking下NDCG@5最高提升23.7%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;频率信号缩放&quot;,&quot;空间翻转&quot;]"><strong><a href="../../venues/KDD/2026/graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;创作者竞争博弈&quot;,&quot;偏差-方差权衡&quot;,&quot;正则化策略&quot;]"><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/lower_bias_higher_welfare_how_creator_competition_reshapes_bias_variance_tr/">Lower Bias, Higher Welfare: How Creator Competition Reshapes Bias-Variance Tradeoff in Recommendation Platforms?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>博弈论建模创作者竞争如何重塑平台正则化选择，理论与实验一致表明策略环境下应弱正则（低偏差）以最大化用户福利。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;流行度去偏&quot;,&quot;嵌入几何校正&quot;]"><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/rethinking_popularity_bias_in_collaborative_filtering_via_analytical_vector/">Rethinking Popularity Bias in Collaborative Filtering via Analytical Vector Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 BPR 优化在嵌入空间刻出“流行度方向”的几何偏差，提出非对称方向分解校正 DDC，在 5 个骨干上 MRR@10 提升 6%–29% 且 AvgPop@10 降 21%–47%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔稠密检索&quot;,&quot;Matryoshka 表征学习&quot;,&quot;多语言多垂类&quot;]"><strong><a href="../../venues/KDD/2026/collaborative_filtering/scaling_multilingual_semantic_search_in_uber_eats_delivery/">Scaling Multilingual Semantic Search in Uber Eats Delivery</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于 gte-Qwen2 双塔统一多语言多垂类语义召回，InfoNCE+Triplet NCE 两阶段训练与 MRL 多维度服务，Store R@20 相对 Legacy 提升 88%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;互惠推荐 RRS&quot;,&quot;可转移效用匹配 (TU Matching)&quot;,&quot;IPFP 均衡求解&quot;]"><strong><a href="../../venues/RecSys/2022/collaborative_filtering/matching_theory_based_recommender_systems_in_online_dating/">Matching Theory-based Recommender Systems in Online Dating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 Choo-Siow 可转移效用匹配替代传统融合函数，把互惠偏好与用户承接容量统一到闭式均衡解，并在 tapple 千万级用户平台落地。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;iALS&quot;,&quot;频率缩放正则化&quot;,&quot;超参调优&quot;]"><strong><a href="../../venues/RecSys/2022/collaborative_filtering/revisiting_the_performance_of_ials_on_item_recommendation_benchmarks/">Revisiting the Performance of iALS on Item Recommendation Benchmarks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>复核 iALS 在四个经典基准上的性能，用频率缩放正则、参考尺度 λ* 与超大嵌入维度这套调参配方，把被广泛认为落后的 iALS 拉回到 SOTA 水平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;兴趣胶囊动态路由&quot;,&quot;辅助任务联合学习&quot;,&quot;物品表征增广&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/a_model_agnostic_framework_for_recommendation_via_interest_aware_item_embed/">A Model-Agnostic Framework for Recommendation via Interest-aware Item Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多兴趣胶囊网络退化为辅助任务并共享物品嵌入，用兴趣梯度增广 DIN/DIEN 精排，Amazon 双集 AUC 稳定提升</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;对抗训练&quot;,&quot;Sharpness-aware Minimization&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/adversarial_collaborative_filtering_for_free/">Adversarial Collaborative Filtering for Free</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SharpCF 以轨迹损失隐式替代对抗训练，免二次反传；全库排序下均超 BPR 18.1%、APR 6.82%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;解耦表征&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/augmented_negative_sampling_for_collaborative_filtering/">Augmented Negative Sampling for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>解耦难/易因子后受控扰动易因子合成负样本，并以增强增益采样，5 数据集全库排序显著超越 DNS/MixGCF/DENS。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;单纯曝光效应&quot;,&quot;重复消费建模&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/ex2vec_characterizing_users_and_items_from_the_mere_exposure_effect/">Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ACT-R 基级激活调制用户-物品相对距离，再经倒 U 型兴趣函数打分；Deezer 新歌重复收听预测 Balanced Acc 64.27%，较最优基线相对提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;TU Matching&quot;,&quot;互惠推荐&quot;]"><strong><a href="../../venues/RecSys/2023/debias_fairness_industrial/fast_and_examination_agnostic_reciprocal_recommendation_in_matching_markets/">Fast and Examination-agnostic Reciprocal Recommendation in Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Choo–Siow 可转移效用匹配做互惠推荐重排，摆脱检查函数依赖并用 (2d+2) 维内积实现 ANN 检索，匹配总量超越 Naive/Reciprocal 且与 SW 持平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;低比特量化&quot;,&quot;梯度直通估计器&quot;,&quot;Hessian 迹估计&quot;]"><strong><a href="../../venues/RecSys/2023/graph_rec/hessian_aware_quantized_node_embeddings_for_recommendation/">Hessian-aware Quantized Node Embeddings for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用二阶 Hessian 迹修正 STE 的梯度失配，实现任意比特 GNN 节点量化嵌入，1-bit 下 Recall@50 较 HashGNN 平均提升约 16%，推理加速 3.6×。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;混合曲率流形&quot;,&quot;几何感知 Margin&quot;]"><strong><a href="../../venues/RecSys/2023/graph_rec/knowledge_based_multiple_adaptive_spaces_fusion_for_recommendation/">Knowledge-based Multiple Adaptive Spaces Fusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习曲率的统一空间兼容双曲/欧氏/球面并注意力融合多子空间，配合几何感知 margin，在三个 KG 数据集采样排序下 HR@20 最高提升 1.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;截断SVD&quot;,&quot;谱聚类与社区检测&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/of_spiky_svds_and_music_recommendation/">Of Spiky SVDs and Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示截断 SVD 音乐嵌入空间中普遍存在的“尖峰”几何，证明其等价于变度社区（DCBM），并给出余弦/点积选择与嵌入稳定性评估的工业实践。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;平均池化嵌入&quot;,&quot;序统计量闭式分析&quot;,&quot;表征自洽性&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/on_the_consistency_of_average_embeddings_for_item_recommendation/">On the Consistency of Average Embeddings for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 Consistency_k 期望精度指标与正态近似闭式解，证明理想 i.i.d. 嵌入下平均向量自洽，而 Deezer 真实嵌入的 Consistency 不足 14%/6%/2%，揭示工业平均池化表征的系统性失配。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排（评分预测）&quot;,&quot;差分隐私&quot;,&quot;集合矩阵分解&quot;,&quot;公共物品特征&quot;]"><strong><a href="../../venues/RecSys/2023/debias_fairness_industrial/private_matrix_factorization_with_public_item_features/">Private Matrix Factorization with Public Item Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出DP-CMF，将公共物品元数据/相似度/公开评分以集合矩阵分解融入用户级DP矩阵分解，在ML10M评分预测上显著缩小隐私-效用差距。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双边序列匹配&quot;,&quot;自蒸馏&quot;,&quot;时间敏感共注意力&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/reciprocal_sequential_recommendation/">Reciprocal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将双向推荐重构为双视角行为序列匹配任务，用时间敏感共注意力做微级匹配并以自蒸馏压回点积宏级匹配，采样排序下 HR@5 最高相对提升 83%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣学习&quot;,&quot;兴趣感知难负采样&quot;,&quot;路由正则化&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/rethinking_multi_interest_learning_for_candidate_matching_in_recommender_sy/">Rethinking Multi-Interest Learning for Candidate Matching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示多兴趣召回中均匀采样负例失效与路由坍缩两大训练缺陷，用兴趣感知难负采样与路由方差正则修复，Recall@20 相对提升最高 19.7%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Bayesian Surprise&quot;,&quot;Serendipity&quot;,&quot;LDA Topic Model&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/topic_level_bayesian_surprise_and_serendipity_for_recommender_systems/">Topic-Level Bayesian Surprise and Serendipity for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 LDA 主题分布上的贝叶斯惊喜（KL 后验-先验散度）结合评分构成事后惊喜度，再用协作过滤找相似用户做惊喜推荐；Goodreads 4 用户 449 本书人工标注，AROW 惊喜检测 F1 56.5%、AROW+vbBLR 推荐 F1 37.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样交叉熵&quot;,&quot;Full-Ranking 评测&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/turning_dross_into_gold_loss_is_bert4rec_really_better_than_sasrec/">Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 SASRec 与 BERT4Rec 的差距实为损失函数差异：统一为全库 softmax 交叉熵后 SASRec 更快更强，ML-1M 的 NDCG@10 提升 20.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;投毒攻击&quot;,&quot;代理模型重训练加速&quot;]"><strong><a href="../../venues/RecSys/2024/debias_fairness_industrial/accelerating_the_surrogate_retraining_for_poisoning_attacks_against_recomme/">Accelerating the Surrogate Retraining for Poisoning Attacks against Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 Gradient Passing，在反传中沿交互图传递梯度以逼近级联效应，一次更新≈多轮重训练，攻击 Recall@50 提升 18%–177% 且耗时降 27%–43%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;条件检索 Conditional Retrieval&quot;,&quot;双塔 + In-batch 负采样&quot;,&quot;特征交叉&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/bootstrapping_conditional_retrieval_for_user_to_item_recommendations/">Bootstrapping Conditional Retrieval for User-to-Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品侧 topic 元数据为普通 user-item 互动样本合成条件，在用户塔内做 user×condition 早期特征交叉，无需条件专属训练数据即可支持条件检索；Pinterest 全量上线带来 +0.26% WAU，服务成本仅为过滤式双塔的 1/4。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;预训练音频表征&quot;,&quot;迁移学习&quot;,&quot;内容-协同混合推荐&quot;]"><strong><a href="../../venues/RecSys/2024/multimodal_rec/comparative_analysis_of_pretrained_audio_representations_in_music_recommend/">Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统评测 6 种预训练音频后端（MusiCNN/MERT/EncodecMAE 等）在 KNN、浅层网络与 BERT4Rec 三种推荐器上的冻结嵌入效果，揭示 MIR 榜单与音乐推荐性能严重错位。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;ADMM 全局变量一致性&quot;,&quot;隐式反馈 ALS&quot;,&quot;用户因子共享&quot;]"><strong><a href="../../venues/RecSys/2024/cross_domain_multi_scenario/cross_domain_latent_factors_sharing_via_implicit_matrix_factorization/">Cross-Domain Latent Factors Sharing via Implicit Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 ADMM 全局变量一致性把隐式 ALS 矩阵分解扩展为跨域版本，仅交换扰动用户因子即可共享隐因子，热启动 NDCG@10 最高相对提升 55.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/全库打分&quot;,&quot;乘积量化&quot;,&quot;子ID推理&quot;,&quot;PQTopK&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/efficient_inference_of_sub_item_id_based_sequential_recommendation_models_w/">Efficient Inference of Sub-Item Id-based Sequential Recommendation Models with Millions of Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用PQTopK重排子ID打分循环，将RecJPQ增强SASRec在百万物品Gowalla的CPU推理加速4.5×，并验证千万级目录可行。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;长尾查询&quot;,&quot;LLM 数据增强&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/embedding_based_retrieval_for_long_tail_search_queries_in_ecommerce/">Embedding based retrieval for long tail search queries in ecommerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Best Buy 用 LLM 合成查询、领域 RoBERTa 预训练、q2q→q2p 微调与权重融合优化双塔长尾召回，线上转化 +3%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双塔离线预计算&quot;,&quot;Poly-Attention 多向量表征&quot;,&quot;LLM 摘要蒸馏监督&quot;]"><strong><a href="../../venues/RecSys/2024/llm_rec/embsum_leveraging_the_summarization_capabilities_of_large_language_models_f/">EmbSum: Leveraging the Summarization Capabilities of Large Language Models for Content-Based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 T5 编码器-解码器 + Poly-Attention 生成用户/内容多向量表征，并以 LLM 摘要作离线监督，双塔离线预计算下 MIND AUC 71.95。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段（召回表征鲁棒训练）&quot;,&quot;对抗训练&quot;,&quot;投毒攻击&quot;,&quot;用户脆弱性&quot;]"><strong><a href="../../venues/RecSys/2024/debias_fairness_industrial/improving_the_shortest_plank_vulnerability_aware_adversarial_training_for_r/">Improving the Shortest Plank: Vulnerability-Aware Adversarial Training for Robust Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户拟合度估计脆弱性并逐用户自适应对抗扰动幅度，全库排序下攻击成功率平均降 21.53%，推荐指标不降反升</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元数据对齐正则&quot;,&quot;闭式 EASE 变体&quot;,&quot;语义特征嵌入&quot;]"><strong><a href="../../venues/RecSys/2024/cold_start_rec/marec_metadata_alignment_for_cold_start_recommendation/">MARec: Metadata Alignment for cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用一条 item-item 相似度对齐正则把元数据嵌入注入 EASE/VAE 骨干，冷启 hr@10 提升 8.4%–53.8%，训练快两个数量级且热启不掉点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;BPR&quot;,&quot;可复现性研究&quot;,&quot;负采样&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/revisiting_bpr_a_replicability_study_of_a_common_recommender_system_baselin/">Revisiting BPR: A Replicability Study of a Common Recommender System Baseline</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统复现 BPR 并解耦四大实现特征，证明调优后的 BPR 在全局时间切分下可追平甚至超越 EASE/Mult-VAE。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可扩展交叉熵&quot;,&quot;MIPS 难负样本&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/scalable_cross_entropy_loss_for_sequential_recommendations_with_large_item_/">Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以随机桶中心做 top-k 分桶，把全库 softmax 近似为桶内小 logits 矩阵，并用 Mix 投影缓解桶塌缩；全库排序下显存最多降 100 倍而指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;隐式反馈负采样&quot;,&quot;中性类集合级排序&quot;,&quot;半监督伪标签&quot;]"><strong><a href="../../venues/RecSys/2024/collaborative_filtering/unlocking_the_hidden_treasures_enhancing_recommendations_with_unlabeled_dat/">Unlocking the Hidden Treasures: Enhancing Recommendations with Unlabeled Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出正-中性-负 (PNN) 三段式集合级排序范式，配合用户感知注意力半监督分类与钳制机制质心排序损失，在 4 个数据集上使 MF/GNN/对比学习骨干一致提升 3%–30%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/偏好推断&quot;,&quot;离散选择模型&quot;,&quot;核密度估计&quot;,&quot;曝光偏差&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/a_non_parametric_choice_model_that_learns_how_users_choose_between_recommen/">A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 sigmoid 核密度估计学习误差分布，无需假设 MNL/ENL 即可同时推断用户偏好与选择机制，抗竞争型曝光偏差。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排&quot;,&quot;因果推断&quot;,&quot;上下文老虎机&quot;,&quot;跨渠道营销&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;子 ID 分解&quot;,&quot;个性化流行度&quot;,&quot;准确率-新颖度权衡&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;采样 Softmax&quot;,&quot;logQ 修正&quot;,&quot;批内负采样&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/correcting_the_logq_correction_revisiting_sampled_softmax_for_large_scale_r/">Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>指出标准 logQ 修正误把正样本当作采样项，提出带置信度权重的改进修正，在公开与工业数据上稳定超越标准 logQ。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;极值理论&quot;,&quot;量子特征选择&quot;,&quot;QUBO/QAOA&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/estimating_quantum_execution_requirements_for_feature_selection_in_recommen/">Estimating Quantum Execution Requirements for Feature Selection in Recommender Systems Using Extreme Value Theory</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用极值理论对 QAOA 求解 QUBO 特征选择的输出建模，估计达到经典 SA 水平所需的量子执行次数，并在模拟器与真实 QPU 上验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;信息瓶颈&quot;,&quot;多行为推荐&quot;,&quot;图去噪&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉语言模型&quot;,&quot;SigLIP 微调&quot;,&quot;ANN 向量检索&quot;]"><strong><a href="../../venues/RecSys/2025/multimodal_rec/improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;Learning-to-Rank&quot;,&quot;Two-Tower&quot;,&quot;GBDT vs DNN&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;实体引导兴趣建模&quot;,&quot;对比预训练&quot;,&quot;双塔用户编码器&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/ip2_entity_guided_interest_probing_for_personalized_news_recommendation/">IP2: Entity-Guided Interest Probing for Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以实体-标题对比预训练探测新闻内实体焦点，并用跨注意力双塔用户编码器建模新闻间实体引导兴趣，MIND-large AUC 达 72.06%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;特征工程/在线推理&quot;,&quot;Spark-Keras 等价翻译&quot;,&quot;训练-服务一致性&quot;,&quot;特征预处理管线&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/kamae_bridging_spark_and_keras_for_seamless_ml_preprocessing/">Kamae: Bridging Spark and Keras for Seamless ML Preprocessing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 PySpark 预处理管线一对一翻译为等价 Keras 层，消除离线/在线特征工程重复实现，Expedia 线上延迟降 61%、成本降 58%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;三元组损失&quot;,&quot;流行度去偏&quot;]"><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/leveraging_geometric_insights_in_hyperbolic_triplet_loss_for_improved_recom/">Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>洛伦兹双曲三元组损失 TriplH，以物品-物品项自适应间隔替代固定 margin，兼顾精度、去流行度偏差与 3 倍推理加速。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多分辨率地理嵌入&quot;,&quot;H3 六边形网格&quot;,&quot;双塔对比学习&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/location_matters_leveraging_multi_resolution_geo_embeddings_for_housing_sea/">Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 H3 多分辨率六边形网格与城市嵌入注入双塔对比学习，缓解房源地理稀疏，离线重排模拟租金转化提升 85%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;离线评估&quot;,&quot;离策略估计&quot;,&quot;元分析&quot;,&quot;BLUE&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/meta_off_policy_estimation/">Meta Off-Policy Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多个 OPE 估计器视为相关研究做固定效应元分析，用 BLUE 闭式解融合并给出保守置信区间，OBD 上方差降超 50%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;离线评测&quot;,&quot;负采样策略&quot;,&quot;曝光偏差&quot;,&quot;元评测&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/on_the_reliability_of_sampling_strategies_in_offline_recommender_evaluation/">On the Reliability of Sampling Strategies in Offline Recommender Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以全观测 KuaiRec 为真值，四维度（分辨率/保真度/鲁棒性/预测力）系统解耦曝光偏差与采样偏差，给出采样策略选择指南。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正交Procrustes&quot;,&quot;低秩SVD&quot;,&quot;嵌入空间稳定化&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/orthogonal_low_rank_embedding_stabilization/">Orthogonal Low Rank Embedding Stabilization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用低秩SVD+正交Procrustes把重训后的用户/物品嵌入对齐到固定参考空间，保内积无损，跨期余弦相似度从0.01升至0.82</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;异步离线预排序&quot;,&quot;级联漏斗解耦&quot;,&quot;候选池扩充&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双效用行为建模&quot;,&quot;局部贪心最优策略&quot;,&quot;站外选项估计&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/recommendation_and_temptation/">Recommendation and Temptation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户选择解耦为 enrichment 与 temptation 双效用并引入站外选项，证明局部贪心最优并给出可估计算子，MovieLens 与仿真全面超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Two-Tower&quot;,&quot;内容特征召回&quot;,&quot;ANN 分层索引&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/suggest_complement_inspire_story_of_two_tower_recommendations_at_allegro_co/">Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以内容特征双塔为统一底座，仅改查询塔与索引逻辑即复用至相似/互补/灵感三类召回，两年线上 A/B 验证 CTR 与 GMV 双增。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;稀疏自编码器&quot;,&quot;嵌入压缩&quot;,&quot;向量检索&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/the_future_is_sparse_embedding_compression_for_scalable_retrieval_in_recomm/">The Future is Sparse: Embedding Compression for Scalable Retrieval in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用余弦重建的稀疏自编码器把稠密向量压成高维 k-稀疏向量，免重训骨干，12× 压缩仅损 1.35% CTR，线上显著优于同尺寸 Matryoshka。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;评测与可复现性审计&quot;,&quot;超参数优化&quot;,&quot;框架默认值&quot;,&quot;早停策略&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/the_hidden_cost_of_defaults_in_recommender_system_evaluation/">The Hidden Cost of Defaults in Recommender System Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>审计 RecBole 未文档化的 early_stop=10 默认策略如何静默截断随机搜索与贝叶斯优化，量化搜索路径不稳定性与可复现性风险。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长序列建模&quot;,&quot;多兴趣表征&quot;,&quot;级联类别检索&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/user_long_term_multi_interest_retrieval_model_for_recommendation/">User Long-Term Multi-Interest Retrieval Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别感知分层双兴趣学习与指针生成式类别→物品级联检索，将召回阶段行为序列扩展至千级，淘宝离线HR@2000达16.55%，线上点击+5.54%、订单+11.01%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Embedding-Native Retrieval&quot;,&quot;Shared LLM Backbone&quot;,&quot;Contrastive Joint Training&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/egr_embedding_native_generative_retrieval_with_a_shared_llm/">EGR: Embedding-Native Generative Retrieval with a Shared LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一共享 LLM 同时编码物品与用户历史，以双对比损失联合训练，免量化免 grounding，线上 CVR +2.91%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;子空间划分&quot;,&quot;多目标检索&quot;]"><strong><a href="../../venues/RecSys/2026/multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;用户表征学习&quot;,&quot;多专家舰队&quot;,&quot;免日志评估&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/mosaic_a_fleet_of_user_embedding_specialists_for_recommendation_at_meta/">Mosaic: A Fleet of User Embedding Specialists for Recommendation at Meta</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以四类异构专家模型族构建用户表征舰队，配合 MRM 与余弦冗余损失最大化增量信息，并提出免日志 CoEval 评估，离线 NE 稳定下降、线上 A/B 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入表哈希&quot;,&quot;零碰撞线性探测&quot;,&quot;嵌入新鲜度&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/multi_probe_zero_collision_hash_mpzch_mitigating_embedding_collisions_and_e/">Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性探测+身份/元数据辅助张量与主动淘汰，在工业级嵌入表上实现零碰撞并提升新内容新鲜度，训练/推理延迟几乎不变。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔语义检索&quot;,&quot;混合召回分数校准&quot;,&quot;多语言检索&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/multilingual_semantic_retrieval_for_apple_music_search/">Multilingual Semantic Retrieval for Apple Music Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>305M 多语双塔 + 课程式多目标微调，配合分位数校准混合召回，离线 Hit@10 相对提升 69%，线上尾部查询 CR 提升 7.93% 且无结果率降 86%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;权重共享&quot;,&quot;参数高效&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/parameter_efficient_single_collaborative_branch_for_recommendation/">Parameter-Efficient Single Collaborative Branch for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单支路权重共享替代用户/物品双塔，参数量减半，精度持平且覆盖率与流行度偏差显著改善。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;混合嵌入&quot;,&quot;增量搜索个性化&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;多智能体架构搜索&quot;,&quot;奖励黑客&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/recevolve_a_knowledge_driven_autonomous_agent_system_for_recommender_system/">RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔召回模型的架构搜索全生命周期交给多智能体闭环，41 次生产级训练换来 NDCG@50 相对 +19.9%、线上满意度 +3.77%，并意外暴露奖励黑客漏洞。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;混合检索&quot;,&quot;自适应路由&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/smart_llm_augmented_hybrid_retrieval_for_dynamic_product_ads/">SMART: LLM-Augmented Hybrid Retrieval for Dynamic Product Ads</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示规则查询擅长 BM25 重定向、LLM 查询擅长 ANN 拓客的检索分解，用输出审计门控仅路由 10% 用户，线上转化率 +27.6%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;查询改写&quot;,&quot;路径式端到端检索&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/spear_selection_aware_personalized_end_to_end_adaptive_rewriting_and_retrie/">SPEAR: Selection-aware Personalized End-to-end Adaptive Rewriting and Retrieval for Community Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以梯度隔离双嵌入+乘性门控+动态改写选择器统一查询改写与检索，离线点击召回@10提升99.5%，线上QVCTR+0.259%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;训练种子方差&quot;,&quot;模型选择稳定性&quot;,&quot;可复现性&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/training_seeds_and_model_selection_stability_in_recommender_system_evaluati/">Training seeds and model-selection stability in recommender-system evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>固定数据划分与超参、仅变训练种子，从用户级指标敏感性、验证选模稳定性、验证-测试一致性与推荐列表一致性四层量化种子噪声，主张种子应纳入评估协议。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回+粗排&quot;,&quot;生成式检索&quot;,&quot;全栈统一&quot;,&quot;KV缓存复用&quot;]"><strong><a href="../../venues/RecSys/2026/collaborative_filtering/unipinrec_unifying_generative_retrieval_and_ranking_at_pinterest_scale/">UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MAM 掩码动作建模 + 混合样本 + 跨阶段 KV 缓存复用，将生成式召回与排序全栈统一，线上延迟降 11.1%、QPS 升 63.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/训练目标&quot;,&quot;损失函数自动生成&quot;,&quot;强化学习搜索&quot;,&quot;奖励过滤&quot;]"><strong><a href="../../venues/SIGIR/2022/collaborative_filtering/autolossgen_automatic_loss_function_generation_for_recommender_systems/">AutoLossGen: Automatic Loss Function Generation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 RNN 控制器在基础算子空间上以 RL 从零搜索损失函数，配合代理测试与奖励过滤克服稀疏奖励，生成 MaxR/SumR 在 AUC 上稳定超过 MSE/BCE/Hinge/Focal。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;端到端训练范式&quot;,&quot;过参数化冗余利用&quot;,&quot;同伴协作权重移植&quot;,&quot;层重要性熵度量&quot;]"><strong><a href="../../venues/SIGIR/2022/collaborative_filtering/enhancing_top_n_item_recommendations_by_peer_collaboration/">Enhancing Top-N Item Recommendations by Peer Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 PCRec：两个同构推荐模型按层熵/范数度量冗余并互相移植权重，在推理零额外开销下稳定提升 Top-N 精度</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;知识图谱聚合&quot;,&quot;门控融合&quot;]"><strong><a href="../../venues/SIGIR/2022/graph_rec/hakg_hierarchy_aware_knowledge_gated_network_for_recommendation/">HAKG: Hierarchy-Aware Knowledge Gated Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户-物品图与知识图谱同时嵌入双曲空间，用双曲关系传递聚合、层级锥角约束与双物品嵌入门控融合，Full-Ranking 下 nDCG@20 最高提升 15.43%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;归纳式嵌入&quot;,&quot;模板基向量&quot;,&quot;免重训练&quot;]"><strong><a href="../../venues/SIGIR/2022/collaborative_filtering/inmo_a_model_agnostic_and_scalable_module_for_inductive_collaborative_filte/">INMO: A Model-Agnostic and Scalable Module for Inductive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用模板用户/物品作基向量、由交互集合线性生成用户/物品嵌入，为 MF/LightGCN 注入归纳能力并大幅压缩参数量；直推与两种归纳场景全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱图滤波&quot;,&quot;图去噪编码器&quot;]"><strong><a href="../../venues/SIGIR/2022/graph_rec/less_is_more_reweighting_important_spectral_graph_features_for_recommendati/">Less is More: Reweighting Important Spectral Graph Features for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>谱域分析发现仅少数平滑/粗糙图特征有效，用指数核带通滤波的 GDE 替代邻域聚合，5 数据集全库排序最优且比 LightGCN 快 12 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤&quot;,&quot;召回（双塔打分）&quot;,&quot;动量对比学习&quot;,&quot;标注偏置去噪&quot;]"><strong><a href="../../venues/SIGIR/2022/collaborative_filtering/mp2_a_momentum_contrast_approach_for_recommendation_with_pointwise_and_pair/">MP2: A Momentum Contrast Approach for Recommendation with Pointwise and Pairwise Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用三塔结构+动量物品塔平滑表示、以双塔差异度重加权逐点标签，联合逐点/成对损失缓解隐式反馈标注偏置，NDCG@5 相对最优基线提升 1.3–3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/端到端排序&quot;,&quot;知识图谱推荐&quot;,&quot;协同注意力&quot;,&quot;包围子图与门控融合&quot;]"><strong><a href="../../venues/SIGIR/2022/graph_rec/unify_local_and_global_information_for_top_n_recommendation/">Unify Local and Global Information for Top-$N$ Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双分支 KADM：协同注意力编码局部交互、关系感知 GNN 编码包围子图，门控融合后在 1:100 采样排序下 Recall@20 提升 3.1%~7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;对抗训练&quot;,&quot;梯度反转层&quot;,&quot;变分自编码器&quot;]"><strong><a href="../../venues/SIGIR/2022/debias_fairness_industrial/unlearning_protected_user_attributes_in_recommendations_with_adversarial_tr/">Unlearning Protected User Attributes in Recommendations with Adversarial Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>在 MultVAE 潜在向量上接梯度反转对抗头，剥离性别信息；攻击者探针 BAcc 从 0.71 降至 0.57，NDCG 仅微降且与末轮基线无显著差异</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Dense Retrieval&quot;,&quot;Personalization&quot;,&quot;Unified Information Access&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/a_personalized_dense_retrieval_framework_for_unified_information_access/">A Personalized Dense Retrieval Framework for Unified Information Access</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出统一信息访问双塔框架UIA，用功能编码+用户历史注意力个性化，在Lowe’s/ESCI全库排序显著超越任务专用与联合训练基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;替代商品推荐&quot;,&quot;弱监督学习&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/a_transformer_based_substitute_recommendation_model_incorporating_weakly_su/">A Transformer-Based Substitute Recommendation Model Incorporating Weakly Supervised Customer Behavior Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以商品标题对+XLMR交互匹配替代关系，PR对数回归与负采样去噪，11市场单模型A/B收入+19%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/全库排序&quot;,&quot;ODE 图滤波&quot;,&quot;Blurring-Sharpening&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/blurring_sharpening_process_models_for_collaborative_filtering/">Blurring-Sharpening Process Models for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出免训练 ODE 模糊-锐化范式直接处理交互矩阵，在三个基准全库排序 Recall/NDCG 超 43 个基线，NDCG 最高提升 3.74%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Embedding 尺寸搜索&quot;,&quot;连续动作空间强化学习&quot;,&quot;随机游走探索&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/continuous_input_embedding_size_search_for_recommender_systems/">Continuous Input Embedding Size Search For Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 TD3 在连续动作空间为每个用户/物品搜索任意粒度 Embedding 尺寸，并以随机游走探索提升收敛，80–95% 稀疏率下 R@20 全面超越离散 RL 搜索。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;点积模型嵌入维度&quot;,&quot;流行度偏差与物品公平性&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/curse_of_low_dimensionality_in_recommender_systems/">Curse of \&quot;Low\&quot; Dimensionality in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>论证点积模型低维嵌入是可表示排序数指数级塌缩的根源，实证其加剧流行度偏差、损害多样性与公平性，并给出上界 $n^{2d}$ 与锥体刻画。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回（端到端生成式全库打分）&quot;,&quot;扩散模型去噪&quot;,&quot;隐空间压缩扩散&quot;,&quot;时间感知重加权&quot;]"><strong><a href="../../venues/SIGIR/2023/generative_rec/diffusion_recommender_model/">Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次将扩散模型引入全物品交互生成式推荐，通过缩小噪声尺度并预测 x0 保留个性化信息，辅以物品聚类隐空间扩散与时间重加权，Full-Ranking 下 R@10 最高提升 10.67%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;意图解缠&quot;,&quot;图对比学习&quot;,&quot;自适应增广&quot;]"><strong><a href="../../venues/SIGIR/2023/collaborative_filtering/disentangled_contrastive_collaborative_filtering/">Disentangled Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 K 个全局意图原型解缠用户/物品表征，并用解缠相似度驱动参数化图掩码做自适应对比增广，全库排序下 Recall@20 相对最优基线提升 2.8%–6.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序/精排&quot;,&quot;流行度去偏&quot;,&quot;IPS 逆倾向得分估计&quot;]"><strong><a href="../../venues/SIGIR/2023/debias_fairness_industrial/popularity_debiasing_from_exposure_to_interaction_in_collaborative_filterin/">Popularity Debiasing from Exposure to Interaction in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 IPL 准则——物品交互量应正比于喜欢它的用户数，用 IPS 估计并构造 std 正则，去偏同时反涨推荐精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;特征过相关&quot;,&quot;过平滑&quot;,&quot;自适应正则&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/afdgcf_adaptive_feature_de_correlation_graph_collaborative_filtering_for_re/">AFDGCF: Adaptive Feature De-correlation Graph Collaborative Filtering for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 GNN-CF 中被忽视的“特征过相关”问题并证明其与过平滑成正比，提出层级自适应的列相关惩罚正则，模型无关地提升 4 类骨干模型最多 14.98%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;行为模式挖掘&quot;,&quot;贝叶斯似然比&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/behavior_pattern_mining_based_multi_behavior_recommendation/">Behavior Pattern Mining-based Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以行为模式路径计数为特征、贝叶斯似然比闭式打分替代GNN，在三个电商数据集Full-Ranking下Recall@10最高提升393.98%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;物品级偏好建模&quot;,&quot;图卷积预训练&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/behavior_contextualized_item_preference_modeling_for_multi_behavior_recomme/">Behavior-Contextualized Item Preference Modeling for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 BCIPM：用「行为上下文化物品偏好网络」只从目标行为聚合物品级偏好做推荐，辅助行为仅参与参数训练以规避噪声；四个数据集 Leave-One-Out + Full-Ranking 下 HR@10 相对提升 12.15%–26.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;参数高效嵌入&quot;,&quot;元嵌入组合&quot;,&quot;矩阵伪逆闭式更新&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/lightweight_embeddings_for_graph_collaborative_filtering/">Lightweight Embeddings for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以可学习稀疏分配矩阵替代固定哈希映射，在扩展图上联合优化元嵌入码本，实现 44× 参数压缩下接近全量嵌入的精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;线性化核注意力&quot;,&quot;结构位置编码&quot;,&quot;度数重加权掩码&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/masked_graph_transformer_for_large_scale_recommendation/">Masked Graph Transformer for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>纯 Transformer 全节点注意力 + SimRFs 线性核 + 可学习相对度数掩码，Full-Ranking 下 Recall@20 最高提升 8.62%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Meal Recommendation Dataset&quot;,&quot;Bundle Recommendation&quot;,&quot;Healthiness&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/mealrec_a_meal_recommendation_dataset_with_meal_course_affiliation_for_pers/">MealRec$^+$: A Meal Recommendation Dataset with Meal-Course Affiliation for Personalization and Healthiness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>SIGIR 2024｜协同过滤与双塔召回｜构建含餐食-菜品关联与FSA/WHO健康分的MealRec⁺数据集，全库排序下CrossCBR Recall@20提升72.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;嵌入演化监控&quot;,&quot;流行度偏置&quot;,&quot;实时训练&quot;]"><strong><a href="../../venues/SIGIR/2024/debias_fairness_industrial/monitoring_the_evolution_of_behavioural_embeddings_in_social_media_recommen/">Monitoring the Evolution of Behavioural Embeddings in Social Media Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在 1.8 亿用户短视频生产系统上对比批量与实时物品嵌入的演化，用成熟度、学习量与 ℓ2 范数指标量化实时更新收敛快 5 倍且显著缓解流行度偏置。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练&quot;,&quot;模型投毒攻击&quot;,&quot;用户级梯度裁剪&quot;]"><strong><a href="../../venues/SIGIR/2024/debias_fairness_industrial/poisoning_decentralized_collaborative_recommender_system_and_its_countermea/">Poisoning Decentralized Collaborative Recommender System and Its Countermeasures</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出面向去中心化协同推荐的 PAMN 自适应投毒与 UCSU 用户级裁剪防御，在 ML/Amazon 上大幅提升目标物品曝光且保持 HR。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/表示学习&quot;,&quot;符号图&quot;,&quot;图Transformer&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/sigformer_sign_aware_graph_transformer_for_recommendation/">SIGformer: Sign-aware Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用Transformer直接建模整张符号图，以符号谱/路径编码融合正负反馈，五数据集Recall@20最高提升15.61%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/用户表征&quot;,&quot;Neural Hawkes Process&quot;,&quot;用户开屏动机预测&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/to_search_or_to_recommend_predicting_open_app_motivation_with_neural_hawkes/">To Search or to Recommend: Predicting Open-App Motivation with Neural Hawkes Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用神经 Hawkes 过程+层级 Transformer 建模用户开屏动机，OAMD 上 F0.5 提升 2.9%、AUC 达 0.9326</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多项式图滤波&quot;,&quot;免训练协同过滤&quot;,&quot;低通滤波&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/turbo_cf_matrix_decomposition_free_graph_filtering_for_fast_recommendation/">Turbo-CF: Matrix Decomposition-Free Graph Filtering for Fast Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用多项式图滤波器替代特征分解实现免训练免分解的协同过滤，全库排序下精度持平SOTA而耗时降至亚秒级。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;早期交互&quot;,&quot;晚期交互&quot;]"><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;可复现性研究&quot;,&quot;意图感知推荐&quot;,&quot;基线调优&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/a_worrying_reproducibility_study_of_intent_aware_recommendation_models/">A Worrying Reproducibility Study of Intent-Aware Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现 5 个意图感知推荐模型，发现 2 个不可复现、全部被调优后的传统非神经基线超越，揭示方法学顽疾。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标 EBR&quot;,&quot;参数高效微调&quot;,&quot;级联选择性掩码&quot;]"><strong><a href="../../venues/SIGIR/2025/multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;切比雪夫插值&quot;]"><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Interest Unit&quot;,&quot;语义对齐检索&quot;,&quot;多兴趣建模&quot;]"><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/ira_adaptive_interest_aware_representation_and_alignment_for_personalized_m/">IRA: Adaptive Interest-aware Representation and Alignment for Personalized Multi-interest Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可累积/剪枝的文本化 Interest Unit 表征多兴趣，语义对齐替代点击信号做 ANN 召回，离线 H@20 提升 20%，线上点击 +5.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;公平采样&quot;,&quot;倾向分免估计&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/unbiased_collaborative_filtering_with_fair_sampling/">Unbiased Collaborative Filtering with Fair Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>通过对称补充采样使每个用户/物品等概率成为正负样本，无需估计倾向分即可消除流行度偏置，全库排序下 Recall/NDCG 与 ARP 双优。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;线性自编码器&quot;,&quot;数据自适应归一化&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/why_is_normalization_necessary_for_linear_recommenders/">Why is Normalization Necessary for Linear Recommenders?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>理论剖析线性自编码器归一化对流行度/邻域偏差的作用，提出数据自适应归一化 DAN，六数据集全库排序下长尾最高提升 128.57%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔模型&quot;,&quot;在线学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/cs3_efficient_online_capability_synergy_for_two_tower_recommendation/">CS3: Efficient Online Capability Synergy for Two-Tower Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以塔内自校正、跨塔EMA同步与级联模型共享三机制增强双塔召回，离线三数据集AUC一致提升，线上广告收入+8.36%且QPS仅降0.59%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双意图空间表征重构&quot;,&quot;对齐-均匀性优化&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/diaurec_dual_intent_space_representation_optimization_for_recommendation/">DIAURec: Dual-Intent Space Representation Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以原型意图与分布意图重构表征，用对齐-均匀性+粗细粒度匹配联合优化，Amazon-book R@20 相对提升 10.6%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;正交约束&quot;,&quot;嵌入坍缩&quot;,&quot;稀疏缩放&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/ocp_orthogonal_constrained_projection_for_sparse_scaling_in_industrial_comm/">OCP: Orthogonal Constrained Projection for Sparse Scaling in Industrial Commodity Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 Item-ID 投影层上施加 Stiefel 流形正交约束，抑制稀疏缩放下的嵌入坍缩，京东线上 UCXR +12.97%、GMV +8.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;LLM推荐&quot;,&quot;召回&quot;,&quot;分布重塑&quot;,&quot;用户画像&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/promax_exploring_the_potential_of_llm_derived_profiles_with_distribution_sh/">ProMax: Exploring the Potential of LLM-derived Profiles with Distribution Shaping for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 画像视为检索信号而非融合特征，用双路分布重塑（监督+自监督）引导基座模型，LightGCN 上 NDCG@10 最高提升 8.44%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;表征对齐&quot;,&quot;互补性融合&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/rethinking_semantic_collaborative_integration_why_alignment_is_not_enough/">Rethinking Semantic–Collaborative Integration: Why Alignment Is Not Enough</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>论证语义-协同全局对齐假设过强，提出共享+私有潜结构与互补性诊断，最小融合在 Amazon 三域 Full-Ranking 上超越对齐范式最高 41.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;GPU 向量检索&quot;,&quot;Bloom Index 特征过滤&quot;,&quot;模型化服务架构&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/silvertorch_a_unified_model_based_system_to_democratize_large_scale_recomme/">SilverTorch: A Unified Model-based System to Democratize Large-Scale Recommendation on GPUs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 ANN 检索与特征过滤下沉为 PyTorch 模型层，GPU Bloom Index 与 Int8 融合 ANN 协同设计，吞吐提升 23.7×、成本效率提升 13.35×。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;相关性蒸馏&quot;,&quot;难负样本挖掘&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/unified_supervision_for_walmart_s_sponsored_search_retrieval_via_joint_sema/">Unified Supervision for Walmart&#x27;s Sponsored Search Retrieval via Joint Semantic Relevance and Behavioral Engagement Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以语义相关性为主监督、互动为相关项内偏好信号，融合交叉编码器分级标签、多通道检索先验与去偏互动构建统一双塔训练目标，离线 P@25 +10.5%、NDCG@25 +5.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;免训练推荐&quot;,&quot;时间衰减打分&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/SIGIR/2026/collaborative_filtering/zorro_a_zero_weight_personalized_recommender_system_for_scalable_news_recom/">ZoRRO: A Zero-Weight Personalized Recommender System for Scalable News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>零权重免训练新闻推荐框架，以时间衰减×表征相似度打分，离线全库排序超越 NRMS，线上 CTR 逼近 SOTA 且推理快 600 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;逆倾向加权&quot;,&quot;线性自编码器&quot;,&quot;流行度去偏&quot;]"><strong><a href="../../venues/SIGIR-AP/2025/debias_fairness_industrial/accurate_and_diverse_recommendations_via_propensity_weighted_linear_autoenc/">Accurate and Diverse Recommendations via Propensity-Weighted Linear Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用对数 Sigmoid 重定义 IPS 倾向分，避免幂律对热门物品过度惩罚，在 EASE 类线性自编码器上以闭式解实现覆盖率大幅提升且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Learned Sparse Retrieval&quot;,&quot;注意力用户建模&quot;]"><strong><a href="../../venues/SIGIR-AP/2026/collaborative_filtering/lsnr_learned_sparse_news_recommendations_with_attentive_user_modeling/">LSNR: Learned Sparse News Recommendations with Attentive User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用独立编码+注意力聚合替代 STAR 的历史拼接，构建双通路稀疏用户向量，MIND NDCG@10 较 STAR 提升 11%，短历史场景反超稠密模型。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;离散哈希编码&quot;,&quot;去噪自编码器&quot;,&quot;成对排序损失&quot;]"><strong><a href="../../venues/TKDE/2022/cold_start_rec/deep_pairwise_hashing_for_cold_start_recommendation/">Deep Pairwise Hashing for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>DAE 内容预训练 + 成对排序离散哈希，将用户/物品映射为二值码用汉明距离检索；10% 稀疏冷启动下 Yelp MRR 较 DDL 提升 23%，在线打分加速 3.7–6.3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;TNNLS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（评分预测）&quot;,&quot;幅值约束&quot;,&quot;球坐标重参数化&quot;]"><strong><a href="../../venues/TKDE/2022/collaborative_filtering/magnitude_bounded_matrix_factorisation_for_recommender_systems/">Magnitude Bounded Matrix Factorisation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>通过球坐标重参数化约束用户/物品特征幅值，实现个体化预测边界与快速SGD，稀疏数据RMSE/F1多数领先。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;字典学习&quot;,&quot;动态用户偏好&quot;,&quot;GRU+Attention&quot;]"><strong><a href="../../venues/TKDE/2022/sequential_rec/modeling_dynamic_user_preference_via_dictionary_learning_for_sequential_rec/">Modeling Dynamic User Preference via Dictionary Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>S2PNM以共享字典把GRU序列编码译为动态偏好，与静态偏好同空间相加；Netflix时间切分RMSE较SVD++降2.18%，Baby Care NDCG@5较SASRec升6.10%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/粗排&quot;,&quot;神经推荐综述&quot;,&quot;图神经网络&quot;,&quot;辅助信息融合&quot;]"><strong><a href="../../venues/TKDE/2023/collaborative_filtering/a_survey_on_accuracy_oriented_neural_recommendation_from_collaborative_filt/">A Survey on Accuracy-oriented Neural Recommendation: From Collaborative Filtering to Information-rich Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以数据使用为轴重构神经推荐综述：统一框架下划分协同过滤、内容增强与时空序列三类，梳理表征/交互建模与辅助信息融合演进及开放问题</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对抗变分推断&quot;,&quot;多峰后验嵌入&quot;]"><strong><a href="../../venues/TKDE/2023/self_supervised_denoising/dual_adversarial_variational_embedding_for_robust_recommendation/">Dual Adversarial Variational Embedding for Robust Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双侧 VAE + 对抗式隐式后验正则实现个性化降噪与多峰偏好建模，Leave-One-Out + 1:99 采样排序下 NDCG@5 提升约 1.3%–8.3%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;正交度量映射&quot;,&quot;对偶学习&quot;]"><strong><a href="../../venues/TKDE/2023/cross_domain_multi_scenario/dual_metric_learning_for_effective_and_efficient_cross_domain_recommendatio/">Dual Metric Learning for Effective and Efficient Cross-Domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用正交隐空间映射+双向对偶迭代做跨域迁移，仅 8 个重叠用户即可超越用全量重叠用户的 SOTA，Book-Movie RMSE 降至 0.2184。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;哈希学习 (Learning to Hash)&quot;,&quot;汉明空间图卷积&quot;,&quot;高阶相似度传播&quot;]"><strong><a href="../../venues/TKDE/2023/graph_rec/hs_gcn_hamming_spatial_graph_convolutional_networks_for_recommendation/">HS-GCN: Hamming Spatial Graph Convolutional Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在汉明空间直接做图卷积传播，显式建模一阶与高阶汉明相似度，7 个数据集全库排序 HR@50 平均超最强哈希基线 26.2%、逼近实值 GNN 且训练/推理提速 2.4×/4.9×。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲表示学习&quot;,&quot;语义邻居聚合&quot;,&quot;陀螺向量空间&quot;]"><strong><a href="../../venues/TKDE/2023/collaborative_filtering/hyperbolic_neural_collaborative_recommender/">Hyperbolic Neural Collaborative Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用热核加权关系图构造用户/物品语义邻居，并在 Poincaré 球内以陀螺向量算子做深度聚合，CTR AUC 较最强基线最高 +4.15%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/评分预测&quot;,&quot;本地差分隐私&quot;,&quot;矩阵分解&quot;,&quot;高斯混合模型&quot;,&quot;有界Laplace&quot;]"><strong><a href="../../venues/TKDE/2023/debias_fairness_industrial/private_and_utility_enhanced_recommendations_with_local_differential_privac/">Private and Utility Enhanced Recommendations with Local Differential Privacy and Gaussian Mixture Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出BLP本地扰动与MoG-MF噪声建模，在LDP下用10折CV、RMSE/F-score验证精度提升与单轮通信优势。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;曝光置信度建模&quot;,&quot;随机游走信息量采样&quot;]"><strong><a href="../../venues/TKDE/2023/collaborative_filtering/samwalker_recommendation_with_informative_sampling_strategy/">SamWalker++: recommendation with informative sampling strategy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用伪社交网络图扩散核自适应分配隐式反馈置信度，并沿网络随机游走采样正比于置信度的信息量实例，稀疏数据 Pre@5 提升 67.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;变分信息瓶颈&quot;,&quot;用户特征去噪&quot;]"><strong><a href="../../venues/TKDE/2023/collaborative_filtering/variational_bandwidth_auto_encoder_for_hybrid_recommender_systems/">Variational Bandwidth Auto-encoder for Hybrid Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户相关的信道带宽动态调节辅助特征信息的流入量，同时缓解评分稀疏不确定性与特征噪声过拟合，三数据集全库排序 Recall@20 最高提升 7.2%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;马尔可夫图扩散&quot;,&quot;InfoBPR 排序损失&quot;,&quot;图稀疏化&quot;]"><strong><a href="../../venues/TKDE/2024/graph_rec/mgdcf_distance_learning_via_markov_graph_diffusion_for_neural_collaborative/">MGDCF: Distance Learning via Markov Graph Diffusion for Neural Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>证明 LightGCN/APPNP 等价于 1 层上下文编码 NRL，用马尔可夫图扩散权衡两类距离，并配 InfoBPR 多负样本损失，Amazon-Book NDCG@20 较 LightGCN 提升 47%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;端侧排序&quot;,&quot;弹性嵌入&quot;,&quot;端侧推荐&quot;]"><strong><a href="../../venues/TKDE/2024/collaborative_filtering/personalized_elastic_embedding_learning_for_on_device_recommendation/">Personalized Elastic Embedding Learning for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>一次训练生成按端侧预算裁剪的用户组个性化弹性物品嵌入，组内精调与控制器赋权，在 Amazon/Yelp 全库排序下优于 RULE。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同嵌入映射&quot;,&quot;两阶段调优&quot;]"><strong><a href="../../venues/TKDE/2025/llm_rec/collm_integrating_collaborative_embeddings_into_large_language_models_for_r/">CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将传统协同过滤模型学到的用户/物品表征经 MLP 映射进 LLM 词嵌入空间，以两阶段调优兼顾暖启动与冷启动，AUC 相对 TALLRec 提升约 8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;意图解耦&quot;,&quot;对比学习&quot;,&quot;高阶关系&quot;]"><strong><a href="../../venues/TKDE/2025/collaborative_filtering/intent_propagation_contrastive_collaborative_filtering/">Intent Propagation Contrastive Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双螺旋消息传播融合高阶关系与意图解耦，对比学习对齐结构/意图视图，Full-Ranking 下 NDCG 平均提升 9.36%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;门控专家网络&quot;,&quot;潜在因子解耦&quot;]"><strong><a href="../../venues/TKDE/2025/collaborative_filtering/latent_factor_modeling_with_expert_network_for_multi_behavior_recommendatio/">Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用门控专家网络显式建模共享潜在因子，配合对比学习保证因子独立与一致，在三个多行为数据集 Full-Ranking 下 HR@10 最高提升 3.95%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Krylov 子空间&quot;,&quot;多项式图滤波&quot;,&quot;免训练推荐&quot;]"><strong><a href="../../venues/TKDE/2026/collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;解耦表征&quot;,&quot;多模态注意力&quot;]"><strong><a href="../../venues/TMM/2023/multimodal_rec/disentangled_multimodal_representation_learning_for_recommendation/">Disentangled Multimodal Representation Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用距离相关正则解耦 ID/文本/视觉特征的 K 个因子，再以权重共享的多模态注意力建模用户对各因子模态偏好，Amazon 五域全库排序 Recall@20 最高提升 15.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回与排序&quot;,&quot;反射式缺失值填补&quot;,&quot;绑定权重自编码器&quot;,&quot;负采样去偏&quot;]"><strong><a href="../../venues/TOIS/2022/collaborative_filtering/collaborative_reflection_augmented_autoencoder_network_for_recommender_syst/">Collaborative Reflection-Augmented Autoencoder Network for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以反射门显式填补未交互评分并联合自编码器去噪，在四个基准改善RMSE与Top-K排序，且免负采样。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;特征级注意力&quot;,&quot;物品相似度建模&quot;]"><strong><a href="../../venues/TOIS/2022/collaborative_filtering/feature_level_attentive_icf_for_recommendation/">Feature-level Attentive ICF for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为 ICF 历史物品生成 d 维特征级注意力向量替代标量权重，跨物品同维 softmax 隐式融合物品级注意力，6 数据集 HR/NDCG@10 稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果推断去偏&quot;,&quot;混淆变量平衡&quot;,&quot;对抗学习&quot;]"><strong><a href="../../venues/TOIS/2023/debias_fairness_industrial/debiased_recommendation_with_user_feature_balancing/">Debiased Recommendation with User Feature Balancing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以用户特征投影+IPM分布平衡替代IPS实现因果去偏，配合裁剪/采样/对抗三种平衡项策略与潜混淆建模，在6个数据集上NDCG@10最高提升6.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;高斯偏好分布&quot;,&quot;协方差偏好关联&quot;,&quot;可解释属性画像&quot;]"><strong><a href="../../venues/TOIS/2023/collaborative_filtering/dual_preference_distribution_learning_for_item_recommendation/">Dual Preference Distribution Learning for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布对用户\&quot;物品级一般偏好+属性级细粒度偏好\&quot;双分布建模，协方差刻画偏好关联并顺带产出属性画像解释；Leave-One-Out + 1:100 采样排序下 NDCG@10 相对提升 1.7%–13.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;学习哈希&quot;,&quot;离散优化&quot;]"><strong><a href="../../venues/TOIS/2023/debias_fairness_industrial/lightfr_lightweight_federated_recommendation_with_privacy_preserving_matrix/">LightFR: Lightweight Federated Recommendation with Privacy-preserving Matrix Factorization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用学习哈希把联邦 MF 的用户/物品表征压成二值码，配联邦离散坐标下降与梯度聚合，精度追平联邦 SOTA，存储与推理开销降约一个量级。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;MIPS 最大内积检索&quot;,&quot;近邻图路由&quot;,&quot;强化学习与奖励塑形&quot;]"><strong><a href="../../venues/TOIS/2023/collaborative_filtering/reinforcement_routing_on_proximity_graph_for_efficient_recommendation/">Reinforcement Routing on Proximity Graph for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Reinforce+自评基线训练 GCN 智能体在近邻图上路由，并以势函数奖励塑形注入少量 ground truth；固定 IPC 预算下全库召回全面超越 ip-NSW/IPDG/Mobius。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;软节点聚类&quot;,&quot;Gumbel-Softmax&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/TOIS/2024/graph_rec/cluster_based_graph_collaborative_filtering/">Cluster-based Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Gumbel-Softmax 软节点聚类构建簇特定图，将高阶图卷积限制在同兴趣簇内以滤除噪声并缓解过平滑，Recall@20 相对 SOTA 提升 2.2%–4.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;采样Softmax损失&quot;,&quot;难负样本挖掘&quot;,&quot;图卷积网络&quot;]"><strong><a href="../../venues/TOIS/2024/collaborative_filtering/on_the_effectiveness_of_sampled_softmax_loss_for_item_recommendation/">On the Effectiveness of Sampled Softmax Loss for Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>理论揭示SSM损失缓解流行度偏差、挖掘难负样本并对齐DCG，图消息传递可补偿其模长学习缺陷；LightGCN+SSM全库排序Recall@20最高提升41.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果/信息论&quot;]"><strong><a href="../../venues/TOIS/2024/debias_fairness_industrial/robust_collaborative_filtering_to_popularity_distribution_shift/">Robust Collaborative Filtering to Popularity Distribution Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>PopGo 用同构 shortcut model 量化交互级流行度捷径并以掩码预测去偏，在 OOD/ID 全库排序上同时提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;嵌入表压缩&quot;,&quot;轻量化推荐基准&quot;,&quot;端侧部署&quot;]"><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/a_thorough_performance_benchmarking_on_lightweight_embedding_based_recommen/">A Thorough Performance Benchmarking on Lightweight Embedding-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一协议横评 7 类轻量嵌入压缩方法，提出 MagPrune 强基线，并首次量化跨任务迁移性与树莓派端侧效率瓶颈。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;数据增强&quot;,&quot;LLM4Rec&quot;,&quot;ID-based Recommendation&quot;,&quot;LoRA 微调&quot;]"><strong><a href="../../venues/TOIS/2025/llm_rec/enhancing_id_based_recommendation_with_large_language_models/">Enhancing ID-based Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LoRA 微调 LLM 仅凭纯 ID 序列生成伪交互数据，经双重过滤后注入任意 ID 基推荐模型，全库排序下平均提升 3%–9%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;组合嵌入量化&quot;,&quot;图重连稀疏化&quot;]"><strong><a href="../../venues/TOIS/2025/collaborative_filtering/lightweight_embeddings_with_graph_rewiring_for_collaborative_filtering/">Lightweight Embeddings with Graph Rewiring for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>量化组合码本+图重连剪枝，在严格存储预算下兼顾 GNN 推荐精度与边缘端推理效率，iFashion 存储降 97.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Graph Transformer&quot;,&quot;位置编码&quot;,&quot;长程协同信号&quot;]"><strong><a href="../../venues/TOIS/2025/graph_rec/position_aware_graph_transformer_for_recommendation/">Position-aware Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用四种为交互图定制的节点位置编码把 Transformer 全局建模能力注入 GCN 推荐骨干，在四个数据集上以 Full-Ranking 协议取得最高 27.69% 的 Recall@20 相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;后处理/模型修正&quot;,&quot;属性遗忘&quot;,&quot;分布距离MMD&quot;,&quot;函数空间正则&quot;]"><strong><a href="../../venues/TOIS/2025/debias_fairness_industrial/post_training_attribute_unlearning_in_recommender_systems/">Post-Training Attribute Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出锚点分布MMD与函数空间排序正则的PoT-AU双损失，多类属性遗忘复杂度降至O(T)且推荐性能近无损。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;流行度去偏&quot;,&quot;逆倾向加权&quot;,&quot;列表级学习&quot;]"><strong><a href="../../venues/TOIS/2026/debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;前门调整&quot;,&quot;未观测混杂&quot;]"><strong><a href="../../venues/TORS/2023/debias_fairness_industrial/deconfounded_causal_collaborative_filtering/">Deconfounded Causal Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以客观物品特征为前门中介，统一去除未观测混杂，并在 Amazon/Yelp 的 SKEW/RAND 采样排序上显著提升 nDCG@5。</p>
</dd>
<dt data-paper-tags="[&quot;TORS 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/排序&quot;,&quot;输出扰动&quot;,&quot;无负采样&quot;]"><strong><a href="../../venues/TORS/2023/self_supervised_denoising/selfcf_a_simple_framework_for_self_supervised_collaborative_filtering/">SelfCF: A Simple Framework for Self-supervised Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TORS 2023 自监督去噪：输出扰动替代负采样，四数据集全库排序下较 BUIR 平均提升 17.79%，训练快 2–4×。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;单位一致性&quot;,&quot;张量补全&quot;]"><strong><a href="../../venues/TORS/2023/collaborative_filtering/tensor_completion_with_provable_consistency_and_fairness_guarantees_for_rec/">Tensor Completion with Provable Consistency and Fairness Guarantees for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以单位一致性约束替代秩/范数优化，实现可证明唯一、共识排序与尺度公平的矩阵/张量补全；MovieLens-1M 扰动测试零违例。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入坍缩&quot;,&quot;谱均衡&quot;,&quot;有效秩&quot;]"><strong><a href="../../venues/TORS/2025/collaborative_filtering/balancing_embedding_spectrum_for_recommendation/">Balancing Embedding Spectrum for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从谱分析视角揭示成对学习中的嵌入坍缩，提出 DirectSpec 全通滤波直接均衡奇异值谱，MF/LightGCN 上 nDCG@10 最高提升 52.6%/42.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;对比学习&quot;,&quot;高阶关联&quot;]"><strong><a href="../../venues/TORS/2025/collaborative_filtering/enhancing_item_level_bundle_representation_for_bundle_recommendation/">Enhancing Item-level Bundle Representation for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 B-U-I 高阶关联与预训练伪交互增强捆绑包物品级表征，全库排序下平均提升 7.5%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;数据投毒攻击&quot;,&quot;矩阵分解&quot;,&quot;用户簇定向攻击&quot;]"><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/evaluating_impact_of_user_cluster_targeted_attacks_in_matrix_factorisation_/">Evaluating Impact of User-Cluster Targeted Attacks in Matrix Factorisation Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统分析 MF 推荐中面向用户簇的投毒攻击，揭示 V 矩阵是攻击传播主通道，簇内真实评分量决定脆弱性。</p>
</dd>
<dt data-paper-tags="[&quot;TWEB&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;元特征学习&quot;,&quot;可视化配置抽象&quot;,&quot;跨数据集迁移&quot;]"><strong><a href="../../venues/TWEB/2022/collaborative_filtering/personalized_visualization_recommendation/">Personalized Visualization Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on the Web">ACM TWEB</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用共享元特征空间+数据无关可视化配置抽象，构建四部图矩阵分解实现跨数据集个性化可视化推荐，HIT@5 相对提升 29.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2020&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;离散哈希码学习&quot;,&quot;直通估计器STE&quot;]"><strong><a href="../../venues/WSDM/2022/graph_rec/learning_to_hash_with_graph_neural_networks_for_recommender_systems/">Learning to Hash with Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>HashGNN 端到端联合学习 GNN 连续表征与二值哈希码，融合链接重建与 Hamming 排序损失，并以 Guidance STE 消除梯度放大；全库排序 HR@50 较 GraphSage 提升 4–10%，检索提速 2.5–3.6 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/评测基准&quot;,&quot;数据采样与 Coreset&quot;,&quot;Kendall&#x27;s Tau 排名保持性&quot;,&quot;Propensity 去偏&quot;]"><strong><a href="../../venues/WSDM/2022/collaborative_filtering/on_sampling_collaborative_filtering_datasets/">On Sampling Collaborative Filtering Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统刻画 16 种 CF 数据采样策略对算法排名保持性的影响，提出 SVP-CF 代理采样与 Data-genie 采样器推荐器，仅用 10% 数据即可保持排名，带来 5.8× 训练加速。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;层级树检索&quot;,&quot;用户兴趣探索&quot;]"><strong><a href="../../venues/WSDM/2022/rl_bandit_rec/show_me_the_whole_world_towards_entire_item_space_exploration_for_interacti/">Show Me the Whole World: Towards Entire Item Space Exploration for Interactive Personalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用共点击表征自底向上聚出物品层级树，把全库 Contextual Bandit 探索改写为根到叶序列决策 (HCB) 与渐进感受野扩展 (pHCB)，IPS 模拟器上累积奖励最高提升 280%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;解耦表征&quot;,&quot;推荐多样性&quot;]"><strong><a href="../../venues/WSDM/2023/collaborative_filtering/disentangled_representation_for_diversified_recommendations/">Disentangled Representation for Diversified Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DCRS 将用户偏好解耦为类别相关/无关分量，用对抗判别器约束，在 Full-Ranking 协议下兼顾准确率与类别多样性。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练(联邦)&quot;,&quot;联邦推荐遗忘&quot;,&quot;更新校准回滚&quot;,&quot;半难负采样&quot;]"><strong><a href="../../venues/WSDM/2023/debias_fairness_industrial/federated_unlearning_for_on_device_recommendation/">Federated Unlearning for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用日志回滚+方向校准实现联邦推荐用户遗忘，配合重要性更新筛选与混合半难负采样，7× 加速且超越全量重训</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/模型压缩&quot;,&quot;知识蒸馏&quot;,&quot;流行度去偏&quot;,&quot;分组排序学习&quot;]"><strong><a href="../../venues/WSDM/2023/debias_fairness_industrial/unbiased_knowledge_distillation_for_recommendation/">Unbiased Knowledge Distillation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示知识蒸馏的流行度偏差，提出按流行度分组的无偏蒸馏 UnKD；在三个数据集上提升 Recall@10 并显著改善非热门物品。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;属性模拟&quot;,&quot;物品共现矩阵&quot;,&quot;多兴趣建模&quot;]"><strong><a href="../../venues/WSDM/2024/collaborative_filtering/attribute_simulation_for_item_embedding_enhancement_in_multi_interest_recom/">Attribute Simulation for Item Embedding Enhancement in Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品共现矩阵近似物品-属性关联矩阵，提出即插即用的 SimEmb 模块替代 ID Embedding 做多兴趣召回，4 个数据集全库排序 Recall@20 平均相对提升 25.59%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Embedding 尺寸搜索&quot;,&quot;神经架构搜索&quot;,&quot;内存预算约束&quot;]"><strong><a href="../../venues/WSDM/2024/collaborative_filtering/budgeted_embedding_table_for_recommender_systems/">Budgeted Embedding Table For Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>BET 用「预算受限的表级动作采样」取代逐实例尺寸搜索，并以 DeepSets 集合表征代理评估动作适应度，在 80–95% 三档稀疏率下稳定超越 CIESS 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;维度坍缩&quot;,&quot;率失真编码长度&quot;,&quot;非对比学习&quot;]"><strong><a href="../../venues/WSDM/2024/collaborative_filtering/towards_mitigating_dimensional_collapse_of_representations_in_collaborative/">Towards Mitigating Dimensional Collapse of Representations in Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以率失真编码长度替代 InfoNCE，消除维度坍缩，免增广免负采样，ML10M Recall@10 提升 14.21%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图谱域扩散&quot;,&quot;各向异性噪声&quot;,&quot;条件去噪网络&quot;]"><strong><a href="../../venues/WSDM/2025/collaborative_filtering/s_diff_an_anisotropic_diffusion_model_for_collaborative_filtering_in_spectr/">S-Diff: An Anisotropic Diffusion Model for Collaborative Filtering in Spectral Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将扩散前向过程搬到物品图拉普拉斯谱域，用特征值参数化各向异性噪声保低频，配合 FiLM 条件去噪，全库排序下 Recall@20 最高提升约 5.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;协作信息量化定义&quot;,&quot;协作关系分布实证&quot;,&quot;序列推荐评测&quot;]"><strong><a href="../../venues/WSDM/2026/collaborative_filtering/have_we_really_understood_collaborative_information_an_empirical_investigat/">Have We Really Understood Collaborative Information? An Empirical Investigation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以物品共现路径长度量化协作信息为0/1/2-hop协作关系，系统实证其在六个基准上的分布、对12种推荐算法的影响与预测依赖，揭示模型强于简单CR、弱于复杂CR的核心规律。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;在线知识蒸馏&quot;,&quot;异构目标一致性&quot;,&quot;多任务梯度归一化&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/consensus_learning_from_heterogeneous_objectives_for_one_class_collaborativ/">Consensus Learning from Heterogeneous Objectives for One-Class Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为 OCCF 目标模型挂异构目标辅助头，用排名一致性生成动态共识做在线蒸馏，全库排序下 R@20 最高提升 30.7%，推理零额外开销</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/排序打分&quot;,&quot;交叉配对排序损失&quot;,&quot;流行度去偏&quot;,&quot;动态难样本采样&quot;]"><strong><a href="../../venues/WWW/2022/debias_fairness_industrial/cross_pairwise_ranking_for_unbiased_item_recommendation/">Cross Pairwise Ranking for Unbiased Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>交叉配对损失抵消用户/物品曝光倾向，实现样本级无偏学习，Recall@20 较 DICE 提升 9.2%–18.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;倒排多索引&quot;,&quot;重要性采样&quot;,&quot;变分自编码器&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/fast_variational_autoencoder_with_inverted_multi_index_for_collaborative_fi/">Fast Variational AutoEncoder with Inverted Multi-Index for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用倒排多索引分解 softmax 概率，构造亚线性时间的无偏/低偏采样器 MIDX_Uni/MIDX_Pop，使 Mult-VAE 训练提速 6× 且 NDCG@50 反超全 softmax 基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双曲空间&quot;,&quot;几何正则化&quot;,&quot;图协同过滤&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/hrcf_enhancing_collaborative_filtering_via_hyperbolic_geometric_regularizat/">HRCF: Enhancing Collaborative Filtering via Hyperbolic Geometric Regularization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用“根对齐 + 原点感知惩罚”把双曲 embedding 推离原点，缓解双曲图聚合的过平滑，3 个数据集的 Recall@20 较 HGCF 提升 1.6%–3.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤&quot;,&quot;精排&quot;,&quot;损失函数设计&quot;,&quot;用户兴趣边界&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/learning_explicit_user_interest_boundary_for_recommendation/">Learning Explicit User Interest Boundary for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>为每个用户学习可学习兴趣边界 $b_u$，令正负样本分别与边界做 pairwise，融合 pointwise 灵活性与 pairwise 排序性，Leave-One-Out + 1:100 采样排序下 Hit@1 平均提升 5.22%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回/粗排&quot;,&quot;跨模型一致性&quot;,&quot;隐式反馈去噪&quot;,&quot;KL 散度正则&quot;]"><strong><a href="../../venues/WWW/2022/self_supervised_denoising/learning_robust_recommenders_through_cross_model_agreement/">Learning Robust Recommenders through Cross-Model Agreement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用跨模型预测一致性作弱监督信号，通过 KL 一致性约束+似然最大化联合去噪正负样本，在 4 个骨干上一致提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;个性化商品搜索&quot;,&quot;跳跃图卷积&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/WWW/2022/graph_rec/modeling_user_behavior_with_graph_convolution_for_personalized_product_sear/">Modeling User Behavior with Graph Convolution for Personalized Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用全体用户短期行为构建连续行为图，以跳跃图卷积丰富商品表征并接入零注意力用户建模，8 个 Amazon 域 NDCG@10 较 ZAM 提升 7.8%~41.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;互正则化&quot;]"><strong><a href="../../venues/WWW/2022/cold_start_rec/mutually_regularized_dual_collaborative_variational_auto_encoder_for_recomm/">Mutually-Regularized Dual Collaborative Variational Auto-encoder for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双协同 VAE，把 UAE 末层权重视为物品嵌入并以内容互正则，配合对称推断实现新物品免重训；三数据集全库排序优于 CVAE/CoVAE 等。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;对比学习&quot;,&quot;反向流正则&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/re4_learning_to_re_contrast_re_attend_re_construct_for_multi_interest_recom/">Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出反向流 Re4：以对比、重注意、重构三项正则约束多兴趣表征；Amazon/Gowalla 全库排序 NDCG@20 相对提升 42%/12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;模型训练后遗忘（召回/排序通用）&quot;,&quot;Machine Unlearning&quot;,&quot;Balanced Sharding&quot;,&quot;Adaptive Aggregation&quot;]"><strong><a href="../../venues/WWW/2022/debias_fairness_industrial/recommendation_unlearning/">Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 RecEraser：相似度平衡分片保留协同信息，注意力自适应聚合子模型；全库排序下高效精确遗忘且性能优于 SISA/GraphEraser。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;链接预测&quot;,&quot;度加权路径分数&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/revisiting_neighborhood_based_link_prediction_for_collaborative_filtering/">Revisiting Neighborhood-based Link Prediction for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>WWW&#x27;22 图学习 Workshop：把推荐还原为二部图链接预测，提出仅 6 个可学习参数的 LinkProp 度加权三跳路径分数，全库排序下 Amazon-Book 的 Recall/NDCG@20 相对 GTN 提升 60%/70%</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;双侧交互网络&quot;,&quot;共检索机制&quot;,&quot;直播推荐&quot;]"><strong><a href="../../venues/WWW/2022/collaborative_filtering/who_to_watch_next_two_side_interactive_networks_for_live_broadcast_recommen/">Who to Watch Next: Two-side Interactive Networks for Live Broadcast Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向直播推荐的三元组（用户-主播-商品）双侧交互网络，用 PNN+RNN 编码静态/动态特征、双注意力挖掘 AND 交互模式并按类目共检索剪枝，线上 ACTR 提升 8.11%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回+精排级联&quot;,&quot;难负例采样&quot;,&quot;排序序蒸馏&quot;,&quot;渐近无偏KL估计&quot;]"><strong><a href="../../venues/WWW/2023/multi_task_reranking/cooperative_retriever_and_ranker_in_deep_recommenders/">Cooperative Retriever and Ranker in Deep Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>CoRR 联合训练召回与精排：用自适应采样器从召回分布抽硬负例训精排，用渐近无偏 KL 估计做排序序一致性蒸馏回灌召回，四个大规模数据集 NDCG@10 提升 5–18%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;易到难课程学习&quot;,&quot;异构模型集成&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/distillation_from_heterogeneous_models_for_top_k_recommendation/">Distillation from Heterogeneous Models for Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用教师训练轨迹构造由易到难的排序知识课程，将异构推荐模型集成蒸馏进轻量学生，R@10 最高提升 24.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;聚类分治难负例&quot;,&quot;Prompt 式多任务学习&quot;,&quot;双塔 ANN 检索&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/divide_and_conquer_towards_better_embedding_based_retrieval_for_recommender/">Divide and Conquer: Towards Better Embedding-based Retrieval for Recommender Systems From a Multi-task Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将候选库语义聚类后分簇并行 EBR，仅用同簇难负例训练并配合 Prompt 式多任务适配，KuaiRand Recall 最高提升 42.7%，线上 A/B 全指标正向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;多轮交互在线召回&quot;,&quot;NDCG导向属性选择&quot;,&quot;用户表征精炼&quot;]"><strong><a href="../../venues/WWW/2023/agent_conversational_rec/enhancing_user_personalization_in_conversational_recommenders/">Enhancing User Personalization in Conversational Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 MGMCR 多正确答案设定与 UpsRec 框架：NDCG 增益贪心属性问询 + 注意力用户表征精炼器，四数据集 NDCG@10 较 UNICORN 提升 23.6%–58.6% 且轮次减半。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;自适应负采样&quot;,&quot;组公平性&quot;,&quot;双层优化&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/fairly_adaptive_negative_sampling_for_recommendations/">Fairly Adaptive Negative Sampling for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次从负采样视角治理物品组公平：用组级 G-BCE 感知不公平、动量自适应调组采样分布并与重要性采样 Mixup，ML1M-2 上 Recall-Disp 降 90.97%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;DP-SGD&quot;,&quot;整数规划公平重排&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/fairness_aware_differentially_private_collaborative_filtering/">Fairness-aware Differentially Private Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 DP-SGD 放大活跃/非活跃用户精度鸿沟（Matthew 效应），提出分离式用户-物品梯度扰动 + 整数规划公平重排的 DP-Fair 两阶段框架，在四项真实数据集上同时提升总效用并压降 F_U。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图划分&quot;,&quot;物品相似度模型&quot;,&quot;谱聚类&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/fine_tuning_partition_aware_item_similarities_for_efficient_and_scalable_re/">Fine-tuning Partition-aware Item Similarities for Efficient and Scalable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用谱划分把物品相似度建模限制在子划分内，并以全局谱信息微调；4 数据集上全库排序指标超越 GCN，提速 10×、省 95% 参数。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;对比学习&quot;,&quot;图神经网络&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/graph_less_collaborative_filtering/">Graph-less Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双层知识蒸馏把 GCN 教师压缩为 MLP 学生，配合梯度自适应对比正则，在 Full-Ranking 协议下兼顾精度与推理效率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦召回训练&quot;,&quot;成员推理攻击&quot;,&quot;本地差分隐私&quot;,&quot;参数容量再分配&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/interaction_level_membership_inference_attack_against_federated_recommender/">Interaction-level Membership Inference Attack Against Federated Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>首次揭示联邦推荐中交互级成员推理攻击：服务器仅凭上传的物品嵌入迭代训练影子模型反推用户正样本，并提出公共参数约束正则以近乎零推荐损失防御。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;不变学习&quot;,&quot;解耦表征&quot;,&quot;流行度分布偏移&quot;]"><strong><a href="../../venues/WWW/2023/debias_fairness_industrial/invariant_collaborative_filtering_to_popularity_distribution_shift/">Invariant Collaborative Filtering to Popularity Distribution Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将流行度偏移重构为分布外泛化问题，用表示增广+距离相关解耦出不变偏好表征，在 5 个真实集与 4 类评测协议下稳定超越主流去偏基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;难负采样&quot;,&quot;OPAUC&quot;,&quot;DRO&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/on_the_theories_behind_hard_negative_sampling_for_recommendation/">On the Theories Behind Hard Negative Sampling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>证明BPR难负采样等价优化OPAUC，并给出可控硬度与按K调采样策略。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;混合频率滤波器&quot;,&quot;非参数协同过滤&quot;]"><strong><a href="../../venues/WWW/2023/graph_rec/personalized_graph_signal_processing_for_collaborative_filtering/">Personalized Graph Signal Processing for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用个性化图信号+增广相似图+混合频率图滤波构建免训练的非参数 CF，随机划分全库排序下 NDCG 平均超越 SOTA 3.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（Top-N 排序）&quot;,&quot;可解释推荐&quot;,&quot;口味簇学习&quot;,&quot;稀疏隶属矩阵&quot;,&quot;直通梯度估计&quot;]"><strong><a href="../../venues/WWW/2023/collaborative_filtering/towards_explainable_collaborative_filtering_with_taste_clusters_learning/">Towards Explainable Collaborative Filtering with Taste Clusters Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用稀疏「口味簇」隶属矩阵实现自解释 CF，Forest 集成补偿稀疏化损失，三数据集 Recall@20 相对最优基线提升 8.0%–14.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;双稳健估计&quot;,&quot;模型校准&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/doubly_calibrated_estimator_for_recommendation_on_data_missing_not_at_rando/">Doubly Calibrated Estimator for Recommendation on Data Missing Not At Random</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出双校准 DR 估计器，用校准专家与三层联合学习同时校准插补误差和倾向性分数，在 Coat/Yahoo!R3/KuaiRec 无偏测试集上 MSE 降低 3.4%–8.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;分布式检索&quot;,&quot;摊销组曝光公平&quot;,&quot;对偶优化&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/fairsync_ensuring_amortized_group_exposure_in_distributed_recommendation_re/">FairSync: Ensuring Amortized Group Exposure in Distributed Recommendation Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将组曝光约束转为对偶空间中的分布式召回，中心聚合历史公平信息并广播；全库KNN评测在Amazon-Book/Taobao保持100% ESP，Recall较Uncalibrated提升3.4%/9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排/排序阶段&quot;,&quot;交叉双边公平&quot;,&quot;锐度感知损失再加权&quot;,&quot;打分归一化&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/intersectional_two_sided_fairness_in_recommendation/">Intersectional Two-sided Fairness in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 ITFR，用锐度感知损失识别劣势交叉群组 + 协作梯度贡献再加权 + 打分归一化，在 3 个数据集上 CV@20 降低 7.8%–43.5% 且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回/精排（通用表征插件）&quot;,&quot;可微向量量化&quot;,&quot;类别树&quot;]"><strong><a href="../../venues/WWW/2024/generative_rec/learning_category_trees_for_id_based_recommendation_exploring_the_power_of_/">Learning Category Trees for ID-Based Recommendation: Exploring the Power of Differentiable Vector Quantization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出端到端可微 VQ 框架 CAGE，为纯 ID 推荐自动学习用户/物品类别树；在列表补全 NDCG/HR 评测中较 FANS 最高提升 21.41%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;隐式图建模&quot;,&quot;方差缩减采样&quot;]"><strong><a href="../../venues/WWW/2024/graph_rec/linear_time_graph_neural_networks_for_scalable_recommendations/">Linear-Time Graph Neural Networks for Scalable Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 PPNP 不动点方程的单层隐式求解 + 高效方差缩减邻居采样，把 GNN 推荐的训练复杂度降到线性 $O(|E|Dd)$，与 MF 同级而精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排（训练目标）&quot;,&quot;部分AUC&quot;,&quot;点式损失&quot;]"><strong><a href="../../venues/WWW/2024/collaborative_filtering/lower_left_partial_auc_an_effective_and_efficient_optimization_metric_for_r/">Lower-Left Partial AUC: An Effective and Efficient Optimization Metric for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出LLPAUC双约束部分AUC与点式minimax损失，全库排序下提升Top-K并抗噪。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Note Compression Prompt&quot;,&quot;生成式对比学习 GCL&quot;,&quot;多任务指令微调 CSFT&quot;]"><strong><a href="../../venues/WWW/2024/llm_rec/notellm_a_retrievable_large_language_model_for_note_recommendation/">NoteLLM: A Retrievable Large Language Model for Note Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Note Compression Prompt 把笔记压缩为单个 [EMB] 隐状态做 I2I 召回，并以话题标签/类目生成做多任务增强；257K 全库 Full-Ranking 下 Avg Recall 94.66%，线上 CTR +16.20%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;Federated Recommendation&quot;,&quot;Poisoning Attack&quot;,&quot;Fake Users&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/poisoning_federated_recommender_systems_with_fake_users/">Poisoning Federated Recommender Systems with Fake Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出免本地数据、免聚合规则知识的假用户投毒攻击 PoisonFRS，仅凭服务器下发的物品嵌入估计流行物品并构造目标模型，在 4 个真实数据集与 7 种聚合规则下以 0.03%–0.05% 假用户将目标物品 HR@5 推至 0.70–1.00。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多兴趣建模&quot;,&quot;生成式语义ID检索&quot;,&quot;层次聚类&quot;]"><strong><a href="../../venues/WWW/2024/collaborative_filtering/rimirec_modeling_refined_multi_interest_in_hierarchical_structure_for_recom/">RimiRec: Modeling Refined Multi-interest in Hierarchical Structure for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用层次 k-means 语义类目 ID + Transformer 生成用户多层兴趣圈，再按子库分区训练多塔向量检索；ML-1M/Lofter 全库 Top-M 召回显著优于 MIND/ComiRec，并已上线 Lofter。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;ADMM 分布式优化&quot;,&quot;曝光公平正则&quot;]"><strong><a href="../../venues/WWW/2024/debias_fairness_industrial/scalable_and_provably_fair_exposure_control_for_large_scale_recommender_sys/">Scalable and Provably Fair Exposure Control for Large-Scale Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将曝光公平正则嵌入 iALS，用 ADMM+近端梯度与 Sherman-Morrison 技巧保住可分性，训练开销接近 iALS，在 ML-20M/MSD 等百万级数据上实现可调曝光控制。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;数据集链接&quot;,&quot;协同嵌入重排&quot;,&quot;零样本 LLM推荐&quot;]"><strong><a href="../../venues/WWW/2025/agent_conversational_rec/bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;扩散推荐&quot;,&quot;伪用户协同信号&quot;]"><strong><a href="../../venues/WWW/2025/generative_rec/collaborative_diffusion_model_for_recommender_system/">Collaborative Diffusion Model for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论词转为伪用户，在扩散去噪过程中聚合真实/伪邻居偏好信息，缓解个性化信息流失，三数据集 R@20 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图滤波&quot;,&quot;多准则推荐&quot;,&quot;免训练&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;隐式反馈加权&quot;,&quot;线性自编码器&quot;,&quot;预条件共轭梯度&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/does_weighting_improve_matrix_factorization_for_recommender_systems/">Does Weighting Improve Matrix Factorization for Recommender Systems?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统实证加权隐式反馈矩阵分解：大秩模型下不加权常持平或更优，并给出加权目标闭式解与预条件共轭梯度高效求解。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;理想高通滤波&quot;,&quot;高阶邻域&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;图卷积权重控制&quot;,&quot;连续时间建模&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入表压缩&quot;,&quot;模块度图聚类&quot;,&quot;哈希碰撞&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;交叉交互特征&quot;,&quot;实时用户序列&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态双塔&quot;,&quot;稠密检索&quot;,&quot;图文对齐&quot;]"><strong><a href="../../venues/WWW/2025/multimodal_rec/multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;个人损失分布重采样&quot;]"><strong><a href="../../venues/WWW/2025/self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图 Transformer&quot;,&quot;排序目标驱动架构&quot;,&quot;线性加速&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mixture-of-Logits&quot;,&quot;近似Top-K检索&quot;,&quot;负载均衡正则&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;Token Alignment&quot;,&quot;Logits Matching&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/training_large_recommendation_models_via_graph_language_token_alignment/">Training Large Recommendation Models via Graph-Language Token Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 LightGCN 用户/物品节点经投影器对齐进 LLaMA-2 词表空间，用 GLLM logits 匹配层做端到端物品预测，Full-Ranking 下 NDCG@10 最高提升 185%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;k-DPP 多样性采样&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/diversity_augmented_negative_sampling_for_implicit_collaborative_filtering/">Diversity-Augmented Negative Sampling for Implicit Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用用户级缓存 + 多样性增强 k-DPP + mixup 合成负样本，解决两阶段负采样同质化问题，Full-Ranking 下 NDCG@20 最高提升 8.76%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;稀疏嵌入&quot;,&quot;渐进剪枝&quot;,&quot;线性自编码器&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/efficient_learning_of_sparse_representations_from_interactions/">Efficient Learning of Sparse Representations from Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 ELSA 线性自编码器训练中渐进剪枝学习高维稀疏物品嵌入，实现最高 100× 压缩仅损 2.5% nDCG，并派生可解释语义分段。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Target-Aware 蒸馏&quot;,&quot;Generate-then-Rank&quot;,&quot;分解因果自注意力&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/grank_towards_target_aware_and_streamlined_industrial_retrieval_with_a_gene/">GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用训练期目标注入把 Target-Aware 能力蒸馏进 MIPS 兼容生成器，再以轻量交叉注意力重排，免结构化索引实现 Recall@500 +30% 与 1.7× QPS。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;用户活跃度重加权&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/identifying_and_upweighting_power_niche_users_to_mitigate_popularity_bias_i/">Identifying and Upweighting Power-Niche Users to Mitigate Popularity Bias in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>识别高活跃-小众偏好用户并对其在BPR损失中双重加权，在三个基准上实现性能与流行度去偏的帕累托双赢。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Item-based CF&quot;,&quot;Trigger-Target 负采样&quot;,&quot;目标注意力&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;联邦训练阶段（客户端梯度投毒）&quot;,&quot;子群投毒攻击&quot;,&quot;联邦推荐安全&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/spattack_subgroup_poisoning_attacks_on_federated_recommender_systems/">Spattack: Subgroup Poisoning Attacks on Federated Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个联邦推荐子群投毒攻击：近似-提升范式+对比排斥与自适应加权，0.1%恶意客户端即达100% ER@5且几乎不伤非目标群。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Top-K 损失函数&quot;,&quot;分位数回归&quot;,&quot;分布鲁棒优化&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/talos_optimizing_top_k_accuracy_in_recommender_systems/">Talos: Optimizing Top-$K$ Accuracy in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用量化阈值把 Top-K 截断改写为分数-阈值比较，配合采样分位回归与约束项，直接优化 Precision@K/Recall@K 并具备 DRO 鲁棒性。</p>
</dd>
</dl>
