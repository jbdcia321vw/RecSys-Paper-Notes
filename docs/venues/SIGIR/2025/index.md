---
hide:
  - toc
---

# 🏛️ SIGIR 2025 推荐系统论文解读

86 篇 SIGIR 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 大语言模型推荐 ×14 · 去偏公平与可信推荐 ×12 · 生成式推荐 ×12 · 多模态推荐 ×6 · 多任务学习 ×5 · 知识蒸馏 ×3 · 跨域序列推荐 ×3 · LLM 语义嵌入 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">33</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体协作" aria-pressed="false">多智能体协作<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询改写" aria-pressed="false">查询改写<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度偏置" aria-pressed="false">流行度偏置<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Agentic Feedback Loop" aria-pressed="false">Agentic Feedback Loop<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="AIGC 内容生态" aria-pressed="false">AIGC 内容生态<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Thought" aria-pressed="false">Chain-of-Thought<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CoT 难样本挖掘" aria-pressed="false">CoT 难样本挖掘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-domain Memory" aria-pressed="false">Cross-domain Memory<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Customer Persona" aria-pressed="false">Customer Persona<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decision Transformer" aria-pressed="false">Decision Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DQN 奖励设计" aria-pressed="false">DQN 奖励设计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GFlowNet" aria-pressed="false">GFlowNet<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSTU" aria-pressed="false">HSTU<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Interest Unit" aria-pressed="false">Interest Unit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Knowledge-Grounded Generation" aria-pressed="false">Knowledge-Grounded Generation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KTO 行为对齐" aria-pressed="false">KTO 行为对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM User Agent" aria-pressed="false">LLM User Agent<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM User Simulation" aria-pressed="false">LLM User Simulation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 指令微调" aria-pressed="false">LLM 指令微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 生成假新闻" aria-pressed="false">LLM 生成假新闻<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-a-Judge" aria-pressed="false">LLM-as-a-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM增强CTR" aria-pressed="false">LLM增强CTR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推理" aria-pressed="false">LLM推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 动态集成" aria-pressed="false">LoRA 动态集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MK-MMD 特征核对齐" aria-pressed="false">MK-MMD 特征核对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP-Mixer" aria-pressed="false">MLP-Mixer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Patch 预训练" aria-pressed="false">Patch 预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Persona-Based Personalization" aria-pressed="false">Persona-Based Personalization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Popularity Modeling" aria-pressed="false">Popularity Modeling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Random Walk Approximation" aria-pressed="false">Random Walk Approximation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE 语义ID" aria-pressed="false">RQ-VAE 语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SENet特征重加权" aria-pressed="false">SENet特征重加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sockpuppeting 仿真" aria-pressed="false">Sockpuppeting 仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Synthetic Query Benchmark" aria-pressed="false">Synthetic Query Benchmark<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Thompson Sampling" aria-pressed="false">Thompson Sampling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Truth Decay" aria-pressed="false">Truth Decay<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="一致性正则" aria-pressed="false">一致性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三专家解耦" aria-pressed="false">三专家解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化因子" aria-pressed="false">个性化因子<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="习题推荐" aria-pressed="false">习题推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="互信息对比学习" aria-pressed="false">互信息对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="互补生成器学习" aria-pressed="false">互补生成器学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交替优化" aria-pressed="false">交替优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪孪生网络" aria-pressed="false">伪孪生网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏差 MAPB" aria-pressed="false">位置偏差 MAPB<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩分解" aria-pressed="false">低秩分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩矩阵分解" aria-pressed="false">低秩矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低阶信号保持" aria-pressed="false">低阶信号保持<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="侧信息融合" aria-pressed="false">侧信息融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="候选空间收缩" aria-pressed="false">候选空间收缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="倾向分免估计" aria-pressed="false">倾向分免估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好型 qrels" aria-pressed="false">偏好型 qrels<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好感知注意力" aria-pressed="false">偏好感知注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好抽取" aria-pressed="false">偏好抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元优化两阶段训练" aria-pressed="false">元优化两阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推理" aria-pressed="false">免训练推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局归一化注意力" aria-pressed="false">全局归一化注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局模态注意力" aria-pressed="false">全局模态注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平采样" aria-pressed="false">公平采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣生命周期" aria-pressed="false">兴趣生命周期<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层用户画像" aria-pressed="false">分层用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="切比雪夫插值" aria-pressed="false">切比雪夫插值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表全面性" aria-pressed="false">列表全面性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级上下文建模" aria-pressed="false">列表级上下文建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态奖励塑形" aria-pressed="false">动态奖励塑形<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态推荐系统" aria-pressed="false">动态推荐系统<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动量蒸馏" aria-pressed="false">动量蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信审计" aria-pressed="false">去偏公平与可信审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数生成网络" aria-pressed="false">参数生成网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双提升攻击目标" aria-pressed="false">双提升攻击目标<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双智能体" aria-pressed="false">双智能体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲空间推荐" aria-pressed="false">双曲空间推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双阶段模态融合" aria-pressed="false">双阶段模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实数据增广" aria-pressed="false">反事实数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反向推理微调" aria-pressed="false">反向推理微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反馈回路 (Feedback Loop)" aria-pressed="false">反馈回路 (Feedback Loop)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排统一" aria-pressed="false">召回+精排统一<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/精排前的域分组阶段" aria-pressed="false">召回/精排前的域分组阶段<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性" aria-pressed="false">可复现性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成数据保真度" aria-pressed="false">合成数据保真度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化聚类" aria-pressed="false">向量量化聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果发现" aria-pressed="false">因果发现<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图不变学习" aria-pressed="false">图不变学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图推荐" aria-pressed="false">图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图提示 (Graph Prompting)" aria-pressed="false">图提示 (Graph Prompting)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图提示学习" aria-pressed="false">图提示学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域聚类" aria-pressed="false">域聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基线调优" aria-pressed="false">基线调优<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="处理效应" aria-pressed="false">处理效应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复合图卷积" aria-pressed="false">复合图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务基准" aria-pressed="false">多任务基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣建模" aria-pressed="false">多兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多周期兴趣建模" aria-pressed="false">多周期兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域序列建模" aria-pressed="false">多域序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域预训练" aria-pressed="false">多域预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性重排生成" aria-pressed="false">多样性重排生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态好友推荐" aria-pressed="false">多模态好友推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态语义图" aria-pressed="false">多模态语义图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态跨注意力" aria-pressed="false">多模态跨注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多步预测" aria-pressed="false">多步预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多生成器集成" aria-pressed="false">多生成器集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标 EBR" aria-pressed="false">多目标 EBR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标偏好对齐" aria-pressed="false">多目标偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多码本向量量化" aria-pressed="false">多码本向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列推荐" aria-pressed="false">多行为序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多视图搜索" aria-pressed="false">多视图搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大五人格推断" aria-pressed="false">大五人格推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子图检索" aria-pressed="false">子图检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗环境生成" aria-pressed="false">对抗环境生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话式商品搜索" aria-pressed="false">对话式商品搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层次化压缩" aria-pressed="false">层次化压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层级信息对齐" aria-pressed="false">层级信息对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层级因果 Transformer" aria-pressed="false">层级因果 Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="布朗桥反馈预判" aria-pressed="false">布朗桥反馈预判<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="干预分布" aria-pressed="false">干预分布<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="平台推荐审计" aria-pressed="false">平台推荐审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="幻觉与事实性" aria-pressed="false">幻觉与事实性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列数据增强" aria-pressed="false">序列数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列生成" aria-pressed="false">序列生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构 Token 化" aria-pressed="false">异构 Token 化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响编码器" aria-pressed="false">影响编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图一致性过滤" aria-pressed="false">意图一致性过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图原型聚类" aria-pressed="false">意图原型聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知对比学习" aria-pressed="false">意图感知对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知推荐" aria-pressed="false">意图感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦表征" aria-pressed="false">意图解耦表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散式跨模态交互" aria-pressed="false">扩散式跨模态交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型用户画像" aria-pressed="false">扩散模型用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑广告" aria-pressed="false">捆绑广告<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序驱动增强器" aria-pressed="false">排序驱动增强器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推测解码" aria-pressed="false">推测解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理蒸馏" aria-pressed="false">推理蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码Softmax损失" aria-pressed="false">掩码Softmax损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据划分" aria-pressed="false">数据划分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据合成与基准构建" aria-pressed="false">数据合成与基准构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据投毒攻击" aria-pressed="false">数据投毒攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据自适应归一化" aria-pressed="false">数据自适应归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集构建" aria-pressed="false">数据集构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集记忆" aria-pressed="false">数据集记忆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据预处理与基准构建" aria-pressed="false">数据预处理与基准构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本投毒攻击" aria-pressed="false">文本投毒攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新物品公平性" aria-pressed="false">新物品公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐可信性" aria-pressed="false">新闻推荐可信性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="无ID推荐" aria-pressed="false">无ID推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="无序集合标识符" aria-pressed="false">无序集合标识符<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="早期交互" aria-pressed="false">早期交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="显隐式双流融合" aria-pressed="false">显隐式双流融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="晚期交互" aria-pressed="false">晚期交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="本地生活推荐" aria-pressed="false">本地生活推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="机器遗忘" aria-pressed="false">机器遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散去噪" aria-pressed="false">条件扩散去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散增强" aria-pressed="false">条件扩散增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="来源偏见 (Source Bias)" aria-pressed="false">来源偏见 (Source Bias)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="松弛验证" aria-pressed="false">松弛验证<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询引导同时生成" aria-pressed="false">查询引导同时生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="梯度自适应加权" aria-pressed="false">梯度自适应加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索增强表征" aria-pressed="false">检索增强表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索池分组" aria-pressed="false">检索池分组<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型血缘追踪" aria-pressed="false">模型血缘追踪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态感知成对匹配" aria-pressed="false">模态感知成对匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态解耦" aria-pressed="false">模态解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="段内排序损失" aria-pressed="false">段内排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="池化评测" aria-pressed="false">池化评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时增强" aria-pressed="false">测试时增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合拍卖" aria-pressed="false">混合拍卖<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="灾难性遗忘" aria-pressed="false">灾难性遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="点击模型" aria-pressed="false">点击模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="片段级兴趣建模" aria-pressed="false">片段级兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态空间模型" aria-pressed="false">状态空间模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式奖励模型" aria-pressed="false">生成式奖励模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐评估" aria-pressed="false">生成式推荐评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户世界模型" aria-pressed="false">用户世界模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户建模" aria-pressed="false">用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="由粗到细检索" aria-pressed="false">由粗到细检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相似边对 (SEP)" aria-pressed="false">相似边对 (SEP)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="眼动追踪数据集" aria-pressed="false">眼动追踪数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱增强推荐" aria-pressed="false">知识图谱增强推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱构建" aria-pressed="false">知识图谱构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识解耦" aria-pressed="false">知识解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵吸收" aria-pressed="false">矩阵吸收<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交去噪" aria-pressed="false">社交去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经重排" aria-pressed="false">神经重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线知识生成" aria-pressed="false">离线知识生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评估" aria-pressed="false">离线评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测基准" aria-pressed="false">离线评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏注意力掩码" aria-pressed="false">稀疏注意力掩码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="竞价词生成" aria-pressed="false">竞价词生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端对话式检索" aria-pressed="false">端到端对话式检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端物品分词" aria-pressed="false">端到端物品分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端策略学习" aria-pressed="false">端到端策略学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="算法审计可复现性" aria-pressed="false">算法审计可复现性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联选择性掩码" aria-pressed="false">级联选择性掩码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性物品-物品模型" aria-pressed="false">线性物品-物品模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缺失数据模拟" aria-pressed="false">缺失数据模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缺失模态生成" aria-pressed="false">缺失模态生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动机制设计" aria-pressed="false">自动机制设计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自蒸馏式自洽" aria-pressed="false">自蒸馏式自洽<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应BPR" aria-pressed="false">自适应BPR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应温度" aria-pressed="false">自适应温度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论偏好画像" aria-pressed="false">评论偏好画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义对齐检索" aria-pressed="false">语义对齐检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语言嵌入融合" aria-pressed="false">语言嵌入融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱滤波" aria-pressed="false">谱滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移抑制" aria-pressed="false">负迁移抑制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="贴纸检索" aria-pressed="false">贴纸检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨框架互操作" aria-pressed="false">跨框架互操作<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态对齐" aria-pressed="false">跨模态对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨类型资源推荐" aria-pressed="false">跨类型资源推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轮播界面位置偏置" aria-pressed="false">轮播界面位置偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="边感知图卷积" aria-pressed="false">边感知图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="迁移增益" aria-pressed="false">迁移增益<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过程监督" aria-pressed="false">过程监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆物品频率IIF" aria-pressed="false">逆物品频率IIF<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐层注意力融合" aria-pressed="false">逐层注意力融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用用户建模" aria-pressed="false">通用用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾学生表征增强" aria-pressed="false">长尾学生表征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列目标注意力" aria-pressed="false">长序列目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期行为序列建模" aria-pressed="false">长期行为序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="闭式解" aria-pressed="false">闭式解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零空间学习" aria-pressed="false">零空间学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非侵入式融合" aria-pressed="false">非侵入式融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域建模" aria-pressed="false">频域建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波" aria-pressed="false">频域滤波<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;早期交互&quot;,&quot;晚期交互&quot;]"><strong><a href="collaborative_filtering/a_learnable_fully_interacted_two_tower_model_for_pre_ranking_system/">A Learnable Fully Interacted Two-Tower Model for Pre-Ranking System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习 Item Meta Matrix 实现显式早期交互、行/列 FC 相似度矩阵实现晚期交互，在保持双塔解耦推理效率下 AUC 相对提升 4.6%–14.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;切比雪夫插值&quot;]"><strong><a href="collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Interest Unit&quot;,&quot;语义对齐检索&quot;,&quot;多兴趣建模&quot;]"><strong><a href="collaborative_filtering/ira_adaptive_interest_aware_representation_and_alignment_for_personalized_m/">IRA: Adaptive Interest-aware Representation and Alignment for Personalized Multi-interest Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可累积/剪枝的文本化 Interest Unit 表征多兴趣，语义对齐替代点击信号做 ANN 召回，离线 H@20 提升 20%，线上点击 +5.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱构建&quot;,&quot;跨类型资源推荐&quot;,&quot;模型血缘追踪&quot;]"><strong><a href="graph_rec/benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱增强推荐&quot;,&quot;LLM 语义嵌入&quot;,&quot;检索增强表征&quot;]"><strong><a href="graph_rec/comprehending_knowledge_graphs_with_large_language_models_for_recommender_s/">Comprehending Knowledge Graphs with Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 理解 KG 局部子图并构建语义物品图做全局检索增强，离线预计算语义嵌入后接入 LightGCN，全库排序下全面超越 12 个基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;SENet特征重加权&quot;]"><strong><a href="self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;查询改写&quot;,&quot;LLM 指令微调&quot;,&quot;意图一致性过滤&quot;]"><strong><a href="sequential_rec/ai_guided_accelerator_for_search_experience/">AI Guided Accelerator For Search Experience</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>挖掘电商查询转移链并做意图一致性过滤，再用 Solar-10B 指令微调生成多样化替代收敛查询，线上 CTR +32.2%、转化 +38.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;零空间学习&quot;,&quot;语言嵌入融合&quot;,&quot;冷启动&quot;]"><strong><a href="sequential_rec/alphafuse_learn_id_embeddings_for_sequential_recommendation_in_null_space_o/">AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将语言嵌入 SVD 分解为语义丰富行空间与语义稀疏零空间，在裁剪后的零空间内学习 ID 嵌入，无需适配器即插即用地融合语义与协同信号。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;干预分布&quot;,&quot;处理效应&quot;]"><strong><a href="sequential_rec/csrec_rethinking_sequential_recommendation_from_a_causal_perspective/">CSRec: Rethinking Sequential Recommendation from A Causal Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将序列推荐重构为因果干预下的决策接受概率建模，用递归约束隔离系统影响，AHR@0.2 提升约 1.5–3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;序列数据增强&quot;,&quot;免训练推理&quot;]"><strong><a href="sequential_rec/data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;侧信息融合&quot;]"><strong><a href="sequential_rec/diff_dual_side_information_filtering_and_fusion_for_sequential_recommendati/">DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>频域滤波去噪 + 早期/中间双重融合的侧信息序列推荐，Full-Ranking 下 R@20 最高提升 14.1%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;频域建模&quot;,&quot;多视图搜索&quot;,&quot;本地生活推荐&quot;]"><strong><a href="sequential_rec/fim_frequency_aware_multi_view_interest_modeling_for_local_life_service_rec/">FIM: Frequency-Aware Multi-View Interest Modeling for Local-Life Service Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>多视图搜索分离本地生活多周期需求，FFT 频域感知演化建模周期兴趣，Kuaishou 线上 GMV +8.46%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构 Token 化&quot;,&quot;层级因果 Transformer&quot;,&quot;多步预测&quot;]"><strong><a href="sequential_rec/heterrec_heterogeneous_information_transformer_for_scalable_sequential_reco/">HeterRec: Heterogeneous Information Transformer for Scalable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将物品拆解为异构特征 token 序列并设计层级因果 Transformer 与列表级多步预测损失，在工业与 Amazon 数据上 Recall@100 提升 3.8%–4.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Decision Transformer&quot;,&quot;反事实数据增广&quot;,&quot;曝光偏差&quot;]"><strong><a href="sequential_rec/improving_sequential_recommenders_through_counterfactual_augmentation_of_sy/">Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Decision Transformer 建模全量曝光序列，并以随机/自改进反事实增广+用户模拟器挖掘未曝光兴趣，KuaiRand 上 Recall@20 相对提升约 87%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图原型聚类&quot;,&quot;条件扩散增强&quot;,&quot;意图感知对比学习&quot;]"><strong><a href="sequential_rec/intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性物品-物品模型&quot;,&quot;知识蒸馏&quot;,&quot;闭式解&quot;]"><strong><a href="sequential_rec/linear_item_item_model_with_neural_knowledge_for_session_based_recommendati/">Linear Item-Item Model with Neural Knowledge for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将神经会话模型的转移知识蒸馏为物品-物品矩阵正则项，与线性自蒸馏相似度统一闭式求解，R@20 最高提升 14.78% 且推理 FLOPs 降低 813 倍。</p>
</dd>
</dl>
<p><a href="sequential_rec/">查看全部12篇「序列与会话推荐」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;缺失数据模拟&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="generative_rec/addressing_missing_data_issue_for_diffusion_based_recommendation/">Addressing Missing Data Issue for Diffusion-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双侧 Thompson 采样在引导序列中模拟缺失数据，以一致性正则化让扩散推荐模型外推抵抗真实缺失，YooChoose NDCG@20 提升 26.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;端到端物品分词&quot;,&quot;RQ-VAE 语义ID&quot;,&quot;交替优化&quot;]"><strong><a href="generative_rec/generative_recommender_with_end_to_end_learnable_item_tokenization/">Generative Recommender with End-to-End Learnable Item Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 ETEGRec 双编码器-解码器框架，用序列-物品对齐与偏好-语义对齐将 RQ-VAE 分词器与 T5 推荐器端到端联合优化，全库排序下 Recall@10 相对 LETTER 提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;召回+精排统一&quot;,&quot;序列生成&quot;,&quot;排序驱动增强器&quot;,&quot;梯度自适应加权&quot;]"><strong><a href="generative_rec/killing_two_birds_with_one_stone_unifying_retrieval_and_ranking_with_a_sing/">Killing Two Birds with One Stone: Unifying Retrieval and Ranking with a Single Generative Recommendation Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将召回与排序统一为单生成模型的位置级序列生成任务，配合排序驱动增强器与梯度自适应加权实现协同与同步优化，全库排序下召回排序双提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;无序集合标识符&quot;,&quot;稀疏注意力掩码&quot;,&quot;查询引导同时生成&quot;]"><strong><a href="generative_rec/order_agnostic_identifier_for_large_language_model_based_generative_recomme/">Order-agnostic Identifier for Large Language Model-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出无序集合标识符范式，用CF+语义多token集合表示物品并同时生成，T5上Recall@10平均提升约50%，推理提速8-18倍。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;CoT 难样本挖掘&quot;,&quot;KTO 行为对齐&quot;,&quot;知识蒸馏&quot;]"><strong><a href="llm_rec/adore_autonomous_domain_oriented_relevance_engine_for_e_commerce/">ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 CoT-LLM 自动挖掘难样本、KTO 对齐用户行为、错误类型感知合成对抗样本，再以关键属性显式蒸馏到轻量 BERT，离线 F1 85.45%、线上 CTR +1.24%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;由粗到细检索&quot;,&quot;子图检索&quot;,&quot;LLM推理&quot;]"><strong><a href="llm_rec/corona_a_coarse_to_fine_framework_for_graph_based_recommendation_with_large/">CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 做偏好/意图推理在交互图上逐级缩小候选子图，再由 GNN 精排，全库排序下 Recall 平均相对提升 18.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;数据集记忆&quot;,&quot;流行度偏置&quot;]"><strong><a href="llm_rec/do_llms_memorize_recommendation_datasets_a_preliminary_study_on_movielens_1/">Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用少样本提示探测 GPT/Llama 对 MovieLens-1M 的物品、用户与交互记忆覆盖率，揭示记忆量与推荐性能强相关且放大流行度偏置。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;离线知识生成&quot;,&quot;推测解码&quot;,&quot;检索池分组&quot;,&quot;松弛验证&quot;]"><strong><a href="llm_rec/efficiency_unleashed_inference_acceleration_for_llm_based_recommender_syste/">Efficiency Unleashed: Inference Acceleration for LLM-based Recommender Systems with Speculative Decoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将检索式推测解码引入 LLM 离线知识生成，用定制检索池与松弛验证实现 3-5× 无损加速，线上省 67% 算力。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回+重排&quot;,&quot;Chain-of-Thought&quot;,&quot;位置偏差 MAPB&quot;,&quot;偏好抽取&quot;]"><strong><a href="llm_rec/improving_llm_powered_recommendations_with_personalized_information/">Improving LLM-powered Recommendations with Personalized Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户偏好与物品感知两条 CoT 显式抽取并注入 CRM 召回与 LLM 重排，Leave-One-Out 全库排序下 Hit@10 最高提升约 11%，并显著降低位置偏差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;意图解耦表征&quot;,&quot;跨模态对齐&quot;,&quot;动量蒸馏&quot;]"><strong><a href="llm_rec/intent_representation_learning_with_large_language_model_for_recommendation/">Intent Representation Learning with Large Language Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 思维链抽取文本意图，与交互意图双塔对齐+动量蒸馏匹配，作为模型无关插件在三个数据集上稳定提升 Recall/NDCG。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双曲空间推荐&quot;,&quot;元优化两阶段训练&quot;,&quot;层级信息对齐&quot;]"><strong><a href="llm_rec/large_language_models_enhanced_hyperbolic_space_recommender_systems/">Large Language Models Enhanced Hyperbolic Space Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HyperLLM 用 LLM 生成层级标签与偏好摘要，经元优化两阶段训练将结构/语义层级信息注入双曲推荐，Recall@20 最高提升 41.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;掩码Softmax损失&quot;,&quot;自适应温度&quot;]"><strong><a href="llm_rec/msl_not_all_tokens_are_what_you_need_for_tuning_llm_as_a_recommender/">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出掩码 Softmax 损失屏蔽虚构物品 token，并配自适应温度策略，四数据集 NDCG@10 平均提升 42.24%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;层次化压缩&quot;,&quot;Patch 预训练&quot;,&quot;长序列建模&quot;]"><strong><a href="llm_rec/multi_grained_patch_training_for_efficient_llm_based_recommendation/">Multi-Grained Patch Training for Efficient LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用层次化 item/session patch 压缩长交互历史，两阶段训练让 LLM 理解压缩表征，同 token 预算下 HR@20 最高提升 72%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;MK-MMD 特征核对齐&quot;,&quot;三专家解耦&quot;,&quot;灾难性遗忘&quot;]"><strong><a href="llm_rec/pre_train_align_and_disentangle_empowering_sequential_recommendation_with_l/">Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MK-MMD 特征核做推荐锚定对齐，再以三专家频率门控解耦，Full-Ranking 下冷物品 HR@10 提升最高 462%。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部14篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;LLM User Agent&quot;,&quot;Cross-domain Memory&quot;,&quot;Popularity Modeling&quot;]"><strong><a href="agent_conversational_rec/agentcf_memory_enhanced_llm_based_agents_for_popularity_aware_cross_domain_/">AgentCF++: Memory-enhanced LLM-based Agents for Popularity-aware Cross-domain Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以双层记忆+两步融合+兴趣组共享记忆，令 LLM 用户智能体在跨域场景显式建模流行度影响，采样排序 MRR 平均超 AgentCF 14.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;Agentic Feedback Loop&quot;,&quot;LLM User Simulation&quot;]"><strong><a href="agent_conversational_rec/agentic_feedback_loop_modeling_improves_recommendation_and_user_simulation/">Agentic Feedback Loop Modeling Improves Recommendation and User Simulation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用记忆驱动的推荐智能体-用户智能体多轮反馈闭环，同时提升下一物品推荐与用户行为模拟，并验证不放大流行度/位置偏差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;多模态语义图&quot;,&quot;图提示学习&quot;]"><strong><a href="agent_conversational_rec/mscrs_multi_modal_semantic_graph_prompt_learning_framework_for_conversation/">MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建协同/文本/图像三模态语义图并以共享嵌入融合，图提示注入 DialoGPT，ReDial 与 INSPIRED 全指标 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端对话式检索&quot;,&quot;对话式商品搜索&quot;,&quot;数据集构建&quot;,&quot;多任务基准&quot;]"><strong><a href="agent_conversational_rec/pscon_product_search_through_conversations/">PSCon: Product Search Through Conversations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出人-人教练式采集协议构建双市场双语真实 CPS 数据集 PSCon，定义六子任务流水线并给出 Transformer 统一基准模型。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;离线评估&quot;,&quot;用户模拟&quot;,&quot;大五人格推断&quot;,&quot;合成数据保真度&quot;]"><strong><a href="agent_conversational_rec/pub_an_llm_enhanced_personality_driven_user_behaviour_simulator_for_recomme/">PUB: An LLM-Enhanced Personality-Driven User Behaviour Simulator for Recommender System Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用大五人格驱动 LLM Agent 生成合成交互日志，Jaccard 相似度 0.31 逼近真实序列，并复现 7 种推荐算法的 nDCG 排序趋势。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;用户模拟&quot;,&quot;生成式奖励模型&quot;]"><strong><a href="agent_conversational_rec/search_based_interaction_for_conversation_recommendation_via_generative_rew/">Search-Based Interaction For Conversation Recommendation via Generative Reward Model Based Simulated User</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用生成式奖励模型构建无标签模拟用户，通过属性评论与生成式打分驱动束搜索交互，INSPIRED 上 Recall@10 相对最优基线提升 83%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;用户世界模型&quot;,&quot;扩散模型用户画像&quot;,&quot;布朗桥反馈预判&quot;]"><strong><a href="agent_conversational_rec/simulating_before_planning_constructing_intrinsic_user_world_model_for_user/">Simulating Before Planning: Constructing Intrinsic User World Model for User-Tailored Dialogue Policy Planning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散模型刻画用户画像、布朗桥预判用户反馈，构建内在用户世界模型以定制对话策略，P4G/ESConv 上 SR 显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;双阶段模态融合&quot;,&quot;复合图卷积&quot;,&quot;自适应BPR&quot;]"><strong><a href="multimodal_rec/cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;扩散式跨模态交互&quot;,&quot;知识解耦&quot;,&quot;非侵入式融合&quot;]"><strong><a href="multimodal_rec/diffusion_based_multi_modal_synergy_interest_network_for_click_through_rate/">Diffusion-based Multi-modal Synergy Interest Network for Click-through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PLE 式共享/独有专家解耦模态知识、以扩散式多步跨模态去噪捕捉协同信号，FDAF 非侵入融合，四数据集 AUC 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态解耦&quot;,&quot;缺失模态生成&quot;,&quot;互信息对比学习&quot;]"><strong><a href="multimodal_rec/disentangling_and_generating_modalities_for_recommendation_in_missing_modal/">Disentangling and Generating Modalities for Recommendation in Missing Modality Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将模态特征解耦为通用/特有分量并用互信息损失分离，再以跨模态对齐特征与用户模态偏好生成缺失模态，缺失场景全库排序 R@20 相对提升 4.7%–8.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;条件扩散去噪&quot;,&quot;多行为序列推荐&quot;]"><strong><a href="multimodal_rec/multi_modal_multi_behavior_sequential_recommendation_with_conditional_diffu/">Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用条件扩散分别对模态与行为特征去噪，并以多专家网络解耦共性/特性兴趣，在 Rec-Tmal/Kuaishou 上 HR@10 提升约 4-8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;片段级兴趣建模&quot;,&quot;多模态跨注意力&quot;,&quot;段内排序损失&quot;]"><strong><a href="multimodal_rec/short_video_segment_level_user_dynamic_interests_modeling_in_personalized_r/">Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将短视频切分为5秒片段，用CLIP视觉+ID双模态跨注意力建模片段级动态兴趣，以段内排序损失解决标签缺失，在跳看预测与视频推荐两任务上大幅超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;贴纸检索&quot;,&quot;用户建模&quot;]"><strong><a href="multimodal_rec/u_sticker_a_large_scale_multi_domain_user_sticker_dataset_for_retrieval_and/">U-Sticker: A Large-Scale Multi-Domain User Sticker Dataset for Retrieval and Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建含 22K 用户、370K 贴纸、8.3M 消息的多领域多语言贴纸数据集，支撑个性化贴纸检索与用户建模。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;多模态好友推荐&quot;,&quot;模态感知成对匹配&quot;,&quot;全局模态注意力&quot;]"><strong><a href="social_group_rec/frog_effective_friend_recommendation_in_online_games_via_modality_aware_use/">FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向在线游戏好友推荐，用模态感知双向匹配+局部/全局双路注意力建模用户偏好，离线 HR/NDCG 最高提升 15.82%/14.59%，线上 accRate 提升 33.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图不变学习&quot;,&quot;社交去噪&quot;,&quot;对抗环境生成&quot;]"><strong><a href="social_group_rec/invariance_matters_empowering_social_recommendation_via_graph_invariant_lea/">Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用图不变学习替代社交图硬过滤：多环境生成器+不变风险最小化+对抗环境探索，Full-Ranking 下 NDCG@20 较 GBSR 提升 5.4%–8.7%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;目标注意力&quot;,&quot;参数生成网络&quot;,&quot;多域序列建模&quot;]"><strong><a href="cross_domain_multi_scenario/adaptive_domain_scaling_for_personalized_sequential_modeling_in_recommender/">Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用域特征动态生成序列物品的个性化MLP权重与候选物品的多查询表示，改造目标注意力实现多域序列建模，抖音双场景线上GMV+0.79%、ADVV+1.00%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图提示 (Graph Prompting)&quot;,&quot;负迁移抑制&quot;,&quot;多域预训练&quot;]"><strong><a href="cross_domain_multi_scenario/adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;LLM 语义嵌入&quot;,&quot;分层用户画像&quot;]"><strong><a href="cross_domain_multi_scenario/bridge_the_domains_large_language_models_enhanced_cross_domain_sequential_r/">Bridge the Domains: Large Language Models Enhanced Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义嵌入+分层画像桥接跨域物品与用户偏好，破解重叠困境与转移复杂性，三数据集 H@10 最高提升 31.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/精排前的域分组阶段&quot;,&quot;域聚类&quot;,&quot;因果发现&quot;,&quot;迁移增益&quot;]"><strong><a href="cross_domain_multi_scenario/cdc_causal_domain_clustering_for_multi_domain_recommendation/">CDC: Causal Domain Clustering for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用孤立/混合双亲和矩阵+因果发现系数自适应融合，联合优化目标域簇与训练源域集，50+域上DomainAUC提升4.14‰/6.76‰，线上eCPM+4.9%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;通用用户建模&quot;,&quot;跨域序列推荐&quot;,&quot;信息瓶颈&quot;]"><strong><a href="cross_domain_multi_scenario/learning_universal_user_representations_leveraging_cross_domain_user_intent/">Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Snapchat 多业务面事件拼接为统一序列训练 Transformer 用户塔，产出跨域通用用户表征，线上多场景 A/B 全面正向。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;LoRA 动态集成&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="cross_domain_multi_scenario/x_cross_dynamic_integration_of_language_models_for_cross_domain_sequential_/">X-Cross: Dynamic Integration of Language Models for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>逐层动态集成多个 LoRA 源域语言模型，仅用 25% 参数与 50-75% 更少微调数据即达/超 LoRA 跨域序列推荐效果。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;生成式推荐&quot;,&quot;多智能体协作&quot;,&quot;候选空间收缩&quot;,&quot;反向推理微调&quot;]"><strong><a href="poi_spatiotemporal_rec/comapoi_a_collaborative_multi_agent_framework_for_next_poi_prediction_bridg/">CoMaPOI: A Collaborative Multi-Agent Framework for Next POI Prediction Bridging the Gap Between Trajectory and Language</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Profiler/Forecaster/Predictor 三智能体把数值轨迹转语言画像、动态收缩候选集并反向推理微调，三数据集全指标提升 4%–12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-ICTIR 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;相似边对 (SEP)&quot;,&quot;边感知图卷积&quot;]"><strong><a href="poi_spatiotemporal_rec/sep_gcn_leveraging_similar_edge_pairs_with_temporal_and_spatial_contexts_fo/">SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以时空相似交互边对构建边级相似图并注入LightGCN传播，在NYC/Gowalla/Brightkite全库排序上Recall@20最高提升14.89%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长期行为序列建模&quot;,&quot;多码本向量量化&quot;,&quot;HSTU&quot;]"><strong><a href="ctr_feature_interaction/deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;显隐式双流融合&quot;,&quot;低阶信号保持&quot;,&quot;逐层注意力融合&quot;]"><strong><a href="ctr_feature_interaction/dlf_enhancing_explicit_implicit_interaction_via_dynamic_low_order_aware_fus/">DLF: Enhancing Explicit-Implicit Interaction via Dynamic Low-Order-Aware Fusion for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以残差感知低阶交互网络与逐层注意力融合，动态平衡显隐式特征交叉，四数据集AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;LLM增强CTR&quot;,&quot;伪孪生网络&quot;]"><strong><a href="ctr_feature_interaction/elec_efficient_large_language_model_empowered_click_through_rate_prediction/">ELEC: Efficient Large Language Model-Empowered Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>伪孪生增益/朴素双网络+分数级CLID与表示级MSE双蒸馏，在线仅跑表格DCNv2，AUC较FLIP提升0.0020/0.0034。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动机制设计&quot;,&quot;混合拍卖&quot;,&quot;捆绑广告&quot;]"><strong><a href="ctr_feature_interaction/hybrid_advertising_in_the_sponsored_search/">Hybrid Advertising in the Sponsored Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出混合广告拍卖模型与 HRegNet 神经网络机制，在近 DSIC/IR 约束下联合优化独立店铺与店铺-品牌捆绑包的广告位分配与定价，真实数据上收益显著超越 VCG 与 JRegNet。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;兴趣生命周期&quot;,&quot;向量量化聚类&quot;,&quot;多任务学习&quot;]"><strong><a href="ctr_feature_interaction/interest_changes_considering_user_interest_life_cycle_in_recommendation_sys/">Interest Changes: Considering User Interest Life Cycle in Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用曝光/点击/交互直方图+VQ聚类刻画兴趣生命周期，经特征重校准与专家网络融合注入MMOE精排，Lofter线上CTR+0.38%、CVR+1.04%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩矩阵分解&quot;,&quot;长序列目标注意力&quot;,&quot;矩阵吸收&quot;]"><strong><a href="ctr_feature_interaction/lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;查询改写&quot;,&quot;多目标偏好对齐&quot;,&quot;竞价词生成&quot;]"><strong><a href="ctr_feature_interaction/multi_objective_aligned_bidword_generation_model_for_e_commerce_search_adve/">Multi-objective Aligned Bidword Generation Model for E-commerce Search Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用判别器提供相关性/真实性/CPM三维奖励，扩展MODPO做多目标偏好对齐，驱动LLM生成高商业价值竞价词，线上广告收入+2.13%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;逆物品频率IIF&quot;,&quot;全局归一化注意力&quot;,&quot;个性化搜索&quot;]"><strong><a href="ctr_feature_interaction/nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;多生成器集成&quot;,&quot;列表全面性&quot;,&quot;互补生成器学习&quot;]"><strong><a href="multi_task_reranking/comprehensive_list_generation_for_multi_generator_reranking/">Comprehensive List Generation for Multi-Generator Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出多生成器-评估器重排框架与列表全面性指标，用 CLIG 自动学习互补生成器，离线 AUC 提升 0.9%、线上观看时长 +0.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标 EBR&quot;,&quot;参数高效微调&quot;,&quot;级联选择性掩码&quot;]"><strong><a href="multi_task_reranking/csmf_cascaded_selective_mask_fine_tuning_for_multi_objective_embedding_base/">CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>借鉴 PEFT 选择性掩码思想，按曝光→点击→转化级联顺序分阶段微调双塔召回，零参数增量实现多目标融合检索。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;神经重排&quot;,&quot;长尾学生表征增强&quot;,&quot;习题推荐&quot;]"><strong><a href="multi_task_reranking/nr4der_neural_re_ranking_for_diversified_exercise_recommendation/">NR4DER: Neural Re-ranking for Diversified Exercise Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 mLSTM 难度过滤 + 活跃学生序列增强 + 神经重排，兼顾长尾学生与个性化学习节奏的习题推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;列表级上下文建模&quot;,&quot;一致性正则&quot;,&quot;自蒸馏式自洽&quot;]"><strong><a href="multi_task_reranking/towards_principled_learning_for_re_ranking_in_recommender_systems/">Towards Principled Learning for Re-ranking in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出收敛一致性与对抗一致性两条重排学习原则，以自洽重排与相邻换位扰动构造额外监督，即插即用地提升多种重排基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端策略学习&quot;,&quot;离线强化学习&quot;,&quot;动态奖励塑形&quot;,&quot;双智能体&quot;]"><strong><a href="rl_bandit_rec/darlr_dual_agent_offline_reinforcement_learning_for_recommender_systems_wit/">DARLR: Dual-Agent Offline Reinforcement Learning for Recommender Systems with Dynamic Reward</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双智能体离线 RL 用选择器聚合相似用户动态重塑奖励与不确定性惩罚，四数据集累积奖励全面超越 ROLeR/DORL。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;新物品公平性&quot;,&quot;DQN 奖励设计&quot;,&quot;动态推荐系统&quot;]"><strong><a href="rl_bandit_rec/enhancing_new_item_fairness_in_dynamic_recommender_systems/">Enhancing New-item Fairness in Dynamic Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 DQN 智能体 + 三重奖励（探索/公平/准确）在动态推荐系统中提升新物品曝光公平性，KuaiRec 上 TGF 降 86%+ 且 HR 反升 39%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (13)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;可复现性研究&quot;,&quot;意图感知推荐&quot;,&quot;基线调优&quot;]"><strong><a href="debias_fairness_industrial/a_worrying_reproducibility_study_of_intent_aware_recommendation_models/">A Worrying Reproducibility Study of Intent-Aware Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现 5 个意图感知推荐模型，发现 2 个不可复现、全部被调优后的传统非神经基线超越，揭示方法学顽疾。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据预处理与基准构建&quot;,&quot;可复现性&quot;,&quot;数据划分&quot;,&quot;跨框架互操作&quot;]"><strong><a href="debias_fairness_industrial/datarec_a_python_library_for_standardized_and_reproducible_data_management_/">DataRec: A Python Library for Standardized and Reproducible Data Management in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>基于 55 篇顶会论文调研构建的统一数据管理库，提供可复现的过滤/划分/版本校验与跨框架导出，覆盖 18 个内置数据集。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据投毒攻击&quot;,&quot;序列推荐&quot;,&quot;双提升攻击目标&quot;,&quot;多样性重排生成&quot;]"><strong><a href="debias_fairness_industrial/diversity_aware_dual_promotion_poisoning_attack_on_sequential_recommendatio/">Diversity-aware Dual-promotion Poisoning Attack on Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示攻击与推荐目标梯度冲突，提出双提升MSE攻击目标与多样性重排序列生成，在保推荐精度的同时显著提升目标物品曝光。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;来源偏见 (Source Bias)&quot;,&quot;反馈回路 (Feedback Loop)&quot;,&quot;AIGC 内容生态&quot;]"><strong><a href="debias_fairness_industrial/exploring_the_escalation_of_source_bias_in_user_data_and_recommender_system/">Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 AIGC 在用户-数据-系统反馈回路中被逐步放大的来源偏见，并提出基于 L1 损失的黑盒去偏方法维持内容生态平衡。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测基准&quot;,&quot;池化评测&quot;,&quot;流行度偏置&quot;,&quot;偏好型 qrels&quot;]"><strong><a href="debias_fairness_industrial/extending_movielens_32m_to_provide_new_evaluation_objectives/">Extending MovieLens-32M to Provide New Evaluation Objectives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 TREC 池化法为 ML-32M 构建 51 用户偏好型 qrels，揭示传统 train/test 评测的流行度偏置并给出新评测目标。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;无ID推荐&quot;,&quot;文本投毒攻击&quot;,&quot;多智能体协作&quot;]"><strong><a href="debias_fairness_industrial/id_free_not_risk_free_llm_powered_agents_unveil_risks_in_id_free_recommende/">ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示无ID推荐器可被LLM多智能体改写文本攻击，提出TextSimu攻击与RewriteDetection检测，Hit@50最高提升数倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Truth Decay&quot;,&quot;LLM 生成假新闻&quot;,&quot;新闻推荐可信性&quot;]"><strong><a href="debias_fairness_industrial/llm_generated_fake_news_induces_truth_decay_in_news_ecosystem_a_case_study_/">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建约 7.5 万条人机混合新闻推荐数据集，模拟 LLM 生成假新闻在候选、历史、训练集三阶段渗透，揭示真实新闻排序优势被侵蚀的 Truth Decay 现象并以困惑度给出解释。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;机器遗忘&quot;,&quot;图推荐&quot;,&quot;影响编码器&quot;]"><strong><a href="debias_fairness_industrial/pre_training_for_recommendation_unlearning/">Pre-training for Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出预训练式影响编码器 UnlearnRec，将遗忘请求映射为嵌入漂移，免重训实现高效图推荐遗忘，10x 加速且保持精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;眼动追踪数据集&quot;,&quot;轮播界面位置偏置&quot;,&quot;点击模型&quot;]"><strong><a href="debias_fairness_industrial/recgaze_the_first_eye_tracking_and_user_interaction_dataset_for_carousel_in/">RecGaze: The First Eye Tracking and User Interaction Dataset for Carousel Interfaces</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个轮播界面眼动+点击+光标+选择解释数据集，87用户3477次交互，揭示F型浏览与位置偏置。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信审计&quot;,&quot;平台推荐审计&quot;,&quot;算法审计可复现性&quot;,&quot;Sockpuppeting 仿真&quot;,&quot;个性化因子&quot;]"><strong><a href="debias_fairness_industrial/revisiting_algorithmic_audits_of_tiktok_poor_reproducibility_and_short_term/">Revisiting Algorithmic Audits of TikTok: Poor Reproducibility and Short-term Validity of Findings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现 TikTok 推荐审计发现可复现性极差、结论仅短期有效，观看行为才是最强个性化信号，且结论高度依赖度量选择。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部13篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
