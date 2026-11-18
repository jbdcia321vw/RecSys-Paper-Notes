---
hide:
  - toc
---

# 🕸️ 图神经网络与知识图谱推荐 (Graph & Knowledge-Aware Recommendation)

> **研究范畴**：LightGCN 谱图卷积改进、超图推荐 (Hypergraph)、异构图与知识图谱推理 (KGRec)

当前方向共收录 **122** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 去偏公平与可信推荐 ×14 · 大语言模型推荐 ×13 · 多模态推荐 ×12 · 对比学习 ×12 · 生成式推荐 ×10 · 多行为推荐 ×6 · 可解释推荐 ×3 · 多任务学习 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">91</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">29</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">12</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱滤波" aria-pressed="false">谱滤波<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KG-RAG" aria-pressed="false">KG-RAG<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Neural ODE" aria-pressed="false">Neural ODE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="OOD 泛化" aria-pressed="false">OOD 泛化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图 Transformer" aria-pressed="false">图 Transformer<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图消息传递" aria-pressed="false">图消息传递<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图滤波" aria-pressed="false">图滤波<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络推荐" aria-pressed="false">图神经网络推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多项式图滤波" aria-pressed="false">多项式图滤波<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子图检索" aria-pressed="false">子图检索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱推荐" aria-pressed="false">知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱路径推理" aria-pressed="false">知识图谱路径推理<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移抑制" aria-pressed="false">负迁移抑制<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="过平滑" aria-pressed="false">过平滑<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Aspect-Conditioned Scoring" aria-pressed="false">Aspect-Conditioned Scoring<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR 精排" aria-pressed="false">CTR 精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Customer Persona" aria-pressed="false">Customer Persona<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Folding-In" aria-pressed="false">Folding-In<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Fourier KAN" aria-pressed="false">Fourier KAN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Graph Transformer" aria-pressed="false">Graph Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GraphRAG" aria-pressed="false">GraphRAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSIC 去偏" aria-pressed="false">HSIC 去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Knowledge Graph RAG" aria-pressed="false">Knowledge Graph RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Krylov 子空间" aria-pressed="false">Krylov 子空间<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN 鲁棒性" aria-pressed="false">LightGCN 鲁棒性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 知识注入解释" aria-pressed="false">LLM 知识注入解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 自动构图" aria-pressed="false">LLM 自动构图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 表征增强" aria-pressed="false">LLM 表征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推理" aria-pressed="false">LLM推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 参数高效微调" aria-pressed="false">LoRA 参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP替代GCN" aria-pressed="false">MLP替代GCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Multi-LoRA" aria-pressed="false">Multi-LoRA<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Pareto 多目标优化" aria-pressed="false">Pareto 多目标优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prompt 补偿" aria-pressed="false">Prompt 补偿<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Random Walk Approximation" aria-pressed="false">Random Walk Approximation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Ricci 曲率" aria-pressed="false">Ricci 曲率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SENet特征重加权" aria-pressed="false">SENet特征重加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Steiner Tree" aria-pressed="false">Steiner Tree<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="UltraGCN" aria-pressed="false">UltraGCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三元闭包重加权" aria-pressed="false">三元闭包重加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文老虎机" aria-pressed="false">上下文老虎机<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="事件序列建模" aria-pressed="false">事件序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="互补信号" aria-pressed="false">互补信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交互级依赖图" aria-pressed="false">交互级依赖图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交叉相关聚合" aria-pressed="false">交叉相关聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务感知子图检索" aria-pressed="false">任务感知子图检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪多任务学习" aria-pressed="false">伪多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏差控制函数" aria-pressed="false">位置偏差控制函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置编码" aria-pressed="false">位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩近似" aria-pressed="false">低秩近似<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低通谱滤波" aria-pressed="false">低通谱滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="作者谱系" aria-pressed="false">作者谱系<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈去噪" aria-pressed="false">信息瓶颈去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元优化两阶段训练" aria-pressed="false">元优化两阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元路径意图建模" aria-pressed="false">元路径意图建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元路径消息传播" aria-pressed="false">元路径消息传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免增广训练" aria-pressed="false">免增广训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练" aria-pressed="false">免训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练反思更新" aria-pressed="false">免训练反思更新<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推荐" aria-pressed="false">免训练推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="关系链对比学习" aria-pressed="false">关系链对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣演化建模" aria-pressed="false">兴趣演化建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动推荐" aria-pressed="false">冷启动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层强化学习" aria-pressed="false">分层强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层排序损失" aria-pressed="false">分层排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布式图学习" aria-pressed="false">分布式图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布鲁棒优化" aria-pressed="false">分布鲁棒优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分数生成模型" aria-pressed="false">分数生成模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="切比雪夫插值" aria-pressed="false">切比雪夫插值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前门调整" aria-pressed="false">前门调整<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态图推荐" aria-pressed="false">动态图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态步长" aria-pressed="false">动态步长<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同记忆图" aria-pressed="false">协同记忆图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单节点触发器" aria-pressed="false">单节点触发器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层图卷积" aria-pressed="false">双层图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲几何" aria-pressed="false">双曲几何<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲空间" aria-pressed="false">双曲空间<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲空间推荐" aria-pressed="false">双曲空间推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双混合对比学习" aria-pressed="false">双混合对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双画像同质性门控" aria-pressed="false">双画像同质性门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双通道图学习" aria-pressed="false">双通道图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双阶段模态融合" aria-pressed="false">双阶段模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实解释" aria-pressed="false">反事实解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性基准" aria-pressed="false">可复现性基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可靠性感知融合" aria-pressed="false">可靠性感知融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化码本" aria-pressed="false">向量量化码本<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="哈希碰撞" aria-pressed="false">哈希碰撞<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="噪声视图" aria-pressed="false">噪声视图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信号处理" aria-pressed="false">图信号处理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信息瓶颈" aria-pressed="false">图信息瓶颈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图剪枝" aria-pressed="false">图剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤" aria-pressed="false">图协同过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤量化" aria-pressed="false">图协同过滤量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积权重控制" aria-pressed="false">图卷积权重控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积溢出影响" aria-pressed="false">图卷积溢出影响<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图去噪" aria-pressed="false">图去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图后门攻击" aria-pressed="false">图后门攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图序列联合训练" aria-pressed="false">图序列联合训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图推荐" aria-pressed="false">图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图掩码自编码器" aria-pressed="false">图掩码自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图提示 (Graph Prompting)" aria-pressed="false">图提示 (Graph Prompting)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图数据增广" aria-pressed="false">图数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图特征插值" aria-pressed="false">图特征插值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经ODE" aria-pressed="false">图神经ODE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经架构搜索" aria-pressed="false">图神经架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图结构化辅助信息" aria-pressed="false">图结构化辅助信息<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图结构注入" aria-pressed="false">图结构注入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图聚合权重" aria-pressed="false">图聚合权重<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图谱结构信号" aria-pressed="false">图谱结构信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图软提示" aria-pressed="false">图软提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图重连" aria-pressed="false">图重连<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图重连稀疏化" aria-pressed="false">图重连稀疏化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="地域公平性" aria-pressed="false">地域公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量推荐" aria-pressed="false">增量推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复合图卷积" aria-pressed="false">复合图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务对比学习" aria-pressed="false">多任务对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多准则推荐" aria-pressed="false">多准则推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多利益相关方可解释推荐" aria-pressed="false">多利益相关方可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多哈希嵌入" aria-pressed="false">多哈希嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域预训练" aria-pressed="false">多域预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头自注意力" aria-pressed="false">多头自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体强化学习" aria-pressed="false">多智能体强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态/社交推荐" aria-pressed="false">多模态/社交推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态图学习" aria-pressed="false">多模态图学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多粒度注意力" aria-pressed="false">多粒度注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为对比学习" aria-pressed="false">多行为对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列建模" aria-pressed="false">多行为序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多跳语义解耦" aria-pressed="false">多跳语义解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多重图消息传递" aria-pressed="false">多重图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大规模图引擎" aria-pressed="false">大规模图引擎<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子图采样表格化" aria-pressed="false">子图采样表格化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子带图传播" aria-pressed="false">子带图传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对抗扰动不稳定性估计" aria-pressed="false">对抗扰动不稳定性估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习损失" aria-pressed="false">对比学习损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对称损失" aria-pressed="false">对称损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐与均匀性" aria-pressed="false">对齐与均匀性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小波分解" aria-pressed="false">小波分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小波包变换" aria-pressed="false">小波包变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层级信息对齐" aria-pressed="false">层级信息对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入表压缩" aria-pressed="false">嵌入表压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业级 GNN 系统" aria-pressed="false">工业级 GNN 系统<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构交互建模" aria-pressed="false">异构交互建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构图消息传递" aria-pressed="false">异构图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构图神经网络" aria-pressed="false">异构图神经网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构图表征" aria-pressed="false">异构图表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构知识图谱" aria-pressed="false">异构知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异质超图" aria-pressed="false">异质超图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响函数" aria-pressed="false">影响函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响编码器" aria-pressed="false">影响编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对表征" aria-pressed="false">成对表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散去噪" aria-pressed="false">扩散去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散增广" aria-pressed="false">扩散增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拉普拉斯位置编码" aria-pressed="false">拉普拉斯位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑剪枝" aria-pressed="false">拓扑剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续图记忆" aria-pressed="false">持续图记忆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序目标驱动架构" aria-pressed="false">排序目标驱动架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐基础模型" aria-pressed="false">推荐基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐遗忘" aria-pressed="false">推荐遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="搜索增强推荐" aria-pressed="false">搜索增强推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="教师蒸馏" aria-pressed="false">教师蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据混合增强" aria-pressed="false">数据混合增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方向性扩散" aria-pressed="false">方向性扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方面级情感聚合" aria-pressed="false">方面级情感聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方面解耦" aria-pressed="false">方面解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序子图" aria-pressed="false">时序子图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序异构图" aria-pressed="false">时序异构图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序状态演化" aria-pressed="false">时序状态演化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序邻居采样" aria-pressed="false">时序邻居采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间即算子" aria-pressed="false">时间即算子<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间引导扩散" aria-pressed="false">时间引导扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时频多分辨率建模" aria-pressed="false">时频多分辨率建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曲率感知传播深度" aria-pressed="false">曲率感知传播深度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="机器遗忘" aria-pressed="false">机器遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="架构解耦" aria-pressed="false">架构解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="根因归因" aria-pressed="false">根因归因<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="梯度反转层" aria-pressed="false">梯度反转层<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索增强表征" aria-pressed="false">检索增强表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块度图聚类" aria-pressed="false">模块度图聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型更新/遗忘" aria-pressed="false">模型更新/遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型血缘追踪" aria-pressed="false">模型血缘追踪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态融合" aria-pressed="false">模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态补全" aria-pressed="false">模态补全<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差向量量化" aria-pressed="false">残差向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差扩散" aria-pressed="false">残差扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差量化" aria-pressed="false">残差量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流形增强" aria-pressed="false">流形增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度自适应融合" aria-pressed="false">流行度自适应融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时增强" aria-pressed="false">测试时增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时自适应" aria-pressed="false">测试时自适应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合双塔架构" aria-pressed="false">混合双塔架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="渐进式提示增强" aria-pressed="false">渐进式提示增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="游玩时长建模" aria-pressed="false">游玩时长建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在扩散模型" aria-pressed="false">潜在扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="热启动" aria-pressed="false">热启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品-物品图" aria-pressed="false">物品-物品图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品互补矩阵" aria-pressed="false">物品互补矩阵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品关系挖掘" aria-pressed="false">物品关系挖掘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品演化图" aria-pressed="false">物品演化图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征互斥 HSIC" aria-pressed="false">特征互斥 HSIC<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征变换" aria-pressed="false">特征变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征频域谱滤波" aria-pressed="false">特征频域谱滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="环境条件建模" aria-pressed="false">环境条件建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="理想高通滤波" aria-pressed="false">理想高通滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式数据增强" aria-pressed="false">生成式数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户-码本图消息传递" aria-pressed="false">用户-码本图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户侧知识图谱" aria-pressed="false">用户侧知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="由粗到细检索" aria-pressed="false">由粗到细检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相似就诊检索" aria-pressed="false">相似就诊检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="相似边对 (SEP)" aria-pressed="false">相似边对 (SEP)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识利用率" aria-pressed="false">知识利用率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识剪枝" aria-pressed="false">知识剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱传播" aria-pressed="false">知识图谱传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱偏好对齐" aria-pressed="false">知识图谱偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱去噪" aria-pressed="false">知识图谱去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱增强推荐" aria-pressed="false">知识图谱增强推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱构建" aria-pressed="false">知识图谱构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱融合" aria-pressed="false">知识图谱融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识追踪" aria-pressed="false">知识追踪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交模块度最大化" aria-pressed="false">社交模块度最大化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经符号推理" aria-pressed="false">神经符号推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离散码向量量化" aria-pressed="false">离散码向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="空间翻转" aria-pressed="false">空间翻转<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="符号图结构学习" aria-pressed="false">符号图结构学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="符号边重加权" aria-pressed="false">符号边重加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精度-多样性权衡" aria-pressed="false">精度-多样性权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性加速" aria-pressed="false">线性加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组合优化约束" aria-pressed="false">组合优化约束<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组合嵌入量化" aria-pressed="false">组合嵌入量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一损失" aria-pressed="false">统一损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缺失模态填补" aria-pressed="false">缺失模态填补<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群体公平" aria-pressed="false">群体公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群体隐式兴趣" aria-pressed="false">群体隐式兴趣<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群组推荐" aria-pressed="false">群组推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对齐" aria-pressed="false">自监督对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应BPR" aria-pressed="false">自适应BPR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应对比学习" aria-pressed="false">自适应对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="营销知识图谱构建" aria-pressed="false">营销知识图谱构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="虚拟邻居增强" aria-pressed="false">虚拟邻居增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为重要性加权" aria-pressed="false">行为重要性加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征解耦" aria-pressed="false">表征解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论驱动推荐" aria-pressed="false">评论驱动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义推理" aria-pressed="false">语义推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义画像生成" aria-pressed="false">语义画像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程推荐" aria-pressed="false">课程推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图对比学习" aria-pressed="false">超图对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图神经网络" aria-pressed="false">超图神经网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超图预训练" aria-pressed="false">超图预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超复数代数" aria-pressed="false">超复数代数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域对比学习" aria-pressed="false">跨域对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域知识迁移" aria-pressed="false">跨域知识迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨类型资源推荐" aria-pressed="false">跨类型资源推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨行为对比学习" aria-pressed="false">跨行为对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="路径解释摘要" aria-pressed="false">路径解释摘要<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跳过行为建模" aria-pressed="false">跳过行为建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轻量化GNN" aria-pressed="false">轻量化GNN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="边信息融合" aria-pressed="false">边信息融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="边感知图卷积" aria-pressed="false">边感知图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="连续时间建模" aria-pressed="false">连续时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="选择性检索" aria-pressed="false">选择性检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻域对齐损失" aria-pressed="false">邻域对齐损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻居对加权" aria-pressed="false">邻居对加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻接层对比视图" aria-pressed="false">邻接层对比视图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="量化感知训练" aria-pressed="false">量化感知训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="金融图推荐" aria-pressed="false">金融图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长程协同信号" aria-pressed="false">长程协同信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="问答推荐" aria-pressed="false">问答推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="随机游走多样性" aria-pressed="false">随机游走多样性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负采样" aria-pressed="false">难负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非参数图卷积" aria-pressed="false">非参数图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称对比" aria-pressed="false">非对称对比<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非递归图表示" aria-pressed="false">非递归图表示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预测上下文采样" aria-pressed="false">预测上下文采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波去噪" aria-pressed="false">频域滤波去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频率信号缩放" aria-pressed="false">频率信号缩放<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频率感知门控融合" aria-pressed="false">频率感知门控融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频谱偏移校正" aria-pressed="false">频谱偏移校正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高阶邻域" aria-pressed="false">高阶邻域<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="鲁棒性评估" aria-pressed="false">鲁棒性评估<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;图神经架构搜索&quot;,&quot;行为重要性加权&quot;,&quot;双层优化&quot;]"><strong><a href="../../venues/AAAI/2025/cross_domain_multi_scenario/behavior_importance_aware_graph_neural_architecture_search_for_cross_domain/">Behavior Importance-Aware Graph Neural Architecture Search for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用一次性免重训超网为各域定制 GNN 架构，并以双层隐式梯度学习源域行为重要性权重，Amazon 与微信广告数据 AUC 最高提升 10.57%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;拓扑剪枝&quot;,&quot;邻域对齐损失&quot;,&quot;MLP替代GCN&quot;]"><strong><a href="../../venues/AAAI/2025/multimodal_rec/beyond_graph_convolution_multimodal_recommendation_with_topology_aware_mlps/">Beyond Graph Convolution: Multimodal Recommendation with Topology-aware MLPs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用拓扑剪枝+邻域对齐损失替代GCN消息传递，以MLP建模多模态物品关系，全库排序下Recall@10最高提升7.53%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户侧知识图谱&quot;,&quot;图掩码自编码器&quot;,&quot;跨域对比学习&quot;]"><strong><a href="../../venues/AAAI/2025/llm_rec/bridging_the_user_side_knowledge_gap_in_knowledge_aware_recommendations_wit/">Bridging the User-side Knowledge Gap in Knowledge-aware Recommendations with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 推断用户兴趣并结构化为 CIKG 异构图，配合兴趣重建与跨域对比学习去噪，在三个数据集全库排序下 R@100 最高提升 9.31%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;跳过行为建模&quot;,&quot;双层图卷积&quot;,&quot;分层排序损失&quot;]"><strong><a href="../../venues/AAAI/2025/multimodal_rec/exploiting_fine_grained_skip_behaviors_for_micro_video_recommendation/">Exploiting Fine-Grained Skip Behaviors for Micro-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将短视频跳过行为细分为高正/低正/负三级，构建双层正图与分层 BPR 损失，两数据集八项指标全面超越 FRAME 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;残差扩散&quot;,&quot;捆绑推荐&quot;]"><strong><a href="../../venues/AAAI/2026/generative_rec/modeling_item_level_dynamic_variability_with_residual_diffusion_for_bundle_/">Modeling Item-Level Dynamic Variability with Residual Diffusion for Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差扩散在潜空间重建捆绑包item级表征，模型无关地适配B-I归属动态变化，Recall/NDCG最高提升约23%，训练开销仅增4%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;动态图推荐&quot;,&quot;任务感知子图检索&quot;]"><strong><a href="../../venues/AAAI/2026/graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;Knowledge Graph RAG&quot;,&quot;Multi-LoRA&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/kerl_knowledge_enhanced_personalized_recipe_recommendation_using_large_lang/">KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 FoodKG 子图作上下文 + 多 LoRA 适配器，统一完成约束食谱推荐、烹饪步骤生成与微量营养素估计，KGQA 基准 F1 达 0.973。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;KG-RAG&quot;,&quot;图软提示&quot;,&quot;选择性检索&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/knowledge_graph_retrieval_augmented_generation_for_llm_based_recommendation/">Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用跳场知识子图索引+流行度选择性检索+GNN软提示，将KG结构知识注入冻结LLM，MovieLens-1M ACC 相对提升58.6%且推理仅慢0.14s</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026 Findings&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;重排&quot;,&quot;持续图记忆&quot;,&quot;测试时自适应&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/graph_rec/continual_graph_memory_for_adaptive_recommendation_under_intent_drift/">Continual Graph Memory for Adaptive Recommendation under Intent Drift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将知识图谱视为可写双记忆（语义图记忆+情景教训记忆），冻结参数下仅靠质量门控图编辑实现意图漂移自适应重排。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;LoRA 参数高效微调&quot;,&quot;图消息传递&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/graphlora_structure_aware_low_rank_adaptation_for_large_language_model_reco/">GraphLoRA: Structure-Aware Low-Rank Adaptation for Large Language Model Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将可训练 GNN 消息传递嵌入 LoRA 瓶颈（A 与 B 之间），让协同拓扑在参数空间传播；在 ML-1M/Amazon-Book 上以 r=8 的极小参数预算超越 CoRA 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;LLM推荐&quot;,&quot;重排&quot;,&quot;协同记忆图&quot;,&quot;架构解耦&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/memrec_collaborative_memory_augmented_agentic_recommender_system/">MemRec: Collaborative Memory-Augmented Agentic Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将记忆管理从推理中架构解耦，用轻量 LM_Mem 在协同记忆图上做零样本规则剪枝与异步传播，仅向重型 LLM_Rec 注入蒸馏后的高信号协同上下文，四基准全库排序 H@1 最高 +28.98%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;游玩时长建模&quot;,&quot;随机游走多样性&quot;]"><strong><a href="../../venues/ACM_MM/2025/multimodal_rec/dual_phase_playtime_guided_recommendation_interest_intensity_exploration_an/">Dual-Phase Playtime-guided Recommendation: Interest Intensity Exploration and Multimodal Random Walks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双 Beta 分布解耦游玩时长强弱兴趣，并以多模态随机游走做跨类目探索，在 Steam 数据上同时提升准确率与覆盖度。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;物品-物品图&quot;,&quot;三元闭包重加权&quot;]"><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/one_graph_multiple_gains_single_high_quality_item_item_graph_for_multimodal/">One Graph, Multiple Gains: Single High-Quality Item-Item Graph for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建一次高质量物品-物品图，经三元闭包重加权后复用于表征、交互图与优化三阶段，四数据集全指标SOTA且训练更快更省显存。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤量化&quot;,&quot;量化感知训练&quot;,&quot;动态步长&quot;]"><strong><a href="../../venues/CIKM/2025/collaborative_filtering/a_node_aware_dynamic_quantization_approach_for_graph_collaborative_filterin/">A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出节点感知动态量化 GNAQ，用参数共享学习每节点量化步长并以图消息传递更新，2-bit 下 Recall@10 平均提升 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;搜索增强推荐&quot;,&quot;向量量化码本&quot;,&quot;用户-码本图消息传递&quot;]"><strong><a href="../../venues/CIKM/2025/cross_domain_multi_scenario/benefit_from_rich_tackling_search_interaction_sparsity_in_search_enhanced_r/">Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM+RQ-VAE 将用户搜索/推荐偏好离散为码本，构建 User-Code 图做消息传递，把搜索富用户信号传给稀疏用户，Qilin 上 HR@1 相对 UniSAR 提升 12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果推断&quot;,&quot;图聚合权重&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/causality_aware_graph_aggregation_weight_estimator_for_popularity_debiasing/">Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将图聚合证明为后门调整，用 VAE 式编码器-解码器估计无偏聚合权重，配合动量更新，小众物品 Recall@20 提升约 20%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;社交模块度最大化&quot;,&quot;特征互斥 HSIC&quot;]"><strong><a href="../../venues/CIKM/2025/social_group_rec/community_aware_social_community_recommendation/">Community-Aware Social Community Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 CASO，用模块度最大化、社交紧密度聚合与协同编码三路解耦用户表征，并以 HSIC 互斥与 KL 社区检测损失优化，在 6 个真实社交网络上 Full-Ranking 全面超越 9 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;增量推荐&quot;,&quot;时序状态演化&quot;,&quot;伪多任务学习&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/cpmr_context_aware_incremental_sequential_recommendation_with_pseudo_multi_/">CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出伪多任务学习范式，用历史图与上下文图双通道演化用户/物品时序状态，在四个数据集上 MRR 平均提升 30.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Fourier KAN&quot;,&quot;特征变换&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/enhancing_graph_collaborative_filtering_with_fourierkan_feature_transformat/">Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Fourier KAN 替代 NGCF 中被误删的交互特征变换 W2，在 7:1:2 时间切分全库排序下 R@20 最高提升 7.8%，且可作即插即用骨干。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;生成式数据增强&quot;,&quot;物品互补矩阵&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;超复数代数&quot;,&quot;Prompt 补偿&quot;,&quot;自监督对齐&quot;]"><strong><a href="../../venues/CIKM/2025/multimodal_rec/hypercomplex_prompt_aware_multimodal_recommendation/">Hypercomplex Prompt-aware Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Cayley-Dickson 超复数多分量嵌入 + 超复数乘法非线性融合 + Prompt 动态补偿，在四个 Amazon 数据集 Full-Ranking 下 Recall@20 最高提升约 13%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;图数据增广&quot;,&quot;HSIC 去偏&quot;,&quot;群体公平&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/improving_recommendation_fairness_via_graph_structure_and_representation_au/">Improving Recommendation Fairness via Graph Structure and Representation Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双先验假设定位敏感边与敏感特征，做图剪枝+特征掩码双增广，HSIC去偏+对比学习保效用，ML-1M 上 DP@10 降 3.4%、EO@10 降 7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;知识图谱去噪&quot;,&quot;流行度自适应融合&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/spark_adaptive_low_rank_knowledge_graph_modeling_in_hybrid_geometric_spaces/">SPARK: Adaptive Low-Rank Knowledge Graph Modeling in Hybrid Geometric Spaces for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 低秩去噪 KG、SVD 初始化双曲/欧氏并行 GNN，并按物品流行度自适应融合多源信号，长尾推荐 R@10 最高提升 10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Folding-In&quot;,&quot;热启动&quot;,&quot;UltraGCN&quot;,&quot;低秩近似&quot;]"><strong><a href="../../venues/CIKM/2025/graph_rec/ultra_fast_warm_start_solution_for_graph_recommendations/">Ultra Fast Warm Start Solution for Graph Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 SVD Folding-In 的闭式解迁移到 UltraGCN，用一次矩阵-向量乘精确更新热用户嵌入，比 SGD 微调快 30 倍且质量更优、覆盖率更高。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;图重连&quot;,&quot;环境条件建模&quot;]"><strong><a href="../../venues/CIKM/2026/graph_rec/beyond_observed_auxiliary_relations_environment_conditioned_modeling_for_mu/">Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以辅助行为可观测性为环境条件，用去偏稠密化与目标引导剪枝双模块重连多行为图，隐藏目标物品 HR@10 最高提升 44.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱融合&quot;,&quot;对抗扰动不稳定性估计&quot;,&quot;自适应对比学习&quot;]"><strong><a href="../../venues/CIKM/2026/graph_rec/do_all_nodes_benefit_equally_from_knowledge_graphs_adaptive_node_aware_kg_f/">Do All Nodes Benefit Equally from Knowledge Graphs? Adaptive Node-Aware KG Fusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以对抗扰动估计节点级 CF 不稳定性，自适应门控 KG 注入强度并配合自适应对比对齐，四数据集 Full-Ranking 平均排名第一。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;拉普拉斯位置编码&quot;,&quot;图谱结构信号&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/enriching_sequential_recommendation_with_graph_laplacian_positional_embeddi/">Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用物品共现图归一化拉普拉斯特征向量冻结替换 SASRec 可学习位置嵌入，四基准多数指标提升且在线开销近零。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;曲率感知传播深度&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/CIKM/2026/graph_rec/neural_tree_collaborative_filtering_rethinking_graph_collaborative_filterin/">Neural Tree Collaborative Filtering: Rethinking Graph Collaborative Filtering as Tree Collaborative Filtering with Curvature-Aware Propagation Depth</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以局部度不平衡曲率代理为每节点分配自适应传播深度，将 GCF 重释为节点根树集成，Full-Ranking 下 R@20 稳定超越 LightGCN 与 SSL 骨干。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;多哈希嵌入&quot;,&quot;时序邻居采样&quot;]"><strong><a href="../../venues/CIKM/2026/social_group_rec/scaling_graph_neural_networks_for_friend_recommendation_multi_hash_user_emb/">Scaling Graph Neural Networks for Friend Recommendation: Multi-Hash User Embeddings and Temporal Neighbor Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用多哈希ID嵌入(表压缩&gt;98%)与时间戳排序CSR二分采样(O(log d+K))支撑194M用户/28B边GNN好友排序，线上好友添加+16%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;知识图谱路径推理&quot;,&quot;残差量化&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/sprig_semantic_id_enhanced_paths_for_knowledge_graph_based_generative_recom/">SPRIG: Semantic-ID-enhanced Paths for Knowledge Graph-based Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将内容派生的分层语义 ID 注入知识图谱路径语言建模，以 768 个共享码替代逐物品词表，nDCG@10 较 KGGLM 提升 44%/58% 且参数与算力更低。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;表征解耦&quot;,&quot;梯度反转层&quot;,&quot;跨域知识迁移&quot;]"><strong><a href="../../venues/DASFAA/2025/cross_domain_multi_scenario/a_unified_framework_for_adaptive_representation_enhancement_and_inversed_le/">A Unified Framework for Adaptive Representation Enhancement and Inversed Learning in Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>解耦共享/专属表征，用 LightGCN 域内增强 + 自注意力跨域自适应融合 + GRL 逆向约束，全库排序下 Recall@20 最高提升 8.31%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;超图预训练&quot;,&quot;边信息融合&quot;,&quot;冷启动推荐&quot;]"><strong><a href="../../venues/DASFAA/2025/graph_rec/genet_unleashing_the_power_of_side_information_for_recommendation_via_hyper/">GENET: Unleashing the Power of Side Information for Recommendation via Hypergraph Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将异构边信息统一构建为超图，通过超边预测与双层自监督对比预训练，再以 LightGCN/GRU 微调，Top-N 与序列推荐最高提升 38%，冷启动提升 175%+。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;异构知识图谱&quot;,&quot;多利益相关方可解释推荐&quot;,&quot;地域公平性&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/okra_an_explainable_heterogeneous_multi_stakeholder_job_recommender_system/">OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>异构知识图谱+多头GATv2构建求职者/企业双侧嵌入，谐波均值融合打分，nDCG@10较六基线最高提升约18%，并揭示城乡地域偏差。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;异构交互建模&quot;,&quot;多头自注意力&quot;,&quot;预测上下文采样&quot;]"><strong><a href="../../venues/ICDE/2025/cold_start_rec/all_in_one_heterogeneous_interaction_modeling_for_cold_start_rating_predict/">All-in-One: Heterogeneous Interaction Modeling for Cold-Start Rating Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HIRE 用纯 MHSA 在采样的预测上下文上端到端学习用户/物品/属性三层异构交互，三种冷启动场景下 Precision/NDCG/MAP 平均超 12 个基线 0.21/0.29/0.22。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;营销知识图谱构建&quot;,&quot;渐进式提示增强&quot;,&quot;教师蒸馏&quot;]"><strong><a href="../../venues/ICDE/2025/llm_rec/making_large_language_models_better_knowledge_miners_for_online_marketing_w/">Making Large Language Models Better Knowledge Miners for Online Marketing with Progressive Prompting Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用渐进式提示增强把 LLM 变成可控的营销知识图谱挖掘器，并以教师蒸馏出可部署的 LightPAIR，受众定向覆盖提升 15%–122%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;重排&quot;,&quot;可解释推荐&quot;,&quot;Steiner Tree&quot;,&quot;路径解释摘要&quot;]"><strong><a href="../../venues/ICDE/2025/graph_rec/path_based_summary_explanations_for_graph_recommenders_extended_version/">Path-based summary explanations for graph recommenders (extended version)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次提出路径解释的摘要化：用 Steiner Tree 与 Prize-Collecting Steiner Tree 将多条用户-物品解释路径压缩为连通子图，在可理解性、多样性、隐私与可扩展性上全面优于逐条路径基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;成对表征&quot;,&quot;混合双塔架构&quot;,&quot;时序异构图&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/graph_rec/contextgnn_beyond_two_tower_recommendation_systems/">ContextGNN: Beyond Two-Tower Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出局部成对表征+全局双塔回退的混合GNN架构，在RelBench十个任务上平均MAP较最强成对基线提升20%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;生成式推荐&quot;,&quot;Neural ODE&quot;,&quot;知识图谱偏好对齐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/spot_trip_dual_preference_driven_out_of_town_trip_recommendation/">SPOT-Trip: Dual-Preference Driven Out-of-Town Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次形式化异地行程推荐，用知识图谱静态偏好+Neural ODE动态偏好解耦建模，F1最高提升17.01%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;互补信号&quot;,&quot;流形增强&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multimodal_rec/can_complementary_signals_bridge_similarity_islands_manifold_augmented_grap/">Can Complementary Signals Bridge Similarity Islands? Manifold-Augmented Graph Embedding for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 MAGE，用离线 Soft Conceptor 残差注入 + 在线非对称双流 GNN 桥接互补品“相似性孤岛”，四个 Amazon 数据集 Full-Ranking 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图信息瓶颈&quot;,&quot;跨行为对比学习&quot;,&quot;多行为推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/gcib_graph_contrastive_information_bottleneck_for_multi_behavior_recommenda/">GCIB: Graph Contrastive Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用图信息瓶颈在结构层剪枝辅助行为噪声边、用跨行为 InfoNCE 在特征层对齐目标语义，Leave-One-Out 全库排序下 NDCG@10 最高提升 37.2%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双画像同质性门控&quot;,&quot;免训练反思更新&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/harp_training_free_dual_profile_agentic_communication_for_llm_based_recomme/">HARP: Training-Free Dual-Profile Agentic Communication for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 推荐重构为四阶段智能体通信，用双画像、同质性门控检索与单次反思三个免训练算子补齐缺失环节，Hit@1 相对提升 6.9%–24.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;低通谱滤波&quot;,&quot;流行度去偏&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/debias_fairness_industrial/low_pass_personalized_subgraph_federated_recommendation/">Low-pass Personalized Subgraph Federated Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用低通谱滤波提取跨子图稳定的结构信号做个性化联邦聚合，并以局部流行度偏差感知 margin 抑制长尾偏置，五数据集 NDCG@20 最高提升 24.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图协同过滤&quot;,&quot;对比学习损失&quot;,&quot;邻居对加权&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/rethinking_contrastive_learning_for_graph_collaborative_filtering_limitatio/">Rethinking Contrastive Learning for Graph Collaborative Filtering: Limitations and a Simple Remedy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>展开 GCF 预测为邻居对加权聚合，揭示 SSM 仅按物品侧相似度更新且忽略邻居对类型，提出类型感知 NT-SSM，NDCG@20 最高提升 21.45%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;图后门攻击&quot;,&quot;单节点触发器&quot;,&quot;LightGCN 鲁棒性&quot;]"><strong><a href="../../venues/IJCAI/2025/debias_fairness_industrial/single_node_trigger_backdoor_attacks_in_graph_based_recommendation_systems/">Single-Node Trigger Backdoor Attacks in Graph-Based Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出单节点触发器后门攻击 SNT-BA，仅注入 1 个虚假用户即可让目标物品在 99% 目标用户 Top-10 中曝光，同时 HR@20 仅降约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;上下文老虎机&quot;,&quot;知识追踪&quot;,&quot;课程推荐&quot;]"><strong><a href="../../venues/IJCAI/2026/rl_bandit_rec/pace_a_personalized_adaptive_curriculum_engine_for_9_1_1_call_taker_trainin/">PACE: A Personalized Adaptive Curriculum Engine for 9-1-1 Call-taker Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用技能图信念传播+个体学习/遗忘动力学+Thompson Sampling 上下文老虎机为 9-1-1 调度员推荐训练场景，Z2H 缩短 19.50%、终局掌握度提升 10.95%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;LLM 自动构图&quot;,&quot;残差向量量化&quot;,&quot;元路径消息传播&quot;]"><strong><a href="../../venues/KDD/2025/graph_rec/an_automatic_graph_construction_framework_based_on_large_language_models_fo/">An Automatic Graph Construction Framework based on Large Language Models for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 逐点生成语义向量 + 残差量化抽取隐因子作额外节点，O(N) 构图并设计元路径聚合，Full-Ranking 下 NDCG@10 最高提升 43.57%，华为广告 A/B 提升 RPM 2.69%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;频谱偏移校正&quot;,&quot;图结构化辅助信息&quot;,&quot;多模态/社交推荐&quot;]"><strong><a href="../../venues/KDD/2025/graph_rec/collaborative_filtering_meets_spectrum_shift_connecting_user_item_interacti/">Collaborative Filtering Meets Spectrum Shift: Connecting User-Item Interaction with Graph-Structured Side Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示增广图频谱右移现象，提出仅含平移/缩放两因子的 SSC 校正，即插即用提升 LightGCN/JGCF 最高 23%，零额外开销。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;组合优化约束&quot;,&quot;负迁移抑制&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/combinatorial_optimization_perspective_based_framework_for_multi_behavior_r/">Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多行为融合视为组合优化问题施加三阶段约束，并用分布拟合多专家网络抑制负迁移，三数据集 HR@10 最高提升 49.91%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;离散码向量量化&quot;,&quot;虚拟邻居增强&quot;]"><strong><a href="../../venues/KDD/2025/self_supervised_denoising/enhancing_graph_contrastive_learning_with_reliable_and_informative_augmenta/">Enhancing Graph Contrastive Learning with Reliable and Informative Augmentation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用端到端多级向量量化学习离散码，将码作为虚拟邻居与语义正样本构建三重对比视图，Full-Ranking 下 NDCG@10 最高提升 17.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;分布式图学习&quot;,&quot;子图采样表格化&quot;,&quot;工业级 GNN 系统&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/gigl_large_scale_graph_neural_networks_at_snapchat/">GiGL: Large-Scale Graph Neural Networks at Snapchat</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Snapchat 开源十亿级图学习库 GiGL，以表格化子图采样+PyG 解耦实现 900M 节点/16.8B 边图 12 小时内端到端训练推理，支撑 35+ 线上发布。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱推荐&quot;,&quot;轻量化GNN&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/KDD/2025/graph_rec/lightkg_efficient_knowledge_aware_recommendations_with_simplified_gnn_archi/">LightKG: Efficient Knowledge-Aware Recommendations with Simplified GNN Architecture</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用标量对编码关系+线性聚合+免子图对比层，在稀疏场景超越12个KGRS基线5.8%并省84.3%训练时间</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Pareto 多目标优化&quot;,&quot;符号图结构学习&quot;,&quot;LLM 知识注入解释&quot;]"><strong><a href="../../venues/KDD/2025/multi_task_reranking/mopi_hfrs_a_multi_objective_personalized_health_aware_food_recommendation_s/">MOPI-HFRS: A Multi-objective Personalized Health-aware Food Recommendation System with LLM-enhanced Interpretation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建首个融合 NHANES 医疗数据的健康感知食物推荐基准，用符号图结构学习 + Pareto 多目标优化平衡偏好/个性化健康/营养多样性，并以知识注入 LLM 生成可解释理由。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;交互级依赖图&quot;,&quot;多粒度注意力&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/multi_grained_preference_enhanced_transformer_for_multi_behavior_sequential/">Multi-Grained Preference Enhanced Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>M-GPT 用交互级图卷积建模跨类型行为依赖，并以多粒度多尺度注意力抽取行为感知偏好，Leave-One-Out 全库排序下 HR@5 相对提升 12.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;异构图表征&quot;,&quot;多任务对比学习&quot;,&quot;大规模图引擎&quot;]"><strong><a href="../../venues/KDD/2025/self_supervised_denoising/omnisage_large_scale_multi_entity_heterogeneous_graph_representation_learni/">OmniSage: Large Scale, Multi-Entity Heterogeneous Graph Representation Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>统一异构图 GNN、内容特征与用户序列的多任务对比学习框架，支撑 Pinterest 十亿级实体通用表征，线上全站 Repin 提升约 2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图神经ODE&quot;,&quot;时间引导扩散&quot;,&quot;物品演化图&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/time_matters_enhancing_sequential_recommendations_with_time_guided_graph_ne/">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间引导扩散补全稀疏用户时间图、图神经ODE对齐用户兴趣与物品演化，五数据集Recall@20最高提升46%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;LLM推荐&quot;,&quot;生成式推荐&quot;,&quot;方面解耦&quot;,&quot;多重图消息传递&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/copersona_collaborative_persona_graphs_for_robust_llm_personalization/">CoPersona: Collaborative Persona Graphs for Robust LLM Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用方面级多重人物图+双分支推理补全稀疏用户画像，Amazon六域生成式评测全面超越DEP/DPL。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;特征频域谱滤波&quot;,&quot;可靠性感知融合&quot;]"><strong><a href="../../venues/KDD/2026/self_supervised_denoising/dynamic_spectral_denoising_with_global_context_attention_for_multi_behavior/">Dynamic Spectral Denoising with Global-Context Attention for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>沿特征维做 RFFT 学习视图自适应谱调制净化表征，再以净化全局锚做可靠性感知注意力融合，全库排序下 Taobao HR@10 相对提升 26.17%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;语义画像生成&quot;,&quot;知识图谱传播&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/enriching_semantic_profiles_into_knowledge_graph_for_recommender_systems_us/">Enriching Semantic Profiles into Knowledge Graph for Recommender Systems Using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 为 KG 全实体生成语义画像，以可逆加性注入+成对匹配对齐，Full-Ranking 下 Recall@40 最高提升 2.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;频率信号缩放&quot;,&quot;空间翻转&quot;]"><strong><a href="../../venues/KDD/2026/graph_rec/how_do_graph_signals_affect_recommendation_unveiling_the_mystery_of_low_and/">How Do Graph Signals Affect Recommendation: Unveiling the Mystery of Low and High-Frequency Graph Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>理论证明同绝对值低频/高频图信号在推荐中等价，提出频率信号缩放器与空间翻转插件，在四个数据集全库排序下 Recall@20 最高提升约 3.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;KG-RAG&quot;,&quot;MoE&quot;,&quot;多智能体强化学习&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/mixture_of_experts_knowledge_graph_retrieval_augmented_generation_for_multi/">Mixture-of-Experts Knowledge Graph Retrieval-Augmented Generation for Multi-Agent LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 多粒度 KG 检索智能体按查询复杂度路由检索粒度，配合知识对齐与对比偏好优化，在 20 候选池协议下 Accuracy 相对提升 5.8%–20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;信息瓶颈&quot;,&quot;多行为推荐&quot;,&quot;图去噪&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;邻接层对比视图&quot;,&quot;LightGCN&quot;]"><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;非参数图卷积&quot;,&quot;CTR 精排&quot;]"><strong><a href="../../venues/RecSys/2025/multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;异构图神经网络&quot;,&quot;对比学习&quot;,&quot;推荐基础模型&quot;]"><strong><a href="../../venues/RecSys/2025/cross_domain_multi_scenario/rankgraph_unified_heterogeneous_graph_learning_for_cross_domain_recommendat/">RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建跨产品异构异质图，用 GPU 加速 RGCN + 对比学习产出图 token 注入推荐基础模型，线上 CTR +0.92%、CVR +2.82%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;Ricci 曲率&quot;,&quot;根因归因&quot;,&quot;金融图推荐&quot;]"><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/ricciflowrec_a_geometric_root_cause_recommender_using_ricci_curvature_on_fi/">RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Ollivier/Forman-Ricci 曲率与 Ricci 流在动态金融图上做根因归因，以曲率风险惩罚重排资产，NDCG@10 提升 4.9%、Top-10 波动降 28%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;统一损失&quot;,&quot;免增广训练&quot;]"><strong><a href="../../venues/RecSys/2025/self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;位置偏差控制函数&quot;,&quot;流行度去偏&quot;,&quot;图消息传递&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/unbiased_recommender_systems_with_implicit_feedback/">Unbiased Recommender Systems with Implicit Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>博士论文愿景：以控制函数法纠偏排序位置偏差、以事后投影与消息传递加权纠偏 GNN 协同过滤的流行度偏差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;图提示 (Graph Prompting)&quot;,&quot;负迁移抑制&quot;,&quot;多域预训练&quot;]"><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/adaptive_graph_integration_for_cross_domain_recommendation_via_heterogeneou/">Adaptive Graph Integration for Cross-Domain Recommendation via Heterogeneous Graph Coordinators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用异质自适应图协调器把多域图拼成统一结构做多域预训练，再以图提示迁移到目标域，Full-Ranking 下全面超越 9 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱构建&quot;,&quot;跨类型资源推荐&quot;,&quot;模型血缘追踪&quot;]"><strong><a href="../../venues/SIGIR/2025/graph_rec/benchmarking_recommendation_classification_and_tracing_based_on_hugging_fac/">Benchmarking Recommendation, Classification, and Tracing Based on Hugging Face Knowledge Graph</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 260 万节点 HuggingKG 与三任务 HuggingBench，揭示稀疏领域 KG 上 KGRec 类方法失效、TransE 反超复杂模型的关键现象。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;双阶段模态融合&quot;,&quot;复合图卷积&quot;,&quot;自适应BPR&quot;]"><strong><a href="../../venues/SIGIR/2025/multimodal_rec/cohesion_composite_graph_convolutional_network_with_dual_stage_fusion_for_m/">COHESION: Composite Graph Convolutional Network with Dual-Stage Fusion for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用行为模态早期精炼+晚期注意力融合的双阶段融合，配合异质/同质复合图与自适应BPR，在三个 Amazon 数据集上 R@20 相对提升 4.4%–6.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;知识图谱增强推荐&quot;,&quot;LLM 语义嵌入&quot;,&quot;检索增强表征&quot;]"><strong><a href="../../venues/SIGIR/2025/graph_rec/comprehending_knowledge_graphs_with_large_language_models_for_recommender_s/">Comprehending Knowledge Graphs with Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 理解 KG 局部子图并构建语义物品图做全局检索增强，离线预计算语义嵌入后接入 LightGCN，全库排序下全面超越 12 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;由粗到细检索&quot;,&quot;子图检索&quot;,&quot;LLM推理&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/corona_a_coarse_to_fine_framework_for_graph_based_recommendation_with_large/">CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 做偏好/意图推理在交互图上逐级缩小候选子图，再由 GNN 精排，全库排序下 Recall 平均相对提升 18.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;谱滤波&quot;,&quot;切比雪夫插值&quot;]"><strong><a href="../../venues/SIGIR/2025/collaborative_filtering/graph_spectral_filtering_with_chebyshev_interpolation_for_recommendation/">Graph Spectral Filtering with Chebyshev Interpolation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用切比雪夫插值逼近非线性图滤波器，直接以原始交互为信号，突破 LightGCN 的截断线性低通瓶颈，Full-Ranking 下全面 SOTA 且推理快 8.1 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;双曲空间推荐&quot;,&quot;元优化两阶段训练&quot;,&quot;层级信息对齐&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/large_language_models_enhanced_hyperbolic_space_recommender_systems/">Large Language Models Enhanced Hyperbolic Space Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>HyperLLM 用 LLM 生成层级标签与偏好摘要，经元优化两阶段训练将结构/语义层级信息注入双曲推荐，Recall@20 最高提升 41.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;机器遗忘&quot;,&quot;图推荐&quot;,&quot;影响编码器&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/pre_training_for_recommendation_unlearning/">Pre-training for Recommendation Unlearning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出预训练式影响编码器 UnlearnRec，将遗忘请求映射为嵌入漂移，免重训实现高效图推荐遗忘，10x 加速且保持精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-ICTIR 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;相似边对 (SEP)&quot;,&quot;边感知图卷积&quot;]"><strong><a href="../../venues/SIGIR/2025/poi_spatiotemporal_rec/sep_gcn_leveraging_similar_edge_pairs_with_temporal_and_spatial_contexts_fo/">SEP-GCN: Leveraging Similar Edge Pairs with Temporal and Spatial Contexts for Location-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以时空相似交互边对构建边级相似图并注入LightGCN传播，在NYC/Gowalla/Brightkite全库排序上Recall@20最高提升14.89%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;SENet特征重加权&quot;]"><strong><a href="../../venues/SIGIR/2025/self_supervised_denoising/squeeze_and_excitation_a_weighted_graph_contrastive_learning_for_collaborat/">Squeeze and Excitation: A Weighted Graph Contrastive Learning for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>仅扰动末层GCN视图并用SENet动态加权特征，在三个数据集上实现最高24.45%的NDCG@20提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Customer Persona&quot;,&quot;Random Walk Approximation&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/you_are_what_you_bought_generating_customer_personas_for_e_commerce_applica/">You Are What You Bought: Generating Customer Personas for E-commerce Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 标注少量原型用户、随机游走传播生成可读人物画像，并以 RevAff 近似加速，画像注入 GCN 推荐最高提升 12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Aspect-Conditioned Scoring&quot;,&quot;异构图消息传递&quot;,&quot;作者谱系&quot;]"><strong><a href="../../venues/SIGIR/2026/graph_rec/aspect_aware_content_based_recommendations_for_mathematical_research_papers/">Aspect-Aware Content-Based Recommendations for Mathematical Research Papers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向数学论文的 aspect 条件异构图推荐，构建 GoldRiM/SilverRiM 数据集，AchGNN 在 Full-Ranking 下 MRR 相对 GraphCL 提升 26%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双通道图学习&quot;,&quot;频率感知门控融合&quot;]"><strong><a href="../../venues/SIGIR/2026/graph_rec/dcgl_dual_channel_graph_learning_with_large_language_models_for_knowledge_a/">DCGL: Dual-Channel Graph Learning with Large Language Models for Knowledge-Aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双通道解耦 LLM 语义与 ID 行为信号，多级对比对齐 + 频率感知门控融合，四数据集全库排序全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;反事实解释&quot;,&quot;可复现性基准&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/from_top_1_to_top_k_a_reproducibility_study_and_benchmarking_of_counterfact/">From Top-1 to Top-K: A Reproducibility Study and Benchmarking of Counterfactual Explanations for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一复现11种反事实解释器，首次提出列表级Top-K评估协议与PN-R位置敏感指标，揭示显式格式下有效性-稀疏性权衡与图解释器可扩展性瓶颈。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;问答推荐&quot;,&quot;多项式图滤波&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2026/graph_rec/qagcf_graph_collaborative_filtering_for_q_a_recommendation/">QAGCF: Graph Collaborative Filtering for Q&amp;A Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将问答对解耦为用户-问题/用户-答案双协同图与问题-答案/问题-问题双语义图，融合为全局异质图后用 Jacobi 多项式图滤波器分离高低频信号，配合噪声对比学习，在知乎与商业数据集 Full-Ranking 下 Recall@10 提升 5.16%–5.34%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态补全&quot;,&quot;子图检索&quot;,&quot;图 Transformer&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/robust_multimodal_recommendation_via_graph_retrieval_enhanced_modality_comp/">Robust Multimodal Recommendation via Graph Retrieval-Enhanced Modality Completion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态感知子图检索替代 K 跳邻居聚合，配合图 Transformer 与稀疏路由码本补全缺失模态，40% 缺失率下 Recall@20 相对提升 4.2%–5.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;时间即算子&quot;,&quot;谱滤波&quot;]"><strong><a href="../../venues/SIGIR/2026/multimodal_rec/timemm_time_as_operator_spectral_filtering_for_dynamic_multimodal_recommend/">TimeMM: Time-as-Operator Spectral Filtering for Dynamic Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将交互新近度映射为时间核加权图算子族，隐式实现免特征分解的多尺度谱滤波，配合自适应算子混合与模态路由，全库排序下 R@20 相对提升 3%–10%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;小波包变换&quot;,&quot;子带图传播&quot;,&quot;时频多分辨率建模&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，在子带内做 Chebyshev 图传播并以能量-谱平坦度门控融合，Full-Ranking 下稀疏数据集 HR@20 最高 +15.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;关系链对比学习&quot;]"><strong><a href="../../venues/TKDE/2025/graph_rec/dual_channel_multiplex_graph_neural_networks_for_recommendation/">Dual-Channel Multiplex Graph Neural Networks for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双通道多路图网络显式建模行为模式、隐式挖掘关系链依赖，三数据集全指标平均提升 10%+</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;知识图谱路径推理&quot;,&quot;分层强化学习&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/explainable_session_based_recommendation_via_path_reasoning/">Explainable Session-based Recommendation via Path Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出分层强化学习框架 PR4SR，用会话级智能体选起点、路径级智能体做知识图谱推理，配合多目标与路径中点奖励，为 5 种 SR 模型提供可解释路径并提升精度。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;交叉相关聚合&quot;,&quot;非递归图表示&quot;,&quot;多跳语义解耦&quot;]"><strong><a href="../../venues/TKDE/2025/graph_rec/graph_cross_correlated_network_for_recommendation/">Graph Cross-Correlated Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用非递归 PGR 保留各跳邻居纯语义，再以 HCC/ECC 显式建模用户-物品子图交叉相关项，全库排序下 NDCG@20 平均提升约 1.8%–2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;元路径意图建模&quot;,&quot;信息瓶颈去噪&quot;]"><strong><a href="../../venues/TKDE/2025/graph_rec/intent_guided_heterogeneous_graph_contrastive_learning_for_recommendation/">Intent-guided Heterogeneous Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用元路径意图构建双对比学习视图，并以信息瓶颈掩码自编码器去噪，六数据集全库排序 Recall@20 最高提升 12.4%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Krylov 子空间&quot;,&quot;多项式图滤波&quot;,&quot;免训练推荐&quot;]"><strong><a href="../../venues/TKDE/2026/collaborative_filtering/memory_is_no_longer_a_bottleneck_memory_efficient_graph_filtering_for_scala/">Memory Is No Longer a Bottleneck: Memory-Efficient Graph Filtering for Scalable Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用逐用户 Krylov 子空间替代全量物品相似度图，实现免训练多项式图滤波，内存降 5.74×、推理加速 26× 且精度反超 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;小波分解&quot;,&quot;模态融合&quot;]"><strong><a href="../../venues/TKDE/2026/multimodal_rec/tackling_the_modality_inhibitance_effect_with_wavelet_based_modality_integr/">Tackling the Modality Inhibitance Effect with Wavelet-based Modality Integration in Multi-modal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模态抑制效应(MIE)概念，用Haar小波多分辨率分解提取各模态本质特征、结构感知多模态图学习与交叉加权自适应调节器，在4数据集Full-Ranking下平均提升6.48%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图特征插值&quot;,&quot;缺失模态填补&quot;]"><strong><a href="../../venues/TKDE/2026/multimodal_rec/training_free_graph_based_imputation_of_missing_modalities_in_multimodal_re/">Training-free Graph-based Imputation of Missing Modalities in Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将缺失模态重铸为物品-物品共购图上的图特征插值，提出四种免训练传播式填补算子，可即插即用并保持甚至扩大多模态与传统推荐的性能差距。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;评测基准&quot;,&quot;知识图谱推荐&quot;,&quot;知识利用率&quot;,&quot;鲁棒性评估&quot;]"><strong><a href="../../venues/TOIS/2025/graph_rec/kg4receval_does_knowledge_graph_really_matter_for_recommender_systems/">KG4RecEval: Does Knowledge Graph Really Matter for Recommender Systems?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 KGER 指标与 KG4RecEval 框架，系统扰动/删除 KG 知识，发现 KG 对推荐精度贡献常为负或近零，挑战 KG 必有益的主流认知。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;组合嵌入量化&quot;,&quot;图重连稀疏化&quot;]"><strong><a href="../../venues/TOIS/2025/collaborative_filtering/lightweight_embeddings_with_graph_rewiring_for_collaborative_filtering/">Lightweight Embeddings with Graph Rewiring for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>量化组合码本+图重连剪枝，在严格存储预算下兼顾 GNN 推荐精度与边缘端推理效率，iFashion 存储降 97.7%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Graph Transformer&quot;,&quot;位置编码&quot;,&quot;长程协同信号&quot;]"><strong><a href="../../venues/TOIS/2025/graph_rec/position_aware_graph_transformer_for_recommendation/">Position-aware Graph Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用四种为交互图定制的节点位置编码把 Transformer 全局建模能力注入 GCN 推荐骨干，在四个数据集上以 Full-Ranking 协议取得最高 27.69% 的 Recall@20 相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;噪声视图&quot;,&quot;对称损失&quot;]"><strong><a href="../../venues/TOIS/2025/self_supervised_denoising/symmetric_graph_contrastive_learning_against_noisy_views_for_recommendation/">Symmetric Graph Contrastive Learning against Noisy Views for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>定义图增广产生的噪声视图，用对称指数对比损失替代 InfoNCE，使 GCL 对噪声视图鲁棒，全库排序下 NDCG@20 最高提升 12.25%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;符号边重加权&quot;,&quot;LLM 表征增强&quot;,&quot;精度-多样性权衡&quot;]"><strong><a href="../../venues/TOIS/2026/graph_rec/cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图对比学习&quot;,&quot;相似就诊检索&quot;]"><strong><a href="../../venues/TOIS/2026/graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;模型更新/遗忘&quot;,&quot;影响函数&quot;,&quot;推荐遗忘&quot;,&quot;图卷积溢出影响&quot;]"><strong><a href="../../venues/TORS/2025/debias_fairness_industrial/recommendation_unlearning_via_influence_function/">Recommendation Unlearning via Influence Function</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用影响函数一步更新模型实现推荐遗忘，联合直接损失影响与图卷积溢出影响，剪枝加速后较重训提速 250 倍以上。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图神经网络&quot;,&quot;方面级情感聚合&quot;,&quot;评论驱动推荐&quot;]"><strong><a href="../../venues/WSDM/2025/graph_rec/an_aspect_performance_aware_hypergraph_neural_network_for_review_based_reco/">An Aspect Performance-aware Hypergraph Neural Network for Review-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用四元组超图建模用户-物品-方面-情感极性，以用户偏好加权聚合冲突情感推断物品方面表现，六数据集 MSE 平均降 2.30%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图剪枝&quot;,&quot;知识蒸馏&quot;,&quot;过平滑&quot;]"><strong><a href="../../venues/WSDM/2025/graph_rec/lightgnn_simple_graph_neural_network_for_recommendation/">LightGNN: Simple Graph Neural Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习边/嵌入剪枝+分层知识蒸馏压缩 GNN 推荐模型，80% 边与 90% 嵌入裁剪下仍超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;异质超图&quot;,&quot;意图解耦&quot;,&quot;多行为对比学习&quot;]"><strong><a href="../../venues/WSDM/2025/graph_rec/mixrec_heterogeneous_graph_collaborative_filtering/">MixRec: Heterogeneous Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>参数化异质超图解耦多行为意图，配合节点/图级自适应对比学习，Leave-One-Out 全库排序下 HR@10 最高提升 9.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;分数生成模型&quot;,&quot;难负采样&quot;]"><strong><a href="../../venues/WSDM/2025/self_supervised_denoising/scone_a_novel_stochastic_sampling_to_generate_contrastive_views_and_hard_ne/">SCONE: A Novel Stochastic Sampling to Generate Contrastive Views and Hard Negative Samples for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分数生成模型统一生成用户-物品特定对比视图与难负样本，6 数据集全库排序 Recall@20 最高提升 1.70%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;频域滤波去噪&quot;,&quot;多模态图学习&quot;]"><strong><a href="../../venues/WSDM/2025/multimodal_rec/spectrum_based_modality_representation_fusion_graph_convolutional_network_f/">Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多模态特征投影至频域做滤波去噪与点积融合，配合模态专属/融合双图学习与模态感知偏好模块，在 Amazon 三数据集全库排序下 Recall@20 相对最优基线提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;捆绑推荐&quot;,&quot;对比学习&quot;,&quot;物品关系挖掘&quot;]"><strong><a href="../../venues/WSDM/2026/cold_start_rec/epiccbr_item_relation_enhanced_dual_scenario_contrastive_learning_for_cold_/">EpicCBR: Item-Relation-Enhanced Dual-Scenario Contrastive Learning for Cold-Start Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过物品对关系增强与流行度感知双场景对比学习，在冷启动捆绑推荐上实现最高 387% 的 Recall@20 提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图滤波&quot;,&quot;多准则推荐&quot;,&quot;免训练&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;分布鲁棒优化&quot;,&quot;扩散去噪&quot;,&quot;OOD 泛化&quot;]"><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/distributionally_robust_graph_out_of_distribution_recommendation_via_diffus/">Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 DRO 推荐对噪声分布过度加权之弊，用 VGAE+扩散去噪、Sinkhorn 距离与熵正则重构最坏情形优化，OOD 场景最高提升 75.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;理想高通滤波&quot;,&quot;高阶邻域&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;GraphRAG&quot;,&quot;可解释推荐&quot;,&quot;知识剪枝&quot;]"><strong><a href="../../venues/WWW/2025/llm_rec/g_refer_graph_retrieval_augmented_large_language_model_for_explainable_reco/">G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用路径级+节点级混合图检索把协同过滤信号显式翻译为文本，经知识剪枝与 RAFT 微调驱动 LLM 生成可解释推荐，BERT-F1 最高提升 2.81%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;图卷积权重控制&quot;,&quot;连续时间建模&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果推断&quot;,&quot;扩散模型&quot;,&quot;OOD 泛化&quot;]"><strong><a href="../../venues/WWW/2025/debias_fairness_industrial/graph_representation_learning_via_causal_diffusion_for_out_of_distribution_/">Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构因果模型定位环境混杂因子，以变分推断+扩散模型学习环境不变图表征，OOD 场景最高提升 36.73%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;对齐与均匀性&quot;,&quot;图序列联合训练&quot;]"><strong><a href="../../venues/WWW/2025/graph_rec/graph_sequential_alignment_and_uniformity_toward_enhanced_recommendation_sy/">Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以共享嵌入层联合训练 LightGCN 与 SASRec，用对齐-均匀性损失实现跨子模块知识迁移，Leave-One-Out 全库排序下 R@10 最高提升 13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入表压缩&quot;,&quot;模块度图聚类&quot;,&quot;哈希碰撞&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;双曲空间&quot;,&quot;潜在扩散模型&quot;,&quot;方向性扩散&quot;]"><strong><a href="../../venues/WWW/2025/generative_rec/hyperbolic_diffusion_recommender_model/">Hyperbolic Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将扩散推荐从欧氏空间迁移至双曲流形，用聚类锚定的方向性噪声替代各向同性高斯噪声，Full-Ranking 下 R@10 最高提升 5.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图滤波&quot;,&quot;群组推荐&quot;]"><strong><a href="../../venues/WWW/2025/social_group_rec/leveraging_member_group_relations_via_multi_view_graph_filtering_for_effect/">Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出免训练多视图图滤波 Group-GF，用三张物品相似图与多项式滤波器聚合成员-群组关系，HR@10 最高提升 16.4% 且运行时间仅 1.55s。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;数据混合增强&quot;,&quot;双混合对比学习&quot;]"><strong><a href="../../venues/WWW/2025/self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图 Transformer&quot;,&quot;排序目标驱动架构&quot;,&quot;线性加速&quot;]"><strong><a href="../../venues/WWW/2025/collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图结构注入&quot;,&quot;对比学习&quot;,&quot;事件序列建模&quot;]"><strong><a href="../../venues/WWW/2026/self_supervised_denoising/beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;语义推理&quot;,&quot;群体隐式兴趣&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/iterative_semantic_reasoning_from_individual_to_group_interests_for_generat/">Iterative Semantic Reasoning from Individual to Group Interests for Generative Recommendation with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 双向 CoT 推理物品/用户语义，构建个体显式兴趣图与相似用户群隐式兴趣图，并以对比蒸馏迭代对齐，直接推荐 H@10 最高提升 24.57%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;神经符号推理&quot;,&quot;前门调整&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/modeling_endogenous_logic_causal_neuro_symbolic_reasoning_model_for_explain/">Modeling Endogenous Logic: Causal Neuro-Symbolic Reasoning Model for Explainable Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用前门调整把多行为链内源逻辑做成可解释因果中介，三路神经逻辑推理在三个数据集 Full-Ranking 上 HR@10 提升 20%+。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时序子图&quot;,&quot;兴趣演化建模&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;扩散增广&quot;,&quot;非对称对比&quot;]"><strong><a href="../../venues/WWW/2026/self_supervised_denoising/radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
</dl>
  </div>
</div>
