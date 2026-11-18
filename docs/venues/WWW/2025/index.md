---
hide:
  - toc
---

# 🏛️ WWW (TheWebConf) 2025 推荐系统论文解读

88 篇 WWW (TheWebConf) 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** WWW 2025 ×88 · 大语言模型推荐 ×18 · 生成式推荐 ×17 · 去偏公平与可信推荐 ×9 · 多模态推荐 ×6 · 多任务学习 ×5 · LLM Agent ×2 · LLM 零样本重排 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">27</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 零样本重排" aria-pressed="false">LLM 零样本重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="OOD 泛化" aria-pressed="false">OOD 泛化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图滤波" aria-pressed="false">图滤波<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期行为序列建模" aria-pressed="false">长期行为序列建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Bounded Rationality" aria-pressed="false">Bounded Rationality<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CF-LLM 融合" aria-pressed="false">CF-LLM 融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Thought 节点图" aria-pressed="false">Chain-of-Thought 节点图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CLIP 图像融合" aria-pressed="false">CLIP 图像融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Combinatorial Semi-Bandit" aria-pressed="false">Contextual Combinatorial Semi-Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Continual Learning" aria-pressed="false">Continual Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Critiquing 评测" aria-pressed="false">Critiquing 评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-Attention" aria-pressed="false">Cross-Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-Encoder" aria-pressed="false">Cross-Encoder<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DHEN 特征交叉集成" aria-pressed="false">DHEN 特征交叉集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Embedding Pruning" aria-pressed="false">Embedding Pruning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GraphRAG" aria-pressed="false">GraphRAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM User Simulator" aria-pressed="false">LLM User Simulator<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Logits Matching" aria-pressed="false">Logits Matching<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 参数高效适配" aria-pressed="false">LoRA 参数高效适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Margin InfoNCE" aria-pressed="false">Margin InfoNCE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Middleman Bias" aria-pressed="false">Middleman Bias<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixture-of-Logits" aria-pressed="false">Mixture-of-Logits<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Neural ODE" aria-pressed="false">Neural ODE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="On-device Recommendation" aria-pressed="false">On-device Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Perceiver 多模态融合" aria-pressed="false">Perceiver 多模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prompt Tuning" aria-pressed="false">Prompt Tuning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Shapley Value" aria-pressed="false">Shapley Value<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Soft Prompt" aria-pressed="false">Soft Prompt<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Stop-Gradient" aria-pressed="false">Stop-Gradient<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SUTVA违背" aria-pressed="false">SUTVA违背<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TD 分解" aria-pressed="false">TD 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Token Alignment" aria-pressed="false">Token Alignment<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Tucker 分解" aria-pressed="false">Tucker 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="VTuber观众行为测量" aria-pressed="false">VTuber观众行为测量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三元知识库" aria-pressed="false">三元知识库<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性分解" aria-pressed="false">不确定性分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性量化" aria-pressed="false">不确定性量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不适推荐过滤" aria-pressed="false">不适推荐过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两步决策与记忆检索" aria-pressed="false">两步决策与记忆检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个人损失分布重采样" aria-pressed="false">个人损失分布重采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化图像生成" aria-pressed="false">个性化图像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化标题生成" aria-pressed="false">个性化标题生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化相关性激励" aria-pressed="false">个性化相关性激励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主动推荐" aria-pressed="false">主动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="乘积量化" aria-pressed="false">乘积量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交叉交互特征" aria-pressed="false">交叉交互特征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪查询-视频匹配" aria-pressed="false">伪查询-视频匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪用户协同信号" aria-pressed="false">伪用户协同信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置编码" aria-pressed="false">位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="候选位置偏差" aria-pressed="false">候选位置偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好优化" aria-pressed="false">偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好对齐" aria-pressed="false">偏好对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好画像" aria-pressed="false">偏好画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="停留时长预测" aria-pressed="false">停留时长预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练" aria-pressed="false">免训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平性-相关性联合评测" aria-pressed="false">公平性-相关性联合评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平正则化" aria-pressed="false">公平正则化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="共生偏差" aria-pressed="false">共生偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="关键词相关性" aria-pressed="false">关键词相关性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容编码器" aria-pressed="false">内容编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层 UCB" aria-pressed="false">分层 UCB<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层提示" aria-pressed="false">分层提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布鲁棒优化" aria-pressed="false">分布鲁棒优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="创作者侧推荐" aria-pressed="false">创作者侧推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动作探索鲁棒性" aria-pressed="false">动作探索鲁棒性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同嵌入重排" aria-pressed="false">协同嵌入重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同检索增强生成" aria-pressed="false">协同检索增强生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单调融合单元" aria-pressed="false">单调融合单元<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原生广告分配" aria-pressed="false">原生广告分配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏推荐" aria-pressed="false">去偏推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲空间" aria-pressed="false">双曲空间<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双混合对比学习" aria-pressed="false">双混合对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双源物品索引" aria-pressed="false">双源物品索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双边市场公平性" aria-pressed="false">双边市场公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实去偏" aria-pressed="false">反事实去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="叙事生成" aria-pressed="false">叙事生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="叙事驱动推荐" aria-pressed="false">叙事驱动推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排" aria-pressed="false">召回+精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排+重排" aria-pressed="false">召回+精排+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序策略优化" aria-pressed="false">召回/排序策略优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可学习偏置项" aria-pressed="false">可学习偏置项<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="哈希碰撞" aria-pressed="false">哈希碰撞<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="困惑度标注" aria-pressed="false">困惑度标注<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图 CoT 推理" aria-pressed="false">图 CoT 推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图 Transformer" aria-pressed="false">图 Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信号处理" aria-pressed="false">图信号处理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图卷积权重控制" aria-pressed="false">图卷积权重控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图序列联合训练" aria-pressed="false">图序列联合训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图文对齐" aria-pressed="false">图文对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线实验/因果评估" aria-pressed="false">在线实验/因果评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域不变兴趣抽取" aria-pressed="false">域不变兴趣抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基础模型" aria-pressed="false">基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="外部知识蒸馏" aria-pressed="false">外部知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务融合" aria-pressed="false">多任务融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务辅助监督" aria-pressed="false">多任务辅助监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多准则均衡" aria-pressed="false">多准则均衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多准则推荐" aria-pressed="false">多准则推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域学习" aria-pressed="false">多域学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体" aria-pressed="false">多智能体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态双塔" aria-pressed="false">多模态双塔<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态奖励模型" aria-pressed="false">多模态奖励模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐与内容理解" aria-pressed="false">多模态推荐与内容理解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多注意力跨域序列建模" aria-pressed="false">多注意力跨域序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多维评测框架" aria-pressed="false">多维评测框架<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大记忆网络" aria-pressed="false">大记忆网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="奖励塑形" aria-pressed="false">奖励塑形<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="安全探索" aria-pressed="false">安全探索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实时用户序列" aria-pressed="false">实时用户序列<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实时竞价出价优化" aria-pressed="false">实时竞价出价优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比差异最大化" aria-pressed="false">对比差异最大化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐与均匀性" aria-pressed="false">对齐与均匀性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层次Softmax" aria-pressed="false">层次Softmax<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入表压缩" aria-pressed="false">嵌入表压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托前沿" aria-pressed="false">帕累托前沿<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="幻觉" aria-pressed="false">幻觉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="广告图像生成" aria-pressed="false">广告图像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习" aria-pressed="false">强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响路径规划" aria-pressed="false">影响路径规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="性别偏差" aria-pressed="false">性别偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="情感对齐评估" aria-pressed="false">情感对齐评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散世界模型" aria-pressed="false">扩散世界模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散去噪" aria-pressed="false">扩散去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型序列推荐" aria-pressed="false">扩散模型序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型数据补全" aria-pressed="false">扩散模型数据补全<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散生成推荐" aria-pressed="false">扩散生成推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拍卖基准" aria-pressed="false">拍卖基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐" aria-pressed="false">捆绑推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序目标驱动架构" aria-pressed="false">排序目标驱动架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理缓存" aria-pressed="false">推理缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示工程" aria-pressed="false">提示工程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示演化" aria-pressed="false">提示演化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示策略评测" aria-pressed="false">提示策略评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="插件化 Agent" aria-pressed="false">插件化 Agent<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="摊销推理" aria-pressed="false">摊销推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="效用评估" aria-pressed="false">效用评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="敏感内容标签" aria-pressed="false">敏感内容标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据为中心" aria-pressed="false">数据为中心<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据再生" aria-pressed="false">数据再生<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据压缩/训练加速" aria-pressed="false">数据压缩/训练加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强/冷启动" aria-pressed="false">数据增强/冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据泄漏抑制" aria-pressed="false">数据泄漏抑制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据混合增强" aria-pressed="false">数据混合增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据蒸馏/训练加速" aria-pressed="false">数据蒸馏/训练加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集构建" aria-pressed="false">数据集构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集浓缩" aria-pressed="false">数据集浓缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集链接" aria-pressed="false">数据集链接<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="方向性扩散" aria-pressed="false">方向性扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="无偏估计" aria-pressed="false">无偏估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="早期融合" aria-pressed="false">早期融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知" aria-pressed="false">时间感知<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="显式时间建模" aria-pressed="false">显式时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件概率融合" aria-pressed="false">条件概率融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询-短视频相关性" aria-pressed="false">查询-短视频相关性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="标题-内容一致性" aria-pressed="false">标题-内容一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="样本选择偏差" aria-pressed="false">样本选择偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块度图聚类" aria-pressed="false">模块度图聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力衰减" aria-pressed="false">注意力衰减<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式分布漂移" aria-pressed="false">流式分布漂移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式训练" aria-pressed="false">流式训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合序列推荐" aria-pressed="false">混合序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在扩散模型" aria-pressed="false">潜在扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征归因重要性采样" aria-pressed="false">特征归因重要性采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征空间对齐" aria-pressed="false">特征空间对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态抽象" aria-pressed="false">状态抽象<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="环境模拟器" aria-pressed="false">环境模拟器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="理想高通滤波" aria-pressed="false">理想高通滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式重排" aria-pressed="false">生成式重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户可用性计算" aria-pressed="false">用户可用性计算<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户行为建模" aria-pressed="false">用户行为建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识剪枝" aria-pressed="false">知识剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识检索增强" aria-pressed="false">知识检索增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测" aria-pressed="false">离线评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稠密检索" aria-pressed="false">稠密检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端联合训练" aria-pressed="false">端到端联合训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="算法放大" aria-pressed="false">算法放大<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="类别感知公平度量" aria-pressed="false">类别感知公平度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粉丝社区文化" aria-pressed="false">粉丝社区文化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性加速" aria-pressed="false">线性加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放律" aria-pressed="false">缩放律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群组推荐" aria-pressed="false">群组推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="聚类用户合成" aria-pressed="false">聚类用户合成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="聚类随机化" aria-pressed="false">聚类随机化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自对弈偏好优化" aria-pressed="false">自对弈偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督未来行为预测" aria-pressed="false">自监督未来行为预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征对齐" aria-pressed="false">表征对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表格数据合成" aria-pressed="false">表格数据合成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视觉语言模型微调" aria-pressed="false">视觉语言模型微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="角色扮演与可控生成" aria-pressed="false">角色扮演与可控生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="记忆参数服务器" aria-pressed="false">记忆参数服务器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="记忆检索" aria-pressed="false">记忆检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评分条件生成" aria-pressed="false">评分条件生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测与仿真" aria-pressed="false">评测与仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测协议" aria-pressed="false">评测协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评论排序辅助任务" aria-pressed="false">评论排序辅助任务<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义一致性正则" aria-pressed="false">语义一致性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负载均衡正则" aria-pressed="false">负载均衡正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="贴纸个性化检索" aria-pressed="false">贴纸个性化检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域对比学习" aria-pressed="false">跨域对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域知识迁移" aria-pressed="false">跨域知识迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨阶段协同" aria-pressed="false">跨阶段协同<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="转化广告 CVR" aria-pressed="false">转化广告 CVR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似Top-K检索" aria-pressed="false">近似Top-K检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似算法" aria-pressed="false">近似算法<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="连续时间建模" aria-pressed="false">连续时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="退火适配器" aria-pressed="false">退火适配器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆向推荐（向主播推荐观众）" aria-pressed="false">逆向推荐（向主播推荐观众）<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻居列表反事实训练" aria-pressed="false">邻居列表反事实训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期价值估计" aria-pressed="false">长期价值估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期兴趣建模" aria-pressed="false">长期兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控融合" aria-pressed="false">门控融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="随机曝光数据集" aria-pressed="false">随机曝光数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式反馈加权" aria-pressed="false">隐式反馈加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式反馈去噪" aria-pressed="false">隐式反馈去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本 LLM推荐" aria-pressed="false">零样本 LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非侵入对齐重建" aria-pressed="false">非侵入对齐重建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非平稳环境" aria-pressed="false">非平稳环境<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非自回归采样生成" aria-pressed="false">非自回归采样生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预条件共轭梯度" aria-pressed="false">预条件共轭梯度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预算平滑" aria-pressed="false">预算平滑<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="风格-内容解耦" aria-pressed="false">风格-内容解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高阶邻域" aria-pressed="false">高阶邻域<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="黑盒 LLM 反思重排" aria-pressed="false">黑盒 LLM 反思重排<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图滤波&quot;,&quot;多准则推荐&quot;,&quot;免训练&quot;]"><strong><a href="collaborative_filtering/criteria_aware_graph_filtering_extremely_fast_yet_accurate_multi_criteria_r/">Criteria-Aware Graph Filtering: Extremely Fast Yet Accurate Multi-Criteria Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>免训练免矩阵分解的多准则图滤波推荐，按准则定制多项式低通滤波器并融合用户准则偏好，BA 数据集 0.2 秒推理、NDCG@5 最高提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;隐式反馈加权&quot;,&quot;线性自编码器&quot;,&quot;预条件共轭梯度&quot;]"><strong><a href="collaborative_filtering/does_weighting_improve_matrix_factorization_for_recommender_systems/">Does Weighting Improve Matrix Factorization for Recommender Systems?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统实证加权隐式反馈矩阵分解：大秩模型下不加权常持平或更优，并给出加权目标闭式解与预条件共轭梯度高效求解。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图信号处理&quot;,&quot;理想高通滤波&quot;,&quot;高阶邻域&quot;]"><strong><a href="collaborative_filtering/frequency_aware_graph_signal_processing_for_collaborative_filtering/">Frequency-aware Graph Signal Processing for Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用级联高通-低通滤波捕获独有/共性特征、并行低通滤波捕获高阶邻域，免训练线性组合在六个数据集上 Full-Ranking 提升 2-5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;图卷积权重控制&quot;,&quot;连续时间建模&quot;]"><strong><a href="collaborative_filtering/graph_neural_controlled_differential_equations_for_collaborative_filtering/">Graph Neural Controlled Differential Equations For Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLP 控制器生成连续时变权重矩阵替代 GODE-CF 的离散权重，在四个 Amazon 5-core 数据集 Full-Ranking 下 Recall@20 全面超越 LightGCN/GODE-CF。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入表压缩&quot;,&quot;模块度图聚类&quot;,&quot;哈希碰撞&quot;]"><strong><a href="collaborative_filtering/graphhash_graph_clustering_enables_parameter_efficiency_in_recommender_syst/">GraphHash: Graph Clustering Enables Parameter Efficiency in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模块度二部图聚类替代 ID 哈希，将图平滑前置到预处理阶段，嵌入表缩减 75%+ 时 Recall@20 平均提升 101.52%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;交叉交互特征&quot;,&quot;实时用户序列&quot;]"><strong><a href="collaborative_filtering/interactrank_personalized_web_scale_search_pre_ranking_with_cross_interacti/">InteractRank: Personalized Web-Scale Search Pre-Ranking with Cross Interaction Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在双塔预排序点积之上叠加历史交互统计特征 IQP 与实时用户序列，以 1.1× FLOPs 实现线上 SIFR +6.5%、F1S +9.2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图 Transformer&quot;,&quot;排序目标驱动架构&quot;,&quot;线性加速&quot;]"><strong><a href="collaborative_filtering/rankformer_a_graph_transformer_for_recommendation_based_on_ranking_objectiv/">Rankformer: A Graph Transformer for Recommendation based on Ranking Objective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从 BPR 梯度下降推导出图 Transformer 层，正负样本分离聚合+基准项+偏移项，线性加速后 4 数据集平均提升 4.48%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;对齐与均匀性&quot;,&quot;图序列联合训练&quot;]"><strong><a href="graph_rec/graph_sequential_alignment_and_uniformity_toward_enhanced_recommendation_sy/">Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以共享嵌入层联合训练 LightGCN 与 SASRec，用对齐-均匀性损失实现跨子模块知识迁移，Leave-One-Out 全库排序下 R@10 最高提升 13.4%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;数据混合增强&quot;,&quot;双混合对比学习&quot;]"><strong><a href="self_supervised_denoising/mixrec_individual_and_collective_mixing_empowers_data_augmentation_for_reco/">MixRec: Individual and Collective Mixing Empowers Data Augmentation for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Beta/Dirichlet 混合在嵌入空间零成本造正负视图，配合双混合对比损失，全库排序下 Recall@20 最高提升 5.05%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;隐式反馈去噪&quot;,&quot;个人损失分布重采样&quot;]"><strong><a href="self_supervised_denoising/personalized_denoising_implicit_feedback_for_robust_recommender_system/">Personalized Denoising Implicit Feedback for Robust Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示全局损失分布中正常/噪声交互严重重叠，提出基于用户个人损失分布的重采样去噪 PLD，BPR 下 Recall@20 最高提升 7.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;信息瓶颈&quot;,&quot;图 CoT 推理&quot;]"><strong><a href="self_supervised_denoising/unleashing_the_power_of_large_language_model_for_denoising_recommendation/">Unleashing the Power of Large Language Model for Denoising Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>LLaRD 用 LLM 生成偏好知识与图 CoT 关系知识，再以信息瓶颈对齐去噪图，Full-Ranking 下 Recall@20 最高提升约 14%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;位置编码&quot;,&quot;目标注意力&quot;,&quot;门控融合&quot;]"><strong><a href="sequential_rec/a_contextual_aware_position_encoding_for_sequential_recommendation/">A Contextual-Aware Position Encoding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个为序列推荐定制的上下文感知位置编码 CAPE：用目标-上下文不相似度累加动态分配位置，配合门控融合异构特征，在 5 个骨干与工业 A/B 上稳定涨点。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;摊销推理&quot;,&quot;早期融合&quot;,&quot;Cross-Attention&quot;]"><strong><a href="sequential_rec/efficient_user_history_modeling_with_amortized_inference_for_deep_learning_/">Efficient user history modeling with amortized inference for deep learning recommendation models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 cross-attention 追加候选替代逐项拼接，使历史编码可摊销推理，LinkedIn Feed/Ads 延迟降 30%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;混合序列推荐&quot;,&quot;Perceiver 多模态融合&quot;,&quot;Critiquing 评测&quot;]"><strong><a href="sequential_rec/flare_fusing_language_models_and_collaborative_architectures_for_recommende/">FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用冻结 mT5+Perceiver 将文本注入 Bert4Rec，并以重调 Bert4Rec 基线、大词表评测与 Critiquing 任务重估 ID+文本增益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;大记忆网络&quot;,&quot;乘积量化&quot;,&quot;记忆参数服务器&quot;]"><strong><a href="sequential_rec/large_memory_network_for_recommendation/">Large Memory Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用百万级全局共享记忆块压缩用户历史行为，以乘积量化双轴激活实现 O(√n d) 检索，抖音电商搜索 AUC +4.68%、订单/人 +0.87%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知&quot;,&quot;长期行为序列建模&quot;,&quot;流式训练&quot;]"><strong><a href="sequential_rec/long_term_interest_clock_fine_grained_time_perception_in_streaming_recommen/">Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用候选物品+当前时刻双查询从长期行为中检索子序列，并以时距感知注意力聚合，实现流式推荐细粒度时间感知，线上活跃天数+0.122%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mixture-of-Logits&quot;,&quot;近似Top-K检索&quot;,&quot;负载均衡正则&quot;]"><strong><a href="sequential_rec/retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;数据蒸馏/训练加速&quot;,&quot;Tucker 分解&quot;,&quot;双层优化&quot;,&quot;特征空间对齐&quot;]"><strong><a href="sequential_rec/td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;扩散推荐&quot;,&quot;伪用户协同信号&quot;]"><strong><a href="generative_rec/collaborative_diffusion_model_for_recommender_system/">Collaborative Diffusion Model for Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将评论词转为伪用户，在扩散去噪过程中聚合真实/伪邻居偏好信息，缓解个性化信息流失，三数据集 R@20 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散模型序列推荐&quot;,&quot;语义ID&quot;,&quot;对比差异最大化&quot;]"><strong><a href="generative_rec/distinguished_quantized_guidance_for_diffusion_based_sequence_recommendatio/">Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用语义码本量化序列增强扩散引导，并以对比差异最大化抑制流行度偏置生成，四数据集全库排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;缩放律&quot;,&quot;数据再生&quot;,&quot;长期行为序列建模&quot;]"><strong><a href="generative_rec/generative_large_recommendation_models_emerging_trends_in_llms_for_recommen/">Generative Large Recommendation Models: Emerging Trends in LLMs for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理生成式大推荐模型范式：数据质量、缩放律、用户行为挖掘与训练推理效率四大议题及未来方向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;双曲空间&quot;,&quot;潜在扩散模型&quot;,&quot;方向性扩散&quot;]"><strong><a href="generative_rec/hyperbolic_diffusion_recommender_model/">Hyperbolic Diffusion Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将扩散推荐从欧氏空间迁移至双曲流形，用聚类锚定的方向性噪声替代各向同性高斯噪声，Full-Ranking 下 R@10 最高提升 5.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (19)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;CF-LLM 融合&quot;,&quot;Soft Prompt&quot;,&quot;叙事生成&quot;]"><strong><a href="llm_rec/beyond_retrieval_generating_narratives_in_conversational_recommender_system/">Beyond Retrieval: Generating Narratives in Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 REGEN 叙事数据集并提出 CF+内容嵌入经 Adapter 注入 LLM 的融合架构，在 Office/Clothing 两域上 BLEU/ROUGE/语义相似度较单模态提升 4–12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;多维评测框架&quot;,&quot;候选位置偏差&quot;,&quot;幻觉&quot;]"><strong><a href="llm_rec/beyond_utility_evaluating_llm_as_recommender/">Beyond Utility: Evaluating LLM as Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出含历史长度敏感性、候选位置偏差、画像生成与幻觉四维的 LLM 推荐多维评测框架，揭示 LLM 在重排与冷启动占优但位置偏差严重。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;三元知识库&quot;,&quot;数据为中心&quot;,&quot;知识检索增强&quot;]"><strong><a href="llm_rec/d2k_turning_historical_data_into_retrievable_knowledge_for_recommender_syst/">D2K: Turning Historical Data into Retrievable Knowledge for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将历史日志转为用户-物品-上下文三元知识库，Transformer 编码 + 个性化适配注入任意 CTR 模型，AUC 相对提升 1.35%–2.90%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;双源物品索引&quot;,&quot;非侵入对齐重建&quot;,&quot;退火适配器&quot;]"><strong><a href="llm_rec/eager_llm_enhancing_large_language_models_as_recommenders_through_exogenous/">EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以双源知识富集物品索引+非侵入多尺度对齐重建+退火适配器，将外源行为/语义信号注入 LLM 生成式推荐，Amazon 三数据集 Full-Ranking 下 Recall@10 最高提升 26.16%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;分层提示&quot;,&quot;新闻推荐&quot;,&quot;数据增强&quot;]"><strong><a href="llm_rec/enhancing_news_recommendation_with_hierarchical_llm_prompting/">Enhancing News Recommendation with Hierarchical LLM Prompting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层提示让 LLM 生成更丰富的新闻标题与实体，作为模型无关的数据增强模块，在 MIND 上以 AUC 68.88 超越 GLoCIM 等 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;外部知识蒸馏&quot;,&quot;流式分布漂移&quot;,&quot;基础模型&quot;]"><strong><a href="llm_rec/external_large_foundation_model_how_to_efficiently_serve_trillions_of_param/">External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以外部蒸馏+数据增强服务解耦教师训练，用辅助头与学生适配器弥合流式分布漂移，实现万亿参数基础模型零延迟增益服务多业务精排模型。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;不适推荐过滤&quot;,&quot;偏好画像&quot;,&quot;多智能体&quot;]"><strong><a href="llm_rec/filtering_discomforting_recommendations_with_large_language_models/">Filtering Discomforting Recommendations with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可编辑偏好画像+对话式规则生成，以插件方式过滤主观不适推荐；离线代理任务与24人一周用户研究验证有效性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;GraphRAG&quot;,&quot;可解释推荐&quot;,&quot;知识剪枝&quot;]"><strong><a href="llm_rec/g_refer_graph_retrieval_augmented_large_language_model_for_explainable_reco/">G-Refer: Graph Retrieval-Augmented Large Language Model for Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用路径级+节点级混合图检索把协同过滤信号显式翻译为文本，经知识剪枝与 RAFT 微调驱动 LLM 生成可解释推荐，BERT-F1 最高提升 2.81%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;叙事驱动推荐&quot;,&quot;提示策略评测&quot;]"><strong><a href="llm_rec/large_language_models_as_narrative_driven_recommenders/">Large Language Models as Narrative-Driven Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统评测 38 个 LLM 在 Reddit 叙事式电影推荐上的表现，发现零样本提示即可显著超越 doc2vec 基线 70%+，中型开源模型接近大型闭源模型。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回+精排&quot;,&quot;端到端联合训练&quot;,&quot;内容编码器&quot;,&quot;推理缓存&quot;]"><strong><a href="llm_rec/legommenders_a_comprehensive_content_based_recommendation_library_with_llm_/">Legommenders: A Comprehensive Content-Based Recommendation Library with LLM Support</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个支持内容编码器与行为/交互模块端到端联合训练的推荐库，内置 LLM 编码与数据增强，覆盖 1000+ 模型组合与 15 个数据集，推理缓存带来最高 50x 加速。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部19篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;插件化 Agent&quot;,&quot;数据泄漏抑制&quot;]"><strong><a href="agent_conversational_rec/a_llm_based_controllable_scalable_human_involved_user_simulator_framework_f/">A LLM-based Controllable, Scalable, Human-Involved User Simulator Framework for Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以插件管理器分阶段管控 LLM 用户模拟器，通过记忆隔离与偏好二分抑制数据泄漏，在 ReDial/OpenDialKG/MovieLens 上显著提升 CRS 评估可信度。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;LLM Agent&quot;,&quot;环境模拟器&quot;,&quot;用户行为建模&quot;]"><strong><a href="agent_conversational_rec/agentsociety_challenge_designing_llm_agents_for_user_modeling_and_recommend/">AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个 WWW LLM Agent 双赛道竞赛：Yelp/Amazon/Goodreads 三源数据 + 交互式模拟器，295 队参赛，Track1/2 最终阶段提升 9.1%/15.9%，并验证模拟 groundtruth 与真实分布强相关。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;数据集链接&quot;,&quot;协同嵌入重排&quot;,&quot;零样本 LLM推荐&quot;]"><strong><a href="agent_conversational_rec/bridging_conversational_and_collaborative_signals_for_conversational_recomm/">Bridging Conversational and Collaborative Signals for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 Reddit-ML32M 跨源数据集，用 SASRec 协同嵌入对 GPT-3.5 零样本推荐做相似度重排，H@5 提升 12.32%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;协同检索增强生成&quot;,&quot;黑盒 LLM 反思重排&quot;]"><strong><a href="agent_conversational_rec/collaborative_retrieval_for_large_language_model_based_conversational_recom/">Collaborative Retrieval for Large Language Model-based Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 EASE 协同检索 + 两步 LLM 反思重排为黑盒 LLM 注入 CF 知识，Reddit-v2/Redial 全库排序 Recall@20 显著超越 Zero-shot LLM。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;可解释推荐&quot;,&quot;情感对齐评估&quot;,&quot;评分条件生成&quot;]"><strong><a href="agent_conversational_rec/disentangling_likes_and_dislikes_in_personalized_generative_explainable_rec/">Disentangling Likes and Dislikes in Personalized Generative Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 重构正负观点分离的可解释推荐数据集，提出情感匹配与正负内容相似度指标，并证明将预测评分作为输入可显著提升情感感知解释生成。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测与仿真&quot;,&quot;LLM User Simulator&quot;,&quot;Bounded Rationality&quot;,&quot;角色扮演与可控生成&quot;]"><strong><a href="agent_conversational_rec/recusersim_a_realistic_and_diverse_user_simulator_for_evaluating_conversati/">RecUserSim: A Realistic and Diverse User Simulator for Evaluating Conversational Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出四模块 LLM Agent 用户模拟器，以有限理性三层 Rating-Action-Response 机制实现真实角色扮演、多样人群覆盖与显式打分，并落地华为 Celia 餐饮助手。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐与内容理解&quot;,&quot;精排&quot;,&quot;停留时长预测&quot;,&quot;LLM 语义嵌入&quot;,&quot;评论排序辅助任务&quot;]"><strong><a href="multimodal_rec/comment_staytime_prediction_with_llm_enhanced_comment_understanding/">Comment Staytime Prediction with LLM-enhanced Comment Understanding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个评论区停留时长数据集 KuaiComt 与两阶段 LCU 框架：LLM 领域微调 + 评论排序辅助任务，离线全指标提升且线上停留时长 +1.27%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;扩散模型数据补全&quot;,&quot;反事实去偏&quot;]"><strong><a href="multimodal_rec/generating_with_fairness_a_modality_diffused_counterfactual_framework_for_i/">Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模态感知扩散补全缺失模态，并以反事实推断剥离可见性偏差，在三个数据集上同时提升准确率与公平性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;伪查询-视频匹配&quot;,&quot;层次Softmax&quot;,&quot;查询-短视频相关性&quot;]"><strong><a href="multimodal_rec/hcmrm_a_high_consistency_multimodal_relevance_model_for_search_ads/">HCMRM: A High-Consistency Multimodal Relevance Model for Search Ads</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用视频关键词切分合成伪查询做三元组预训练，并以对称层次 Softmax 对齐四档相关性标签，快手搜索广告线上无关广告率降 6.1%、收入增 1.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;显式时间建模&quot;,&quot;多任务辅助监督&quot;]"><strong><a href="multimodal_rec/hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态双塔&quot;,&quot;稠密检索&quot;,&quot;图文对齐&quot;]"><strong><a href="multimodal_rec/multimodal_semantic_retrieval_for_product_search/">Multimodal semantic retrieval for product search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在电商产品检索中系统对比 4-tower/3-tower 图文融合双塔，用 FAISS 全库 Cosine 检索验证多模态表征在购买召回与相关性精度上的增益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;生成式推荐&quot;,&quot;个性化图像生成&quot;,&quot;偏好对齐&quot;]"><strong><a href="multimodal_rec/personalized_image_generation_with_large_multimodal_models/">Personalized Image Generation with Large Multimodal Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pigeon 用 LMM+掩码生成器从噪声历史图中抽取视觉偏好，两阶段偏好对齐（掩码重建+DPO）实现贴纸/海报个性化生成，个性化指标提升 7%~31%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;召回+重排&quot;,&quot;视觉语言模型微调&quot;,&quot;贴纸个性化检索&quot;,&quot;效用评估&quot;]"><strong><a href="multimodal_rec/persrv_personalized_sticker_retrieval_with_vision_language_model/">PerSRV: Personalized Sticker Retrieval with Vision-Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 SFT-LLaVA 生成贴纸语义、三路众包效用分与 CLIP 风格聚类，实现召回+个性化排序的贴纸检索，WeChat 数据 M-MRR@20 提升 8.95%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图滤波&quot;,&quot;群组推荐&quot;]"><strong><a href="social_group_rec/leveraging_member_group_relations_via_multi_view_graph_filtering_for_effect/">Leveraging Member-Group Relations via Multi-View Graph Filtering for Effective Group Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出免训练多视图图滤波 Group-GF，用三张物品相似图与多项式滤波器聚合成员-群组关系，HR@10 最高提升 16.4% 且运行时间仅 1.55s。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;VTuber观众行为测量&quot;,&quot;逆向推荐（向主播推荐观众）&quot;,&quot;粉丝社区文化&quot;]"><strong><a href="social_group_rec/virtual_stars_real_fans_understanding_the_vtuber_ecosystem/">Virtual Stars, Real Fans: Understanding the VTuber Ecosystem</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>基于Bilibili 270万场直播的VTuber核心粉丝（会员）行为测量研究，逆向构建FanRanker将潜在付费会员推荐给主播，Top-50命中85%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;意图解耦&quot;,&quot;标题-内容一致性&quot;,&quot;新闻推荐&quot;]"><strong><a href="cold_start_rec/crown_a_novel_approach_to_comprehending_users_preferences_for_accurate_pers/">CROWN: A Novel Approach to Comprehending Users&#x27; Preferences for Accurate Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别引导意图解耦、标题-内容一致性加权与 GNN 混合用户表征，统一解决新闻推荐的多意图、读后偏好差异与冷启动三大难题。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;扩散生成推荐&quot;,&quot;捆绑推荐&quot;]"><strong><a href="cold_start_rec/personalized_diffusion_model_reshapes_cold_start_bundle_recommendation/">Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用个性化条件扩散在分布空间生成捆绑包，配合解耦自注意力与KL正则，冷启动捆绑推荐全库排序Recall@20最高提升1155%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;LoRA 参数高效适配&quot;,&quot;域不变兴趣抽取&quot;]"><strong><a href="cross_domain_multi_scenario/abxi_invariant_interest_adaptation_for_task_guided_cross_domain_sequential_/">ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用任务引导对齐消除预测错配，并以 dLoRA/iLoRA 双 LoRA 抽取域不变兴趣，三数据集 MRR 大幅超越 CDSR 基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;CLIP 图像融合&quot;,&quot;多注意力跨域序列建模&quot;]"><strong><a href="cross_domain_multi_scenario/image_fusion_for_cross_domain_sequential_recommendation/">Image Fusion for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用冻结 CLIP 图像嵌入补足 ID 表征，并以多注意力分别建模域内/跨域序列，在 Amazon 双域场景 MRR 提升约 5%–18%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;信息瓶颈&quot;,&quot;参数高效微调&quot;,&quot;多域学习&quot;]"><strong><a href="cross_domain_multi_scenario/pre_train_and_fine_tune_recommenders_as_large_models/">Pre-train and Fine-tune: Recommenders as Large Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以信息瓶颈理论解释推荐模型微调，提出即插即用的信息感知自适应核 IAK，冻结十亿参数预训练推荐大模型，仅用一周域内数据微调即获 CTR-AUC 平均 +0.26%~+1.28% 与线上 GMV +0.74%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域对比学习&quot;,&quot;课程学习&quot;,&quot;Stop-Gradient&quot;]"><strong><a href="cross_domain_multi_scenario/separated_contrastive_learning_for_matching_in_cross_domain_recommendation_/">Separated Contrastive Learning for Matching in Cross-domain Recommendation with Curriculum Scheduling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 intra-CL 与 inter-CL 解耦为两阶段课程训练，配合 stop-gradient 与基于 Katz 中心性的负样本难度调度，稳定跨域召回训练并显著提升 HIT@100。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;重排&quot;,&quot;LLM 零样本重排&quot;,&quot;跨域知识迁移&quot;,&quot;提示工程&quot;]"><strong><a href="cross_domain_multi_scenario/uncovering_cross_domain_recommendation_ability_of_large_language_models/">Uncovering Cross-Domain Recommendation Ability of Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用源域购买序列+跨域引导提示驱动 LLM 零样本重排目标域候选，揭示域间隙与模型规模对跨域迁移的决定性影响。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;自动出价&quot;,&quot;预算平滑&quot;,&quot;拍卖基准&quot;]"><strong><a href="ctr_feature_interaction/bat_benchmark_for_auto_bidding_task/">BAT: Benchmark for Auto-bidding Task</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布 Avito 双拍卖格式（VCG/FP）真实自动出价数据集，配套 ALM/TA-PID 等 5 个基线，覆盖预算平滑与 CPC 约束两类 RTB 任务。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;生成式推荐&quot;,&quot;多模态奖励模型&quot;,&quot;偏好优化&quot;,&quot;广告图像生成&quot;]"><strong><a href="ctr_feature_interaction/ctr_driven_advertising_image_generation_with_multimodal_large_language_mode/">CTR-Driven Advertising Image Generation with Multimodal Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MLLM 双分支奖励模型把 CTR 相对比较转化为偏好信号，结合产品中心偏好优化 PCPO 驱动广告图生成，线上 CTR 提升 2%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;重排&quot;,&quot;原生广告分配&quot;,&quot;近似算法&quot;,&quot;注意力衰减&quot;]"><strong><a href="ctr_feature_interaction/efficient_and_practical_approximation_algorithms_for_advertising_in_content/">Efficient and Practical Approximation Algorithms for Advertising in Content Feeds</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>针对信息流原生广告位分配，提出反向贪心 2-近似算法，突破前人 4-近似瓶颈，并在合成与真实数据上首次系统评测。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;Middleman Bias&quot;,&quot;Cross-Encoder&quot;,&quot;关键词相关性&quot;]"><strong><a href="ctr_feature_interaction/middleman_bias_in_advertising_aligning_relevance_of_keyphrase_recommendatio/">Middleman Bias in Advertising: Aligning Relevance of Keyphrase Recommendations with Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示广告关键词相关性过滤中的“中间人偏差”，用Search相关性判定重构训练标签，并以轻量Cross-Encoder在eBay线上取得CVR+11.6%的收益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;DHEN 特征交叉集成&quot;,&quot;自监督未来行为预测&quot;,&quot;转化广告 CVR&quot;]"><strong><a href="ctr_feature_interaction/on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;条件概率融合&quot;,&quot;个性化相关性激励&quot;,&quot;语义一致性正则&quot;]"><strong><a href="ctr_feature_interaction/prectr_a_synergistic_framework_for_integrating_personalized_search_relevanc/">PRECTR: A Synergistic Framework for Integrating Personalized Search Relevance Matching and CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用条件概率融合把相关性分级与CTR统一进单模型，配两阶段训练、语义一致性正则与个性化相关性激励，线上CTR+0.4%、GMV+1.1%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;LLM Agent&quot;,&quot;实时竞价出价优化&quot;,&quot;两步决策与记忆检索&quot;]"><strong><a href="ctr_feature_interaction/rtbagent_a_llm_based_agent_system_for_real_time_bidding/">RTBAgent: A LLM-based Agent System for Real-Time Bidding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个 LLM Agent 出价系统，用两步决策+多类记忆+专家策略工具在 iPinYou 上以调整因子方式超越规则/RL/扩散基线。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;粗排&quot;,&quot;跨阶段协同&quot;,&quot;Margin InfoNCE&quot;,&quot;样本选择偏差&quot;]"><strong><a href="multi_task_reranking/a_hybrid_cross_stage_coordination_pre_ranking_model_for_online_recommendati/">A Hybrid Cross-Stage Coordination Pre-ranking Model for Online Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 HCCP 粗排模型，融合上游召回与下游精排多级未曝光样本，用 Margin InfoNCE 联合优化一致性与长尾精度，JD 线上 UCVR+14.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;LLM 零样本重排&quot;,&quot;Chain-of-Thought 节点图&quot;,&quot;多准则均衡&quot;]"><strong><a href="multi_task_reranking/llm4rerank_llm_based_auto_reranking_framework_for_recommendations/">LLM4Rerank: LLM-based Auto-Reranking Framework for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将准确性/多样性/公平性抽象为全连接图节点，用零样本 LLM 以 CoT 多跳自动重排，三数据集上多准则均衡超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;邻居列表反事实训练&quot;,&quot;非自回归采样生成&quot;]"><strong><a href="multi_task_reranking/nlgr_utilizing_neighbor_lists_for_generative_rerank_in_personalized_recomme/">NLGR: Utilizing Neighbor Lists for Generative Rerank in Personalized Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用邻居列表相对奖励对齐评估器-生成器目标，并以采样式非自回归生成灵活跳转组合空间，美团线上 CTR +3.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;Prompt Tuning&quot;,&quot;Continual Learning&quot;]"><strong><a href="multi_task_reranking/pcl_prompt_based_continual_learning_for_user_modeling_in_recommender_system/">PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用位置级+上下文提示做持续用户建模，冻结骨干仅训提示，缓解灾难性遗忘，6/8 任务超越 MTL 上界。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;多任务融合&quot;,&quot;强化学习&quot;,&quot;单调融合单元&quot;]"><strong><a href="multi_task_reranking/xmtf_a_formula_free_model_for_reinforcement_learning_based_multi_task_fusio/">xMTF: A Formula-Free Model for Reinforcement-Learning-Based Multi-Task Fusion in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习单调融合单元替代预定义融合公式，配合两阶段混合训练，离线总观看时长提升约7%，线上日观看时长+0.833%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;非平稳环境&quot;,&quot;状态抽象&quot;,&quot;安全探索&quot;]"><strong><a href="rl_bandit_rec/auro_reinforcement_learning_for_adaptive_user_retention_optimization_in_rec/">AURO: Reinforcement Learning for Adaptive User Retention Optimization in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用值对齐状态抽象+拒绝采样守护探索，解决留存优化RL在非平稳环境下的适应与隐式冷启动难题，KuaiSim与线上A/B全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;Contextual Combinatorial Semi-Bandit&quot;,&quot;可学习偏置项&quot;,&quot;分层 UCB&quot;]"><strong><a href="rl_bandit_rec/interactive_visualization_recommendation_with_hier_sucb/">Interactive Visualization Recommendation with Hier-SUCB</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将可视化推荐建模为带可学习偏置项的分层上下文组合半臂老虎机，把遗憾界从 $O(\sqrt{T\ln^3(nm^2T\ln T)})$降至$O(\sqrt{T\ln^3(m^2T\ln T)})$，在 Plot.ly 上 80/160 轮超越离线 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;离线强化学习&quot;,&quot;扩散世界模型&quot;,&quot;奖励塑形&quot;]"><strong><a href="rl_bandit_rec/reward_balancing_revisited_enhancing_offline_reinforcement_learning_for_rec/">Reward Balancing Revisited: Enhancing Offline Reinforcement Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散世界模型显式建模奖励不确定性，并以衰减加权的交互/熵双惩罚平衡多样性与长期收益，在三个真实数据集上刷新离线RL推荐SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/排序策略优化&quot;,&quot;TD 分解&quot;,&quot;长期价值估计&quot;,&quot;动作探索鲁棒性&quot;]"><strong><a href="rl_bandit_rec/value_function_decomposition_in_markov_recommendation_process/">Value Function Decomposition in Markov Recommendation Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将标准 TD 损失分解为策略随机与用户环境随机两个子目标，实现更快收敛、更强探索鲁棒性的长期价值估计。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回+精排+重排&quot;,&quot;创作者侧推荐&quot;,&quot;用户可用性计算&quot;,&quot;双边市场公平性&quot;]"><strong><a href="debias_fairness_industrial/creator_side_recommender_system_challenges_designs_and_applications/">Creator-Side Recommender System: Challenges, Designs, and Applications</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出创作者侧推荐系统 DualRec，将用户侧召回/排序镜像为物品找用户，并用用户可用性计算 UAC 解决用户可达性问题，快手线上 DAC 提升 2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;敏感内容标签&quot;,&quot;算法放大&quot;,&quot;数据集构建&quot;]"><strong><a href="debias_fairness_industrial/datasets_for_navigating_sensitive_topics_in_recommendation_systems/">Datasets for Navigating Sensitive Topics in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 ML-DDD 与 AO3 两个带敏感内容警告标签的推荐数据集，提出 Warning Amplification 指标，发现个性化 CF 反而去放大敏感内容。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测协议&quot;,&quot;无偏估计&quot;,&quot;随机曝光数据集&quot;,&quot;去偏推荐&quot;]"><strong><a href="debias_fairness_industrial/debias_can_be_unreliable_mitigating_bias_issue_in_evaluating_debiasing_reco/">Debias Can be Unreliable: Mitigating Bias Issue in Evaluating Debiasing Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示随机曝光数据集上传统 Recall 评估与全曝光真值不一致，提出 URE 无偏估计方案并重估去偏方法。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;分布鲁棒优化&quot;,&quot;扩散去噪&quot;,&quot;OOD 泛化&quot;]"><strong><a href="debias_fairness_industrial/distributionally_robust_graph_out_of_distribution_recommendation_via_diffus/">Distributionally Robust Graph Out-of-Distribution Recommendation via Diffusion Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 DRO 推荐对噪声分布过度加权之弊，用 VGAE+扩散去噪、Sinkhorn 距离与熵正则重构最坏情形优化，OOD 场景最高提升 75.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;因果推断&quot;,&quot;扩散模型&quot;,&quot;OOD 泛化&quot;]"><strong><a href="debias_fairness_industrial/graph_representation_learning_via_causal_diffusion_for_out_of_distribution_/">Graph Representation Learning via Causal Diffusion for Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构因果模型定位环境混杂因子，以变分推断+扩散模型学习环境不变图表征，OOD 场景最高提升 36.73%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;重排&quot;]"><strong><a href="debias_fairness_industrial/joint_evaluation_of_fairness_and_relevance_in_recommender_systems_with_pare/">Joint Evaluation of Fairness and Relevance in Recommender Systems with Pareto Frontier</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 DPFR：以测试集构造相关性-公平性帕累托前沿，用模型得分到前沿的欧氏距离作为联合评测指标，揭示现有联合指标排序与帕累托最优解不一致。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;Shapley Value&quot;,&quot;Embedding Pruning&quot;,&quot;On-device Recommendation&quot;]"><strong><a href="debias_fairness_industrial/on_device_content_based_recommendation_with_single_shot_embedding_pruning_a/">On-device Content-based Recommendation with Single-shot Embedding Pruning: A Cooperative Game Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Shapley 值一次性剪枝 CTR 嵌入表，配合场感知码本替代零填充，无需重训即可适配任意端侧存储预算。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;在线实验/因果评估&quot;,&quot;共生偏差&quot;,&quot;SUTVA违背&quot;,&quot;聚类随机化&quot;]"><strong><a href="debias_fairness_industrial/reducing_symbiosis_bias_through_better_a_b_tests_of_recommendation_algorith/">Reducing Symbiosis Bias Through Better A/B Tests of Recommendation Algorithms</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示推荐算法共享训练数据导致的共生偏差，理论建模并仿真对比三种实验设计，用国家级分流实验验证偏差存在。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;类别感知公平度量&quot;,&quot;公平正则化&quot;,&quot;性别偏差&quot;]"><strong><a href="debias_fairness_industrial/unmasking_gender_bias_in_recommendation_systems_and_enhancing_category_awar/">Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出6个类别感知的性别偏差度量（含排序折扣）与GBS聚合，并以类别覆盖差作正则项，在3数据集上偏差降50–77%而性能损失有限。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
