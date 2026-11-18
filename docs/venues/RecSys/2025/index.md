---
hide:
  - toc
---

# 🏛️ RecSys 2025 推荐系统论文解读

86 篇 RecSys 2025 论文解读，覆盖 15 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 生成式推荐 ×10 · 冷启动 ×9 · 去偏公平与可信推荐 ×9 · 多模态推荐 ×9 · 多任务学习 ×5 · 大语言模型推荐 ×4 · 对比学习 ×3 · 曝光偏差 ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">38</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">17</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Two-Tower" aria-pressed="false">Two-Tower<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间切分" aria-pressed="false">全局时间切分<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平性评估" aria-pressed="false">公平性评估<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度偏差" aria-pressed="false">流行度偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测" aria-pressed="false">离线评测<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏自编码器" aria-pressed="false">稀疏自编码器<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本迁移" aria-pressed="false">零样本迁移<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ACT-R 认知记忆建模" aria-pressed="false">ACT-R 认知记忆建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Algorithm Performance Space" aria-pressed="false">Algorithm Performance Space<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ANN 分层索引" aria-pressed="false">ANN 分层索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ANN 向量检索" aria-pressed="false">ANN 向量检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Beta 后验过滤" aria-pressed="false">Beta 后验过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="BLUE" aria-pressed="false">BLUE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Thought Tokenization" aria-pressed="false">Chain-of-Thought Tokenization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CLIP 对比微调" aria-pressed="false">CLIP 对比微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CLIP 嵌入空间个性化" aria-pressed="false">CLIP 嵌入空间个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit 视角选择" aria-pressed="false">Contextual Bandit 视角选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR 精排" aria-pressed="false">CTR 精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DCN-v2" aria-pressed="false">DCN-v2<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GBDT vs DNN" aria-pressed="false">GBDT vs DNN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="H3 六边形网格" aria-pressed="false">H3 六边形网格<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Interval-Infused Attention" aria-pressed="false">Interval-Infused Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Journey-Aware Sparse Attention" aria-pressed="false">Journey-Aware Sparse Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KNN 邻居扩展" aria-pressed="false">KNN 邻居扩展<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Learning-to-Rank" aria-pressed="false">Learning-to-Rank<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 反思 (Reflection)" aria-pressed="false">LLM 反思 (Reflection)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 文本增强" aria-pressed="false">LLM 文本增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 画像过滤" aria-pressed="false">LLM 画像过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 知识增强" aria-pressed="false">LLM 知识增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-a-Judge" aria-pressed="false">LLM-as-a-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Annotator" aria-pressed="false">LLM-as-Annotator<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Evaluator" aria-pressed="false">LLM-as-Evaluator<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec 隐私" aria-pressed="false">LLM4Rec 隐私<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM群组决策" aria-pressed="false">LLM群组决策<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="logQ 修正" aria-pressed="false">logQ 修正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Off-Policy Evaluation" aria-pressed="false">Off-Policy Evaluation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="p-core 过滤" aria-pressed="false">p-core 过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PCA 可视化" aria-pressed="false">PCA 可视化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="QUBO/QAOA" aria-pressed="false">QUBO/QAOA<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RecSys GenAIECommerce" aria-pressed="false">RecSys GenAIECommerce<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Ricci 曲率" aria-pressed="false">Ricci 曲率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-KMeans" aria-pressed="false">RQ-KMeans<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Scaling Law" aria-pressed="false">Scaling Law<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Serendipity" aria-pressed="false">Serendipity<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SigLIP 微调" aria-pressed="false">SigLIP 微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Slate 生成" aria-pressed="false">Slate 生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Spark-Keras 等价翻译" aria-pressed="false">Spark-Keras 等价翻译<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="System-2 反思精炼" aria-pressed="false">System-2 反思精炼<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Thompson Sampling" aria-pressed="false">Thompson Sampling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TransE 平移建模" aria-pressed="false">TransE 平移建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Transformer 序列建模" aria-pressed="false">Transformer 序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Visual Grounding" aria-pressed="false">Visual Grounding<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三元组损失" aria-pressed="false">三元组损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文感知POI推荐" aria-pressed="false">上下文感知POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文老虎机" aria-pressed="false">上下文老虎机<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上游-下游范式" aria-pressed="false">上游-下游范式<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化受限召回" aria-pressed="false">个性化受限召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化流行度" aria-pressed="false">个性化流行度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主题偏置" aria-pressed="false">主题偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="价值敏感设计" aria-pressed="false">价值敏感设计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩SVD" aria-pressed="false">低秩SVD<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="保形风险控制" aria-pressed="false">保形风险控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="候选池扩充" aria-pressed="false">候选池扩充<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好优化" aria-pressed="false">偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="儿童用户群体" aria-pressed="false">儿童用户群体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元分析" aria-pressed="false">元分析<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元评测" aria-pressed="false">元评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免增广训练" aria-pressed="false">免增广训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣探索" aria-pressed="false">兴趣探索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容初始化" aria-pressed="false">内容初始化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容特征召回" aria-pressed="false">内容特征召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷热阈值" aria-pressed="false">冷热阈值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="准确率-新颖度权衡" aria-pressed="false">准确率-新颖度权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前缀 n-gram 参数化" aria-pressed="false">前缀 n-gram 参数化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数共享" aria-pressed="false">参数共享<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向匹配推荐" aria-pressed="false">双向匹配推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔对比学习" aria-pressed="false">双塔对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔用户编码器" aria-pressed="false">双塔用户编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层公平评估" aria-pressed="false">双层公平评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双效用行为建模" aria-pressed="false">双效用行为建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲几何" aria-pressed="false">双曲几何<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双注意力机制" aria-pressed="false">双注意力机制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实离线评估" aria-pressed="false">反事实离线评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实解释" aria-pressed="false">反事实解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/偏好推断" aria-pressed="false">召回/偏好推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/向量检索" aria-pressed="false">召回/向量检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回前置意图判别" aria-pressed="false">召回前置意图判别<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可控推荐" aria-pressed="false">可控推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量检索" aria-pressed="false">向量检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量模长重标定" aria-pressed="false">向量模长重标定<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去偏" aria-pressed="false">因果去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图去噪" aria-pressed="false">图去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线知识蒸馏" aria-pressed="false">在线知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域语义偏置" aria-pressed="false">域语义偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域跷跷板" aria-pressed="false">域跷跷板<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复现研究" aria-pressed="false">复现研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务排序" aria-pressed="false">多任务排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣表征" aria-pressed="false">多兴趣表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣解耦" aria-pressed="false">多兴趣解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多准则权衡" aria-pressed="false">多准则权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多分辨率地理嵌入" aria-pressed="false">多分辨率地理嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多利益相关者" aria-pressed="false">多利益相关者<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态交叉注意力" aria-pressed="false">多模态交叉注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列推荐" aria-pressed="false">多行为序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子 ID 分解" aria-pressed="false">子 ID 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实体引导兴趣建模" aria-pressed="false">实体引导兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实体表征解耦" aria-pressed="false">实体表征解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比自监督" aria-pressed="false">对比自监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比预训练" aria-pressed="false">对比预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话性能预测" aria-pressed="false">对话性能预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="局部贪心最优策略" aria-pressed="false">局部贪心最优策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入压缩" aria-pressed="false">嵌入压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入反演攻击" aria-pressed="false">嵌入反演攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入空间稳定化" aria-pressed="false">嵌入空间稳定化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入量化防御" aria-pressed="false">嵌入量化防御<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业数据集" aria-pressed="false">工业数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="差分隐私审计" aria-pressed="false">差分隐私审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托前沿近似" aria-pressed="false">帕累托前沿近似<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列模式聚类" aria-pressed="false">序列模式聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构图神经网络" aria-pressed="false">异构图神经网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构序列建模" aria-pressed="false">异构序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构用户建模" aria-pressed="false">异构用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异步离线预排序" aria-pressed="false">异步离线预排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="弱监督代理标签" aria-pressed="false">弱监督代理标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="情感感知跨域迁移" aria-pressed="false">情感感知跨域迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成员推断攻击 (MIA)" aria-pressed="false">成员推断攻击 (MIA)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散参数生成" aria-pressed="false">扩散参数生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐公平性" aria-pressed="false">扩散推荐公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散生成增强" aria-pressed="false">扩散生成增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="批内负采样" aria-pressed="false">批内负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="捆绑推荐公平性" aria-pressed="false">捆绑推荐公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="探索-利用权衡" aria-pressed="false">探索-利用权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理期搜索" aria-pressed="false">推理期搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理模型裁剪" aria-pressed="false">推理模型裁剪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐基础模型" aria-pressed="false">推荐基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="播客推荐" aria-pressed="false">播客推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="效价-唤醒对齐" aria-pressed="false">效价-唤醒对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数字档案" aria-pressed="false">数字档案<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据切分" aria-pressed="false">数据切分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据集元特征" aria-pressed="false">数据集元特征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据驱动任务划分" aria-pressed="false">数据驱动任务划分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐多样性" aria-pressed="false">新闻推荐多样性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="早停策略" aria-pressed="false">早停策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序泄漏" aria-pressed="false">时序泄漏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔建模" aria-pressed="false">时间间隔建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有效维度" aria-pressed="false">有效维度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有机/算法交互解耦" aria-pressed="false">有机/算法交互解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有界 delta 校正" aria-pressed="false">有界 delta 校正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散先验" aria-pressed="false">条件扩散先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="极值理论" aria-pressed="false">极值理论<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="核密度估计" aria-pressed="false">核密度估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="根因归因" aria-pressed="false">根因归因<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="框架默认值" aria-pressed="false">框架默认值<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块化基准评测" aria-pressed="false">模块化基准评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型反演攻击" aria-pressed="false">模型反演攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型更新策略" aria-pressed="false">模型更新策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="正交Procrustes" aria-pressed="false">正交Procrustes<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度偏置" aria-pressed="false">流行度偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度校准" aria-pressed="false">流行度校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时增强" aria-pressed="false">测试时增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时自适应" aria-pressed="false">测试时自适应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="消费者与提供者公平" aria-pressed="false">消费者与提供者公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品中心探索" aria-pressed="false">物品中心探索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品公平性" aria-pressed="false">物品公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征工程/在线推理" aria-pressed="false">特征工程/在线推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征预处理管线" aria-pressed="false">特征预处理管线<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像" aria-pressed="false">用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户研究" aria-pressed="false">用户研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="画像生成" aria-pressed="false">画像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目录失败" aria-pressed="false">目录失败<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交图种子扩散" aria-pressed="false">社交图种子扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社会选择聚合" aria-pressed="false">社会选择聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经元转向" aria-pressed="false">神经元转向<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离散选择模型" aria-pressed="false">离散选择模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离策略估计" aria-pressed="false">离策略估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线回放评估" aria-pressed="false">离线回放评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线在线一致性" aria-pressed="false">离线在线一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评估" aria-pressed="false">离线评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测协议" aria-pressed="false">离线评测协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀有转化预估" aria-pressed="false">稀有转化预估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="站外选项估计" aria-pressed="false">站外选项估计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联漏斗解耦" aria-pressed="false">级联漏斗解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联类别检索" aria-pressed="false">级联类别检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性上下文 Bandit" aria-pressed="false">线性上下文 Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组公平与个体公平" aria-pressed="false">组公平与个体公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一损失" aria-pressed="false">统一损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联合搜索推荐" aria-pressed="false">联合搜索推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自然语言查询推荐" aria-pressed="false">自然语言查询推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自编码器" aria-pressed="false">自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="虚拟用户生成" aria-pressed="false">虚拟用户生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为序列建模" aria-pressed="false">行为序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="规范目标分布" aria-pressed="false">规范目标分布<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视觉语言模型" aria-pressed="false">视觉语言模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解释一致性审计" aria-pressed="false">解释一致性审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="训练-服务一致性" aria-pressed="false">训练-服务一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="训练数据治理/模型遗忘" aria-pressed="false">训练数据治理/模型遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评估协议" aria-pressed="false">评估协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测与元评估" aria-pressed="false">评测与元评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测与可复现性审计" aria-pressed="false">评测与可复现性审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测体系" aria-pressed="false">评测体系<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准与数据集选择" aria-pressed="false">评测基准与数据集选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负反馈去害" aria-pressed="false">负反馈去害<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样策略" aria-pressed="false">负采样策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超参数优化" aria-pressed="false">超参数优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域公平性" aria-pressed="false">跨域公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域知识蒸馏" aria-pressed="false">跨域知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨渠道营销" aria-pressed="false">跨渠道营销<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨设备搜索" aria-pressed="false">跨设备搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆倾向加权" aria-pressed="false">逆倾向加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆倾向得分" aria-pressed="false">逆倾向得分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用文本嵌入" aria-pressed="false">通用文本嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用用户表征" aria-pressed="false">通用用户表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="邻接层对比视图" aria-pressed="false">邻接层对比视图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="采样 Softmax" aria-pressed="false">采样 Softmax<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复消费替换" aria-pressed="false">重复消费替换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="量子特征选择" aria-pressed="false">量子特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="金融图推荐" aria-pressed="false">金融图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列用户建模" aria-pressed="false">长序列用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="随机游走" aria-pressed="false">随机游走<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护" aria-pressed="false">隐私保护<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私风险评分" aria-pressed="false">隐私风险评分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本跨域迁移" aria-pressed="false">零样本跨域迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非参数图卷积" aria-pressed="false">非参数图卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="音乐偏好诱导" aria-pressed="false">音乐偏好诱导<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="音频表征" aria-pressed="false">音频表征<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (17)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/偏好推断&quot;,&quot;离散选择模型&quot;,&quot;核密度估计&quot;,&quot;曝光偏差&quot;]"><strong><a href="collaborative_filtering/a_non_parametric_choice_model_that_learns_how_users_choose_between_recommen/">A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 sigmoid 核密度估计学习误差分布，无需假设 MNL/ENL 即可同时推断用户偏好与选择机制，抗竞争型曝光偏差。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;重排&quot;,&quot;因果推断&quot;,&quot;上下文老虎机&quot;,&quot;跨渠道营销&quot;]"><strong><a href="collaborative_filtering/agentic_personalisation_of_cross_channel_marketing_experiences/">Agentic Personalisation of Cross-Channel Marketing Experiences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 DiD 因果估计 + Thompson 采样构建模块化营销智能体，在 640 万用户线上 RCT 中显著提升漏斗事件与 GMV。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;子 ID 分解&quot;,&quot;个性化流行度&quot;,&quot;准确率-新颖度权衡&quot;]"><strong><a href="collaborative_filtering/balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;采样 Softmax&quot;,&quot;logQ 修正&quot;,&quot;批内负采样&quot;]"><strong><a href="collaborative_filtering/correcting_the_logq_correction_revisiting_sampled_softmax_for_large_scale_r/">Correcting the LogQ Correction: Revisiting Sampled Softmax for Large-Scale Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>指出标准 logQ 修正误把正样本当作采样项，提出带置信度权重的改进修正，在公开与工业数据上稳定超越标准 logQ。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;极值理论&quot;,&quot;量子特征选择&quot;,&quot;QUBO/QAOA&quot;]"><strong><a href="collaborative_filtering/estimating_quantum_execution_requirements_for_feature_selection_in_recommen/">Estimating Quantum Execution Requirements for Feature Selection in Recommender Systems Using Extreme Value Theory</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用极值理论对 QAOA 求解 QUBO 特征选择的输出建模，估计达到经典 SA 水平所需的量子执行次数，并在模拟器与真实 QPU 上验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;信息瓶颈&quot;,&quot;多行为推荐&quot;,&quot;图去噪&quot;]"><strong><a href="collaborative_filtering/hierarchical_graph_information_bottleneck_for_multi_behavior_recommendation/">Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用信息瓶颈原理重构多行为层次推荐，以 InfoNCE 保真、HSIC 压缩、可学习边丢弃去噪，全库排序下 HR@10 最高提升 15.08%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;Learning-to-Rank&quot;,&quot;Two-Tower&quot;,&quot;GBDT vs DNN&quot;]"><strong><a href="collaborative_filtering/industry_insights_from_comparing_deep_learning_and_gbdt_models_for_e_commer/">Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 OTTO 电商 LTR 场景系统对比 DNN 与生产级 LambdaMART，Two-Tower+Softmax CE 离线 NDCG_c +4.32%，8 周线上点击 +1.86%、营收 +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;实体引导兴趣建模&quot;,&quot;对比预训练&quot;,&quot;双塔用户编码器&quot;]"><strong><a href="collaborative_filtering/ip2_entity_guided_interest_probing_for_personalized_news_recommendation/">IP2: Entity-Guided Interest Probing for Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以实体-标题对比预训练探测新闻内实体焦点，并用跨注意力双塔用户编码器建模新闻间实体引导兴趣，MIND-large AUC 达 72.06%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;特征工程/在线推理&quot;,&quot;Spark-Keras 等价翻译&quot;,&quot;训练-服务一致性&quot;,&quot;特征预处理管线&quot;]"><strong><a href="collaborative_filtering/kamae_bridging_spark_and_keras_for_seamless_ml_preprocessing/">Kamae: Bridging Spark and Keras for Seamless ML Preprocessing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 PySpark 预处理管线一对一翻译为等价 Keras 层，消除离线/在线特征工程重复实现，Expedia 线上延迟降 61%、成本降 58%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;多分辨率地理嵌入&quot;,&quot;H3 六边形网格&quot;,&quot;双塔对比学习&quot;]"><strong><a href="collaborative_filtering/location_matters_leveraging_multi_resolution_geo_embeddings_for_housing_sea/">Location Matters: Leveraging Multi-Resolution Geo-Embeddings for Housing Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 H3 多分辨率六边形网格与城市嵌入注入双塔对比学习，缓解房源地理稀疏，离线重排模拟租金转化提升 85%。</p>
</dd>
</dl>
<p><a href="collaborative_filtering/">查看全部17篇「协同过滤与双塔召回」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;邻接层对比视图&quot;,&quot;LightGCN&quot;]"><strong><a href="self_supervised_denoising/nlgcl_naturally_existing_neighbor_layers_graph_contrastive_learning_for_rec/">NLGCL: Naturally Existing Neighbor Layers Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 GNN 相邻层天然异构节点作正样本对替代图增广，免视图构建，全库排序下 NDCG@10 最高提升 12.41% 且训练最快。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;统一损失&quot;,&quot;免增广训练&quot;]"><strong><a href="self_supervised_denoising/sgcl_unifying_self_supervised_and_supervised_learning_for_graph_recommendat/">SGCL: Unifying Self-Supervised and Supervised Learning for Graph Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 BPR 与 InfoNCE 合并为单一监督图对比损失，免增广免负采样，精度与训练效率双超 SGL/SimGCL。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Transformer 序列建模&quot;,&quot;模块化基准评测&quot;]"><strong><a href="sequential_rec/esasrec_enhancing_transformer_based_recommendations_in_a_modular_fashion/">eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统消融 Transformer 序列推荐各模块增强，提出 SASRec+LiGR+SS 组合 eSASRec，在时间切分与覆盖率指标下与 HSTU 同处帕累托前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;LLM 反思 (Reflection)&quot;,&quot;Contextual Bandit 视角选择&quot;]"><strong><a href="sequential_rec/more_a_mixture_of_reflectors_framework_for_large_language_model_based_seque/">MoRE: A Mixture of Reflectors Framework for Large Language Model-Based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用显式/隐式/协同三视角离线反思 + 元反思器自改进与 Contextual Bandit 在线选视角，LLM 序列推荐全库候选下 NDCG@10 最高提升 10.1%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间间隔建模&quot;,&quot;LLM4Rec&quot;,&quot;Interval-Infused Attention&quot;]"><strong><a href="sequential_rec/not_just_what_but_when_integrating_irregular_intervals_to_llm_for_sequentia/">Not Just What, But When: Integrating Irregular Intervals to LLM for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将不规则时间间隔经 MLP 编码与 Interval-Infused Attention 注入 LLaMA-2，配合选项化提示，在 3 个 Amazon 数据集上 HR@1 平均提升 4.4%，并提出“间隔冷启动”新评测视角。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;离线评测协议&quot;,&quot;数据切分&quot;,&quot;全局时间切分&quot;,&quot;时序泄漏&quot;]"><strong><a href="sequential_rec/time_to_split_exploring_data_splitting_strategies_for_offline_evaluation_of/">Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统对比 LOO 与全局时间切分及目标/验证变体，揭示 LOO 时序泄漏与排序失真，推荐 GTS+Last/Random/Successive 协议。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长序列建模&quot;,&quot;多兴趣表征&quot;,&quot;级联类别检索&quot;]"><strong><a href="sequential_rec/user_long_term_multi_interest_retrieval_model_for_recommendation/">User Long-Term Multi-Interest Retrieval Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别感知分层双兴趣学习与指针生成式类别→物品级联检索，将召回阶段行为序列扩展至千级，淘宝离线HR@2000达16.55%，线上点击+5.54%、订单+11.01%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;RQ-VAE&quot;,&quot;前缀 n-gram 参数化&quot;]"><strong><a href="generative_rec/enhancing_embedding_representation_stability_in_recommendation_systems_with/">Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 Semantic ID prefix-ngram 分层前缀参数化，用语义聚类碰撞替代随机哈希，缓解 ID 漂移与长尾稀疏，Meta 线上 NE 提升 0.15% 且 A/A 方差降 43%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;Chain-of-Thought Tokenization&quot;,&quot;Journey-Aware Sparse Attention&quot;,&quot;多行为序列推荐&quot;]"><strong><a href="generative_rec/grace_generative_recommendation_via_journey_aware_sparse_attention_on_chain/">GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PKG 属性链式 CoT token 增强语义 ID，并以旅程感知四路稀疏注意力替代全注意力，Home 域 HR@10 相对提升 106.9%，长序列注意力计算降 48%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;扩散推荐公平性&quot;,&quot;消费者与提供者公平&quot;,&quot;多准则权衡&quot;]"><strong><a href="generative_rec/how_fair_is_your_diffusion_recommender_model/">How Fair is Your Diffusion Recommender Model?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个针对 DiffRec/L-DiffRec 的公平性实证研究，在 ML1M/FTKY 上以 Full-Ranking 评估效用与消费者/提供者公平，揭示扩散模型放大偏见而 L-DiffRec 可缓解。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散模型&quot;,&quot;Slate 生成&quot;]"><strong><a href="generative_rec/prompt_to_slate_diffusion_models_for_prompt_conditioned_slate_generation/">Prompt-to-Slate: Diffusion Models for Prompt-Conditioned Slate Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用潜空间扩散 Transformer 从纯文本 Prompt 直接生成整组 Slate，离线 NDCG 提升 17%、线上重复曝光降 13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;联合搜索推荐&quot;,&quot;RQ-KMeans&quot;]"><strong><a href="generative_rec/semantic_ids_for_joint_generative_search_and_recommendation/">Semantic IDs for Joint Generative Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统对比任务专属与跨任务语义ID构建策略，发现多任务双塔嵌入+统一RQ-KMeans语义ID在联合搜索推荐中取得最佳权衡。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;RQ-VAE&quot;,&quot;参数共享&quot;]"><strong><a href="generative_rec/semantic_ids_for_music_recommendation/">Semantic IDs for Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 RQ-VAE 残差量化生成音乐语义 ID 替代逐曲嵌入，参数量降 75%–99% 而精度不降，线上多样性显著提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;RAG&quot;,&quot;模型更新策略&quot;,&quot;兴趣探索&quot;]"><strong><a href="llm_rec/balancing_fine_tuning_and_rag_a_hybrid_strategy_for_dynamic_llm_recommendat/">Balancing Fine-tuning and RAG: A Hybrid Strategy for Dynamic LLM Recommendation Updates</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以十亿用户短视频兴趣探索系统为案例，量化对比微调与RAG的更新代价/敏捷性，提出月度微调+周内RAG刷新的混合更新策略，线上A/B显著提升满意度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;画像生成&quot;,&quot;用户研究&quot;,&quot;因果去偏&quot;]"><strong><a href="llm_rec/biases_in_llm_generated_musical_taste_profiles_for_recommendation/">Biases in LLM-Generated Musical Taste Profiles for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>通过 64 人用户研究揭示 LLM 生成音乐品味画像存在流派/国别系统性偏差，且用户认同度与下游推荐指标仅弱相关。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;大语言模型推荐&quot;,&quot;离线评测&quot;,&quot;LLM-as-a-Judge&quot;,&quot;用户画像&quot;,&quot;播客推荐&quot;]"><strong><a href="llm_rec/evaluating_podcast_recommendations_with_profile_aware_llm_as_a_judge/">Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用90天收听历史蒸馏自然语言用户画像，驱动零样本LLM做剧集级与模型级离线评判，47人用户研究中与人工判断高度一致。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;异构用户建模&quot;,&quot;域跷跷板&quot;]"><strong><a href="llm_rec/heterogeneous_user_modeling_for_llm_based_recommendation/">Heterogeneous User Modeling for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用压缩提示+[USER]锚点token将跨域异构行为压成单一用户表征，并以域重要性平滑加权缓解域跷跷板，六域全库排序R@10最高提升约30%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;对话性能预测&quot;,&quot;目录失败&quot;,&quot;自编码器&quot;]"><strong><a href="agent_conversational_rec/failure_prediction_in_conversational_recommendation_systems/">Failure Prediction in Conversational Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出监督式对话性能预测任务，用AE与L1收缩多轮预测器区分系统失败与目录失败，揭示目录缺失场景下预测精度显著下降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;用户模拟&quot;,&quot;偏好优化&quot;,&quot;推理期搜索&quot;]"><strong><a href="agent_conversational_rec/usb_rec_an_effective_framework_for_improving_conversational_recommendation_/">USB-Rec: An Effective Framework for Improving Conversational Recommendation Capability of Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用带标签用户模拟器自动构造偏好对做 RL 训练，再在推理期用内部用户模拟器+树搜索自增强，iEval 全面超越 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;评测基准与数据集选择&quot;,&quot;Algorithm Performance Space&quot;,&quot;PCA 可视化&quot;,&quot;数据集元特征&quot;]"><strong><a href="multimodal_rec/aps_explorer_navigating_algorithm_performance_spaces_for_informed_dataset_s/">APS Explorer: Navigating Algorithm Performance Spaces for Informed Dataset Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 29 算法 × 96 数据集的性能向量投影为可交互 PCA 性能空间，用可视化工具支撑可辩护的数据集选择。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;ACT-R 认知记忆建模&quot;,&quot;音频表征&quot;,&quot;序列与会话推荐&quot;]"><strong><a href="multimodal_rec/beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys GenAIECommerce&quot;,&quot;多模态推荐&quot;,&quot;召回前置意图判别&quot;,&quot;跨设备搜索&quot;,&quot;弱监督代理标签&quot;]"><strong><a href="multimodal_rec/image_seeking_intent_prediction_for_cross_device_product_search/">Image-Seeking Intent Prediction for Cross-Device Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用语音查询+商品摘要预测图像意图，以可微精度损失抑制误触发，离线 Precision 达 78.5。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉语言模型&quot;,&quot;SigLIP 微调&quot;,&quot;ANN 向量检索&quot;]"><strong><a href="multimodal_rec/improving_visual_recommendation_on_e_commerce_platforms_using_vision_langua/">Improving Visual Recommendation on E-commerce Platforms Using Vision-Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 Mercari C2C 平台微调 SigLIP 图像编码器做视觉相似召回，离线 nDCG@5 +9.1%，线上 CTR +50%、CVR +14%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;TransE 平移建模&quot;,&quot;多模态交叉注意力&quot;,&quot;自然语言查询推荐&quot;]"><strong><a href="multimodal_rec/just_ask_for_music_jam_multimodal_and_personalized_natural_language_music_r/">Just Ask for Music (JAM): Multimodal and Personalized Natural Language Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将查询建模为用户到物品的向量平移，用交叉注意力聚合音频/歌词/CF 多模态，在 JAMSessions 全库排序上 NDCG@10 相对 TalkRec 提升约 105%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;LLM-as-Annotator&quot;,&quot;知识蒸馏&quot;,&quot;个性化受限召回&quot;]"><strong><a href="multimodal_rec/llm_powered_nuanced_video_attribute_annotation_for_enhanced_recommendations/">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Gemini 多模态 LLM 做短视频“氛围”属性标注，经金/银集评估与蒸馏扩量，接入个性化受限召回，线上参与度 +0.49%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;生成式推荐&quot;,&quot;条件扩散先验&quot;,&quot;CLIP 嵌入空间个性化&quot;]"><strong><a href="multimodal_rec/personalized_image_generation_for_recommendations_beyond_catalogs/">Personalized Image Generation for Recommendations Beyond Catalogs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用轻量条件扩散先验从隐式反馈采样用户个性化 CLIP 嵌入，冻结 SD+IP-Adapter 解码，免微调实现可扩展个性化图像生成。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;Visual Grounding&quot;,&quot;LLM 文本增强&quot;,&quot;CLIP 对比微调&quot;]"><strong><a href="multimodal_rec/vl_clip_enhancing_multimodal_recommendations_via_visual_grounding_and_llm_a/">VL-CLIP: Enhancing Multimodal Recommendations via Visual Grounding and LLM-Augmented CLIP Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Grounding DINO 裁剪商品主体 + LLM 自反思改写商品描述，再微调 CLIP 双塔，Walmart 线上 CTR +18.6%、GMV +4.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;有机/算法交互解耦&quot;]"><strong><a href="multimodal_rec/yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;社交与群组推荐&quot;,&quot;评测体系&quot;,&quot;多利益相关者&quot;,&quot;价值敏感设计&quot;,&quot;数字档案&quot;]"><strong><a href="social_group_rec/a_multistakeholder_approach_to_value_driven_co_design_of_recommender_system/">A Multistakeholder Approach to Value-Driven Co-Design of Recommender System Evaluation Metrics in Digital Archives</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>通过 5×5 焦点小组将档案领域五类利益相关者价值映射为四阶段研究漏斗评估指标框架，提出研究路径质量、元数据加权相关性等指标方向。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;LLM群组决策&quot;,&quot;解释一致性审计&quot;,&quot;社会选择聚合&quot;]"><strong><a href="social_group_rec/consistent_explainers_or_unreliable_narrators_understanding_llm_generated_g/">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用社会选择聚合策略作多基线对4个LLM的群组推荐与解释做一致性审计，发现推荐近似ADD、解释常称平均但夹带未定义流行度/相似度/多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;保形风险控制&quot;,&quot;负反馈去害&quot;,&quot;重复消费替换&quot;]"><strong><a href="social_group_rec/you_don_t_bring_me_flowers_mitigating_unwanted_recommendations_through_conf/">You Don&#x27;t Bring Me Flowers: Mitigating Unwanted Recommendations Through Conformal Risk Control</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用保形风险控制对任意排序器做后处理，以“重复安全内容替换”替代纯删除，在 KuaiRand 上可证明地把不想要内容比例压到用户指定 α 以下且保住 nDCG。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;通用文本嵌入&quot;,&quot;零样本迁移&quot;,&quot;有效维度&quot;]"><strong><a href="cold_start_rec/do_we_really_need_specialization_evaluating_generalist_text_embeddings_for_/">Do We Really Need Specialization? Evaluating Generalist Text Embeddings for Zero-Shot Recommendation and Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>实证通用文本嵌入(GTE)零样本即超越领域微调模型，并以PCA有效维度揭示空间利用率与压缩收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;评测与元评估&quot;,&quot;LLM-as-Evaluator&quot;,&quot;Serendipity&quot;,&quot;用户模拟&quot;]"><strong><a href="cold_start_rec/exploring_the_potential_of_llms_for_serendipity_evaluation_in_recommender_s/">Exploring the Potential of LLMs for Serendipity Evaluation in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 模拟用户对推荐结果做惊喜度（serendipity）打分，在淘宝与 MovieLens 两个真实用户调研数据集上做元评估，最优配置 Pearson 相关达 21.5%，显著超越传统代理指标。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;物品中心探索&quot;,&quot;Beta 后验过滤&quot;]"><strong><a href="cold_start_rec/item_centric_exploration_for_cold_start_problem/">Item-centric Exploration for Cold Start Problem</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以物品为中心的 Beta 后验过滤控制，在排序后剔除“错配受众”曝光，线上探索曝光降 20% 而满意度升 50%、可推荐语料扩 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;内容初始化&quot;,&quot;有界 delta 校正&quot;,&quot;序列推荐&quot;]"><strong><a href="cold_start_rec/let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;零样本跨域迁移&quot;,&quot;域语义偏置&quot;,&quot;序列模式聚类&quot;]"><strong><a href="cold_start_rec/llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;流行度偏置&quot;,&quot;向量模长重标定&quot;,&quot;物品公平性&quot;]"><strong><a href="cold_start_rec/on_inherited_popularity_bias_in_cold_start_item_recommendation/">On Inherited Popularity Bias in Cold-Start Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示生成式冷启动模型继承热模型流行度偏置，提出基于向量模长重标定的免训练后处理去偏方法，在三个多媒体数据集上显著提升物品公平性与曝光多样性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;重排&quot;,&quot;主题偏置&quot;,&quot;公平性评估&quot;]"><strong><a href="cold_start_rec/reading_between_the_lines_a_study_of_thematic_bias_in_book_recommender_syst/">Reading Between the Lines: A Study of Thematic Bias in Book Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 BERTopic 为 Book-Crossing 打主题标签，构建数据-推荐-用户三阶段主题偏置评估框架，揭示长尾小众读者被主流主题同质化的现象。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;冷热阈值&quot;,&quot;p-core 过滤&quot;,&quot;评估协议&quot;]"><strong><a href="cold_start_rec/recommendation_is_a_dish_better_served_warm/">Recommendation Is a Dish Better Served Warm</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统量化冷启动阈值：物品侧训练交互扫描+用户侧推理历史截断，揭示 6–15 条交互的冷热分界并批判 p-core 过滤的任意性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;社交图种子扩散&quot;,&quot;KNN 邻居扩展&quot;]"><strong><a href="cold_start_rec/socripple_a_two_stage_framework_for_cold_start_video_recommendations/">SocRipple: A Two-Stage Framework for Cold-Start Video Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>社交种子+嵌入KNN涟漪扩散的两阶段冷启视频召回框架，Recall@200 提升至 12.8–13.2%，线上冷启曝光 +36% 且互动率持平。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;情感感知跨域迁移&quot;,&quot;音乐偏好诱导&quot;,&quot;效价-唤醒对齐&quot;]"><strong><a href="cross_domain_multi_scenario/affect_aware_cross_domain_recommendation_for_art_therapy_via_music_preferen/">Affect-aware Cross-Domain Recommendation for Art Therapy via Music Preference Elicitation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以音乐偏好诱导替代纯视觉诱导，提出 Mozart/Haydn/Salieri 三种情感感知跨域艺术治疗推荐，200 人用户研究验证情绪改善。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;复现研究&quot;,&quot;流行度偏差&quot;,&quot;儿童用户群体&quot;]"><strong><a href="cross_domain_multi_scenario/impacts_of_mainstream_driven_algorithms_on_recommendations_for_children_acr/">Impacts of Mainstream-Driven Algorithms on Recommendations for Children Across Domains: A Reproducibility Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>跨电影/音乐/图书三域复现儿童与主流用户偏好偏差，并引入流行度偏差指标，揭示推荐算法对儿童群体的域特异性失效。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;虚拟用户生成&quot;,&quot;跨域公平性&quot;,&quot;双注意力机制&quot;]"><strong><a href="cross_domain_multi_scenario/leave_no_one_behind_fairness_aware_cross_domain_recommender_systems_for_non/">Leave No One Behind: Fairness-Aware Cross-Domain Recommender Systems for Non-Overlapping Users</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>为无源域数据的非重叠用户生成虚拟源域画像，双注意力+限制器实现模型无关的跨域公平性纠偏，UGF最高降94.13%且整体精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;异构图神经网络&quot;,&quot;对比学习&quot;,&quot;推荐基础模型&quot;]"><strong><a href="cross_domain_multi_scenario/rankgraph_unified_heterogeneous_graph_learning_for_cross_domain_recommendat/">RankGraph: Unified Heterogeneous Graph Learning for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建跨产品异构异质图，用 GPU 加速 RGCN + 对比学习产出图 token 注入推荐基础模型，线上 CTR +0.92%、CVR +2.82%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;跨域知识蒸馏&quot;,&quot;多任务排序&quot;,&quot;零样本迁移&quot;]"><strong><a href="cross_domain_multi_scenario/zero_shot_cross_domain_knowledge_distillation_a_case_study_on_youtube_music/">Zero-shot Cross-domain Knowledge Distillation: A Case study on YouTube Music</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 YouTube 视频大教师零样本蒸馏到低流量 YouTube Music 双排序模型，离线 AUC 微增但线上新发内容互动 +11.39%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;重排&quot;,&quot;流行度校准&quot;,&quot;上下文感知POI推荐&quot;,&quot;流行度偏差&quot;]"><strong><a href="poi_spatiotemporal_rec/exploring_the_effect_of_context_awareness_and_popularity_calibration_on_pop/">Exploring the Effect of Context-Awareness and Popularity Calibration on Popularity Bias in POI Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统评测上下文感知模型与校准流行度(CP)重排在POI推荐中的去偏效果，发现二者组合可在保精度同时最贴近用户流行度画像。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;通用用户表征&quot;,&quot;DCN-v2&quot;,&quot;异构序列建模&quot;]"><strong><a href="ctr_feature_interaction/blending_sequential_embeddings_graphs_and_engineered_features_4th_place_sol/">Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>异构因果Transformer序列编码+TwHIN图+300维手工特征经DCN-v2融合，构建跨6任务通用用户画像，获挑战赛第4名。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;Thompson Sampling&quot;,&quot;逆倾向加权&quot;,&quot;反事实离线评估&quot;]"><strong><a href="ctr_feature_interaction/counterfactual_inference_under_thompson_sampling/">Counterfactual Inference under Thompson Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>推导 Thompson Sampling 下动作倾向性的精确可计算闭式解，使 IPS 类离线评估在 TS 策略下首次可行，OBP 仿真验证无偏收敛。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;实体表征解耦&quot;,&quot;对比自监督&quot;,&quot;上游-下游范式&quot;]"><strong><a href="ctr_feature_interaction/decoupled_entity_representation_learning_for_pinterest_ads_ranking/">Decoupled Entity Representation Learning for Pinterest Ads Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出解耦式上游-下游实体表征框架 DERM，三塔+对比自监督+滑动平均聚合，离线 AUC 提升 0.19%、线上 CTR +1.38%、CPA -1.61%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;在线知识蒸馏&quot;,&quot;行为序列建模&quot;]"><strong><a href="ctr_feature_interaction/exploring_scaling_laws_of_ctr_model_for_online_performance_improvement/">Exploring Scaling Laws of CTR Model for Online Performance Improvement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 SUAN 统一注意力块实现 CTR 模型三级量级缩放律，再经稀疏注意力+并行推理+在线蒸馏落地线上，CTR +2.81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;离线在线一致性&quot;,&quot;帕累托前沿近似&quot;,&quot;会话推荐&quot;]"><strong><a href="ctr_feature_interaction/identifying_offline_metrics_that_predict_online_impact_a_pragmatic_strategy/">Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用帕累托前沿近似单模型多偏好分流，在 OTTO 线上实验验证 Recall@20→CTR、OD@20→CVR、乘积→销量 的离线-在线对齐关系。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长序列用户建模&quot;,&quot;扩散生成增强&quot;,&quot;多兴趣解耦&quot;]"><strong><a href="ctr_feature_interaction/modeling_long_term_user_behaviors_with_diffusion_driven_multi_interest_netw/">Modeling Long-term User Behaviors with Diffusion-driven Multi-interest Network for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用目标正交分解构造兴趣通道做多兴趣抽取，再用条件扩散生成增强兴趣，工业 A/B 上 CTR +1.52%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;System-2 反思精炼&quot;,&quot;LLM 知识增强&quot;]"><strong><a href="ctr_feature_interaction/r4ec_a_reasoning_reflection_and_refinement_framework_for_recommendation_sys/">R4ec: A Reasoning, Reflection, and Refinement Framework for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 actor/reflection 双小模型迭代反思精炼 LLM 知识，注入 CTR 主干，AUC 相对提升 2.5%+，线上收入 +2.2%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;随机游走&quot;,&quot;规范目标分布&quot;,&quot;新闻推荐多样性&quot;]"><strong><a href="multi_task_reranking/d_rdw_diversity_driven_random_walks_for_news_recommender_systems/">D-RDW: Diversity-Driven Random Walks for News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将规范目标分布(NTD)作为线性约束嵌入随机游走采样，实现可解释、低能耗的新闻多样化重排，能耗降低一个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;非参数图卷积&quot;,&quot;CTR 精排&quot;]"><strong><a href="multi_task_reranking/non_parametric_graph_convolution_for_re_ranking_in_recommendation_systems/">Non-parametric Graph Convolution for Re-ranking in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将图卷积从训练期搬到测试期，用相似用户/物品构造增强对并加权聚合打分，重排平均提升 8.1% 而额外开销仅 0.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;扩散参数生成&quot;,&quot;测试时自适应&quot;,&quot;可控推荐&quot;]"><strong><a href="multi_task_reranking/paragon_parameter_generation_for_controllable_multi_task_recommendation/">Paragon: Parameter Generation for Controllable Multi-Task Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散模型按偏好权重条件生成适配器参数，免重训实现测试时可控多任务推荐，响应时间降低 94.6%+。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;稀有转化预估&quot;,&quot;数据驱动任务划分&quot;,&quot;推理模型裁剪&quot;]"><strong><a href="multi_task_reranking/practical_multi_task_learning_for_rare_conversions_in_ad_tech/">Practical Multi-Task Learning for Rare Conversions in Ad Tech</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以历史CVR统计自动划分软/硬转化任务，共享底层+双塔MTL提升稀有转化预估，离线AUC+0.69%、线上CPA降2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;异步离线预排序&quot;,&quot;级联漏斗解耦&quot;,&quot;候选池扩充&quot;]"><strong><a href="multi_task_reranking/radar_recall_augmentation_through_deferred_asynchronous_retrieval/">RADAR: Recall Augmentation through Deferred Asynchronous Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用离线异步全复杂度精排模型对 50 倍候选池预打分并缓存 Top-200，线上绕过召回与粗排直接注入精排，Recall@200 翻倍、线上 +0.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;离线回放评估&quot;,&quot;线性上下文 Bandit&quot;,&quot;探索-利用权衡&quot;]"><strong><a href="rl_bandit_rec/exploitation_over_exploration_unmasking_the_bias_in_linear_bandit_recommend/">Exploitation Over Exploration: Unmasking the Bias in Linear Bandit Recommender Offline Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>大规模实证揭示线性 Bandit 离线回放评估系统性偏袒纯利用策略，贪心模型在 89% 数据集上不劣于探索变体。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Off-Policy Evaluation&quot;,&quot;双向匹配推荐&quot;,&quot;逆倾向得分&quot;]"><strong><a href="rl_bandit_rec/off_policy_evaluation_and_learning_for_matching_markets/">Off-Policy Evaluation and Learning for Matching Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向双向匹配市场提出 DiPS/DPR 两阶段奖励 OPE 估计器，利用一阶段 scout 信号降方差，合成+Wantedly 真实 A/B 日志验证 MSE 与策略选择显著优于 DM/IPS/DR。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;捆绑推荐公平性&quot;,&quot;曝光偏差&quot;,&quot;双层公平评估&quot;]"><strong><a href="debias_fairness_industrial/a_reproducibility_study_of_product_side_fairness_in_bundle_recommendation/">A Reproducibility Study of Product-side Fairness in Bundle Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个捆绑推荐商品侧公平性复现研究，揭示捆绑级与物品级曝光分布不一致、指标间结论分歧，并发现用户捆绑偏好越强曝光越公平。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;三元组损失&quot;,&quot;流行度去偏&quot;]"><strong><a href="debias_fairness_industrial/leveraging_geometric_insights_in_hyperbolic_triplet_loss_for_improved_recom/">Leveraging Geometric Insights in Hyperbolic Triplet Loss for Improved Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>洛伦兹双曲三元组损失 TriplH，以物品-物品项自适应间隔替代固定 margin，兼顾精度、去流行度偏差与 3 倍推理加速。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;反事实解释&quot;,&quot;流行度去偏&quot;,&quot;LLM 画像过滤&quot;]"><strong><a href="debias_fairness_industrial/mitigating_popularity_bias_in_counterfactual_explanations_using_large_langu/">Mitigating Popularity Bias in Counterfactual Explanations using Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成用户画像并贪心剔除“出格”历史项，作为 ACCENT 反事实解释的即插即用预处理，使解释流行度分布更贴合用户偏好。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;稀疏自编码器&quot;,&quot;神经元转向&quot;,&quot;流行度去偏&quot;]"><strong><a href="debias_fairness_industrial/opening_the_black_box_interpretable_remedies_for_popularity_bias_in_recomme/">Opening the Black Box: Interpretable Remedies for Popularity Bias in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用稀疏自编码器定位流行度神经元并做激活转向，在 ML-1M/Last.fm 上以极小精度损失换取显著曝光公平提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;模型反演攻击&quot;,&quot;LLM4Rec 隐私&quot;]"><strong><a href="debias_fairness_industrial/privacy_risks_of_llm_empowered_recommender_systems_an_inversion_attack_pers/">Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次系统研究 LLM 推荐系统的 logits 反演攻击，提出相似度引导精炼，可恢复 65% 交互物品与 87% 年龄性别。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;训练数据治理/模型遗忘&quot;,&quot;成员推断攻击 (MIA)&quot;,&quot;差分隐私审计&quot;,&quot;隐私风险评分&quot;]"><strong><a href="debias_fairness_industrial/recps_privacy_risk_scoring_for_recommender_systems/">RecPS: Privacy Risk Scoring for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>基于差分隐私推导交互级隐私分数，用 RecLiRA 成员推断攻击估计 TPR/FPR，指导细粒度交互移除以兼顾隐私与效用。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回/向量检索&quot;,&quot;嵌入反演攻击&quot;,&quot;嵌入量化防御&quot;,&quot;隐私保护&quot;]"><strong><a href="debias_fairness_industrial/rethinking_the_privacy_of_text_embeddings_a_reproducibility_study_of_text_e/">Rethinking the Privacy of Text Embeddings: A Reproducibility Study of \&quot;Text Embeddings Reveal (Almost) As Much As Text\&quot;</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现 Vec2Text 嵌入反演攻击，揭示其长度敏感性，并提出免调参 8-bit 量化作为轻量隐私防御。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;Ricci 曲率&quot;,&quot;根因归因&quot;,&quot;金融图推荐&quot;]"><strong><a href="debias_fairness_industrial/ricciflowrec_a_geometric_root_cause_recommender_using_ricci_curvature_on_fi/">RicciFlowRec: A Geometric Root Cause Recommender Using Ricci Curvature on Financial Graphs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Ollivier/Forman-Ricci 曲率与 Ricci 流在动态金融图上做根因归因，以曲率风险惩罚重排资产，NDCG@10 提升 4.9%、Top-10 波动降 28%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;公平性评估&quot;,&quot;组公平与个体公平&quot;,&quot;LLM推荐&quot;]"><strong><a href="debias_fairness_industrial/stairway_to_fairness_connecting_group_and_individual_fairness/">Stairway to Fairness: Connecting Group and Individual Fairness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次用同一族度量（SD/Gini/Atk）在 3 数据集 8 组 LLMRec 运行上实证组公平与个体公平的错位，揭示组间公平掩盖组内与个体不公。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
