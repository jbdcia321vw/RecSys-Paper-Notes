---
hide:
  - toc
---

# 🏛️ CIKM 2025 推荐系统论文解读

72 篇 CIKM 2025 论文解读，覆盖 16 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 去偏公平与可信推荐 ×9 · 多模态推荐 ×9 · 生成式推荐 ×9 · 多任务学习 ×6 · 大语言模型推荐 ×6 · 冷启动 ×4 · MoE ×3 · 语义ID ×3

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">32</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">23</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续学习" aria-pressed="false">持续学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放定律" aria-pressed="false">缩放定律<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit" aria-pressed="false">Contextual Bandit<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decision Transformer" aria-pressed="false">Decision Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="EASE 闭式解" aria-pressed="false">EASE 闭式解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Expectile 回归" aria-pressed="false">Expectile 回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Folding-In" aria-pressed="false">Folding-In<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Fourier KAN" aria-pressed="false">Fourier KAN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Greedy Removal" aria-pressed="false">Greedy Removal<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习" aria-pressed="false">GRPO 强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSIC 去偏" aria-pressed="false">HSIC 去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSTU" aria-pressed="false">HSTU<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存" aria-pressed="false">KV缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LightGCN" aria-pressed="false">LightGCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 先验" aria-pressed="false">LLM 先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 知识抽取" aria-pressed="false">LLM 知识抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-a-Judge" aria-pressed="false">LLM-as-a-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM2Vec" aria-pressed="false">LLM2Vec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LVLM 嵌入" aria-pressed="false">LVLM 嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MMD 分布正则" aria-pressed="false">MMD 分布正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Next Session Prediction" aria-pressed="false">Next Session Prediction<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="OPQ 量化" aria-pressed="false">OPQ 量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PhotoSet Transformer" aria-pressed="false">PhotoSet Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prompt Engineering" aria-pressed="false">Prompt Engineering<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prompt 补偿" aria-pressed="false">Prompt 补偿<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="QLoRA适配器" aria-pressed="false">QLoRA适配器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Rerank-DPO" aria-pressed="false">Rerank-DPO<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RR 集" aria-pressed="false">RR 集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Scaling Law" aria-pressed="false">Scaling Law<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Serendipity" aria-pressed="false">Serendipity<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Session Mask" aria-pressed="false">Session Mask<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sinkhorn散度" aria-pressed="false">Sinkhorn散度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="UltraGCN" aria-pressed="false">UltraGCN<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一动作损失" aria-pressed="false">下一动作损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="专家轨迹引导" aria-pressed="false">专家轨迹引导<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交叉特征保留" aria-pressed="false">交叉特征保留<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务自适应预训练" aria-pressed="false">任务自适应预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪多任务学习" aria-pressed="false">伪多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩近似" aria-pressed="false">低秩近似<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="侧信息增强" aria-pressed="false">侧信息增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="停留时间预测" aria-pressed="false">停留时间预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局先验分布" aria-pressed="false">全局先验分布<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="关键词剪枝" aria-pressed="false">关键词剪枝<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣分布自编码器" aria-pressed="false">兴趣分布自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容无关审核" aria-pressed="false">内容无关审核<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷物品先验" aria-pressed="false">冷物品先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分组超参优化" aria-pressed="false">分组超参优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级重排" aria-pressed="false">列表级重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前缀提示学习" aria-pressed="false">前缀提示学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态步长" aria-pressed="false">动态步长<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态用户画像" aria-pressed="false">动态用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态配额分配" aria-pressed="false">动态配额分配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="半监督伪标签迁移" aria-pressed="false">半监督伪标签迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效侧调" aria-pressed="false">参数高效侧调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向信息流" aria-pressed="false">双向信息流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔模型" aria-pressed="false">双塔模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲几何" aria-pressed="false">双曲几何<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分推断" aria-pressed="false">变分推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+精排" aria-pressed="false">召回+精排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/投放策略" aria-pressed="false">召回/投放策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可扩展混合专家" aria-pressed="false">可扩展混合专家<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化码本" aria-pressed="false">向量量化码本<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去偏" aria-pressed="false">因果去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去偏标注" aria-pressed="false">因果去偏标注<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果解耦" aria-pressed="false">因果解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图协同过滤量化" aria-pressed="false">图协同过滤量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图数据增广" aria-pressed="false">图数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图聚合权重" aria-pressed="false">图聚合权重<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="场景共享-特化建模" aria-pressed="false">场景共享-特化建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="场景特定表征" aria-pressed="false">场景特定表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基准复现" aria-pressed="false">基准复现<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量推荐" aria-pressed="false">增量推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多场景 CTR 预估" aria-pressed="false">多场景 CTR 预估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头子空间匹配" aria-pressed="false">多头子空间匹配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多尺度序列抽取" aria-pressed="false">多尺度序列抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多归因学习" aria-pressed="false">多归因学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多查询注意力" aria-pressed="false">多查询注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态交叉注意力" aria-pressed="false">多模态交叉注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态兴趣交叉融合" aria-pressed="false">多模态兴趣交叉融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态原生表征" aria-pressed="false">多模态原生表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态基准评测" aria-pressed="false">多模态基准评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态对齐" aria-pressed="false">多模态对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态嵌入" aria-pressed="false">多模态嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态检索" aria-pressed="false">多模态检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态融合" aria-pressed="false">多模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多源数据融合" aria-pressed="false">多源数据融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标召回" aria-pressed="false">多目标召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为推荐" aria-pressed="false">多行为推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多语言LLM" aria-pressed="false">多语言LLM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多语言相关性判别" aria-pressed="false">多语言相关性判别<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比对齐" aria-pressed="false">对比对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入坍缩" aria-pressed="false">嵌入坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列去噪" aria-pressed="false">序列去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构信息交互" aria-pressed="false">异构信息交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异质传播概率" aria-pressed="false">异质传播概率<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="影响力最大化" aria-pressed="false">影响力最大化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="微视频推荐数据集" aria-pressed="false">微视频推荐数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="患者画像聚类" aria-pressed="false">患者画像聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="慢思考推理" aria-pressed="false">慢思考推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散策略" aria-pressed="false">扩散策略<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散规划器" aria-pressed="false">扩散规划器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理系统优化" aria-pressed="false">推理系统优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理阶段数据治理" aria-pressed="false">推理阶段数据治理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示迭代" aria-pressed="false">提示迭代<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="搜索增强推荐" aria-pressed="false">搜索增强推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据最小化" aria-pressed="false">数据最小化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本协同过滤" aria-pressed="false">文本协同过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序感知提示" aria-pressed="false">时序感知提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序状态演化" aria-pressed="false">时序状态演化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有序回归" aria-pressed="false">有序回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有序多token预测" aria-pressed="false">有序多token预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询窗口矩阵" aria-pressed="false">查询窗口矩阵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="树状上下文缓存" aria-pressed="false">树状上下文缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="树索引" aria-pressed="false">树索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="树结构学习" aria-pressed="false">树结构学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="桥接向量" aria-pressed="false">桥接向量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="检索式查询增强" aria-pressed="false">检索式查询增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模仿学习" aria-pressed="false">模仿学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态对齐" aria-pressed="false">模态对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差量化" aria-pressed="false">残差量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式推荐" aria-pressed="false">流式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式训练" aria-pressed="false">流式训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度自适应融合" aria-pressed="false">流行度自适应融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="灾难性遗忘" aria-pressed="false">灾难性遗忘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="热启动" aria-pressed="false">热启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品互补矩阵" aria-pressed="false">物品互补矩阵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征丢弃" aria-pressed="false">特征丢弃<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征互斥 HSIC" aria-pressed="false">特征互斥 HSIC<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征交叉" aria-pressed="false">特征交叉<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征变换" aria-pressed="false">特征变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征选择" aria-pressed="false">特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式数据增强" aria-pressed="false">生成式数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式表征对齐" aria-pressed="false">生成式表征对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式重排" aria-pressed="false">生成式重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户-码本图消息传递" aria-pressed="false">用户-码本图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像推断" aria-pressed="false">用户画像推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户留存" aria-pressed="false">用户留存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户级样本聚合" aria-pressed="false">用户级样本聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户自适应" aria-pressed="false">用户自适应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标条件化 Transformer" aria-pressed="false">目标条件化 Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱去噪" aria-pressed="false">知识图谱去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱融合" aria-pressed="false">知识图谱融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交模块度最大化" aria-pressed="false">社交模块度最大化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线仿真评测" aria-pressed="false">离线仿真评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测" aria-pressed="false">离线评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏MoE" aria-pressed="false">稀疏MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稳定性-可塑性权衡" aria-pressed="false">稳定性-可塑性权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="空洞卷积注意力" aria-pressed="false">空洞卷积注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="立场中立" aria-pressed="false">立场中立<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端可学习特征" aria-pressed="false">端到端可学习特征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端推荐" aria-pressed="false">端到端推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端视频编码" aria-pressed="false">端到端视频编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗排" aria-pressed="false">粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器" aria-pressed="false">线性自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="终身序列建模" aria-pressed="false">终身序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="终身行为序列" aria-pressed="false">终身行为序列<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缓存缩放律" aria-pressed="false">缓存缩放律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群体公平" aria-pressed="false">群体公平<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="群体推荐" aria-pressed="false">群体推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="翻译增强" aria-pressed="false">翻译增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="能量引导采样" aria-pressed="false">能量引导采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动出价" aria-pressed="false">自动出价<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对齐" aria-pressed="false">自监督对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自蒸馏去噪" aria-pressed="false">自蒸馏去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为克隆" aria-pressed="false">行为克隆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征去噪" aria-pressed="false">表征去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观点极化" aria-pressed="false">观点极化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观看时长建模" aria-pressed="false">观看时长建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观看时长预测" aria-pressed="false">观看时长预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视觉画像生成" aria-pressed="false">视觉画像生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="认知不确定性" aria-pressed="false">认知不确定性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义-行为对齐" aria-pressed="false">语义-行为对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义对齐" aria-pressed="false">语义对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移" aria-pressed="false">负迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样公平性" aria-pressed="false">负采样公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超复数代数" aria-pressed="false">超复数代数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态兴趣对齐" aria-pressed="false">跨模态兴趣对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软分区动态参数" aria-pressed="false">软分区动态参数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软提示调优" aria-pressed="false">软提示调优<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="辅助训练" aria-pressed="false">辅助训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="选择性聚合" aria-pressed="false">选择性聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用物品表征" aria-pressed="false">通用物品表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="部分增量训练" aria-pressed="false">部分增量训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="量化感知训练" aria-pressed="false">量化感知训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾推荐" aria-pressed="false">长尾推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="闭式解知识蒸馏" aria-pressed="false">闭式解知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私合规" aria-pressed="false">隐私合规<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="集成CVAE" aria-pressed="false">集成CVAE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非负矩阵分解" aria-pressed="false">非负矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波" aria-pressed="false">频域滤波<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;图协同过滤量化&quot;,&quot;量化感知训练&quot;,&quot;动态步长&quot;]"><strong><a href="collaborative_filtering/a_node_aware_dynamic_quantization_approach_for_graph_collaborative_filterin/">A Node-Aware Dynamic Quantization Approach for Graph Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出节点感知动态量化 GNAQ，用参数共享学习每节点量化步长并以图消息传递更新，2-bit 下 Recall@10 平均提升 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;非负矩阵分解&quot;,&quot;患者画像聚类&quot;,&quot;多源数据融合&quot;]"><strong><a href="collaborative_filtering/patient_clustering_via_integrated_profiling_of_clinical_and_digital_data/">Patient Clustering via Integrated Profiling of Clinical and Digital Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用掩码约束非负低秩分解统一融合诊断码与门户浏览/搜索行为，同时产出可解释软聚类与患者嵌入，聚类与推荐双任务全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Fourier KAN&quot;,&quot;特征变换&quot;,&quot;LightGCN&quot;]"><strong><a href="graph_rec/enhancing_graph_collaborative_filtering_with_fourierkan_feature_transformat/">Enhancing Graph Collaborative Filtering with FourierKAN Feature Transformation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Fourier KAN 替代 NGCF 中被误删的交互特征变换 W2，在 7:1:2 时间切分全库排序下 R@20 最高提升 7.8%，且可作即插即用骨干。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;知识图谱去噪&quot;,&quot;流行度自适应融合&quot;]"><strong><a href="graph_rec/spark_adaptive_low_rank_knowledge_graph_modeling_in_hybrid_geometric_spaces/">SPARK: Adaptive Low-Rank Knowledge Graph Modeling in Hybrid Geometric Spaces for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 低秩去噪 KG、SVD 初始化双曲/欧氏并行 GNN，并按物品流行度自适应融合多源信号，长尾推荐 R@10 最高提升 10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;Folding-In&quot;,&quot;热启动&quot;,&quot;UltraGCN&quot;,&quot;低秩近似&quot;]"><strong><a href="graph_rec/ultra_fast_warm_start_solution_for_graph_recommendations/">Ultra Fast Warm Start Solution for Graph Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 SVD Folding-In 的闭式解迁移到 UltraGCN，用一次矩阵-向量乘精确更新热用户嵌入，比 SGD 微调快 30 倍且质量更优、覆盖率更高。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多行为推荐&quot;,&quot;对比学习&quot;]"><strong><a href="self_supervised_denoising/a_self_supervised_mixture_of_experts_framework_for_multi_behavior_recommend/">A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示多行为推荐在已访问/未访问物品上的性能鸿沟，提出双专家 MoE + 专家自适应 SSL（访问过滤/新颖性推断对比 + 行为生成）框架，全库排序下 HR@20 最高提升 65.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨模态兴趣对齐&quot;,&quot;序列去噪&quot;,&quot;LLM2Vec&quot;]"><strong><a href="self_supervised_denoising/empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;生成式数据增强&quot;,&quot;物品互补矩阵&quot;]"><strong><a href="self_supervised_denoising/generative_data_augmentation_in_graph_contrastive_learning_for_recommendati/">Generative Data Augmentation in Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 VAE 式生成器自适应生成噪声替代固定随机扰动构造对比视图，并引入物品互补矩阵，在三个稀疏数据集上 Full-Ranking 提升约 3–6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;多尺度序列抽取&quot;,&quot;KV缓存&quot;]"><strong><a href="sequential_rec/climber_toward_efficient_scaling_laws_for_large_recommendation_models/">Climber: Toward Efficient Scaling Laws for Large Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向推荐场景改造 Transformer：多尺度子序列抽取降复杂度、自适应温度调制注意力，配合单用户多物品批处理与 KV 缓存，线上指标 +12.19%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;持续学习&quot;,&quot;灾难性遗忘&quot;,&quot;稳定性-可塑性权衡&quot;]"><strong><a href="sequential_rec/continual_recommender_systems/">Continual Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理持续推荐的两大技术路线（经验回放与正则化），并延伸至资源受限与序列交互场景，给出可信性、基础模型与搜推统一三大开放方向。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;增量推荐&quot;,&quot;时序状态演化&quot;,&quot;伪多任务学习&quot;]"><strong><a href="sequential_rec/cpmr_context_aware_incremental_sequential_recommendation_with_pseudo_multi_/">CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出伪多任务学习范式，用历史图与上下文图双通道演化用户/物品时序状态，在四个数据集上 MRR 平均提升 30.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;用户自适应&quot;,&quot;MoE&quot;]"><strong><a href="sequential_rec/muffin_mixture_of_user_adaptive_frequency_filtering_for_sequential_recommen/">MUFFIN: Mixture of User-Adaptive Frequency Filtering for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双分支频域滤波+用户自适应滤波器，兼顾全谱与频带局部建模，五数据集全库排序平均 R@10 提升 4.12%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;多查询注意力&quot;,&quot;查询窗口矩阵&quot;]"><strong><a href="sequential_rec/multi_item_query_attention_for_stable_sequential_recommendation/">Multi-Item-Query Attention for Stable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多物品查询窗口+虚拟物品替代单查询掩码注意力，缓解序列噪声敏感，ML-1M HR@5 提升 21.2%，LastFM 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;终身行为序列&quot;,&quot;下一动作损失&quot;,&quot;推理系统优化&quot;]"><strong><a href="sequential_rec/transact_v2_lifelong_user_action_sequence_modeling_on_pinterest_recommendat/">TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 NN 检索压缩终身序列 + 下一动作对比损失 + Triton 融合推理，Pinterest 线上 Repin +6.35%、Hide -12.80%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;扩散推荐&quot;,&quot;特征丢弃&quot;,&quot;表征去噪&quot;]"><strong><a href="generative_rec/asymmetric_diffusion_recommendation_model/">Asymmetric Diffusion Recommendation Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用特征丢弃替代高斯噪声做离散前向、在隐空间做非对称反向去噪，抖音音乐全量A/B活跃天数+0.131%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;端到端推荐&quot;,&quot;语义ID&quot;,&quot;对比对齐&quot;,&quot;因果去偏&quot;]"><strong><a href="generative_rec/das_dual_aligned_semantic_ids_empowered_industrial_recommender_system/">DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出单阶段双对齐语义ID框架，联合优化RQ-VAE量化与CF去偏对齐，快手全场景A/B eCPM+3.48%、冷启动+8.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;多模态嵌入&quot;,&quot;嵌入坍缩&quot;]"><strong><a href="generative_rec/empowering_large_language_model_for_sequential_recommendation_via_multimoda/">Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 LLM4SR 的嵌入坍缩与语义ID灾难性遗忘，用多模态嵌入+MMD重建+码本初始化破局，全库排序下 nDCG@5 提升 4.4%–10.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;HSTU&quot;,&quot;用户级样本聚合&quot;,&quot;交叉特征保留&quot;]"><strong><a href="generative_rec/mtgr_industrial_scale_generative_recommendation_framework_in_meituan/">MTGR: Industrial-Scale Generative Recommendation Framework in Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以用户级样本聚合复用 HSTU 编码器并保留 DLRM 交叉特征，实现 65× FLOPs 下线上 CTCVR +1.02%、推理成本降 12%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;残差量化&quot;,&quot;多模态交叉注意力&quot;]"><strong><a href="generative_rec/progressive_semantic_residual_quantization_for_multimodal_joint_interest_mo/">Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出前缀语义约束的渐进残差量化 PSRQ 与多码本交叉注意力 MCCA，缓解语义漂移与跨模态建模缺口，冷启动 AUC 显著提升并完成工业上线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Next Session Prediction&quot;,&quot;Session Mask&quot;,&quot;部分增量训练&quot;]"><strong><a href="generative_rec/tbgrecall_a_generative_retrieval_model_for_e_commerce_recommendation_scenar/">TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 Next Session Prediction 会话级非自回归生成召回，配合 Session Mask、MSP/MoE 与部分增量训练，淘宝线上成交额 +2.16%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;多语言相关性判别&quot;,&quot;任务自适应预训练&quot;,&quot;翻译增强&quot;]"><strong><a href="llm_rec/alibaba_international_e_commerce_product_search_competition_dcuragons_team_/">Alibaba International E-commerce Product Search Competition DcuRAGONs Team Technical Report</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以 Gemma-3-12B 为骨干，用翻译增强+任务自适应预训练+类别感知查询分组划分，在阿里国际电商多语言相关性竞赛夺冠（F1_avg≈89.1）。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;文本协同过滤&quot;,&quot;缩放定律&quot;,&quot;通用物品表征&quot;]"><strong><a href="llm_rec/exploring_the_upper_limits_of_text_based_collaborative_filtering_using_larg/">Exploring the Upper Limits of Text-Based Collaborative Filtering Using Large Language Models: Discoveries and Insights</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 TCF 物品编码器从 125M 一路放大到 175B，系统检验缩放极限、通用表征、对 IDCF 的竞争力与零样本迁移四项命题。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;线性自编码器&quot;,&quot;闭式解知识蒸馏&quot;,&quot;长尾推荐&quot;]"><strong><a href="llm_rec/llm_enhanced_linear_autoencoders_for_recommendation/">LLM-Enhanced Linear Autoencoders for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义矩阵构造 item-item 相关阵并以知识蒸馏正则注入 EASE 闭式解，全库排序下 R@20 平均提升 27.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;软提示调优&quot;,&quot;用户画像推断&quot;,&quot;向量量化&quot;]"><strong><a href="llm_rec/prompt_tuning_as_user_inherent_profile_inference_machine/">Prompt Tuning as User Inherent Profile Inference Machine</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户潜在画像建模为可训练软提示，用因果掩码+EM 推断、码本量化成协同 ID，AUC 最高相对提升 4.77%，已上线华为应用市场。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;特征选择&quot;,&quot;桥接向量&quot;,&quot;提示迭代&quot;]"><strong><a href="llm_rec/self_surrogate_light_feature_selection_with_large_language_models_in_deep_r/">SELF: Surrogate-light Feature Selection with Large Language Models in Deep Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多 LLM 迭代提示生成语义特征排序，再以可学习桥接向量对齐任务空间，实现轻量特征选择，AUC 全面领先且线上 CTR +3.01%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;LLM 知识抽取&quot;,&quot;语义-行为对齐&quot;]"><strong><a href="llm_rec/trawl_external_knowledge_enhanced_recommendation_with_llm_assistance/">TRAWL: External Knowledge-Enhanced Recommendation with LLM Assistance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 从原始外部文本抽取推荐知识，再以对比学习适配器对齐语义与行为空间，ML-1M 上 AUC 稳定提升且微信线上 A/B 三项指标正向。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;慢思考推理&quot;,&quot;GRPO 强化学习&quot;,&quot;用户模拟&quot;]"><strong><a href="agent_conversational_rec/starec_an_efficient_agent_framework_for_recommender_systems_via_autonomous_/">STARec: An Efficient Agent Framework for Recommender Systems via Autonomous Deliberate Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将每位用户建模为具备快/慢双认知的 LLM Agent，用 SFT 蒸馏 + GRPO 排序奖励两阶段训练，仅用 0.4% 数据即在 ML-1M/CDs 上超越全量训练基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;课程学习&quot;,&quot;知识图谱融合&quot;,&quot;前缀提示学习&quot;]"><strong><a href="agent_conversational_rec/step_stepwise_curriculum_learning_for_context_knowledge_fusion_in_conversat/">STEP: Stepwise Curriculum Learning for Context-Knowledge Fusion in Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用三阶段课程学习驱动 F-Former 对齐对话与知识图谱，再以双前缀提示注入冻结 PLM，ReDial R@1 提升 15.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;离线仿真评测&quot;,&quot;用户模拟&quot;,&quot;时序感知提示&quot;,&quot;动态用户画像&quot;]"><strong><a href="agent_conversational_rec/temporal_aware_user_behaviour_simulation_with_large_language_models_for_rec/">Temporal-Aware User Behaviour Simulation with Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用动态画像更新、时序模式抽取与自适应聚合构建 LLM 用户模拟器，在 ML-1M 上 nDCG@5 相对 RecAgent 提升约 27.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;微视频推荐数据集&quot;,&quot;端到端视频编码&quot;,&quot;多模态基准评测&quot;]"><strong><a href="multimodal_rec/a_content_driven_micro_video_recommendation_dataset_at_scale/">A Content-Driven Micro-Video Recommendation Dataset at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布十亿级含原始视频模态的微视频推荐数据集 MicroLens，并系统基准 IDRec/VIDRec/VideoRec 三类范式，揭示端到端视频编码优于冻结特征的关键结论。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;视觉画像生成&quot;,&quot;PhotoSet Transformer&quot;,&quot;OPQ 量化&quot;]"><strong><a href="multimodal_rec/bilisting_modality_alignment_for_listings/">BiListing: Modality Alignment for Listings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 生成视觉画像 + PhotoSet Transformer 聚合多图，将 CLIP 图文对齐扩展到多图多文本的房源实体，线上 NDCG +0.425%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;多模态兴趣交叉融合&quot;,&quot;兴趣分布自编码器&quot;]"><strong><a href="multimodal_rec/distribution_guided_auto_encoder_for_user_multimodal_interest_cross_fusion/">Distribution-Guided Auto-Encoder for User Multimodal Interest Cross Fusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以行为粒度做模态内/模态间交叉融合，并用兴趣分布解码器作自监督约束，工业 A/B 上 CTR +12.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;LVLM 嵌入&quot;,&quot;多模态原生表征&quot;,&quot;语义对齐&quot;]"><strong><a href="multimodal_rec/do_recommender_systems_really_leverage_multimodal_content_a_comprehensive_a/">Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用受控噪声实验证明多模态增益源于语义而非模型容量，并以 LVLM 结构化提示生成「原生多模态」[EOS] 嵌入替代晚期融合，Borda 聚合排名第一。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;流式推荐&quot;,&quot;参数高效侧调&quot;,&quot;可扩展混合专家&quot;]"><strong><a href="multimodal_rec/efficient_multimodal_streaming_recommendation_via_expandable_side_mixture_o/">Efficient Multimodal Streaming Recommendation via Expandable Side Mixture-of-Experts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>冻结 BERT/ViT 主干，用可扩展侧调 MoE 增量吸收新时段知识并按利用率剪枝，在三个流式数据集上 HR@10 全面超越 SOTA 且显存/训练时间更省。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;超复数代数&quot;,&quot;Prompt 补偿&quot;,&quot;自监督对齐&quot;]"><strong><a href="multimodal_rec/hypercomplex_prompt_aware_multimodal_recommendation/">Hypercomplex Prompt-aware Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Cayley-Dickson 超复数多分量嵌入 + 超复数乘法非线性融合 + Prompt 动态补偿，在四个 Amazon 数据集 Full-Ranking 下 Recall@20 最高提升约 13%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;树索引&quot;,&quot;终身序列建模&quot;,&quot;多模态检索&quot;]"><strong><a href="multimodal_rec/miss_multi_modal_tree_indexing_and_searching_with_lifelong_sequential_behav/">MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多模态嵌入构建树索引，并以 Co-GSU/MM-GSU 双视角检索终身行为，快手线上召回@800 相对提升 37.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态对齐&quot;,&quot;空洞卷积注意力&quot;,&quot;MMD 分布正则&quot;]"><strong><a href="multimodal_rec/modality_alignment_with_multi_scale_bilateral_attention_for_multimodal_reco/">Modality Alignment with Multi-scale Bilateral Attention for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 MambaRec，用多尺度空洞卷积+双注意力做局部图文对齐、MMD+InfoNCE 做全局分布对齐，Amazon 三数据集 Recall@20 最高提升约 5.4%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回+精排&quot;,&quot;语义ID量化&quot;,&quot;多模态对齐&quot;,&quot;端到端可学习特征&quot;]"><strong><a href="multimodal_rec/qarm_quantitative_alignment_multi_modal_recommendation_at_kuaishou/">QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用业务 item-item 对微调 MLLM 做表示对齐，再以 VQ/RQ 量化成语义码实现端到端可学习，快手广告与电商线上 GMV+2.296%、冷启曝光+5.324%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;社交模块度最大化&quot;,&quot;特征互斥 HSIC&quot;]"><strong><a href="social_group_rec/community_aware_social_community_recommendation/">Community-Aware Social Community Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 CASO，用模块度最大化、社交紧密度聚合与协同编码三路解耦用户表征，并以 HSIC 互斥与 KL 社区检测损失优化，在 6 个真实社交网络上 Full-Ranking 全面超越 9 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;影响力最大化&quot;,&quot;RR 集&quot;,&quot;异质传播概率&quot;]"><strong><a href="social_group_rec/heterogeneous_influence_maximization_in_user_recommendation/">Heterogeneous Influence Maximization in User Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将异质传播概率与 RR 集影响力最大化引入好友邀请推荐，HeteroIR/HeteroIM 在 NSpread/Recall/NDCG 全面领先并落地腾讯游戏 A/B 提升 8.5%/10%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;多语言LLM&quot;,&quot;自蒸馏去噪&quot;,&quot;检索式查询增强&quot;]"><strong><a href="cold_start_rec/csrm_llm_embracing_multilingual_llms_for_cold_start_relevance_matching_in_e/">CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>多语言 LLM 联合翻译任务、检索式查询增强与多轮自蒸馏，解决新兴市场零标注相关性匹配，离线 F1 0.965、线上缺陷率降 45.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;认知不确定性&quot;,&quot;集成CVAE&quot;,&quot;Sinkhorn散度&quot;]"><strong><a href="cold_start_rec/harnessing_light_for_cold_start_recommendations_leveraging_epistemic_uncert/">Harnessing Light for Cold-Start Recommendations: Leveraging Epistemic Uncertainty to Enhance Performance in User-Item Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用集成 CVAE 生成暖化嵌入，并以 Sinkhorn 散度版 PaiDE 估计认知不确定性做优化，在冷启动 CTR 上稳定超越 CVAR 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;多模态融合&quot;]"><strong><a href="cold_start_rec/multi_modal_adaptive_mixture_of_experts_for_cold_start_recommendation/">Multi-modal Adaptive Mixture of Experts for Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双层 MoE（模态专属专家+动态门控融合）替代拼接/平均融合，在 Amazon 三域冷启动全库排序上 Recall@10 最高提升 16.17%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;EASE 闭式解&quot;,&quot;侧信息增强&quot;,&quot;冷物品先验&quot;]"><strong><a href="cold_start_rec/shallow_autoencoding_recommender_with_cold_start_handling_via_side_features/">Shallow AutoEncoding Recommender with Cold Start Handling via Side Features</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户/物品侧特征拼入 EASE 输入矩阵，并用内容相似度矩阵作冷物品权重先验，闭式解统一处理双冷启动。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;软分区动态参数&quot;,&quot;半监督伪标签迁移&quot;]"><strong><a href="cross_domain_multi_scenario/a_soft_partitioned_semi_supervised_collaborative_transfer_learning_approach/">A Soft-partitioned Semi-supervised Collaborative Transfer Learning Approach for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用软分区动态参数缓解主导域压制、用主导域伪标签加权迁移缓解稀疏域过拟合，离线全库AUC与美团线上A/B双验证。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;搜索增强推荐&quot;,&quot;向量量化码本&quot;,&quot;用户-码本图消息传递&quot;]"><strong><a href="cross_domain_multi_scenario/benefit_from_rich_tackling_search_interaction_sparsity_in_search_enhanced_r/">Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM+RQ-VAE 将用户搜索/推荐偏好离散为码本，构建 User-Code 图做消息传递，把搜索富用户信号传给稀疏用户，Qilin 上 HR@1 相对 UniSAR 提升 12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;变分推断&quot;,&quot;场景特定表征&quot;,&quot;全局先验分布&quot;]"><strong><a href="cross_domain_multi_scenario/global_distribution_aware_scenario_specific_variational_representation_lear/">Global-Distribution Aware Scenario-Specific Variational Representation Learning Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用变分推断为每个场景下的用户/物品学习分布表征，并以全局行为模式与物品属性构造先验，缓解稀疏场景表征欠学习，即插即用提升 PEPNet/MARIA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多场景 CTR 预估&quot;,&quot;基准复现&quot;,&quot;场景共享-特化建模&quot;]"><strong><a href="cross_domain_multi_scenario/scenario_wise_rec_a_multi_scenario_recommendation_benchmark/">Scenario-Wise Rec: A Multi-Scenario Recommendation Benchmark</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个多场景推荐统一基准，整合 6 个公开数据集与 12 个 MSR 模型，统一数据处理/训练/评测管线并附工业广告数据验证。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;LLM4Rec&quot;,&quot;群体推荐&quot;,&quot;QLoRA适配器&quot;]"><strong><a href="poi_spatiotemporal_rec/harnessing_large_language_models_for_group_poi_recommendations/">Harnessing Large Language Models for Group POI Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用语义增强 POI Token + QLoRA 双适配器 + 行程目的自监督，破解群体 POI 推荐偏好多样与数据极稀疏难题，HR@10 平均提升 4.4%–9.0%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;粗排&quot;,&quot;双塔模型&quot;,&quot;生成式表征对齐&quot;,&quot;多头子空间匹配&quot;]"><strong><a href="ctr_feature_interaction/hit_model_a_hierarchical_interaction_enhanced_two_tower_model_for_pre_ranki/">HIT Model: A Hierarchical Interaction-Enhanced Two-Tower Model for Pre-Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双生成器预生成跨塔粗粒度向量+多头子空间细粒度匹配，在保持双塔预缓存效率下逼近全交互模型精度，腾讯线上GMV+1.66%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构信息交互&quot;,&quot;双向信息流&quot;,&quot;选择性聚合&quot;]"><strong><a href="ctr_feature_interaction/interformer_effective_heterogeneous_interaction_learning_for_click_through_/">InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出交错式双向信息流的 InterFormer，用独立 Cross Arch 做选择性聚合，在三个公开集与 Meta 工业集上取得 SOTA，线上 NE 提升 0.15% 且 QPS 提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;观看时长预测&quot;,&quot;有序回归&quot;,&quot;树结构学习&quot;]"><strong><a href="ctr_feature_interaction/personalized_tree_based_progressive_regression_model_for_watch_time_predict/">Personalized Tree-Based Progressive Regression Model for Watch-Time Prediction in Short Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将TPM固定满二叉树改为样本级可学习剪枝树，并用IPS校正条件建模选择偏差，离线MAE/XAUC全面领先，线上观看时长+0.428%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉&quot;,&quot;稀疏MoE&quot;,&quot;缩放定律&quot;]"><strong><a href="ctr_feature_interaction/rankmixer_scaling_up_ranking_models_in_industrial_recommenders/">RankMixer: Scaling Up Ranking Models in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用无参多头Token Mixing替代自注意力、Per-token FFN隔离特征子空间，配合ReLU路由稀疏MoE，在抖音万亿级数据上将MFU从4.5%提至45%，参数扩100倍而延迟不增。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;多归因学习&quot;,&quot;多任务学习&quot;,&quot;辅助训练&quot;]"><strong><a href="ctr_feature_interaction/see_beyond_a_single_view_multi_attribution_learning_leads_to_better_convers/">See Beyond a Single View: Multi-Attribution Learning Leads to Better Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多归因标签作为辅助任务联合学习，用笛卡尔积辅助训练增强监督信号，离线GAUC+0.51%、线上ROI+2.6%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;生成式重排&quot;,&quot;有序多token预测&quot;,&quot;Rerank-DPO&quot;]"><strong><a href="multi_task_reranking/gref_a_unified_generative_framework_for_efficient_reranking_via_ordered_mul/">GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双向编码器+动态匹配自回归解码器统一生成式重排，Rerank-DPO 端到端对齐用户偏好，OMTP 多token并行解码将延迟压至 12.97ms，快手线上转发+2.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;多目标召回&quot;,&quot;目标条件化 Transformer&quot;,&quot;动态配额分配&quot;]"><strong><a href="multi_task_reranking/mpformer_adaptive_framework_for_industrial_multi_task_personalized_sequenti/">MPFormer: Adaptive Framework for Industrial Multi-Task Personalized Sequential Retriever</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用目标条件化 Transformer 统一多目标召回，并以排序分数监督动态配额分配，在快手 4 亿 DAU 上实现 +0.43% 总时长与 66.7% 推理资源节省。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;因果解耦&quot;,&quot;停留时间预测&quot;,&quot;负迁移&quot;]"><strong><a href="multi_task_reranking/orca_mitigating_over_reliance_for_multi_task_dwell_time_prediction_with_cau/">ORCA: Mitigating Over-Reliance for Multi-Task Dwell Time Prediction with Causal Decoupling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用因果解耦显式建模并减去CTR对停留时间预测的负迁移，恢复中时长分桶语义，DT指标平均提升10.6%且不损CTR。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;列表级重排&quot;,&quot;树状上下文缓存&quot;]"><strong><a href="multi_task_reranking/you_only_evaluate_once_a_tree_based_rerank_method_at_meituan/">You Only Evaluate Once: A Tree-based Rerank Method at Meituan</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用树状多尺度上下文提取+上下文缓存实现单阶段全排列重排，离线GAUC显著提升并落地美团外卖。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (5)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;扩散策略&quot;,&quot;离线强化学习&quot;,&quot;能量引导采样&quot;]"><strong><a href="rl_bandit_rec/energy_guided_diffusion_sampling_for_long_term_user_behavior_prediction_in_/">Energy-Guided Diffusion Sampling for Long-Term User Behavior Prediction in Reinforcement Learning-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用扩散策略替代行为克隆以提升离线RL4RS策略表达能力，并以Q值能量引导采样降低随机性，六数据集与仿真环境全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;扩散规划器&quot;,&quot;自动出价&quot;,&quot;专家轨迹引导&quot;]"><strong><a href="rl_bandit_rec/expert_guided_diffusion_planner_for_auto_bidding/">Expert-Guided Diffusion Planner for Auto-bidding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用专家轨迹引导条件扩散规划器生成出价序列，配合混合强制训练与跳步采样，线上转化+11.29%、收入+12.36%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/投放策略&quot;,&quot;LLM Agent&quot;,&quot;Contextual Bandit&quot;,&quot;关键词剪枝&quot;]"><strong><a href="rl_bandit_rec/kp_agent_keyword_pruning_in_sponsored_search_advertising_via_llm_powered_co/">KP-Agent: Keyword Pruning in Sponsored Search Advertising via LLM-Powered Contextual Bandits</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将关键词剪枝建模为上下文赌博机，用 LLM Agent 生成代码片段执行剪枝，美团真实数据上累计利润最高提升 49.28%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;Decision Transformer&quot;,&quot;Expectile 回归&quot;,&quot;LLM 先验&quot;]"><strong><a href="rl_bandit_rec/maximum_in_support_return_modeling_for_dynamic_recommendation_with_language/">Maximum In-Support Return Modeling for Dynamic Recommendation with Language Model Prior</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将轨迹拼接从训练期移至动作推理期，用 expectile 回归估计最大在支撑回报并自适应截断历史，配合 LLM 先验初始化 DT，在 5 个离线数据集与 VirtualTB 仿真上全面超越 EDT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;模仿学习&quot;,&quot;用户留存&quot;,&quot;行为克隆&quot;]"><strong><a href="rl_bandit_rec/stratified_expert_cloning_for_retention_aware_recommendation_at_scale/">Stratified Expert Cloning for Retention-Aware Recommendation at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分层专家行为克隆替代 RL 探索，配合动作熵正则与自适应专家选择，在快手双平台 A/B 中提升活跃天数 0.098%/0.122%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评测&quot;,&quot;LLM-as-a-Judge&quot;,&quot;Serendipity&quot;,&quot;Prompt Engineering&quot;]"><strong><a href="debias_fairness_industrial/a_universal_framework_for_offline_serendipity_evaluation_in_recommender_sys/">A Universal Framework for Offline Serendipity Evaluation in Recommender Systems via Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM-as-a-Judge 给推荐列表逐项打 1-5 级惊喜度分，绕开人工真值与阈值定义，实现跨数据集跨模型的离线惊喜度统一评测。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果推断&quot;,&quot;图聚合权重&quot;]"><strong><a href="debias_fairness_industrial/causality_aware_graph_aggregation_weight_estimator_for_popularity_debiasing/">Causality-aware Graph Aggregation Weight Estimator for Popularity Debiasing in Top-K Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将图聚合证明为后门调整，用 VAE 式编码器-解码器估计无偏聚合权重，配合动量更新，小众物品 Recall@20 提升约 20%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;内容无关审核&quot;,&quot;立场中立&quot;,&quot;观点极化&quot;]"><strong><a href="debias_fairness_industrial/content_agnostic_moderation_for_stance_neutral_recommendation/">Content-Agnostic Moderation for Stance-Neutral Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出不依赖内容特征的立场中立化后处理审核，用双聚类离散化打散用户-物品共簇，仿真闭环中显著降低立场偏度且 CTR 仅降约 11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;负采样公平性&quot;,&quot;分组超参优化&quot;]"><strong><a href="debias_fairness_industrial/evaluating_and_addressing_fairness_across_user_groups_in_negative_sampling_/">Evaluating and Addressing Fairness Across User Groups in Negative Sampling for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>实证揭示负采样比 K 增大使活跃用户增益、非活跃用户受损，提出分组特定负采样比并用带先验 Hyperband 优化，全库排序下 NDCG 提升 2–10%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;持续学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="debias_fairness_industrial/federated_continual_recommendation/">Federated Continual Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 FCRec 新任务与 F3CRec 框架，客户端自适应回放+服务端物品级时序均值，在四个数据集三个 FedRec 骨干上 N@20 平均提升 5.6%–32.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;图数据增广&quot;,&quot;HSIC 去偏&quot;,&quot;群体公平&quot;]"><strong><a href="debias_fairness_industrial/improving_recommendation_fairness_via_graph_structure_and_representation_au/">Improving Recommendation Fairness via Graph Structure and Representation Augmentation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双先验假设定位敏感边与敏感特征，做图剪枝+特征掩码双增广，HSIC去偏+对比学习保效用，ML-1M 上 DP@10 降 3.4%、EO@10 降 7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;观看时长建模&quot;,&quot;因果去偏标注&quot;,&quot;多任务学习&quot;]"><strong><a href="debias_fairness_industrial/leveraging_watch_time_feedback_for_short_video_recommendations_a_causal_lab/">Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用分位数把观看时长转成 WPR 与二值标签，并在标注阶段按视频时长分组做因果去偏，离线全指标最优、线上观看时长 +2.23%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;缓存缩放律&quot;,&quot;目标注意力&quot;,&quot;流式训练&quot;]"><strong><a href="debias_fairness_industrial/marm_unlocking_the_future_of_recommendation_systems_through_memory_augmenta/">MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以外部缓存替代多层自注意力，将序列兴趣建模复杂度从 O(n²d) 降至 O(n·d)，离线 GAUC +0.43%、线上人均时长 +2.079%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;推理阶段数据治理&quot;,&quot;数据最小化&quot;,&quot;Greedy Removal&quot;,&quot;隐私合规&quot;]"><strong><a href="debias_fairness_industrial/what_data_is_really_necessary_a_feasibility_study_of_inference_data_minimiz/">What Data is Really Necessary? A Feasibility Study of Inference Data Minimization for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出性能约束式推理数据最小化问题与 Greedy Removal 算法，在三个隐式反馈数据集上实现最高 84.5% 数据削减且 PRR 保持 98%。</p>
</dd>
</dl>
</section>
</div>
