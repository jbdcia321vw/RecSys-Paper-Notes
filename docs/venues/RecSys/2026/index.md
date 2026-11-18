---
hide:
  - toc
---

# 🏛️ RecSys 2026 推荐系统论文解读

82 篇 RecSys 2026 论文解读，覆盖 13 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 生成式推荐 ×21 · 去偏公平与可信推荐 ×14 · 大语言模型推荐 ×11 · 多任务学习 ×7 · LLM-as-a-Judge ×4 · 冷启动 ×4 · 评测基准 ×4 · 语义ID ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">23</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">21</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-a-Judge" aria-pressed="false">LLM-as-a-Judge<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent" aria-pressed="false">LLM Agent<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="BPR 温度缩放" aria-pressed="false">BPR 温度缩放<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Capacity-constrained SID" aria-pressed="false">Capacity-constrained SID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Attribute" aria-pressed="false">Chain-of-Attribute<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contrastive Joint Training" aria-pressed="false">Contrastive Joint Training<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DPO 对齐" aria-pressed="false">DPO 对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DPP 多样性正则" aria-pressed="false">DPP 多样性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Embedding-Native Retrieval" aria-pressed="false">Embedding-Native Retrieval<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习" aria-pressed="false">GRPO 强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GTV 预估" aria-pressed="false">GTV 预估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="In-Context Learning" aria-pressed="false">In-Context Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Inverse Theory of Mind" aria-pressed="false">Inverse Theory of Mind<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存" aria-pressed="false">KV缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存复用" aria-pressed="false">KV缓存复用<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LambdaRank" aria-pressed="false">LambdaRank<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM Agent 编排" aria-pressed="false">LLM Agent 编排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 列表重排" aria-pressed="false">LLM 列表重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 反事实推理" aria-pressed="false">LLM 反事实推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 红队攻防" aria-pressed="false">LLM 红队攻防<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Judge" aria-pressed="false">LLM-as-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Optimizer" aria-pressed="false">LLM-as-Optimizer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Margin 稳定性证书" aria-pressed="false">Margin 稳定性证书<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MMoE" aria-pressed="false">MMoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="pass^k 可靠性" aria-pressed="false">pass^k 可靠性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Persona Synthesis" aria-pressed="false">Persona Synthesis<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Reveal-Tagged Elicitation" aria-pressed="false">Reveal-Tagged Elicitation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RLVR" aria-pressed="false">RLVR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Shared LLM Backbone" aria-pressed="false">Shared LLM Backbone<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Token Merge" aria-pressed="false">Token Merge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Tool Learning" aria-pressed="false">Tool Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三方博弈" aria-pressed="false">三方博弈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文感知列表重排" aria-pressed="false">上下文感知列表重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下游奖励建模" aria-pressed="false">下游奖励建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="串并解码" aria-pressed="false">串并解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="云边协同" aria-pressed="false">云边协同<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="人格路由" aria-pressed="false">人格路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="代理嵌入对齐" aria-pressed="false">代理嵌入对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="代理标签学习" aria-pressed="false">代理标签学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务冲突隔离" aria-pressed="false">任务冲突隔离<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏差控制函数" aria-pressed="false">位置偏差控制函数<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏置" aria-pressed="false">位置偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩缓存" aria-pressed="false">低秩缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息流生态" aria-pressed="false">信息流生态<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="候选集采样" aria-pressed="false">候选集采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免日志评估" aria-pressed="false">免日志评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全栈统一" aria-pressed="false">全栈统一<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动语义塔" aria-pressed="false">冷启动语义塔<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动路由" aria-pressed="false">冷启动路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层向量量化" aria-pressed="false">分层向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分布级对比监督" aria-pressed="false">分布级对比监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分数压缩" aria-pressed="false">分数压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分段加权" aria-pressed="false">分段加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="加权二分类后训练" aria-pressed="false">加权二分类后训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态分块" aria-pressed="false">动态分块<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同信号注入" aria-pressed="false">协同信号注入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协方差建模" aria-pressed="false">协方差建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效" aria-pressed="false">参数高效<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双哈希变更分类" aria-pressed="false">双哈希变更分类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔检索" aria-pressed="false">双塔检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔语义检索" aria-pressed="false">双塔语义检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲流形对齐" aria-pressed="false">双曲流形对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双表面策略解耦" aria-pressed="false">双表面策略解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实DPO" aria-pressed="false">反事实DPO<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实因果推断" aria-pressed="false">反事实因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实数据增强" aria-pressed="false">反事实数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反思式重试" aria-pressed="false">反思式重试<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+粗排" aria-pressed="false">召回+粗排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/特征增强" aria-pressed="false">召回/特征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性" aria-pressed="false">可复现性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性基准" aria-pressed="false">可复现性基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释性" aria-pressed="false">可解释性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释性诊断" aria-pressed="false">可解释性诊断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐评估" aria-pressed="false">可解释推荐评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可验证奖励" aria-pressed="false">可验证奖励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成多轮对话" aria-pressed="false">合成多轮对话<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化" aria-pressed="false">向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="响应生成评估" aria-pressed="false">响应生成评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果去偏" aria-pressed="false">因果去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="固定效应回归" aria-pressed="false">固定效应回归<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图消息传递" aria-pressed="false">图消息传递<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线实验平台" aria-pressed="false">在线实验平台<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线持续学习" aria-pressed="false">在线持续学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线解码" aria-pressed="false">在线解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基准效度" aria-pressed="false">基准效度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量向量索引" aria-pressed="false">增量向量索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量搜索个性化" aria-pressed="false">增量搜索个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多Token预测" aria-pressed="false">多Token预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多专家舰队" aria-pressed="false">多专家舰队<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣表征" aria-pressed="false">多兴趣表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体协同过滤" aria-pressed="false">多智能体协同过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体架构搜索" aria-pressed="false">多智能体架构搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态内容表征" aria-pressed="false">多模态内容表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态大模型" aria-pressed="false">多模态大模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态数据集" aria-pressed="false">多模态数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标检索" aria-pressed="false">多目标检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多语言检索" aria-pressed="false">多语言检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="奖励工程" aria-pressed="false">奖励工程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="奖励黑客" aria-pressed="false">奖励黑客<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子空间划分" aria-pressed="false">子空间划分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="定性访谈研究" aria-pressed="false">定性访谈研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实体接地" aria-pressed="false">实体接地<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="容量探针" aria-pressed="false">容量探针<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比熵惊讶度" aria-pressed="false">对比熵惊讶度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话音乐推荐" aria-pressed="false">对话音乐推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小语言模型" aria-pressed="false">小语言模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层级建模" aria-pressed="false">层级建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性偏好预测" aria-pressed="false">属性偏好预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入新鲜度" aria-pressed="false">嵌入新鲜度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入表哈希" aria-pressed="false">嵌入表哈希<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业级全链路架构与评测基准" aria-pressed="false">工业级全链路架构与评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="差分隐私后处理" aria-pressed="false">差分隐私后处理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并列感知评估" aria-pressed="false">并列感知评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="延迟反馈" aria-pressed="false">延迟反馈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构内容排序" aria-pressed="false">异构内容排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异步在线推理" aria-pressed="false">异步在线推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化微调对齐" aria-pressed="false">强化微调对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成员推理攻击" aria-pressed="false">成员推理攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对排序学习" aria-pressed="false">成对排序学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对比较特征" aria-pressed="false">成对比较特征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对转移模型" aria-pressed="false">成对转移模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑蒸馏" aria-pressed="false">拓扑蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑连通性安全" aria-pressed="false">拓扑连通性安全<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="控制器式对话编排" aria-pressed="false">控制器式对话编排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理崩溃" aria-pressed="false">推理崩溃<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提供方公平性" aria-pressed="false">提供方公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="收敛效度" aria-pressed="false">收敛效度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="整页生成" aria-pressed="false">整页生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序用户建模" aria-pressed="false">时序用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时长去偏" aria-pressed="false">时长去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="智能体即用户" aria-pressed="false">智能体即用户<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="智能体自改进" aria-pressed="false">智能体自改进<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光份额预测" aria-pressed="false">曝光份额预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光公平性审计" aria-pressed="false">曝光公平性审计<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光感知日志" aria-pressed="false">曝光感知日志<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="末项依赖" aria-pressed="false">末项依赖<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="权重共享" aria-pressed="false">权重共享<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询改写" aria-pressed="false">查询改写<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="标签偏移" aria-pressed="false">标签偏移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="概率用户聚类" aria-pressed="false">概率用户聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型选择稳定性" aria-pressed="false">模型选择稳定性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态偏置" aria-pressed="false">模态偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差主导" aria-pressed="false">残差主导<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式用户画像" aria-pressed="false">流式用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合召回分数校准" aria-pressed="false">混合召回分数校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合嵌入" aria-pressed="false">混合嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合检索" aria-pressed="false">混合检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="渐进式后训练" aria-pressed="false">渐进式后训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在扩散模型" aria-pressed="false">潜在扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在推理" aria-pressed="false">潜在推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征重要性排序" aria-pressed="false">特征重要性排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式检索" aria-pressed="false">生成式检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户基础模型" aria-pressed="false">用户基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像蒸馏" aria-pressed="false">用户画像蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户研究" aria-pressed="false">用户研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户表征学习" aria-pressed="false">用户表征学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社会选择聚合" aria-pressed="false">社会选择聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离策略评估" aria-pressed="false">离策略评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线候选池选择" aria-pressed="false">离线候选池选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评估" aria-pressed="false">离线评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏反馈偏差" aria-pressed="false">稀疏反馈偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏嵌入" aria-pressed="false">稀疏嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏特征选择" aria-pressed="false">稀疏特征选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗到细语义分层" aria-pressed="false">粗到细语义分层<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="约束解码" aria-pressed="false">约束解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构保持投影" aria-pressed="false">结构保持投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放律" aria-pressed="false">缩放律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="置换一致性" aria-pressed="false">置换一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联合分布逼近" aria-pressed="false">联合分布逼近<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自动机器学习" aria-pressed="false">自动机器学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自定义Feed生态" aria-pressed="false">自定义Feed生态<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督序列预训练" aria-pressed="false">自监督序列预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应层次聚类" aria-pressed="false">自适应层次聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应路由" aria-pressed="false">自适应路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="虚拟用户生成" aria-pressed="false">虚拟用户生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="规范对齐" aria-pressed="false">规范对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视频推荐用户研究" aria-pressed="false">视频推荐用户研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="训练种子方差" aria-pressed="false">训练种子方差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="记忆反思机制" aria-pressed="false">记忆反思机制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测协议" aria-pressed="false">评测协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID生成式检索" aria-pressed="false">语义ID生成式检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义画像" aria-pressed="false">语义画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="货架生成" aria-pressed="false">货架生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长序列压缩" aria-pressed="false">超长序列压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长序列建模" aria-pressed="false">超长序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨平台代理" aria-pressed="false">跨平台代理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="路径式端到端检索" aria-pressed="false">路径式端到端检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轮播界面点击模型" aria-pressed="false">轮播界面点击模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软量化语义对齐" aria-pressed="false">软量化语义对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轻量排序头" aria-pressed="false">轻量排序头<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="辅助头" aria-pressed="false">辅助头<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="连续训练" aria-pressed="false">连续训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐维门控" aria-pressed="false">逐维门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="锚点混合" aria-pressed="false">锚点混合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾罕见药推荐" aria-pressed="false">长尾罕见药推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列压缩" aria-pressed="false">长序列压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期留存优化" aria-pressed="false">长期留存优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐变量测量模型" aria-pressed="false">隐变量测量模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐式CoT蒸馏" aria-pressed="false">隐式CoT蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护推荐" aria-pressed="false">隐私保护推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私泄露" aria-pressed="false">隐私泄露<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零碰撞线性探测" aria-pressed="false">零碰撞线性探测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零观测唤醒" aria-pressed="false">零观测唤醒<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (8)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;用户表征学习&quot;,&quot;多专家舰队&quot;,&quot;免日志评估&quot;]"><strong><a href="collaborative_filtering/mosaic_a_fleet_of_user_embedding_specialists_for_recommendation_at_meta/">Mosaic: A Fleet of User Embedding Specialists for Recommendation at Meta</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以四类异构专家模型族构建用户表征舰队，配合 MRM 与余弦冗余损失最大化增量信息，并提出免日志 CoEval 评估，离线 NE 稳定下降、线上 A/B 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入表哈希&quot;,&quot;零碰撞线性探测&quot;,&quot;嵌入新鲜度&quot;]"><strong><a href="collaborative_filtering/multi_probe_zero_collision_hash_mpzch_mitigating_embedding_collisions_and_e/">Multi-Probe Zero Collision Hash (MPZCH): Mitigating Embedding Collisions and Enhancing Model Freshness in Large-Scale Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性探测+身份/元数据辅助张量与主动淘汰，在工业级嵌入表上实现零碰撞并提升新内容新鲜度，训练/推理延迟几乎不变。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔语义检索&quot;,&quot;混合召回分数校准&quot;,&quot;多语言检索&quot;]"><strong><a href="collaborative_filtering/multilingual_semantic_retrieval_for_apple_music_search/">Multilingual Semantic Retrieval for Apple Music Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>305M 多语双塔 + 课程式多目标微调，配合分位数校准混合召回，离线 Hit@10 相对提升 69%，线上尾部查询 CR 提升 7.93% 且无结果率降 86%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;权重共享&quot;,&quot;参数高效&quot;]"><strong><a href="collaborative_filtering/parameter_efficient_single_collaborative_branch_for_recommendation/">Parameter-Efficient Single Collaborative Branch for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单支路权重共享替代用户/物品双塔，参数量减半，精度持平且覆盖率与流行度偏差显著改善。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;混合嵌入&quot;,&quot;增量搜索个性化&quot;]"><strong><a href="collaborative_filtering/personalizing_incremental_video_search_with_hybrid_text_and_id_embeddings/">Personalizing Incremental Video Search with Hybrid Text and ID Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 Apple TV 逐键增量搜索，用文本语义与 ID 协同双嵌入注入 XGBoost 排序器，短前缀 NDCG@10 提升 8.63%，线上 CTR +1.14%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;多智能体架构搜索&quot;,&quot;奖励黑客&quot;]"><strong><a href="collaborative_filtering/recevolve_a_knowledge_driven_autonomous_agent_system_for_recommender_system/">RecEvolve: A Knowledge-Driven Autonomous Agent System for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔召回模型的架构搜索全生命周期交给多智能体闭环，41 次生产级训练换来 NDCG@50 相对 +19.9%、线上满意度 +3.77%，并意外暴露奖励黑客漏洞。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;混合检索&quot;,&quot;自适应路由&quot;]"><strong><a href="collaborative_filtering/smart_llm_augmented_hybrid_retrieval_for_dynamic_product_ads/">SMART: LLM-Augmented Hybrid Retrieval for Dynamic Product Ads</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示规则查询擅长 BM25 重定向、LLM 查询擅长 ANN 拓客的检索分解，用输出审计门控仅路由 10% 用户，线上转化率 +27.6%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回+粗排&quot;,&quot;生成式检索&quot;,&quot;全栈统一&quot;,&quot;KV缓存复用&quot;]"><strong><a href="collaborative_filtering/unipinrec_unifying_generative_retrieval_and_ranking_at_pinterest_scale/">UniPinRec: Unifying Generative Retrieval and Ranking at Pinterest Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MAM 掩码动作建模 + 混合样本 + 跨阶段 KV 缓存复用，将生成式召回与排序全栈统一，线上延迟降 11.1%、QPS 升 63.6%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;容量探针&quot;,&quot;基准效度&quot;,&quot;成对转移模型&quot;]"><strong><a href="sequential_rec/do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态分块&quot;,&quot;对比熵惊讶度&quot;,&quot;长序列压缩&quot;]"><strong><a href="sequential_rec/dp_rec_towards_dynamic_patching_for_efficient_long_sequence_recommendation/">DP-Rec: Towards Dynamic Patching for Efficient Long-Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用对比熵惊讶度动态切分长交互序列为变长行为块，在 FLOPs 匹配下 NDCG@10 最高提升 21.1%，实现更优效率-精度前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;超长序列建模&quot;,&quot;Token Merge&quot;,&quot;KV缓存&quot;]"><strong><a href="sequential_rec/longer_scaling_up_long_sequence_modeling_in_industrial_recommenders/">LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局Token+Token Merge+混合因果注意力把工业超长行为序列端到端建模推到1万长度，离线AUC+1.57%、线上广告与电商双场景显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;潜在推理&quot;,&quot;多兴趣表征&quot;]"><strong><a href="sequential_rec/recrec_latent_interests_recursive_reasoning_for_sequential_recommendation/">RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦推理态与预测态、以 R 个潜在兴趣做递归推理的免 RL 序列推荐框架，Full-Ranking 下 NDCG@10 最高提升 14.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;残差主导&quot;,&quot;末项依赖&quot;,&quot;可解释性诊断&quot;]"><strong><a href="sequential_rec/residual_dominance_as_a_structural_account_of_last_item_reliance_in_causal_/">Residual Dominance as a Structural Account of Last-Item Reliance in Causal Self-Attention Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用范数分解揭示残差连接主导同位置信息保留，并以推理期残差缩放诊断因果自注意力推荐器的末项依赖机制。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;超长序列压缩&quot;,&quot;低秩缓存&quot;]"><strong><a href="sequential_rec/sequenceo1_end_to_end_ultra_long_100k_sequence_modeling_in_recommendation_w/">SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可学习原型压缩 100K 历史为定长 sketch 并缓存复用，端到端精排训练/推理 FLOPs 降 49.9×/63.9×，线上 Finish +2.33%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;零观测唤醒&quot;,&quot;逐维门控&quot;]"><strong><a href="sequential_rec/zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (12)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;双曲流形对齐&quot;,&quot;协同信号注入&quot;]"><strong><a href="generative_rec/addressing_cross_stage_decoupling_of_semantic_and_collaborative_signals_in_/">Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SCRec，通过协同增强分词、语义引导生成与双曲流形对齐三模块双向补全语义与协同信号，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 相对提升 18.7%–26.6%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;生成式推荐&quot;,&quot;粗到细语义分层&quot;,&quot;分层向量量化&quot;]"><strong><a href="generative_rec/cofirec_coarse_to_fine_tokenization_for_generative_recommendation/">CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将物品元数据按类目→标题→描述→CF信号分层独立量化，自回归由粗到细解码，Leave-One-Out全库排序下平均超越SOTA约6.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;分布级对比监督&quot;,&quot;软量化语义对齐&quot;]"><strong><a href="generative_rec/distribution_level_contrastive_supervision_for_generative_recommendation/">Distribution-Level Contrastive Supervision for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分词器多层码本概率分布作软语义目标，以负对称KL+BPR对比监督对齐推荐器表征，即插即用提升多种生成式骨干。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Embedding-Native Retrieval&quot;,&quot;Shared LLM Backbone&quot;,&quot;Contrastive Joint Training&quot;]"><strong><a href="generative_rec/egr_embedding_native_generative_retrieval_with_a_shared_llm/">EGR: Embedding-Native Generative Retrieval with a Shared LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一共享 LLM 同时编码物品与用户历史，以双对比损失联合训练，免量化免 grounding，线上 CVR +2.91%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;跨域序列推荐&quot;,&quot;串并解码&quot;]"><strong><a href="generative_rec/empowering_cross_domain_sequential_recommendation_with_hybrid_tokenization_/">Empowering Cross-Domain Sequential Recommendation with Hybrid Tokenization and Serial-Parallel Decoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>跨域共享-专属多塔 RQ-VAE 混合分词 + 两段式串并解码，Leave-One-Out 全库排序下精度 +1.5%、时延 -85.1%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;整页生成&quot;,&quot;加权二分类后训练&quot;,&quot;约束解码&quot;]"><strong><a href="generative_rec/genpage_towards_end_to_end_generative_homepage_construction_at_netflix/">GenPage: Towards End-to-End Generative Homepage Construction at Netflix</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单个 Transformer 将用户上下文 token 化后自回归生成整页多行首页，WBC/RL 后训练，线上 +0.24% 核心参与度且延迟降 20%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;货架生成&quot;,&quot;语义ID生成式检索&quot;,&quot;LLM-as-a-Judge&quot;]"><strong><a href="generative_rec/hypothesis_driven_shelf_generation_for_personalised_recommendation/">Hypothesis-Driven Shelf Generation for Personalised Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自然语言货架假设替代固定模板，解耦规划与目录履约，生成式检索+货架对齐显著提升个性化供给</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;自适应层次聚类&quot;]"><strong><a href="generative_rec/not_all_branches_are_equal_adaptive_semantic_id_construction_for_generative/">Not All Branches Are Equal: Adaptive Semantic ID Construction for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将语义 ID 重构为树路径，用轮廓系数自适应分支+概率质量约束构建 AST，五域 Amazon 全库排序平均相对提升 13.08%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;向量量化&quot;,&quot;在线解码&quot;]"><strong><a href="generative_rec/tokens_are_all_you_need_dual_purpose_semantic_ids_for_achieving_llm_level_i/">Tokens are All You Need: Dual-purpose Semantic IDs for Achieving LLM-Level I/O Efficiency in Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 RQ-VAE 语义ID 同时承担协同身份与内容重建双职责，以 SiDec 在线解码替代稠密向量 I/O，检索 HitRate@100 提升至 0.2910 且吞吐较原始稠密嵌入快 20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID量化&quot;,&quot;拓扑蒸馏&quot;,&quot;RQ-VAE&quot;]"><strong><a href="generative_rec/topology_aware_tokenization_for_generative_recommendation/">Topology-Aware Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 RQ-VAE 语义 ID 量化中的拓扑失真问题，提出由粗到细的三级拓扑蒸馏 TopoTok，在 Amazon 三数据集 Full-Ranking 下 Recall@5 最高提升 9.42%。</p>
</dd>
</dl>
<p><a href="generative_rec/">查看全部12篇「生成式推荐与离散语义 ID」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (11)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回/特征增强&quot;,&quot;小语言模型&quot;,&quot;属性偏好预测&quot;,&quot;时序用户建模&quot;]"><strong><a href="llm_rec/deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;反事实数据增强&quot;,&quot;长尾罕见药推荐&quot;]"><strong><a href="llm_rec/improving_rare_medication_recommendation_with_counterfactual_data_augmentat/">Improving Rare Medication Recommendation with Counterfactual Data Augmentation and Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 反事实数据增强 + 病历摘要指令微调 + 逐药自回归生成，缓解罕见药长尾失效，Group-1 Jaccard 最高 +30.2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;RLVR&quot;,&quot;推理崩溃&quot;,&quot;人格路由&quot;]"><strong><a href="llm_rec/llm_reasoning_for_subjective_tasks_failure_modes_mitigation_and_dynamic_rea/">LLM Reasoning for Subjective Tasks: Failure Modes, Mitigation, and Dynamic Reasoning Routing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示主观验证任务上显式推理与标准 RLVR 会引发推理崩溃，提出条件长度惩罚奖励与人格路由蓝图，Netflix 四任务 macro-F1 最高回升至 0.851。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户画像蒸馏&quot;,&quot;异步在线推理&quot;]"><strong><a href="llm_rec/llm_based_user_personas_for_recommendations_at_scale/">LLM-Based User Personas for Recommendations at Scale</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>工业级实时生成自然语言用户兴趣画像，蒸馏+异步推理支撑十亿用户在线服务，A/B 测试观看时长 +0.04%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;可解释推荐&quot;,&quot;离线候选池选择&quot;,&quot;成对排序学习&quot;]"><strong><a href="llm_rec/pairwise_ranking_outperforms_single_action_rl_for_offline_explanation_selec/">Pairwise Ranking Outperforms Single-Action RL for Offline Explanation Selection: A Practical Lesson</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 解释生成离线化为冻结候选池，用 CPU 端 LambdaRank 逐候选打分选择，F1=0.500 超越 G-Refer/XRec 且延迟&lt;100ms。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;LLM Agent&quot;,&quot;Tool Learning&quot;,&quot;记忆反思机制&quot;]"><strong><a href="llm_rec/personalized_recommendation_tool_learning_via_autonomous_language_agents/">Personalized Recommendation Tool Learning via Autonomous Language Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>让 LLM 智能体做个性化工具选择、传统推荐模型做全库打分，以架构设计绕开幻觉与上下文长度限制，Full-Ranking 下 R@10 最高提升 68.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;渐进式后训练&quot;,&quot;强化微调对齐&quot;]"><strong><a href="llm_rec/progressive_alignment_of_recommender_foundation_model_through_multi_phase_p/">Progressive Alignment of Recommender Foundation Model through Multi-Phase Post-Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将推荐基础模型后训练拆为 LP→FFT→RFT 三阶段，用稠密隐式反馈训练策略、稀疏业务指标仅训练奖励模型，离线 Rank NDCG 0.437→0.463，线上全漏斗指标显著优于非 FM 基线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;流式用户画像&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="llm_rec/recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化画像更新器。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;LLM Agent&quot;,&quot;奖励工程&quot;,&quot;自动机器学习&quot;]"><strong><a href="llm_rec/self_evolving_recommendation_system_end_to_end_autonomous_model_optimizatio/">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Gemini 双智能体（快循环提名+慢循环上线）自动改写优化器/架构/奖励函数，在 YouTube 线上 A/B 取得显著北极星指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;查询改写&quot;,&quot;路径式端到端检索&quot;]"><strong><a href="llm_rec/spear_selection_aware_personalized_end_to_end_adaptive_rewriting_and_retrie/">SPEAR: Selection-aware Personalized End-to-end Adaptive Rewriting and Retrieval for Community Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以梯度隔离双嵌入+乘性门控+动态改写选择器统一查询改写与检索，离线点击召回@10提升99.5%，线上QVCTR+0.259%。</p>
</dd>
</dl>
<p><a href="llm_rec/">查看全部11篇「大语言模型推荐 (LLM4Rec)」论文 →</a></p>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (9)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;可验证奖励&quot;,&quot;pass^k 可靠性&quot;,&quot;Reveal-Tagged Elicitation&quot;]"><strong><a href="agent_conversational_rec/rec_a_verifiable_benchmark_for_agentic_recommender_systems/">$τ$-Rec: A Verifiable Benchmark for Agentic Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可验证目录谓词奖励、reveal-tagged 信息揭示机制与 pass^k 可靠性指标替代 LLM-as-judge，揭示智能体推荐在隐藏约束下的可靠性悬崖。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;智能体即用户&quot;,&quot;三方博弈&quot;,&quot;跨平台代理&quot;]"><strong><a href="agent_conversational_rec/a_position_paper_on_recommender_systems_in_the_era_of_autonomous_agents/">A Position Paper on Recommender Systems in the Era of Autonomous Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出人-智能体-平台三方混合推荐生态，将智能体视为用户侧独立代理，重构任务定义、评估范式与四大研究方向。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;合成多轮对话&quot;,&quot;智能体自改进&quot;,&quot;LLM-as-a-Judge&quot;]"><strong><a href="agent_conversational_rec/bootstrapping_conversational_recommendation_agents_at_spotify_synthetic_dat/">Bootstrapping Conversational Recommendation Agents At Spotify: Synthetic Data Generation and Self-Improvement Loops</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用多轮合成对话+方差归因自改进循环冷启动对话推荐智能体，线上A/B提升14%收听与5%周活。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;召回&quot;,&quot;增量向量索引&quot;,&quot;双哈希变更分类&quot;,&quot;控制器式对话编排&quot;]"><strong><a href="agent_conversational_rec/conversational_recommendation_over_live_e_commerce_catalogues_with_self_ref/">Conversational Recommendation over Live E-Commerce Catalogues with Self-Refreshing Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向实时电商商品库的自刷新检索式对话购物助手，以双哈希增量同步替代全量重建，同步开销降至全量 1.8%–12.3%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;Inverse Theory of Mind&quot;,&quot;Persona Synthesis&quot;,&quot;LLM 反事实推理&quot;]"><strong><a href="agent_conversational_rec/inverse_theory_of_mind_modeling_for_content_recommendation_from_web_browsin/">Inverse Theory of Mind Modeling for Content Recommendation: From Web Browsing to Dynamic Intelligent Interfaces</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 从浏览轨迹反向推断信念并合成多假设人格画像，在 OPeRA 上超越真实人格基线并驱动 VisionOS 空间界面。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;评测基准&quot;,&quot;LLM-as-a-Judge&quot;,&quot;响应生成评估&quot;,&quot;对话音乐推荐&quot;]"><strong><a href="agent_conversational_rec/llm_as_a_judge_for_evaluating_system_responses_in_conversational_music_reco/">LLM-as-a-Judge for Evaluating System Responses in Conversational Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个以 400 条专家标注实证 LLM-as-a-Judge 在对话音乐推荐响应评估中与人类判断的对齐度，并给出条件化与规模选择指南。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;生成式推荐&quot;,&quot;LLM Agent 编排&quot;,&quot;实体接地&quot;,&quot;反思式重试&quot;]"><strong><a href="agent_conversational_rec/melo_a_production_llm_powered_music_recommendation_agent/">Melo: A Production LLM-Powered Music Recommendation Agent</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>网易云音乐生产级 LLM 音乐推荐 Agent，以五节点确定性状态图 + 推理时实体接地 + 反思式重试解决实体幻觉与长尾塌缩，线上 A/B 留存提升超 2pp。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM Agent&quot;,&quot;语义画像&quot;,&quot;DPO 对齐&quot;]"><strong><a href="agent_conversational_rec/shape_your_feed_an_llm_based_agentic_system_for_conversational_recommendati/">Shape Your Feed: An LLM-based Agentic System for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三层 Agentic 架构（感知/服务/自进化）实现多模态实时用户可控重排，离线对齐打分 98.85% 准确率，线上 dislike 降 2.74%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;LLM-as-a-Judge&quot;,&quot;可解释推荐评估&quot;,&quot;用户研究&quot;]"><strong><a href="agent_conversational_rec/the_utility_of_llms_in_recommender_systems_explanation_evaluation/">The Utility of LLMs in Recommender Systems Explanation Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 14 个 LLM 作为推荐解释评审员与 216 名人类评分的一致性，发现排序相关中等但绝对评分一致性低，并给出四条实用准则。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;多模态数据集&quot;,&quot;可复现性基准&quot;]"><strong><a href="multimodal_rec/binge_watch_reproducible_multimodal_benchmarks_datasets_for_large_scale_mov/">Binge Watch: Reproducible Multimodal Benchmark Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>为 ML-10M/20M 构建可复现多模态基准 M3L（剧情/海报/预告片文本-图像-视频-音频），保留&gt;90%交互并验证音频模态的关键增益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;隐变量测量模型&quot;,&quot;时长去偏&quot;]"><strong><a href="multimodal_rec/mend_the_measurement_gap_latent_user_preference_modeling_for_short_form_vid/">Mend the Measurement Gap: Latent User Preference Modeling for Short-Form Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将异构行为反馈视为隐价值状态的噪声测量，用受限基线+因子化路由瓶颈解耦时长混杂，YouTube Shorts 线上主指标提升 2.67%</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;社交与群组推荐&quot;,&quot;评测协议&quot;,&quot;并列感知评估&quot;,&quot;分数压缩&quot;,&quot;BPR 温度缩放&quot;]"><strong><a href="social_group_rec/are_we_really_making_progress_in_group_recommendation_unmasking_the_tie_bre/">Are We Really Making Progress in Group Recommendation? Unmasking the Tie-Breaking Illusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示群组推荐中训练期 sigmoid 分数压缩与评估期确定性破平交互导致的系统性评估偏差，提出并列感知 HR/NDCG 精确期望协议，使多个 SOTA 方法报告增益大幅缩水。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;LLM-as-Judge&quot;,&quot;社会选择聚合&quot;]"><strong><a href="social_group_rec/consensus_vs_dissent_dynamic_llm_modeling_of_subjective_preferences_in_grou/">Consensus vs. Dissent: Dynamic LLM Modeling of Subjective Preferences in Group Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用人类问卷+DeepSeek-V3.1 蒸馏 CoT 微调 Llama/OLMo 作实时判断模型，动态挑选社会选择聚合策略，用户研究(n=284)中公平与共识得分最高。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;多模态大模型&quot;,&quot;代理嵌入对齐&quot;]"><strong><a href="cold_start_rec/idproxy_ctr_prediction_with_multimodal_llms_for_cold_start_recommendation_a/">IDProxy: CTR Prediction with Multimodal LLMs for Cold-Start Recommendation at Xiaohongshu</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MLLM 生成代理 ID 嵌入，经粗到细两阶段对齐工业 CTR 排序器，新笔记在线 AUC 提升约 2 倍于全局流量。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;稀疏嵌入&quot;,&quot;多模态内容表征&quot;]"><strong><a href="cold_start_rec/learning_sparse_representations_of_multimodal_content_for_enhanced_cold_ite/">Learning Sparse Representations of Multimodal Content for Enhanced Cold Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将多模态内容冷启动表征改为 top-K 稀疏嵌入，并用 α-entmax 预稀疏激活诱导锐化与去噪，冷启动 NDCG@20 提升 16.6%–75.5% 且存储更低。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;冷启动路由&quot;,&quot;LambdaRank&quot;,&quot;信息流生态&quot;]"><strong><a href="cold_start_rec/predicting_custom_feed_returns_for_new_bluesky_posts_a_prospective_study/">Predicting Custom-Feed Returns for New Bluesky Posts: A Prospective Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将新帖作为查询、信息流作为候选的冷启动路由任务，构建 collect-first/label-later 基准，LambdaRank 达 cR@10 0.7361。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;课程学习&quot;,&quot;流行度去偏&quot;,&quot;连续训练&quot;]"><strong><a href="cold_start_rec/soft_curriculum_learning_for_optimizing_fresh_and_generalized_recommendatio/">Soft Curriculum Learning for Optimizing Fresh and Generalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用损失退火与图内权重替代刚性数据过滤，在连续训练中打破流行度反馈环，线上冷启动互动 +5.70%、满意度 +0.54% 且吞吐无损。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;生成式推荐&quot;,&quot;隐式CoT蒸馏&quot;,&quot;反事实DPO&quot;]"><strong><a href="poi_spatiotemporal_rec/generative_spatiotemporal_intent_sequence_recommendation_via_implicit_reaso/">Generative Spatiotemporal Intent Sequence Recommendation via Implicit Reasoning in Amap</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将LLM显式CoT压缩为隐式潜token并配合时空反事实DPO，在Amap工业数据与线上A/B中实现低延迟意图序列生成。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;虚拟用户生成&quot;,&quot;提供方公平性&quot;,&quot;潜在扩散模型&quot;]"><strong><a href="poi_spatiotemporal_rec/give_the_long_tail_more_space_promoting_provider_fairness_in_next_poi_recom/">Give the Long-tail More SPACE: Promoting Provider Fairness in Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以社区推断+非平衡最优传输配额+约束引导潜扩散生成虚拟用户，在物理可达与供给约束下提升长尾POI曝光，公平性最高提升94.11%且精度不降。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="ctr_feature_interaction">🎯 CTR/CVR 预估与特征交叉 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户基础模型&quot;,&quot;自监督序列预训练&quot;,&quot;LLM-as-Optimizer&quot;]"><strong><a href="ctr_feature_interaction/building_a_user_foundation_model_for_the_open_web/">Building a User Foundation Model for the Open Web</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在身份碎片化的开放网络 RTB 场景下，用 MLM+时序切分对比预训练 Transformer 用户序列编码器，并以 LLM-in-the-loop NAS 优化训练管线，线上 A/B 取得 +2.13% CTR、-1.13% eCPC。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;GTV 预估&quot;,&quot;层级建模&quot;,&quot;代理标签学习&quot;]"><strong><a href="ctr_feature_interaction/hierarchical_gtv_estimation_bridging_the_gap_between_container_level_rankin/">Hierarchical GTV Estimation: Bridging the Gap between Container-Level Ranking and Item-Level Conversion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向容器式电商的层级 GTV 预估框架 HGE，用集合感知编码器 + 代理标签分解 + 残差融合解决粒度错配，离线 XAUC +0.63pp、线上 RPS +1.42%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;稀疏特征选择&quot;,&quot;特征重要性排序&quot;]"><strong><a href="ctr_feature_interaction/lo_far_a_cost_aware_local_filter_for_sparse_feature_ranking_in_industrial_a/">LO-FAR: A Cost-Aware Local Filter for Sparse Feature Ranking in Industrial Ad Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用CPU级逐特征独立局部估计器替代GPU重训式置换/门控重要性，在475个稀疏ID-list特征上以约2 CPU小时完成排序并保持下游NE增益。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (7)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;子空间划分&quot;,&quot;多目标检索&quot;]"><strong><a href="multi_task_reranking/embedding_subspace_partitioning_for_dynamic_multi_objective_retrieval/">Embedding Subspace Partitioning for Dynamic Multi-Objective Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双塔嵌入切分为任务感知正交子空间，用服务时可调权重的加权内积替代单一点积，单模型即可在线遍历多目标 Pareto 前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;异构内容排序&quot;,&quot;MoE&quot;]"><strong><a href="multi_task_reranking/heterogeneous_ranking_in_industrial_scale_recommender_systems_a_case_study/">Heterogeneous Ranking in Industrial-Scale Recommender Systems: A Case Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向 Google Discover 异构信息流的 HA-MoE 多门控专家架构，配合 LENS 可观测性与 DL-AUC 双层评估，离线与线上均获稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;辅助头&quot;,&quot;联合分布逼近&quot;,&quot;协方差建模&quot;]"><strong><a href="multi_task_reranking/learned_cross_task_relationships_in_multi_task_models/">Learned Cross-Task Relationships in Multi-Task Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用交叉标签/无条件辅助头显式逼近任务两两联合分布，在 YouTube 三大场景线上 A/B 验证迁移与 RL 状态增强双重收益。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;轻量排序头&quot;,&quot;在线持续学习&quot;,&quot;任务冲突隔离&quot;]"><strong><a href="multi_task_reranking/lightweight_ranking_heads_accelerating_multi_task_experimentation_in_produc/">Lightweight Ranking Heads: Accelerating Multi-Task Experimentation in Production Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 stop-gradient + 无状态日训的轻量头动态注入新任务，免骨干重训，将多任务实验周期从数周压缩至数天。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;延迟反馈&quot;,&quot;分段加权&quot;,&quot;MMoE&quot;]"><strong><a href="multi_task_reranking/multi_objective_ranking_for_live_streaming_balancing_fresh_and_delayed_sign/">Multi-Objective Ranking for Live-Streaming: Balancing Fresh and Delayed Signals with Segment-Aware Targeting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Twitch 直播精排用延迟窗口+新鲜/延迟双模型分离+分段推理加权+MMoE，DAV +0.09%、D 段 ARPU +0.56%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;重排&quot;,&quot;位置偏置&quot;,&quot;置换一致性&quot;,&quot;LLM 列表重排&quot;]"><strong><a href="multi_task_reranking/position_bias_undermines_preference_consistency_in_listwise_llm_based_reran/">Position Bias Undermines Preference Consistency in Listwise LLM-Based Reranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 LLM 列表重排的候选置换视为诱导偏好系统，提出 PPI/GPI/LOC 三层一致性度量，揭示有效性、曝光均衡与置换一致性三者可背离。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;多任务学习&quot;,&quot;精排&quot;,&quot;成对比较特征&quot;,&quot;上下文感知列表重排&quot;]"><strong><a href="multi_task_reranking/towards_full_candidate_interaction_a_comprehensive_comparison_network_for_b/">Towards Full Candidate Interaction: A Comprehensive Comparison Network for Better Route Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出比较优先范式，用非重叠路段构造显式比较张量并在成对空间直接推理，线上CR +1.2%、偏航率 -0.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;下游奖励建模&quot;,&quot;长期留存优化&quot;]"><strong><a href="rl_bandit_rec/long_term_user_engagement_optimization_through_model_agnostic_downstream_re/">Long-term User Engagement Optimization through Model-agnostic Downstream Rewards Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 P2P 兔子洞深度、浅层关闭负奖励与新用例采纳三类下游奖励替代稀疏留存标签，在 Pinterest 多场景线上 A/B 中稳定提升会话与留存指标。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (14)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;多智能体协同过滤&quot;,&quot;拓扑连通性安全&quot;,&quot;LLM 红队攻防&quot;]"><strong><a href="debias_fairness_industrial/attacking_and_defending_multi_agent_collaborative_filtering_systems_through/">Attacking and Defending Multi-Agent Collaborative Filtering Systems Through Connectivity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 AgentCF 上系统复现 6 类攻击与 4 类防御，揭示候选数 k 与目录集中度 ρ 对用户/物品分区的非对称、非单调调控规律。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;差分隐私后处理&quot;,&quot;Margin 稳定性证书&quot;,&quot;锚点混合&quot;]"><strong><a href="debias_fairness_industrial/decoupled_learning_and_selection_in_slate_recommendation_for_privacy_and_st/">Decoupled Learning and Selection in Slate Recommendation for Privacy and Stability Under Noisy Scores</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将列表推荐拆为随机打分器+确定性选择器，给出隐私后处理契约与可审计的 margin 稳定性证书，锚点加权使 Flip@K 降 53–81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;工业级全链路架构与评测基准&quot;,&quot;新闻推荐&quot;,&quot;可解释性&quot;,&quot;定性访谈研究&quot;]"><strong><a href="debias_fairness_industrial/do_we_care_about_personalization_and_explainability_an_interview_study_with/">Do We Care About Personalization and Explainability? An Interview Study with News Recommendation Engineers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>对 9 家新闻机构 15 位工程师的半结构化访谈，揭示个性化采纳受编辑控制/隐私/资源约束，可解释性几乎只用于内部调试而非终端用户。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;轮播界面点击模型&quot;,&quot;离策略评估&quot;]"><strong><a href="debias_fairness_industrial/from_click_modeling_to_offline_and_off_policy_evaluation_in_carousel_recomm/">From Click Modeling to Offline and Off-Policy Evaluation in Carousel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>基于 RecGaze 眼动数据推翻轮播界面四大行为假设，提出以可观测变量为核心的点击模型设计框架、离散选择点击模型与 N2DCG 重构，并规划轮播专用 OPE 路线。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;离线评估&quot;,&quot;曝光份额预测&quot;,&quot;反事实因果推断&quot;]"><strong><a href="debias_fairness_industrial/impression_share_prediction_an_offline_evaluation_task_for_ranking_systems/">Impression Share Prediction: An Offline Evaluation Task for Ranking Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出曝光份额预测作为离线评估新任务，用因果DAG证明反事实可识别，PatchTST编码器在首小时恢复+22% L1。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;成员推理攻击&quot;,&quot;In-Context Learning&quot;,&quot;隐私泄露&quot;]"><strong><a href="debias_fairness_industrial/membership_inference_attacks_on_in_context_examples_in_llm_based_recommende/">Membership Inference Attacks on In-Context Examples in LLM-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首次针对 ICL 式 LLM 推荐系统提出 ItemMem 与 RecInertia 两种成员推理攻击，黑盒下攻击优势最高近 1.0，现有提示抽取防御基本失效。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;标签偏移&quot;,&quot;规范对齐&quot;]"><strong><a href="debias_fairness_industrial/normative_alignment_of_recommender_systems_via_internal_label_shift/">Normative Alignment of Recommender Systems via Internal Label Shift</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将编辑/公平目标建模为属性层内部标签偏移，用免重训的闭式重加权实现系统级规范对齐，KL 大幅下降且 AUC 基本无损。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;收敛效度&quot;,&quot;候选集采样&quot;,&quot;稀疏反馈偏差&quot;]"><strong><a href="debias_fairness_industrial/on_the_convergent_validity_of_offline_evaluation_designs_for_recommender_sy/">On the Convergent Validity of Offline Evaluation Designs for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以稠密真值反馈为参照，用 Kendall&#x27;s τ 检验稀疏离线评测设计的收敛效度，发现无普适最优设计且 KuaiRec 上排序近乎反转。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;云边协同&quot;,&quot;隐私保护推荐&quot;,&quot;DPP 多样性正则&quot;]"><strong><a href="debias_fairness_industrial/pricorec_a_privacy_aware_cloud_device_collaborative_framework_for_ad_recomm/">PriCoRec: A Privacy-Aware Cloud-Device Collaborative Framework for Ad Recommendation under Feature Constraints</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将敏感特征留在端侧，云侧预排加 DPP 多样性正则、端侧轻量排序加云端 logit 蒸馏，三数据集 gAUC/R@10 全面领先且时延可控。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;因果去偏&quot;,&quot;概率用户聚类&quot;]"><strong><a href="debias_fairness_industrial/probabilistic_residual_learning_for_online_recommendations/">Probabilistic Residual Learning for Online Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以即插即用的层次贝叶斯残差模型对固定基座推荐器做概率用户聚类与 do-calculus 混杂去偏，冷启动跨域 Recall@20 最高提升约 7.6 倍。</p>
</dd>
</dl>
<p><a href="debias_fairness_industrial/">查看全部14篇「去偏公平、可信隐私与工业架构」论文 →</a></p>
</section>
</div>
