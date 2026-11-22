---
hide:
  - toc
---

# ⏱️ 序列与会话推荐 (Sequential & Session-Based Recommendation)

> **研究范畴**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

当前方向共收录 **713** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 大语言模型推荐 ×55 · 生成式推荐 ×55 · 去偏公平与可信推荐 ×21 · 对比学习 ×19 · 端到端生成式 ×18 · 多模态推荐 ×15 · 知识蒸馏 ×11 · 冷启动 ×10

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">381</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">297</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">184</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">55</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">55</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">44</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">27</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">25</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">24</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">21</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端生成式" aria-pressed="false">端到端生成式<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">16</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/排序" aria-pressed="false">召回/排序<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动与长尾推荐" aria-pressed="false">冷启动与长尾推荐<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR预估与特征交叉" aria-pressed="false">CTR预估与特征交叉<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI本地生活推荐" aria-pressed="false">时空轨迹与POI本地生活推荐<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/精排" aria-pressed="false">召回/精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣表征" aria-pressed="false">多兴趣表征<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列建模" aria-pressed="false">多行为序列建模<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐与离散语义ID" aria-pressed="false">生成式推荐与离散语义ID<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性注意力" aria-pressed="false">线性注意力<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波" aria-pressed="false">频域滤波<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式端到端" aria-pressed="false">生成式端到端<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Scaling Law" aria-pressed="false">Scaling Law<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列去噪" aria-pressed="false">序列去噪<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续学习" aria-pressed="false">持续学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线强化学习" aria-pressed="false">离线强化学习<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏注意力" aria-pressed="false">稀疏注意力<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decision Transformer" aria-pressed="false">Decision Transformer<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Flow Matching" aria-pressed="false">Flow Matching<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存" aria-pressed="false">KV缓存<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP-Mixer" aria-pressed="false">MLP-Mixer<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="侧信息融合" aria-pressed="false">侧信息融合<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分自编码器" aria-pressed="false">变分自编码器<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/端到端" aria-pressed="false">召回/端到端<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习与列表级重排" aria-pressed="false">多任务学习与列表级重排<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列推荐" aria-pressed="false">多行为序列推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="排序" aria-pressed="false">排序<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期行为序列建模" aria-pressed="false">长期行为序列建模<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="BERT4Rec" aria-pressed="false">BERT4Rec<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Next Basket Recommendation" aria-pressed="false">Next Basket Recommendation<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Transformer" aria-pressed="false">Transformer<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="乘积量化" aria-pressed="false">乘积量化<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话图神经网络" aria-pressed="false">会话图神经网络<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间切分" aria-pressed="false">全局时间切分<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列数据增强" aria-pressed="false">序列数据增强<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图对比学习" aria-pressed="false">意图对比学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指令微调" aria-pressed="false">指令微调<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="旋转位置编码" aria-pressed="false">旋转位置编码<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔建模" aria-pressed="false">时间间隔建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式训练" aria-pressed="false">流式训练<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态空间模型" aria-pressed="false">状态空间模型<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户行为建模" aria-pressed="false">用户行为建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="社交与群组推荐" aria-pressed="false">社交与群组推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测协议" aria-pressed="false">离线评测协议<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排（Next-Item 排序）" aria-pressed="false">精排（Next-Item 排序）<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自蒸馏" aria-pressed="false">自蒸馏<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复消费建模" aria-pressed="false">重复消费建模<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高斯分布表征" aria-pressed="false">高斯分布表征<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR预估与特征交叉" aria-pressed="false">CTR/CVR预估与特征交叉<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decoder-only Transformer" aria-pressed="false">Decoder-only Transformer<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Exposure Bias" aria-pressed="false">Exposure Bias<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Hawkes Process" aria-pressed="false">Hawkes Process<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSTU" aria-pressed="false">HSTU<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 指令微调" aria-pressed="false">LLM 指令微调<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Machine Unlearning" aria-pressed="false">Machine Unlearning<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Neural ODE" aria-pressed="false">Neural ODE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prior-Fitted Network" aria-pressed="false">Prior-Fitted Network<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sequential Recommendation" aria-pressed="false">Sequential Recommendation<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Test-Time Training" aria-pressed="false">Test-Time Training<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Transformer 序列建模" aria-pressed="false">Transformer 序列建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Wasserstein 自注意力" aria-pressed="false">Wasserstein 自注意力<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Wasserstein 距离" aria-pressed="false">Wasserstein 距离<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一购物篮推荐" aria-pressed="false">下一购物篮推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化流行度" aria-pressed="false">个性化流行度<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="互信息最大化" aria-pressed="false">互信息最大化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置编码" aria-pressed="false">位置编码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩分解" aria-pressed="false">低秩分解<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好对齐" aria-pressed="false">偏好对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好漂移" aria-pressed="false">偏好漂移<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习 MAML" aria-pressed="false">元学习 MAML<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间线" aria-pressed="false">全局时间线<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="加权物品转移图" aria-pressed="false">加权物品转移图<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态参数生成" aria-pressed="false">动态参数生成<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态图推荐" aria-pressed="false">动态图推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态图神经网络" aria-pressed="false">动态图神经网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态记忆网络" aria-pressed="false">动态记忆网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同信号注入" aria-pressed="false">协同信号注入<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与工业架构" aria-pressed="false">去偏公平与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平可信隐私与工业架构" aria-pressed="false">去偏公平可信隐私与工业架构<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图注意力" aria-pressed="false">图注意力<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线知识蒸馏" aria-pressed="false">在线知识蒸馏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量学习" aria-pressed="false">增量学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量推理" aria-pressed="false">增量推理<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣解耦" aria-pressed="false">多兴趣解耦<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多意图解耦" aria-pressed="false">多意图解耦<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列" aria-pressed="false">多行为序列<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比对齐" aria-pressed="false">对比对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业数据集" aria-pressed="false">工业数据集<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并行解码" aria-pressed="false">并行解码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列模式挖掘" aria-pressed="false">序列模式挖掘<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异质超图" aria-pressed="false">异质超图<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习策略" aria-pressed="false">强化学习策略<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知对比学习" aria-pressed="false">意图感知对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续图学习" aria-pressed="false">持续图学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序点过程" aria-pressed="false">时序点过程<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与 POI 推荐" aria-pressed="false">时空轨迹与 POI 推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间位置编码" aria-pressed="false">时间位置编码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件变分自编码器" aria-pressed="false">条件变分自编码器<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式推荐" aria-pressed="false">流式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="灾难性遗忘" aria-pressed="false">灾难性遗忘<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户疲劳建模" aria-pressed="false">用户疲劳建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经元转向" aria-pressed="false">神经元转向<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端云协同" aria-pressed="false">端云协同<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧推理" aria-pressed="false">端侧推理<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排/重排" aria-pressed="false">精排/重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="终身行为序列" aria-pressed="false">终身行为序列<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述与分类体系" aria-pressed="false">综述与分类体系<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="胶囊动态路由" aria-pressed="false">胶囊动态路由<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督预训练" aria-pressed="false">自监督预训练<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应数据增广" aria-pressed="false">自适应数据增广<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解耦表征学习" aria-pressed="false">解耦表征学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超网络" aria-pressed="false">超网络<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长用户序列" aria-pressed="false">超长用户序列<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨视图对比学习" aria-pressed="false">跨视图对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软标签" aria-pressed="false">软标签<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重听行为建模" aria-pressed="false">重听行为建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾用户" aria-pressed="false">长尾用户<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长短期兴趣解耦" aria-pressed="false">长短期兴趣解耦<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控融合" aria-pressed="false">门控融合<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护" aria-pressed="false">隐私保护<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负例挖掘" aria-pressed="false">难负例挖掘<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域去噪" aria-pressed="false">频域去噪<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域建模" aria-pressed="false">频域建模<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回/端到端策略学习&quot;,&quot;Offline RL&quot;,&quot;外推误差约束&quot;,&quot;离散随机策略&quot;]"><strong><a href="../../venues/AAAI/2021/rl_bandit_rec/a_general_offline_reinforcement_learning_framework_for_interactive_recommen/">A General Offline Reinforcement Learning Framework for Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将交互推荐写成概率生成模型的变分推断，用离散随机 Actor-Critic 加五种行为策略约束抑制外推误差，全库排序下 NDCG@10 提升 13.8%~25.2%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习少样本匹配&quot;,&quot;序列推荐&quot;,&quot;冷启动物品&quot;]"><strong><a href="../../venues/AAAI/2021/cold_start_rec/cold_start_sequential_recommendation_via_meta_learner/">Cold-start Sequential Recommendation via Meta Learner</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>Mecos 用元学习把冷启动物品推荐重构为「查询序列–支持集」度量匹配，仅靠交互数据即可即插即用地提升 SR-GNN 等模型，HR@10 平均提升 70%–99%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态记忆网络&quot;,&quot;长序列建模&quot;]"><strong><a href="../../venues/AAAI/2021/sequential_rec/dynamic_memory_based_attention_network_for_sequential_recommendation/">Dynamic Memory based Attention Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>DMAN 将超长行为序列切段并逐段训练，用循环注意力与动态胶囊记忆分别抽取短期/长期兴趣，全库排序下 HR@50 最高提升 15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;分解式对抗学习&quot;,&quot;正交正则&quot;,&quot;敏感属性去偏&quot;]"><strong><a href="../../venues/AAAI/2021/debias_fairness_industrial/fairrec_fairness_aware_news_recommendation_with_decomposed_adversarial_lear/">FairRec: Fairness-aware News Recommendation with Decomposed Adversarial Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>把用户表征解耦为偏见感知/无偏见双路，用属性预测＋对抗学习＋正交正则剥离性别偏置，全库排序下性别可推断率逼近随机，AUC 仅降 1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;会话推荐&quot;,&quot;图神经网络&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/AAAI/2021/sequential_rec/graph_enhanced_multi_task_learning_of_multi_level_transition_dynamics_for_s/">Graph-Enhanced Multi-Task Learning of Multi-Level Transition Dynamics for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>MTD 用位置感知注意力建模会话内转移、用全局图编码器+互信息最大化建模跨会话转移，多任务联合学习，三数据集 Pre@10 最高提升 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;非侵入式注意力&quot;,&quot;侧信息融合&quot;,&quot;BERT4Rec&quot;]"><strong><a href="../../venues/AAAI/2021/sequential_rec/non_invasive_self_attention_for_side_information_fusion_in_sequential_recom/">Non-invasive Self-attention for Side Information Fusion in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>NOVA 让侧信息只参与注意力 Q/K 计算、V 保持纯 ID 向量空间，全库排序下 ML-1m HR@10 相对提升 13.51%，开销几乎可忽略。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;超图卷积&quot;,&quot;对比自监督&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/AAAI/2021/sequential_rec/self_supervised_hypergraph_convolutional_networks_for_session_based_recomme/">Self-Supervised Hypergraph Convolutional Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>会话即超边构建双通道超图卷积，并以会话级互信息自监督增强，全库排序下 P@20 最高相对提升 19.7%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;元学习 MAML&quot;,&quot;动态矩阵分解&quot;,&quot;时间敏感冷启动&quot;]"><strong><a href="../../venues/AAAI/2022/cold_start_rec/a_dynamic_meta_learning_model_for_time_sensitive_cold_start_recommendations/">A Dynamic Meta-Learning Model for Time-Sensitive Cold-Start Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户隐偏好可加分解为时间特定（MAML 少样本适配）与时间演化（嵌套 RNN）两因子，在 Netflix/Last.fm/ML-1M 分期预测上 RMSE 降 1.7–5.1%、NDCG 升 3.3–7.7%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Contextual Bandit&quot;,&quot;隐空间探索&quot;,&quot;长期奖励&quot;]"><strong><a href="../../venues/AAAI/2022/rl_bandit_rec/context_uncertainty_in_contextual_bandits_with_applications_to_recommender_/">Context Uncertainty in Contextual Bandits with Applications to Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 REN，在 RNN 隐空间用 DPP 多样性项 + 表征不确定性 UCB 项联合打分，兼顾相关性与探索，理论上保持率最优次线性 regret，在线滚动评估中长期奖励显著超越 GRU4Rec/TCN/HRNN。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;序列建模/记忆压缩&quot;,&quot;可微数据选择&quot;,&quot;双层优化&quot;,&quot;梯度近似&quot;]"><strong><a href="../../venues/AAAI/2022/sequential_rec/dips_differentiable_policy_for_sketching_in_recommender_systems/">DiPS: Differentiable Policy for Sketching in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以双层优化端到端学习可微草图（sketching）策略，并用队列式近似梯度替代全历史重算，全库排序下同等精度省 50% 草图槽位。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;轨迹式元学习&quot;,&quot;时序动态建模&quot;]"><strong><a href="../../venues/AAAI/2022/sequential_rec/leaping_through_time_with_gradient_based_adaptation_for_recommendation/">Leaping Through Time with Gradient-based Adaptation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用轨迹式元学习 Leap 替代 RNN 建模时序动态：GTL 抓跨时段共性、OTL 抓短期趋势，时间切分+1:99 采样排序下 HR@1 最高相对提升 18.59%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;决策上下文建模&quot;,&quot;上下文交互单元&quot;,&quot;自适应兴趣聚合&quot;]"><strong><a href="../../venues/AAAI/2023/ctr_feature_interaction/decision_making_context_interaction_network_for_click_through_rate_predicti/">Decision-Making Context Interaction Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 DCIN 建模页面内显式比较与粗排候选隐式决策上下文，CIU 过滤无关项、AIAU 聚合目标兴趣；MeituanAds AUC+0.0042，线上 CTR+2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列&quot;,&quot;胶囊动态路由&quot;,&quot;动态GRU&quot;]"><strong><a href="../../venues/AAAI/2023/sequential_rec/dynamic_multi_behavior_sequence_modeling_for_next_item_recommendation/">Dynamic Multi-Behavior Sequence Modeling for Next Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用胶囊动态路由按用户相关性融合多行为序列，并下沉到动态 GRU 实现 item 级个性化，全库排序 H@10 最高提升 38.5%</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;集合到序列 (Set-to-Sequence)&quot;,&quot;策略梯度&quot;,&quot;知识追踪辅助任务&quot;]"><strong><a href="../../venues/AAAI/2023/sequential_rec/set_to_sequence_ranking_based_concept_aware_learning_path_recommendation/">Set-to-Sequence Ranking-based Concept-aware Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将学习路径推荐形式化为集合到序列生成，用概念自注意力编码 + LSTM 指针解码 + 策略梯度与知识追踪辅助任务，在 ASSIST09/Junyi/工业集上经 DKT/CoKT 模拟器回测显著超越规则与 RL 基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多轮自适应检索&quot;,&quot;表征适配器&quot;,&quot;FFT 可学习滤波&quot;]"><strong><a href="../../venues/AAAI/2024/sequential_rec/ada_retrieval_an_adaptive_multi_round_retrieval_paradigm_for_sequential_rec/">Ada-Retrieval: An Adaptive Multi-Round Retrieval Paradigm for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>模型无关的多轮自适应召回：物品/用户表征适配器逐轮修正用户向量并补召回，Full-Ranking 下 NDCG@50 最高 +9.07%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;傅里叶变换&quot;,&quot;过平滑缓解&quot;]"><strong><a href="../../venues/AAAI/2024/sequential_rec/an_attentive_inductive_bias_for_sequential_recommendation_beyond_the_self_a/">An Attentive Inductive Bias for Sequential Recommendation beyond the Self-Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示自注意力低通滤波导致过平滑，用 DFT 注入注意力归纳偏置并重缩放频率，6 数据集全库排序平均提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;Mobility Tree&quot;,&quot;时段偏好建模&quot;,&quot;Tree-LSTM&quot;]"><strong><a href="../../venues/AAAI/2024/poi_spatiotemporal_rec/learning_time_slot_preferences_via_mobility_tree_for_next_poi_recommendatio/">Learning Time Slot Preferences via Mobility Tree for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 Mobility Tree 分层编码签到轨迹，用四步 IAC/IRC 节点交互与多任务学习刻画多粒度时段偏好，三数据集全指标超越 10 个 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;扩散模型插件&quot;,&quot;去噪重加权与负采样&quot;]"><strong><a href="../../venues/AAAI/2024/sequential_rec/plug_in_diffusion_model_for_sequential_recommendation/">Plug-in Diffusion Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用时间间隔扩散模型对全库物品生成偏好，再以历史行为重加权、软正样本增广与无噪负采样三种插件策略回注序列推荐器，采样排序下 SASRec 提升最高 13.9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;relevant-visit selection&quot;,&quot;target-aware attention&quot;,&quot;用药推荐&quot;]"><strong><a href="../../venues/AAAI/2024/sequential_rec/vita_carefully_chosen_and_weighted_less_is_better_in_medication_recommendat/">VITA: &#x27;Carefully Chosen and Weighted Less&#x27; Is Better in Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用可微 Gumbel 选访 + 目标感知注意力替代 RNN 顺序聚合，仅聚合与当前就诊相关的历史；MIMIC 上 Jaccard 相对最优基线提升 3.83%/5.56%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;主动学习实例选择&quot;,&quot;知识蒸馏&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/AAAI/2025/llm_rec/active_large_language_model_based_knowledge_distillation_for_session_based_/">Active Large Language Model-based Knowledge Distillation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 max-min 主动学习从全量会话中挑出少量高增益实例喂给 GPT-4 教师蒸馏，500 条样本即超越全量 KD 与 10 倍参数教师。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;Mixup&quot;,&quot;表征级融合&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/augmenting_sequential_recommendation_with_balanced_relevance_and_diversity/">Augmenting Sequential Recommendation with Balanced Relevance and Diversity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出即插即用平衡增强插件 BASRec，以单序列随机算子+表征级 mixup 与跨序列非线性 mixup 兼顾相关性与多样性，全库排序下 SASRec 平均提升 33.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;可持续推荐&quot;,&quot;多目标损失&quot;]"><strong><a href="../../venues/AAAI/2025/debias_fairness_industrial/bites_of_tomorrow_personalized_recommendations_for_a_healthier_and_greener_/">Bites of Tomorrow: Personalized Recommendations for a Healthier and Greener Plate</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出绿色食品推荐任务 GRAPE，用双注意力建模用户对可持续指标的动态偏好，并以两种 Green Loss 在保精度前提下提升推荐列表绿色度。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;软标签&quot;,&quot;困难负采样&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/future_sight_and_tough_fights_revolutionizing_sequential_recommendation_wit/">Future Sight and Tough Fights: Revolutionizing Sequential Recommendation with FENRec</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间相关软标签挖掘未来交互、用锚-负混合构造持久困难负样本，四数据集全库排序平均提升 6.16%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;共享账户推荐&quot;,&quot;图胶囊网络&quot;,&quot;子空间对齐&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/lightweight_yet_fine_grained_a_graph_capsule_convolutional_network_with_sub/">Lightweight yet Fine-grained: A Graph Capsule Convolutional Network with Subspace Alignment for Shared-account Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用线性注意力胶囊图+低秩子空间对齐，在共享账户混合序列上做细粒度潜在用户解耦，精度大幅领先且参数/耗时更低。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;联邦基础模型&quot;,&quot;组门控路由&quot;,&quot;多面向用户建模&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/multifaceted_user_modeling_in_recommendation_a_federated_foundation_models_/">Multifaceted User Modeling in Recommendation: A Federated Foundation Models Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>联邦场景下用组门控网络将用户分层路由到组级 FFN，配合用户私有 FFN 实现多面向个性化，CTR 任务 AUC 最高提升 1.16%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;LLM 日志语义增强&quot;,&quot;个性化软提示&quot;,&quot;智能空间行为预测&quot;]"><strong><a href="../../venues/AAAI/2025/llm_rec/predicting_user_behavior_in_smart_spaces_with_llm_enhanced_logs_and_persona/">Predicting User Behavior in Smart Spaces with LLM-Enhanced Logs and Personalized Prompts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义增强智能空间事件日志 + 个性化图软提示，在智能车/智能家居四数据集上显著提升下一行为预测，稀有事件 H@1 提升超 40%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;Test-Time Training&quot;,&quot;TTT-Linear&quot;,&quot;线性注意力&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/preliminary_evaluation_of_the_test_time_training_layers_in_recommendation_s/">Preliminary Evaluation of the Test-Time Training Layers in Recommendation System (Student Abstract)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 TTT-Linear 作为序列推荐特征提取层，逐 token 梯度更新隐状态，在 Beauty/Electronics/ML-1M 上以 1 epoch 追平 SASRec/ComiRec 数十轮训练效果。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;自反思提示知识库&quot;,&quot;PPO 检索智能体&quot;]"><strong><a href="../../venues/AAAI/2025/llm_rec/re2llm_reflective_reinforcement_large_language_model_for_session_based_reco/">Re2LLM: Reflective Reinforcement Large Language Model for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 自反思生成纠错提示构建知识库，再以 PPO 训练轻量检索智能体按会话选提示，冻结主干即超越全量训练基线。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba/SSM&quot;,&quot;双向序列建模&quot;,&quot;长尾用户&quot;]"><strong><a href="../../venues/AAAI/2025/sequential_rec/sigma_selective_gated_mamba_for_sequential_recommendation/">SIGMA: Selective Gated Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用部分翻转双向 Mamba + 输入敏感稠密门控 + FE-GRU 解决 Mamba 单向性与短序列状态不稳问题，五数据集 Full-Ranking 上 HR@10 最高提升 8.82%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2025&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;召回&quot;,&quot;知识蒸馏&quot;,&quot;时空知识图谱&quot;,&quot;空间位置编码&quot;]"><strong><a href="../../venues/AAAI/2025/poi_spatiotemporal_rec/spatial_temporal_knowledge_distillation_for_takeaway_recommendation/">Spatial-Temporal Knowledge Distillation for Takeaway Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 STKG 教师模型蒸馏时空知识给 ST-Transformer 学生模型，融合地理距离与功能区序列建模外卖动态偏好，三城 HR@10 提升约 1.5–2.8%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长尾推荐&quot;,&quot;混合意图&quot;,&quot;对比约束损失&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/bid_farewell_to_seesaw_towards_accurate_long_tail_session_based_recommendat/">Bid Farewell to Seesaw: Towards Accurate Long-tail Session-based Recommendation via Dual Constraints of Hybrid Intents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出混合意图学习与意图约束损失的双约束即插即用框架 HID，在三个数据集上同时提升会话推荐准确率与长尾覆盖，打破 see-saw 困境。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;多模态LLM聚类&quot;,&quot;终身行为序列&quot;,&quot;HSTU&quot;]"><strong><a href="../../venues/AAAI/2026/ctr_feature_interaction/dmgin_how_multimodal_llms_enhance_large_recommendation_models_for_lifelong_/">DMGIN: How Multimodal LLMs Enhance Large Recommendation Models for Lifelong User Post-click Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用MLLM离线聚类店铺压缩终身后点击序列，组内统计+Transformer与组间HSTU保留兴趣演化；工业与Amazon AUC/GAUC领先，线上CTR+4.7%、RPM+2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;紧迫度感知&quot;,&quot;时间位置编码&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/driving_engagement_in_daily_fantasy_sports_with_a_scalable_and_urgency_awar/">Driving Engagement in Daily Fantasy Sports with a Scalable and Urgency-Aware Ranking Engine</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 DIN 上注入实时紧迫度特征与时间位置编码，配合 neuralNDCG 列表损失，在 6.5 亿用户级工业数据上 nDCG@1 相对 LightGBM 提升 9%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域建模&quot;,&quot;跨会话信息&quot;,&quot;频域一致性损失&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/exploiting_inter_session_information_with_frequency_enhanced_dual_path_netw/">Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>FreqRec 用可学习复数频域 MLP 双路建模跨会话群体节奏与用户局部高频，并加频域一致性损失，全库排序下 NDCG@10 最高提升 7.38%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;位置前馈网络&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/hymoerec_hybrid_mixture_of_experts_for_sequential_recommendation/">HyMoERec: Hybrid Mixture-of-Experts for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用共享+稀疏双分支混合专家替换均匀 PFFN，配合自适应融合与负载均衡，在 ML-1M/Beauty 全库排序上稳定超越 Mamba4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;排序&quot;,&quot;双曲图卷积&quot;,&quot;Mamba-MoE&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/hyperbolic_enhanced_mixture_of_experts_mamba_for_sequential_recommendation/">Hyperbolic-Enhanced Mixture-of-Experts Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双曲GCN+VGAE重构物品图，RoPE-MoE Mamba动态建模；四数据集HR/NDCG最高提升6.12%，但划分/排序协议未明。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;序列长度条件化&quot;,&quot;注意力温度调制&quot;]"><strong><a href="../../venues/AAAI/2026/ctr_feature_interaction/length_adaptive_interest_network_for_balancing_long_and_short_sequence_mode/">Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将序列长度作为条件信号注入提示与注意力温度，缓解短序列用户注意力极化，五骨干平均AUC最高+1.15%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;动态图推荐&quot;,&quot;任务感知子图检索&quot;]"><strong><a href="../../venues/AAAI/2026/graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;虚假订单检测&quot;,&quot;影响函数免重训纠偏&quot;]"><strong><a href="../../venues/AAAI/2026/debias_fairness_industrial/unbiased_rectification_for_sequential_recommender_systems_under_fake_orders/">Unbiased Rectification for Sequential Recommender Systems Under Fake Orders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出双视图识别+影响函数筛选+定向梯度上升的免重训序列推荐纠偏框架 DITaR，在留一法采样排序下逼近甚至超过干净数据上界且收敛仅需 5 个 epoch。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;小波变换&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/wavelet_enhanced_adaptive_frequency_filter_for_sequential_recommendation/">Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用上下文动态调制频域滤波器并叠加 Haar 小波高频增强，在 4 个基准 Full-Ranking 下 HR@20 最高提升 11.4%，长序列下训练开销低于 BSARec/SLIME4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化主题注意力&quot;,&quot;神经话题模型&quot;,&quot;用户-标签互动预测&quot;]"><strong><a href="../../venues/ACL_EMNLP/2021/sequential_rec/howyoutagtweets_learning_user_hashtagging_preferences_via_personalized_topi/">#HowYouTagTweets: Learning User Hashtagging Preferences via Personalized Topic Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以 BERTweet+神经话题模型联合训练并对话题标签上下文做用户查询式个性化主题注意力，预测用户未来是否会使用某话题标签；全局时间切分、全库排序下 MAP 0.3114 超越 LSTM-ATT 12.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;层级兴趣建模&quot;,&quot;多粒度兴趣匹配&quot;]"><strong><a href="../../venues/ACL_EMNLP/2021/sequential_rec/hierec_hierarchical_user_interest_modeling_for_personalized_news_recommenda/">HieRec: Hierarchical User Interest Modeling for Personalized News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用「子话题—话题—用户」三级兴趣树替代单一用户向量，并按候选新闻所属话题/子话题做多粒度分层匹配，MIND 上 AUC 相对提升 3.1%、nDCG@5 提升 7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2021 Findings&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/粗排&quot;,&quot;新闻推荐&quot;,&quot;跨选择门控&quot;,&quot;层次聚类图&quot;]"><strong><a href="../../venues/ACL_EMNLP/2021/graph_rec/neural_news_recommendation_with_collaborative_news_encoding_and_structural_/">Neural News Recommendation with Collaborative News Encoding and Structural User Encoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 CNE+SUE，用标题-内容跨选择/交叉注意力和层次聚类图编码用户历史；在 200K-MIND 的 AUC 69.55，较最佳基线 +0.92。</p>
</dd>
<dt data-paper-tags="[&quot;ACL/EMNLP 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;位置偏差&quot;,&quot;对抗学习&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2022/debias_fairness_industrial/debiasgan_eliminating_position_bias_in_news_recommendation_with_adversarial/">DebiasGAN: Eliminating Position Bias in News Recommendation with Adversarial Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>通过位置感知/位置随机双分支与对抗学习消除新闻推荐位置偏差，在 News/Uniform 上 AUC 相对提升 1.1–1.7%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;基础设施与评测&quot;,&quot;模块化设计&quot;,&quot;可复现研究&quot;]"><strong><a href="../../venues/ACL_EMNLP/2023/sequential_rec/newsreclib_a_pytorch_lightning_library_for_neural_news_recommendation/">NewsRecLib: A PyTorch-Lightning Library for Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>EMNLP 2023 序列与会话推荐，提出高度模块化的神经新闻推荐开源库，统一评估协议与组件消融，解决研究不可复现与不公平比较痛点。</p>
</dd>
<dt data-paper-tags="[&quot;Findings of EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;召回与精排&quot;,&quot;预训练&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2023/sequential_rec/punr_pre_training_with_user_behavior_modeling_for_news_recommendation/">PUNR: Pre-training with User Behavior Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Findings of EMNLP 2023 序列与会话推荐：提出用户行为掩码与生成双预训练任务，结合孪生编码器优化新闻推荐用户表征。</p>
</dd>
<dt data-paper-tags="[&quot;NAACL 2024 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;指令微调&quot;,&quot;掩码物品建模&quot;,&quot;贝叶斯个性化排序&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/llm_rec/aligning_large_language_models_with_recommendation_knowledge/">Aligning Large Language Models with Recommendation Knowledge</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 MIM/MLM/BPR 经典算子改写为自然语言辅助任务微调 LLM，配合去 user-ID 的精简推荐提示，在 Amazon 三域全库检索上超越 TIGER 达 19%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2024&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;理由蒸馏&quot;,&quot;提示蒸馏&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/llm_rec/rdrec_rationale_distillation_for_llm_based_recommendation/">RDRec: Rationale Distillation for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Llama-2-7b + CoT 从评论蒸馏「用户偏好/物品属性」双句理由，作为第四个生成任务注入 T5-small 的 POD 提示蒸馏框架；Leave-One-Out 生成式全库排序下 Top-N 提升 12–42%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;领域持续预训练&quot;,&quot;指令微调&quot;,&quot;文本化物品表征&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/llm_rec/recgpt_generative_pre_training_for_text_based_recommendation/">RecGPT: Generative Pre-training for Text-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个领域自适应全量预训练的 7B 推荐 LLM（20.5B token 推荐语料持续预训练 + 10 万指令微调），以语义相似度对齐解码结果，在 4 个基准上刷新评分预测与序列推荐 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;序列与会话推荐&quot;,&quot;生成式端到端&quot;,&quot;LLM 参数高效微调 (LoRA/QLoRA)&quot;,&quot;Mixup 混合检索&quot;,&quot;服饰电商冷启动&quot;]"><strong><a href="../../venues/ACL_EMNLP/2024/sequential_rec/sequential_llm_framework_for_fashion_recommendation/">Sequential LLM Framework for Fashion Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以推荐专用 Prompt + 搜索记忆 + LoRA 微调的 Falcon-7B 生成候选，再用 ID/标题双通道 Mixup 检索排序；Leave-One-Out 全库排序下 Clothing 类目 Recall@10 较 CORE 提升 30.4%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;情感认知状态偏好&quot;,&quot;文本序列推荐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/a_text_based_recommender_system_that_leverages_explicit_affective_state_pre/">A Text-Based Recommender System that Leverages Explicit Affective State Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出情感-认知状态显式偏好推荐任务，用 GPT-4o 从书评挖掘 AC 语句构建数据集，Transformer 融合长短期偏好与 AC 描述，全库排序 HR@10 达 15.6%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2025 Findings&quot;,&quot;生成式推荐&quot;,&quot;时间感知提示&quot;,&quot;趋势感知推理&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/generative_rec/enhancing_time_awareness_in_generative_recommendation/">Enhancing Time Awareness in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将绝对时间戳与目标相对间隔verbalize进提示，并叠加全局转移图与免训练趋势重排，Full-Ranking下R@5最高提升15.4%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;售前咨询动机建模&quot;,&quot;混合注意力专家池化&quot;,&quot;对比对齐&quot;]"><strong><a href="../../venues/ACL_EMNLP/2025/llm_rec/maps_motivation_aware_personalized_search_via_llm_driven_consultation_align/">MAPS: Motivation-Aware Personalized Search via LLM-Driven Consultation Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM+MoAE 将售前咨询动机与查询/商品/历史双向对齐，Commercial 与 Amazon 上排序 HR@10 提升约 20%/35%。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026 Findings&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;前缀微调&quot;,&quot;时序注意力&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/beyond_retrieval_learning_compact_user_representations_for_scalable_llm_per/">Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用双轨前缀（用户态+查询感知时序记录）替代提示序列化与逐用户适配器，LaMP 六任务全面领先且每用户参数降 130×</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026&quot;,&quot;生成式推荐&quot;,&quot;掩码历史重建&quot;,&quot;熵引导掩码&quot;,&quot;课程学习&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/generative_rec/from_past_to_path_masked_history_learning_for_next_item_prediction_in_gener/">From Past To Path: Masked History Learning for Next-Item Prediction in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在自回归生成式推荐上叠加掩码历史重建辅助任务，用熵引导掩码与课程调度弥合训练-推理鸿沟，Amazon 三域 Full-Ranking 全面 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ACL 2026 Findings&quot;,&quot;跨域与多场景推荐&quot;,&quot;评测基准&quot;,&quot;时间外推&quot;,&quot;跨域用户建模&quot;,&quot;LLM4Rec&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/cross_domain_multi_scenario/horizon_a_benchmark_for_in_the_wild_user_behaviour_modeling/">HORIZON: A Benchmark for In-the-wild User Behaviour Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>重构 Amazon Reviews 为 54M 用户跨域长程基准，提出时间-用户双轴四象限评估与 LLM 查询改写/长程建模任务，揭示 ID 模型在时间外推下崩塌。</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;元数据知识图谱&quot;,&quot;Top-K 位置奖励&quot;,&quot;结构感知重排&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/tatk_triple_aware_top_k_learning_with_knowledge_grounded_verification_for_l/">TATK: Triple-Aware Top-K Learning with Knowledge-Grounded Verification for LLM-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用元数据物品图贯穿提示接地、Top-K 位置奖励与推理期 top-M 重排，全库排序下 NDCG@10 最高提升 27.03%</p>
</dd>
<dt data-paper-tags="[&quot;EMNLP 2026 Findings&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;状态空间模型&quot;,&quot;非随机缺失 (MNAR)&quot;,&quot;非对称评论结转&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/sequential_rec/two_sided_state_space_models_for_sequential_recommendation_with_non_random_/">Two-Sided State-Space Models for Sequential Recommendation with Non-Random Multimodal Review Feedback</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将评论视为双侧状态空间中的内生事件，用 MNAR 观测模式、非对称评论结转与局部图传播联合演化用户/物品状态，Full-Ranking 下 R@20 较 BSARec 提升 14.8%–18.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ACL&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;偏好对齐&quot;,&quot;自适应奖励边际&quot;]"><strong><a href="../../venues/ACL_EMNLP/2026/llm_rec/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_int/">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示二元偏好对齐丢失偏好强度与时序上下文，提出自适应奖励边际 RecPO，五数据集 HR@1 全面领先 S-DPO。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2021&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;视觉印象建模&quot;,&quot;记忆网络融合&quot;]"><strong><a href="../../venues/ACM_MM/2021/multimodal_rec/why_do_we_click_visual_impression_aware_news_recommendation/">Why Do We Click: Visual Impression-aware News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以新闻卡片渲染快照为“视觉印象”，局部记忆注意力+全局门控融合，在 IM-MIND 上 AUC/NDCG@5 相对提升约 1.0%/3.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;动态图神经网络&quot;,&quot;混合度量学习&quot;]"><strong><a href="../../venues/ACM_MM/2022/cross_domain_multi_scenario/ddghm_dual_dynamic_graph_with_hybrid_metric_training_for_cross_domain_seque/">DDGHM: Dual Dynamic Graph with Hybrid Metric Training for Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双层动态图分别刻画域内/域间序列转移，用融合注意力门迁移知识，并以协同+对比双度量学习缓解稀疏，全库排序 HR@20 提升 4%~9%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;知识图谱提示&quot;,&quot;知识树掩码&quot;]"><strong><a href="../../venues/ACM_MM/2023/sequential_rec/knowledge_prompt_tuning_for_sequential_recommendation/">Knowledge Prompt-tuning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将知识图谱经关系模板转成知识提示，并用知识树掩码去噪，与 T5 掩码个性化提示联合调优；Leave-One-Out + Full-Ranking 下 books NDCG@5 较 P5 提升 40.65%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态特征融合&quot;,&quot;在线知识蒸馏&quot;,&quot;去偏 In-batch 损失&quot;]"><strong><a href="../../venues/ACM_MM/2023/sequential_rec/online_distillation_enhanced_multi_modal_transformer_for_sequential_recomme/">Online Distillation-enhanced Multi-modal Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ID-aware 多模态 Transformer 与在线蒸馏解耦 ID/模态特征的优化冲突，四数据集 Recall@10 相对最优基线提升 7.4%–13.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布表征&quot;]"><strong><a href="../../venues/ACM_MM/2023/sequential_rec/personalized_behavior_aware_transformer_for_multi_behavior_sequential_recom/">Personalized Behavior-Aware Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯分布刻画物品/行为/位置并借 Wasserstein 距离重构自注意力，把个性化多行为模式与行为-时间协同注入 Transformer；Leave-One-Out + HR/NDCG 上 Taobao HR@5 相对最优基线提升 6.37%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;动态参数生成&quot;,&quot;语义码本量化&quot;]"><strong><a href="../../venues/ACM_MM/2024/sequential_rec/semantic_codebook_learning_for_dynamic_recommendation_models/">Semantic Codebook Learning for Dynamic Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用语义序列主干+裁剪物品分支双路生成动态参数，并以语义码本量化去噪压缩搜索空间；Leave-One-Out/1:99 采样排序下 NDCG@10 最高提升约 10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;端侧重排&quot;,&quot;混合精度量化&quot;,&quot;设备-云协同&quot;,&quot;超网络&quot;]"><strong><a href="../../venues/ACM_MM/2025/sequential_rec/chord_customizing_hybrid_precision_on_device_model_for_sequential_recommend/">CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端超网络按用户画像生成逐通道混合精度量化策略（2-bit 编码），端侧单次前向完成个性化适配与推理，精度反超全精度基线且传输降 173.8×。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图信号低通滤波&quot;,&quot;频域序列去噪&quot;]"><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/精排&quot;,&quot;联邦协作预训练&quot;,&quot;语义软交叉熵&quot;,&quot;LoRA 适配&quot;]"><strong><a href="../../venues/ACM_MM/2026/cross_domain_multi_scenario/from_transfer_to_collaboration_a_federated_framework_for_cross_market_seque/">From Transfer to Collaboration: A Federated Framework for Cross-Market Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>FeCoSR 以联邦多对多协作与语义软交叉熵 S2CE 缓解跨市场异质性，在 XMarket 八市场 Leave-One-Out Full-Ranking 上 NDCG@10 最高提升 17.28%。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/物品表征&quot;,&quot;侧适配器&quot;,&quot;流感知路由&quot;]"><strong><a href="../../venues/ACM_MM/2026/sequential_rec/stream_aware_side_adaptation_for_large_pre_trained_multimodal_embedding_mod/">Stream-aware Side Adaptation for Large Pre-trained Multimodal Embedding Models in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结Qwen3-VL侧适配新框架SHAF/ReSA，Leave-One-Out全库排序，HR@10最高提升7.4%</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba/状态空间对偶&quot;,&quot;多模态迁移学习&quot;,&quot;时间感知结构化掩码&quot;]"><strong><a href="../../venues/ACM_MM/2026/sequential_rec/towards_transfer_efficient_multi_modal_sequential_recommendation_with_state/">Towards Transfer-Efficient Multi-modal Sequential Recommendation with State Space Duality</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 SSD 时间衰减+跨模态共享掩码对齐+双通道傅里叶滤波融合，替代对比学习式多模态预训练，跨域微调收敛加速约 10 倍。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;连续时间编码&quot;,&quot;时序图协同注意力&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/continuous_time_sequential_recommendation_with_temporal_graph_collaborative/">Continuous-Time Sequential Recommendation with Temporal Graph Collaborative Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构造连续时间二分图，用时间核编码+协同注意力统一序列模式与跨用户时序协同信号，Recall@10 平均绝对提升 22.5%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;时序点过程&quot;,&quot;社交激励&quot;,&quot;事件级直接范式&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/extracting_attentive_social_temporal_excitation_for_sequential_recommendati/">Extracting Attentive Social Temporal Excitation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用时序点过程在事件级直接建模好友行为的细粒度激励，STEN 在三数据集 LOO 全库排序下 Recall@10 最高提升 5.79%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双曲空间超图卷积&quot;,&quot;自监督预训练&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/hyperbolic_hypergraphs_for_sequential_recommendation/">Hyperbolic Hypergraphs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用月度/季度/年度分层双曲超图卷积 + 三项自监督预训练（含超边链接预测）建模序列推荐，在 AMT/Goodreads 采样排序下 NDCG@1 最高提 31.61%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;时间切片图&quot;,&quot;时序点过程&quot;,&quot;双重动态表征&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/learning_dual_dynamic_representations_on_time_sliced_user_item_interaction_/">Learning Dual Dynamic Representations on Time-Sliced User-Item Interaction Graphs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>按等长时间切片构建全局用户-物品二部图，用切片 GNN + 双侧 GRU 建模用户/物品双重动态，再以时序点过程辅助任务补偿离散化时间损失；1:100 采样排序下 NDCG@10 相对 SCoRe 提升 3.4%–10.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;精排&quot;,&quot;Neural ODE&quot;,&quot;Next-POI 序列预测&quot;]"><strong><a href="../../venues/CIKM/2021/poi_spatiotemporal_rec/lightmove_a_lightweight_next_poi_recommendation_for_taxicab_rooftop_adverti/">LightMove: A Lightweight Next-POI Recommendation for Taxicab Rooftop Advertising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用含跳步与自适应参数生成的 NODE 轻量预测出租车下一位置，全库排序、逐车时序切分下 Foursquare Hits@1 提升 23.6% 且参数减 78%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;组合嵌入&quot;,&quot;双注意力&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/lightweight_self_attentive_sequential_recommendation/">Lightweight Self-Attentive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>LSAN 用商余数动态组合嵌入压缩物品表，并以卷积/自注意力双分支建模局部/全局偏好；Leave-One-Out+Full-Ranking 下 HR@20 最高较 SASRec 提升 99.65%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;高斯分布表征&quot;,&quot;Wasserstein 距离&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/modeling_sequences_as_distributions_with_uncertainty_for_sequential_recomme/">Modeling Sequences as Distributions with Uncertainty for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>CIKM 2021][序列与会话推荐] 用椭圆高斯分布刻画物品与序列不确定性，双 Transformer 分别建模均值/协方差，以 2-Wasserstein 距离排序，全库采样排序下 Recall@1 最高提升 32.4%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图协同训练&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/CIKM/2021/sequential_rec/self_supervised_graph_co_training_for_session_based_recommendation/">Self-Supervised Graph Co-Training for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以物品视图/会话视图双图构建互补充自监督信号，用协同训练互传伪标签并加对抗散度约束，Tmall P@20 提升 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;周期兴趣建模&quot;,&quot;时空上下文切序列&quot;]"><strong><a href="../../venues/CIKM/2021/poi_spatiotemporal_rec/st_pil_spatial_temporal_periodic_interest_learning_for_next_point_of_intere/">ST-PIL: Spatial-Temporal Periodic Interest Learning for Next Point-of-Interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用星期/时段/geohash三重时空上下文切分历史轨迹构造周期兴趣序列，配合双层注意力融合长短兴趣，Leave-One-Out 全库排序下 Acc@1 提升约 70%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;重排&quot;,&quot;异构行为序列&quot;,&quot;搜索推荐联合建模&quot;]"><strong><a href="../../venues/CIKM/2021/cross_domain_multi_scenario/user_a_unified_information_search_and_recommendation_model_based_on_integra/">USER: A Unified Information Search and Recommendation Model based on Integrated Behavior Sequence</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将搜索与推荐行为融合为异构序列，用统一排序框架同时服务两任务，统一数据下搜索 MAP 提升 3.49%、推荐 MAP 提升 71.96%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;个性化兴趣持续性&quot;,&quot;时序频次分箱&quot;,&quot;稀疏历史补充&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/beyond_learning_from_next_item_sequential_recommendation_via_personalized_i/">Beyond Learning from Next Item: Sequential Recommendation via Personalized Interest Sustainability</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用「训练期末段消费预测」任务刻画个性化兴趣持续性(PIS)，并以内在/外在双路稀疏历史补充方案与原型度量偏好学习联合训练，11 数据集全库采样排序全面超越 10 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;对比学习&quot;,&quot;互信息最大化&quot;]"><strong><a href="../../venues/CIKM/2022/cross_domain_multi_scenario/contrastive_cross_domain_sequential_recommendation/">Contrastive Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>联合建模序列内/间物品关系，双目标学单域/跨域表征，并以对比 Infomax 互信息约束跨域迁移；修正时间穿越划分后 HR@10 相对 SOTA 提升 14%+</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;变分自编码器&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/contrastvae_contrastive_variational_autoencoder_for_sequential_recommendati/">ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双视角ContrastELBO耦合双分支VAE与潜空间对比增强，缓解后验坍缩，Full-Ranking下R@40最高提升19.34%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;双网络解耦&quot;,&quot;训练-推理鸿沟&quot;,&quot;双向KL知识迁移&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/disentangling_past_future_modeling_in_sequential_recommendation_via_dual_ne/">Disentangling Past-Future Modeling in Sequential Recommendation via Dual Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双网络分离过去/未来建模，并以双向KL迁移互补知识，缓解训练-推理鸿沟；4数据集采样排序上HR@1平均提升12.03%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;逆倾向得分 IPS&quot;,&quot;序列推荐&quot;,&quot;双视角去偏&quot;]"><strong><a href="../../venues/CIKM/2022/debias_fairness_industrial/dually_enhanced_propensity_score_estimation_in_sequential_recommendation/">Dually Enhanced Propensity Score Estimation in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出用户/物品双视角时序 IPS 与双 Transformer，理论无偏并显著提升序列推荐。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平与工业架构&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;回声室缓解&quot;]"><strong><a href="../../venues/CIKM/2022/debias_fairness_industrial/dynamic_causal_collaborative_filtering/">Dynamic Causal Collaborative Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用含环因果图与马尔可夫吸收态分析刻画推荐回声室，提出后门调整+反事实历史的 ∂CCF，内容多样性变化平均降低 7.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Neural ODE&quot;,&quot;连续时间动态图&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/evolutionary_preference_learning_via_graph_nested_gru_ode_for_session_based/">Evolutionary Preference Learning via Graph Nested GRU ODE for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将会话建模为连续时间动态图，以GNG-ODE与t-Alignment学习偏好连续演化，全库排序下MRR@20最高提升6.05%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;解释引导数据增强&quot;,&quot;对比学习视图选择&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/explanation_guided_contrastive_learning_for_sequential_recommendation/">Explanation Guided Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用解释方法（Saliency/Occlusion）估计序列物品重要性，指导正负视图增强与自监督+监督对比学习；四数据集 Full-Ranking 下 NDCG@5 较 CL4SRec 提升 14%–22%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;图检索中心节点&quot;,&quot;多场景短期意图&quot;,&quot;个性化门控融合&quot;]"><strong><a href="../../venues/CIKM/2022/ctr_feature_interaction/graph_based_long_term_and_short_term_interest_model_for_click_through_rate_/">Graph Based Long-Term And Short-Term Interest Model for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用图检索结构替代长行为截断/硬类目过滤，结合多场景 GRU 短期意图与个性化门控融合，离线 AUC 达 0.9830，线上 CTR +4.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（端到端 Next-Item 预测）&quot;,&quot;隐式会话上下文&quot;,&quot;图聚类与上下文预测&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/implicit_session_contexts_for_next_item_recommendations/">Implicit Session Contexts for Next-Item Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>会话-物品图 GraphSAGE 嵌入 + K-means 聚类生成隐式会话上下文，用双 Bi-LSTM 上下文预测器作先验特征，Foursquare MRR 相对最优基线 +36.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;会话内上下文&quot;,&quot;曝光偏置建模&quot;,&quot;多任务学习&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/intra_session_context_aware_feed_recommendation_in_live_systems/">Intra-session Context-aware Feed Recommendation in Live Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将信息流会话建模为点击/滑动的马尔可夫事件链，以会话内曝光序列编码上下文，多任务联合学习并用 GPT 式贪心序列生成优化总曝光与总点击；线上人均曝光 +1.16%、转化 +1.56%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;数据基准与离线评估&quot;,&quot;Exposure Bias&quot;,&quot;Missing-at-Random&quot;,&quot;多任务反馈信号&quot;]"><strong><a href="../../venues/CIKM/2022/debias_fairness_industrial/kuairand_an_unbiased_sequential_recommendation_dataset_with_randomly_expose/">KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>快手无感插入百万级随机曝光视频，构建首个带丰富特征与12类反馈的大规模无偏序列推荐数据集，支撑 IPS 全库评估。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;多视图图神经网络&quot;,&quot;自监督信号&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/multi_level_contrastive_learning_framework_for_sequential_recommendation/">Multi-level Contrastive Learning Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MCLSR 构建序列/用户-物品/用户-用户/物品-物品四视图，在兴趣层与特征层做跨视图对比学习以补足稀疏监督信号，四数据集 Full-Ranking 下 Recall@20 平均相对提升约 7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Hawkes Process&quot;,&quot;Hierarchical RNN&quot;,&quot;Entire Space Multi-Task&quot;]"><strong><a href="../../venues/CIKM/2022/ctr_feature_interaction/multi_scale_user_behavior_network_for_entire_space_multi_task_learning/">Multi-Scale User Behavior Network for Entire Space Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层循环网络 HEROES 用 Hawkes 过程与门控跨层通道，在 observation→click→conversion 全空间多尺度建模用户行为，CVR AUC 最高相对提升 11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序评测/鲁棒性分析&quot;,&quot;Rank List Sensitivity&quot;,&quot;交互级扰动&quot;,&quot;级联效应&quot;]"><strong><a href="../../venues/CIKM/2022/debias_fairness_industrial/rank_list_sensitivity_of_recommender_systems_to_interaction_perturbations/">Rank List Sensitivity of Recommender Systems to Interaction Perturbations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出排序列表敏感度 RLS 指标与 CASPER 级联扰动法，量化单条交互扰动致全库排序剧变，并揭示低准确率用户更脆弱</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;LSH哈希采样&quot;,&quot;长序列用户兴趣建模&quot;,&quot;端到端训练&quot;]"><strong><a href="../../venues/CIKM/2022/ctr_feature_interaction/sampling_is_all_you_need_on_modeling_long_term_user_behaviors_for_ctr_predi/">Sampling Is All You Need on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用多轮 SimHash 碰撞概率替代 target attention，直接 gather 同签名行为构造长序列用户兴趣，AUC 媲美全注意力且训练快 5–11×，美团线上升 +2.98% CTR。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长用户行为序列建模&quot;,&quot;稀疏注意力记忆网络&quot;,&quot;实时推理效率&quot;]"><strong><a href="../../venues/CIKM/2022/ctr_feature_interaction/sparse_attentive_memory_network_for_click_through_rate_prediction_with_long/">Sparse Attentive Memory Network for Click-through Rate Prediction with Long Sequences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以目标物品为 Query、千长行为序列为知识库，用目标+记忆双查询稀疏注意力与 O(1) 次迭代记忆更新，实现 O(L) 复杂度长序列 CTR 建模，线上 A/B 提升 CTR 7.30%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;层级自注意力&quot;,&quot;时间门控融合&quot;]"><strong><a href="../../venues/CIKM/2022/sequential_rec/time_lag_aware_sequential_recommendation/">Time Lag Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用层级自注意力分离全局稳定/局部波动的用户偏好，并以时间差驱动的维度级门向量融合长短期表征，Amazon Book 上 Hit@20 相对提升 55%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;局部-全局混合&quot;,&quot;Squeeze-Excitation Attention&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/adamct_adaptive_mixture_of_cnn_transformer_for_sequential_recommendation/">AdaMCT: Adaptive Mixture of CNN-Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将局部卷积滤波注入 Transformer，并用层感知、用户自适应的混合系数与 Sigmoid 挤压-激励注意力统一建模长短期兴趣；在三个 Amazon 基准的 1:100 采样排序下 Recall@10 相对最优基线提升 4.1%–8.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态融合顺序&quot;,&quot;异质图注意力&quot;,&quot;非侵入式融合&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/adaptive_multi_modalities_fusion_in_sequential_recommendation_systems/">Adaptive Multi-Modalities Fusion in Sequential Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用模态图 + 双注意力异步门控，让每个节点自适应选择早期/晚期融合顺序，Amazon 六域 HR@5 平均 +8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回（序列召回）&quot;,&quot;自注意力校准&quot;,&quot;对抗扰动去噪&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/attention_calibration_for_transformer_based_sequential_recommendation/">Attention Calibration for Transformer-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐中注意力权重错配现象，提出空间校准器与对抗校准器直接修正注意力分布，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 8.65%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自编码器协同过滤&quot;,&quot;增量推理&quot;,&quot;物品转移矩阵&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/autoseqrec_autoencoder_for_efficient_sequential_recommendation/">AutoSeqRec: Autoencoder for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用单个共享编码器同时重建用户-物品矩阵与转移矩阵行/列，增量推理只更新输入矩阵不更新参数，ML-100K 上 Recall@10 相对提升 35%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;端到端训练与在线推理服务&quot;,&quot;LSH 动态稀疏&quot;,&quot;CPU 推理加速&quot;]"><strong><a href="../../venues/CIKM/2023/debias_fairness_industrial/bolt_an_automated_deep_learning_framework_for_training_and_deploying_large_/">BOLT: An Automated Deep Learning Framework for Training and Deploying Large-Scale Search and Recommendation Models on Commodity CPU Hardware</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 LSH 动态稀疏层在商用 CPU 上训练十亿级搜索推荐模型，辅以稀疏超参自动调优与稀疏推理，精度持平 GPU 基线而延迟与碳排降一个数量级。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;展示上下文&quot;,&quot;位置偏置&quot;,&quot;用户行为建模&quot;]"><strong><a href="../../venues/CIKM/2023/ctr_feature_interaction/deep_context_interest_network_for_click_through_rate_prediction/">Deep Context Interest Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以点击级展示上下文与位置注意力建模用户兴趣，31天工业数据AUC 0.66475，线上CTR/RPM各+1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;数据增强(召回前处理)&quot;,&quot;Diffusion Model&quot;,&quot;SU-Net&quot;,&quot;长尾用户&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/diffusion_augmentation_for_sequential_recommendation/">Diffusion Augmentation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用扩散模型一次性生成高质量伪前序物品序列做数据增强，配合 SU-Net 与两种引导策略，在三个数据集三种骨干上稳定提升 HR/NDCG，尤其缓解长尾用户。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;属性感知激活&quot;,&quot;动态参数生成&quot;,&quot;相似性与多样性偏好&quot;]"><strong><a href="../../venues/CIKM/2023/ctr_feature_interaction/dpan_dynamic_preference_based_and_attribute_aware_network_for_relevant_reco/">DPAN: Dynamic Preference-based and Attribute-aware Network for Relevant Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>属性级双激活+双维压缩重表达分离相似/多样表征，条件动态参数生成融合，AUC 0.6098（RelaImpr +10.69%），线上 CTR +7.62%</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排（评分预测）&quot;,&quot;元学习&quot;,&quot;自适应损失&quot;]"><strong><a href="../../venues/CIKM/2023/cold_start_rec/meta_learning_with_adaptive_weighted_loss_for_imbalanced_cold_start_recomme/">Meta-Learning with Adaptive Weighted Loss for Imbalanced Cold-Start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 MELO，用序列状态编码器生成用户级自适应损失权重修正 MAML 内循环，在四个评分数据集上显著降低 RMSE/MAE。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;随机播放&quot;,&quot;会话推荐&quot;,&quot;转移增广&quot;]"><strong><a href="../../venues/CIKM/2023/self_supervised_denoising/muse_music_recommender_system_with_shuffle_play_recommendation_enhancement/">MUSE: Music Recommender System with Shuffle Play Recommendation Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>针对音乐随机播放会话唯一转移率过高的问题，提出转移频次引导的会话增广与细粒度匹配，在 MSSD 全库排序上稳定超越 12 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长期行为序列&quot;,&quot;两阶段相关检索&quot;,&quot;参与度门控注意力&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/query_dominant_user_interest_network_for_large_scale_search_ranking/">Query-dominant User Interest Network for Large-Scale Search Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>两阶段相关行为检索+参与度融合注意力，Amazon四集全指标平均提升21.19%，线上CTR +7.6%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;文本匹配双塔&quot;,&quot;流行度去偏&quot;,&quot;注意力稀疏&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/text_matching_improves_sequential_recommendation_by_reducing_popularity_bia/">Text Matching Improves Sequential Recommendation by Reducing Popularity Biases</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TASTE 用模板将物品与交互历史文本化，经 T5 双塔文本匹配做下一物品推荐，配合会话切分注意力稀疏建模长历史，Full-Ranking 下平均提升约 18%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;精排&quot;,&quot;时间提示 (Temporal Prompt)&quot;,&quot;移位窗口地理编码&quot;]"><strong><a href="../../venues/CIKM/2023/poi_spatiotemporal_rec/timestamps_as_prompts_for_geography_aware_location_recommendation/">Timestamps as Prompts for Geography-Aware Location Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以时间戳为解码 Prompt 显式建模待预测时刻，并用移位窗口缓解网格硬边界，Five Dataset Full 采样排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理与更新&quot;,&quot;组合码&quot;,&quot;更新压缩&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/towards_communication_efficient_model_updating_for_on_device_session_based_/">Towards Communication-Efficient Model Updating for On-Device Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出组合码压缩物品嵌入，并用栈/队列与MMD自适应压缩端侧更新，在Gowalla/Lastfm上以60倍更小更新逼近云端重训精度。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;重复消费时间间隔&quot;,&quot;序列匹配&quot;]"><strong><a href="../../venues/CIKM/2023/sequential_rec/tsrec_enhancing_repeat_aware_recommendation_from_a_temporal_sequential_pers/">TSRec: Enhancing Repeat-Aware Recommendation from a Temporal-Sequential Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>从时间间隔与序列匹配双视角建模重复消费，三大模块在三个数据集的时间切分采样排序协议下 HR@5 最高相对提升 5.47%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;结构化行列式点过程 SDPP&quot;,&quot;集合级优化准则&quot;,&quot;多样性建模&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/a_universal_sets_level_optimization_framework_for_next_set_recommendation/">A Universal Sets-level Optimization Framework for Next Set Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用结构化行列式点过程 (SDPP) 将时序集合整体视为元素并设计集合级条件似然，联合共现表征与类别感知多样性核，在四个数据集上 Recall/NDCG 提升 6–25% 且同时提升多样性。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性循环单元&quot;,&quot;并行扫描 CUDA 核&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/behavior_dependent_linear_recurrent_units_for_efficient_sequential_recommen/">Behavior-Dependent Linear Recurrent Units for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以行为依赖的实数线性循环单元 BD-LRU + Triton 并行扫描核，同时达成强性能、快训练与低推理成本，Full-Ranking 下 ML-1M HR@10 提升 7.46%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;医疗处方推荐&quot;,&quot;意图对比学习&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/contrastive_learning_on_medical_intents_for_sequential_prescription_recomme/">Contrastive Learning on Medical Intents for Sequential Prescription Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ARCI 用跨就诊 Transformer 抽取多条“时间路径”，并以意图对比学习正则化多头注意力，在 MIMIC-III/AKI 全库排序上命中率较 SOTA 提升 5–7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;自校正点过程&quot;,&quot;行列式多样性&quot;,&quot;指针式序列解码&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/divnet_diversity_aware_self_correcting_sequential_recommendation_networks/">DivNet: Diversity-Aware Self-Correcting Sequential Recommendation Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将整页集体推荐改写为带因果掩码的指针式序列选择，用自校正解码建模物品相互激发，并以相似度行列式注入多样性；线上 page CTR +8.7%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;属性感知文本编码&quot;,&quot;多向量匹配 MaxSim&quot;,&quot;零样本迁移&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/mars_matching_attribute_aware_representations_for_text_based_sequential_rec/">MARS: Matching Attribute-aware Representations for Text-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用属性级多向量文本编码 + 逐属性 MaxSim 匹配替代单一序列向量，在全库排序下 R@10 最高提升 24.43%，零样本迁移平均提升 13.11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;多行为序列建模&quot;,&quot;语义ID分词器&quot;,&quot;位置路由MoE&quot;]"><strong><a href="../../venues/CIKM/2024/generative_rec/multi_behavior_generative_recommendation/">Multi-Behavior Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将多行为序列推荐重构为「先预测行为、再预测物品」的生成式两步解码，配合均衡语义ID与位置路由稀疏MoE，全库排序相对提升31%~70%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;近似最大内积搜索&quot;,&quot;交叉熵损失近似&quot;,&quot;硬负采样&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/rece_reduced_cross_entropy_loss_for_large_catalogue_sequential_recommenders/">RECE: Reduced Cross-Entropy Loss for Large-Catalogue Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 GPU 友好的 LSH 分桶+MIPS 近似筛选高梯度负类，将全库 cross-entropy 的 logits 张量切片计算，峰值显存最多降 12 倍且质量持平或超越。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;文本侧信息融合&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/sequence_level_semantic_representation_fusion_for_recommender_systems/">Sequence-level Semantic Representation Fusion for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>频域互滤波实现文本-ID 序列级语义融合，等价于上下文卷积；全库排序下 ML-1M/OR 提升 14%/38%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;超长行为序列&quot;,&quot;层次聚类&quot;,&quot;目标注意力&quot;]"><strong><a href="../../venues/CIKM/2024/ctr_feature_interaction/twin_v2_scaling_ultra_long_user_behavior_sequence_modeling_for_enhanced_ctr/">TWIN V2: Scaling Ultra-Long User Behavior Sequence Modeling for Enhanced CTR Prediction at Kuaishou</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>层次聚类压缩生命周期行为，簇感知目标注意力提升快手 CTR 与推荐多样性。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列均匀性&quot;,&quot;物品频次课程学习&quot;,&quot;多维时间建模&quot;]"><strong><a href="../../venues/CIKM/2024/sequential_rec/unirec_a_dual_enhancement_of_uniformity_and_frequency_in_sequential_recomme/">UniRec: A Dual Enhancement of Uniformity and Frequency in Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用序列时间均匀性与物品频次构建双向增强分支，配合多维时间混合注意力，在 4 个数据集采样排序协议下 NDCG@10 相对最强基线提升 2.2%–8.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/序列排序&quot;,&quot;模型水印&quot;,&quot;自回归OOD序列&quot;]"><strong><a href="../../venues/CIKM/2024/debias_fairness_industrial/watermarking_recommender_systems/">Watermarking Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AOW 自回归 OOD 水印，为序列推荐植入黑盒可验证版权序列；四数据集 Full-Ranking 达 100% R@1，兼顾效用与抗蒸馏/微调。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;多尺度序列抽取&quot;,&quot;KV缓存&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/climber_toward_efficient_scaling_laws_for_large_recommendation_models/">Climber: Toward Efficient Scaling Laws for Large Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向推荐场景改造 Transformer：多尺度子序列抽取降复杂度、自适应温度调制注意力，配合单用户多物品批处理与 KV 缓存，线上指标 +12.19%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;持续学习&quot;,&quot;灾难性遗忘&quot;,&quot;稳定性-可塑性权衡&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/continual_recommender_systems/">Continual Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理持续推荐的两大技术路线（经验回放与正则化），并延伸至资源受限与序列交互场景，给出可信性、基础模型与搜推统一三大开放方向。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;增量推荐&quot;,&quot;时序状态演化&quot;,&quot;伪多任务学习&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/cpmr_context_aware_incremental_sequential_recommendation_with_pseudo_multi_/">CPMR: Context-Aware Incremental Sequential Recommendation with Pseudo-Multi-Task Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出伪多任务学习范式，用历史图与上下文图双通道演化用户/物品时序状态，在四个数据集上 MRR 平均提升 30.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;跨模态兴趣对齐&quot;,&quot;序列去噪&quot;,&quot;LLM2Vec&quot;]"><strong><a href="../../venues/CIKM/2025/self_supervised_denoising/empowering_denoising_sequential_recommendation_with_large_language_model_em/">Empowering Denoising Sequential Recommendation with Large Language Model Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义嵌入与协同嵌入做跨模态兴趣对齐，缓解冷门物品过度去噪，四数据集全排序 HR/NDCG 显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦推荐&quot;,&quot;持续学习&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/federated_continual_recommendation/">Federated Continual Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 FCRec 新任务与 F3CRec 框架，客户端自适应回放+服务端物品级时序均值，在四个数据集三个 FedRec 骨干上 N@20 平均提升 5.6%–32.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;异构信息交互&quot;,&quot;双向信息流&quot;,&quot;选择性聚合&quot;]"><strong><a href="../../venues/CIKM/2025/ctr_feature_interaction/interformer_effective_heterogeneous_interaction_learning_for_click_through_/">InterFormer: Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出交错式双向信息流的 InterFormer，用独立 Cross Arch 做选择性聚合，在三个公开集与 Meta 工业集上取得 SOTA，线上 NE 提升 0.15% 且 QPS 提升 24%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;缓存缩放律&quot;,&quot;目标注意力&quot;,&quot;流式训练&quot;]"><strong><a href="../../venues/CIKM/2025/debias_fairness_industrial/marm_unlocking_the_future_of_recommendation_systems_through_memory_augmenta/">MARM: Unlocking the Future of Recommendation Systems through Memory Augmentation and Scalable Complexity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以外部缓存替代多层自注意力，将序列兴趣建模复杂度从 O(n²d) 降至 O(n·d)，离线 GAUC +0.43%、线上人均时长 +2.079%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;树索引&quot;,&quot;终身序列建模&quot;,&quot;多模态检索&quot;]"><strong><a href="../../venues/CIKM/2025/multimodal_rec/miss_multi_modal_tree_indexing_and_searching_with_lifelong_sequential_behav/">MISS: Multi-Modal Tree Indexing and Searching with Lifelong Sequential Behavior for Retrieval Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多模态嵌入构建树索引，并以 Co-GSU/MM-GSU 双视角检索终身行为，快手线上召回@800 相对提升 37.9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;用户自适应&quot;,&quot;MoE&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/muffin_mixture_of_user_adaptive_frequency_filtering_for_sequential_recommen/">MUFFIN: Mixture of User-Adaptive Frequency Filtering for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双分支频域滤波+用户自适应滤波器，兼顾全谱与频带局部建模，五数据集全库排序平均 R@10 提升 4.12%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;多查询注意力&quot;,&quot;查询窗口矩阵&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/multi_item_query_attention_for_stable_sequential_recommendation/">Multi-Item-Query Attention for Stable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多物品查询窗口+虚拟物品替代单查询掩码注意力，缓解序列噪声敏感，ML-1M HR@5 提升 21.2%，LastFM 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Next Session Prediction&quot;,&quot;Session Mask&quot;,&quot;部分增量训练&quot;]"><strong><a href="../../venues/CIKM/2025/generative_rec/tbgrecall_a_generative_retrieval_model_for_e_commerce_recommendation_scenar/">TBGRecall: A Generative Retrieval Model for E-commerce Recommendation Scenarios</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 Next Session Prediction 会话级非自回归生成召回，配合 Session Mask、MSP/MoE 与部分增量训练，淘宝线上成交额 +2.16%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;离线仿真评测&quot;,&quot;用户模拟&quot;,&quot;时序感知提示&quot;,&quot;动态用户画像&quot;]"><strong><a href="../../venues/CIKM/2025/agent_conversational_rec/temporal_aware_user_behaviour_simulation_with_large_language_models_for_rec/">Temporal-Aware User Behaviour Simulation with Large Language Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用动态画像更新、时序模式抽取与自适应聚合构建 LLM 用户模拟器，在 ML-1M 上 nDCG@5 相对 RecAgent 提升约 27.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;终身行为序列&quot;,&quot;下一动作损失&quot;,&quot;推理系统优化&quot;]"><strong><a href="../../venues/CIKM/2025/sequential_rec/transact_v2_lifelong_user_action_sequence_modeling_on_pinterest_recommendat/">TransAct V2: Lifelong User Action Sequence Modeling on Pinterest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 NN 检索压缩终身序列 + 下一动作对比损失 + Triton 融合推理，Pinterest 线上 Repin +6.35%、Hide -12.80%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;序列建模&quot;,&quot;晚融合&quot;,&quot;工业推理优化&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/an_industrial_scale_sequential_recommender_for_linkedin_feed_ranking/">An Industrial-Scale Sequential Recommender for LinkedIn Feed Ranking</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>LinkedIn Feed SR 用交错 post/action 序列的 decoder-only Transformer 替换 DCNv2 精排，配合晚融合特征与 CPU/GPU 分离推理，线上时长 +2.10%、互动 +3.52%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;角度相似度打分&quot;,&quot;参数高效短视适配&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/closing_the_long_short_view_gap_in_sequential_recommendation_without_cached/">Closing the Long-Short View Gap in Sequential Recommendation without Cached History</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用角度相似度打分+softmax1 修正长视训练几何缺陷，再仅微调 Bias/LayerNorm 实现无缓存短视推理，短窗下反超全序列基线。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;表征级迁移&quot;,&quot;纵向联邦学习&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/CIKM/2026/cross_domain_multi_scenario/cross_domain_sequential_recommendation_with_representation_level_user_trans/">Cross-Domain Sequential Recommendation with Representation-Level User Transfer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>仅通过用户表征级迁移实现跨域序列推荐，融合层注入源域用户向量，兼容纵向联邦与部分重叠用户，全库排序下 NDCG@10 提升 7.4–24.1%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;时间位置编码&quot;,&quot;时间门控注意力偏置&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/decoupled_temporal_encoding_for_generative_recommendation/">Decoupled Temporal Encoding for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将宏观时间动态与微观序列顺序解耦的双模块位置编码，工业数据集 GAUC +1.16pt，线上 CTR +1.8%、RPM +3.0%，延迟仅 +0.3%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;拉普拉斯位置编码&quot;,&quot;图谱结构信号&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/enriching_sequential_recommendation_with_graph_laplacian_positional_embeddi/">Enriching Sequential Recommendation with Graph Laplacian Positional Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用物品共现图归一化拉普拉斯特征向量冻结替换 SASRec 可学习位置嵌入，四基准多数指标提升且在线开销近零。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;生成式推荐&quot;,&quot;进化搜索专家合成&quot;,&quot;非对称 Actor-Critic&quot;,&quot;免模拟器部署推理&quot;]"><strong><a href="../../venues/CIKM/2026/rl_bandit_rec/evol_simulator_guided_evolutionary_expert_synthesis_for_deployment_free_lea/">EVOL: Simulator-Guided Evolutionary Expert Synthesis for Deployment-Free Learning Path Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用知识追踪模拟器内进化搜索合成专家路径，蒸馏为仅凭初始掌握度盲规划的非对称 Actor-Critic，实现免模拟器部署的学习路径推荐。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;注意力校准&quot;,&quot;MoE&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/from_overlooked_to_explored_recovering_item_relations_via_mixture_of_perspe/">From Overlooked to Explored: Recovering Item Relations via Mixture of Perspectives for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示自注意力相似性偏置导致异构关系被系统性忽略，提出 K 个视角透镜（亲和/对比双视图）校准注意力，7 个基准全库排序平均提升 3–11%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;组件级嫁接&quot;,&quot;对比蒸馏&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/god_enhancing_generalization_via_deep_grafting_for_sequential_recommendatio/">GOD: Enhancing Generalization via Deep Grafting for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用「嫁接」构造教师条件混合源模型，对学生嵌入表/编码器做组件级蒸馏，Full-Ranking 下最高相对提升 13.92%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;乘积量化&quot;,&quot;物品码&quot;,&quot;嵌入压缩&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/higher_capacity_transfer_into_jpq_item_codes_for_scalable_sequential_recomm/">Higher Capacity Transfer into JPQ Item Codes for Scalable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将高维SVD bootstrap经PQ压缩为固定预算JPQ物品码，在leave-last-out全库nDCG@10上较RecJPQ平均约30%，保持内存高效。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;Query-Driven Decoder&quot;,&quot;时间对齐负采样&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/intsr_an_integrated_generative_framework_for_search_and_recommendation/">IntSR: An Integrated Generative Framework for Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 query 模态统一搜索/推荐与召回/排序四任务，提出 query 驱动解码器与时间对齐负采样，Amap 线上 GMV+9.34%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;归一化流&quot;,&quot;条件变分自编码器&quot;,&quot;LLM 语义融合&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/lsf_sr_latent_semantic_fusion_for_sequential_recommendation_via_flow_based_/">LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用归一化流增强的 CVAE 将 LLM 语义作为条件概率融合进 ID 协同嵌入，Full-Ranking 下 Recall@20 最高提升 12.98%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多框架后端&quot;,&quot;可复现基准&quot;,&quot;声明式配置&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/newsrex_an_open_source_multi_framework_for_neural_news_recommendation/">NewsReX: An Open-Source Multi-Framework for Neural News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>声明式YAML统一PyTorch/JAX双后端的神经新闻推荐框架，跨框架奇偶校验+多seed基准，注意力模型JAX提速1.1–2.2×。</p>
</dd>
<dt data-paper-tags="[&quot;GRAIL@CIKM 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;生成式用户表征&quot;,&quot;探索加权损失&quot;,&quot;采样Softmax难负例&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/palate_pretrained_autoregressive_latent_embeddings/">PALATE: Pretrained Autoregressive LATent Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用GPT式自回归模型对消费者门店互动序列建模，导出末词元隐状态为冻结256维用户嵌入，供DCNv2广告排序器跨场景复用，离线AUC+0.59%/+0.57%，线上收入+0.79%/+0.94%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;子词合并&quot;,&quot;注意力负载&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to/">Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>指出固定长度SID复用于历史侧引发“物品内注意力过载”，提出历史侧语义子词重分词(IST)+行为共现增强(BCA)的非对称分词，三数据集三骨干全库排序NDCG@10最高+13.8%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排/重排&quot;,&quot;检索增强&quot;,&quot;测试时自适应&quot;,&quot;协同记忆&quot;,&quot;熵融合&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/retrieve_then_adapt_retrieval_augmented_test_time_adaptation_for_sequential/">Retrieve-then-Adapt: Retrieval-Augmented Test-Time Adaptation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结序列模型，测试时检索协同记忆，并以交叉注意力与熵融合校正预测，全库排序下稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;行为线索路由&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/routerec_behavior_guided_sparse_routing_for_sequential_recommendation/">RouteRec: Behavior-Guided Sparse Routing for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用会话行为线索（节奏/焦点/记忆/流行度）驱动分层稀疏 MoE 路由，六数据集 18 指标平均排名 1.61 显著优于次优 4.11。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;稀疏KV记忆&quot;,&quot;持续学习&quot;,&quot;演化冲突&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/self_evolving_memory_for_generative_recommendation/">Self-Evolving Memory for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示生成式推荐持续演化中的“演化冲突”问题，提出稀疏 KV 记忆层 LION，以隔离记忆+巩固损失实现异构行为模式的持续演化。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态丢弃&quot;,&quot;鲁棒序列推荐&quot;]"><strong><a href="../../venues/CIKM/2026/multimodal_rec/sequential_modality_dropout_for_robust_multi_modal_sequential_recommendatio/">Sequential Modality Dropout for Robust Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在融合点注入逐样本伯努利模态掩码，四行代码即插即用，使四种多模态序列骨干在模态缺失下 HR@10 留存率提升 1.0–3.2×。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;推理端/精排&quot;,&quot;Machine Unlearning&quot;,&quot;Sequential Recommendation&quot;,&quot;Influence Subtraction&quot;]"><strong><a href="../../venues/CIKM/2026/debias_fairness_industrial/surf_subtractive_updates_for_recommender_forgetting/">SURF: Subtractive Updates for Recommender Forgetting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出推理期减影遗忘框架 SURF，kNN 邻域辅助模型减分，NDCG@20 最高 +32%，仅 2% 重训预算。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同视角内容理解&quot;,&quot;GRPO 强化学习精炼&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/towards_comprehensible_recommendation_with_large_language_model_fine_tuning/">Towards Comprehensible Recommendation with Large Language Model Fine-tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 GRPO 两阶段对齐 LLM 生成协同视角的用户模式与推荐理由，并以推荐模型打分为奖励精炼，Full-Ranking 下 Recall@5 相对提升 14–18%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;持续推荐&quot;,&quot;SPC 三难困境&quot;,&quot;Procrustes 投影&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/tracer_balancing_stability_plasticity_cognitivity_trilemma_for_llm_enhanced/">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 LLM 增强持续推荐中的 SPC 三难困境，用 Procrustes 投影、置信门控凝聚与语义同步三模块协同破局，Full-Ranking 下 ACC 最高提升 14.38%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;社交与群组推荐&quot;,&quot;精排/排序&quot;,&quot;图注意力&quot;,&quot;动态静态表示&quot;,&quot;物品相关图&quot;]"><strong><a href="../../venues/DASFAA/2022/social_group_rec/graph_neural_networks_with_dynamic_and_static_representations_for_social_re/">Graph Neural Networks with Dynamic and Static Representations for Social Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>GNN-DSR用LSTM与图注意力建模用户/物品的动态-静态表示，并聚合社交图与物品相关图，在Ciao/Epinions/Delicious上提升评分预测与物品排序。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;类别感知会话图&quot;,&quot;目标类别意图迁移&quot;]"><strong><a href="../../venues/DASFAA/2022/sequential_rec/intention_adaptive_graph_neural_network_for_category_aware_session_based_re/">Intention Adaptive Graph Neural Network for Category-aware Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出类别感知会话推荐新任务 CSBR，构建含目标类别节点的异构会话图，用意图自适应 GAT 将与目标类别相关的历史交互迁移到目标类别域，P@20 最高提升 6.30%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;CTR/CVR预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;频域FFT&quot;,&quot;隐式负反馈&quot;]"><strong><a href="../../venues/DASFAA/2023/ctr_feature_interaction/fan_fatigue_aware_network_for_click_through_rate_prediction_in_e_commerce_r/">FAN: Fatigue-Aware Network for Click-Through Rate Prediction in E-commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 FFT 把非点击序列变换到频域建模用户疲劳，并以类目调制与门控置信度融入 CTR 精排，离线 AUC 0.7249、在线 PCTR +1.63%、曝光类目数 +11.13%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2023&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;轻量化GCN&quot;,&quot;外部注意力&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="../../venues/DASFAA/2023/cross_domain_multi_scenario/towards_lightweight_cross_domain_sequential_recommendation_via_external_att/">Towards Lightweight Cross-domain Sequential Recommendation via External Attention-enhanced Graph Convolution Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>LEA-GCN 用单层聚合 GCN 与双通道外部注意力实现轻量跨域序列推荐，参数量降 6 倍仍全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双流架构&quot;,&quot;分层对比学习&quot;,&quot;跨模态对齐&quot;]"><strong><a href="../../venues/DASFAA/2025/sequential_rec/empowering_sequential_recommendation_from_collaborative_signals_and_semanti/">Empowering Sequential Recommendation from Collaborative Signals and Semantic Relatedness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双流 Transformer 分别编码 ID 与内容序列，用用户级/物品级分层对比对齐跨模态语义鸿沟，全库排序下 NDCG@10 最高提升 15.44%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;知识提示 (Knowledge Prompt)&quot;,&quot;推理提示 (Reasoning Prompt)&quot;,&quot;内容增强语言建模&quot;]"><strong><a href="../../venues/DASFAA/2025/sequential_rec/reformulating_sequential_recommendation_learning_dynamic_user_interest_with/">Reformulating Sequential Recommendation: Learning Dynamic User Interest with Content-enriched Language Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用知识提示+推理提示将序列推荐改写为文本生成任务，GPT-2 冻结主干下在三个数据集全排序协议上取得 Recall@5 相对提升约 5%–12%。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;跨模态一致性&quot;,&quot;可信决策&quot;,&quot;证据深度学习&quot;]"><strong><a href="../../venues/DASFAA/2025/multimodal_rec/truthsr_trustworthy_sequential_recommender_systems_via_user_generated_multi/">TruthSR: Trustworthy Sequential Recommender Systems via User-generated Multimodal Content</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用评论-图片跨模态一致性降噪 + 主观逻辑证据融合做可信序列推荐，四数据集 Full-Ranking 全面超越 MMSRec。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;安全审计(黑盒抽取+投毒攻击)&quot;,&quot;认知分布对齐&quot;,&quot;行为一致污染生成&quot;]"><strong><a href="../../venues/DASFAA/2026/debias_fairness_industrial/a_cognitive_distribution_and_behavior_consistent_framework_for_black_box_at/">A Cognitive Distribution and Behavior-Consistent Framework for Black-Box Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用位置衰减认知分布对齐蒸馏黑箱序列推荐模型，并融合协同先验与梯度双信号生成行为一致污染序列，黑箱攻击成功率0.93–0.98。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;偏好优化&quot;,&quot;动态负采样&quot;]"><strong><a href="../../venues/DASFAA/2026/llm_rec/dynamicpo_dynamic_preference_optimization_for_recommendation/">DynamicPO: Dynamic Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示多负样本 DPO 的偏好优化坍缩现象，提出动态边界负样本选择与双边际动态 β 调整，即插即用提升 LLM 推荐 HitRatio@1。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;序列与会话推荐&quot;,&quot;生成式端到端&quot;,&quot;Flow Matching&quot;,&quot;行为先验&quot;,&quot;确定性ODE推理&quot;]"><strong><a href="../../venues/DASFAA/2026/sequential_rec/flowrec_prior_informed_flow_matching_for_efficient_sequential_recommendatio/">FlowRec: Prior-Informed Flow Matching for Efficient Sequential Recommendation Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用行为先验替代高斯先验、直线流匹配替代扩散去噪，配合单步对齐损失实现确定性 ODE 生成，Full-Ranking 下 HR@5 相对提升最高 9.87%，推理提速约 6.7 倍。</p>
</dd>
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端策略学习&quot;,&quot;RL exploration&quot;,&quot;Ranking alignment&quot;,&quot;Collaborative regularization&quot;]"><strong><a href="../../venues/DASFAA/2026/rl_bandit_rec/personalized_education_with_ranking_alignment_recommendation/">Personalized Education with Ranking Alignment Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 RAR，以学习目标对称差与推荐序列距离构造排序对齐损失，增强 RL 题目推荐探索效率；在 KSS 与四个 KT 模拟器上显著提升学习效果。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;深度度量学习&quot;,&quot;Triplet Loss&quot;,&quot;会话编码器&quot;]"><strong><a href="../../venues/ECIR/2021/sequential_rec/metric_learning_for_session_based_recommendations/">Metric Learning for Session-based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将会话推荐重构为共享嵌入空间的度量学习，用位置加权 Triplet/NCAS 损失与轻量编码器实现 SOTA，并给出与 pairwise LTR 的同构性分析与完整消融。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长程依赖记忆参数&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="../../venues/ECIR/2022/sequential_rec/recommender_systems_when_memory_matters/">Recommender Systems: When Memory Matters</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以平稳性与长程依赖记忆参数 d 筛除&#x27;行为飘忽&#x27;用户，配合块级正负反馈在线学习，在大规模稀疏数据上 MAP 提升 7–9%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向 Transformer&quot;,&quot;多正样本对比学习&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/contrastive_learning_with_bidirectional_transformers_for_sequential_recomme/">Contrastive Learning with Bidirectional Transformers for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用双向 Transformer + 滑窗切分 + 完形/ dropout 混合增广构造多正样本对比学习，并动态重加权双损失；在 Beauty/Toys/ML-1M 全库排序下 NDCG@10 最高提升 21.04%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;序列 Next-Item 排序&quot;,&quot;离线强化学习&quot;,&quot;Critic Regularized Regression&quot;,&quot;DistilGPT2&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/integrating_offline_reinforcement_learning_with_transformers_for_sequential/">Integrating Offline Reinforcement Learning with Transformers for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DistilGPT2 初始化 CRR 离线 RL 做两阶段序列推荐；在 ML-25M/Yoochoose 全库排序 HR@10 显著超越 SASRec。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序（Next-Item）&quot;,&quot;模态遗忘&quot;,&quot;相关性蒸馏&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/lightweight_modality_adaptation_to_sequential_recommendation_via_correlatio/">Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 KDSR，以整体/分解模态相关性蒸馏和异步训练缓解两阶段序列推荐的模态遗忘，5 个数据集 HR@5 最高提升 23.9%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;GNN&quot;,&quot;可复现性评测&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/performance_comparison_of_session_based_recommendation_algorithms_based_on_/">Performance Comparison of Session-based Recommendation Algorithms based on GNNs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一复现8个GNN会话推荐模型，发现简单kNN/规则基线在MRR@20上不劣甚至更优，并揭示种子与嵌入维度敏感。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/下一物品排序&quot;,&quot;自对比学习&quot;,&quot;表示均匀性&quot;]"><strong><a href="../../venues/ECIR/2024/sequential_rec/self_contrastive_learning_for_session_based_recommendation/">Self Contrastive Learning for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将自对比损失作为物品表示均匀性正则，无需数据增广/正负样本构造，在三个会话推荐基准全库排序下显著提升 P@10/MRR@10。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;CVaR-DRO&quot;,&quot;Minimax Group Fairness&quot;,&quot;SASRec&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/improving_minimax_group_fairness_in_sequential_recommendation/">Improving Minimax Group Fairness in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 CVaR-DRO 首次引入 SASRec 序列推荐，无需组标注即可提升小众/短序列用户的最差组 NDCG@20，Retailrocket 整体提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;重复偏差&quot;,&quot;混合整数线性规划&quot;,&quot;下一篮推荐&quot;]"><strong><a href="../../venues/ECIR/2025/debias_fairness_industrial/repeat_bias_aware_optimization_of_beyond_accuracy_metrics_for_next_basket_r/">Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MILP 重排同时优化多样性/物品公平与重复偏差，在三个零售数据集上以可接受 Recall 损失显著改善 DS、logDP 与 RepBias。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Next Basket Recommendation&quot;,&quot;频率感知模块&quot;,&quot;Transformer 序列建模&quot;]"><strong><a href="../../venues/ECIR/2025/sequential_rec/saferec_self_attention_and_frequency_enriched_model_for_next_basket_recomme/">SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 SASRec 上并联频率感知打分模块，用物品出现频次与历史指示向量补足重复购买建模，Leave-One-Basket 全库排序下 Recall@10 最高提升 8%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;频域重标定&quot;,&quot;可复现性&quot;]"><strong><a href="../../venues/ECIR/2026/sequential_rec/a_systematic_reproducibility_study_of_bsarec_for_sequential_recommendation/">A Systematic Reproducibility Study of BSARec for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统复现BSARec，提出scaled DC频率度量，检验DWT/残差与padding，在7个基准上给出可复现性边界。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户画像蒸馏&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/ECIR/2026/llm_rec/pre_trained_llms_meet_sequential_recommenders_efficient_user_centric_knowle/">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预训练 LLM 生成文本用户画像并蒸馏进 SASRec/BERT4Rec 隐层，两阶段训练实现零推理开销的语义增强，Recall@10 最高提升 23.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;协同过滤&quot;,&quot;召回&quot;,&quot;用户邻域&quot;,&quot;归纳式表征&quot;]"><strong><a href="../../venues/ICDE/2021/collaborative_filtering/explore_user_neighborhood_for_real_time_e_commerce_recommendation/">Explore User Neighborhood for Real-time E-commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用归纳式 UI 模型在线推理用户向量，实时检索相似用户邻域，与全局打分融合，实现工业级实时用户邻域召回；全库排序下平均提升 SASRec 8.9% HR@20，线上点击+2.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;去偏公平与工业架构&quot;,&quot;召回&quot;,&quot;工业级全链路架构&quot;,&quot;会话推荐近邻方法&quot;,&quot;在线 A/B 延迟优化&quot;]"><strong><a href="../../venues/ICDE/2021/debias_fairness_industrial/learnings_from_a_retail_recommendation_system_on_billions_of_interactions_a/">Learnings from a Retail Recommendation System on Billions of Interactions at bol.com</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>电商 ABO 系统实证：item-CF 共现打分+九路召回集成优于神经会话模型，且 p90 延迟 32→15ms 带来订单 +2.19%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端训练（训练期增广）&quot;,&quot;扩散模型数据增广&quot;,&quot;上下文条件生成&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/diffusion_based_contrastive_learning_for_sequential_recommendation/">Context Matters: Enhancing Sequential Recommendation with Context-aware Diffusion-based Contrastive Learning (CaDiRec)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用双向上下文条件扩散模型为序列生成语义合理的替代物品构造对比正视图，5 数据集全库排序 HR@10 最高提升 11.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;序列数据增强&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/contrastive_learning_for_sequential_recommendation/">Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>CL4SRec 用裁剪/掩码/重排三种序列增强构造视图，以 InfoNCE 对比损失联合下一项预测，在四个数据集 Full-Ranking 上 HR@5 平均提升 11.02%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频率感知对比学习&quot;,&quot;自适应数据增广&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/frequency_aware_adaptive_contrastive_learning_for_sequential_recommendation/">Frequency-aware Adaptive Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用频率自适应扰动与序列级重加权缓解低频物品和稀疏用户在对比增广中的过扰动，全库排序 HR@10 最高提升约 3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;微行为建模&quot;,&quot;多重图神经网络&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/micro_behavior_encoding_for_session_based_recommendation/">Micro-Behavior Encoding for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将会话转为带星节点的有向多重图，用GRU边特征与操作感知自注意力联合编码微行为序列/二元关系，全库排序H@10最高提升14.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;多意图解耦&quot;,&quot;意图感知对比学习&quot;,&quot;K-means 原型聚类&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/multi_intent_aware_contrastive_learning_for_sequential_recommendation/">Multi-intent Aware Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MCLRec 用 K-means 意图原型、Top-R 过滤与距离衰减项构建多意图感知对比学习，在四个 5-core 数据集全库排序下 HR/NDCG 最高提升 15.64%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;软硬双层序列去噪&quot;,&quot;跨信号对比学习&quot;,&quot;S形课程学习&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/multi_level_sequence_denoising_with_cross_signal_contrastive_learning_for_s/">Multi-Level Sequence Denoising with Cross-Signal Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>MSDCCL 用软/硬双层去噪与跨信号对比学习互补消除序列噪声，并引入 S 形课程学习；在 5 个数据集全库排序下 HR@20 较最强去噪基线最高提升 118.9%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;可学习序列增强&quot;,&quot;质量感知对比学习&quot;]"><strong><a href="../../venues/ICDE/2022/self_supervised_denoising/quality_aware_collaborative_multi_positive_contrastive_learning_for_sequent/">Quality-Aware Collaborative Multi-Positive Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以可学习协同增强模块生成同目标/相似双视图，用增强置信度加权多正样本 InfoNCE，Beauty 上 HR@10 较 CL4SRec 提升 11.4%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;相对对比学习&quot;,&quot;相似正样本选择&quot;]"><strong><a href="../../venues/ICDE/2022/sequential_rec/relative_contrastive_learning_for_sequential_recommendation_with_similarity/">Relative Contrastive Learning for Sequential Recommendation with Similarity-based Positive Pair Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将同目标序列设为强正、相似序列设为弱正，用带 max 边界的加权相对对比损失训练 SASRec/FMLP，全库排序下 6 数据集平均提升 4.88%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;频域滤波&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/ICDE/2023/sequential_rec/contrastive_enhanced_slide_filter_mixer_for_sequential_recommendation/">Contrastive Enhanced Slide Filter Mixer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用频域滑动滤波与对比学习替代自注意力，在5个基准全库排序下 HR@10 最高相对提升21.39%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;影响路径生成&quot;,&quot;个性化注意力掩码&quot;]"><strong><a href="../../venues/ICDE/2023/sequential_rec/influential_recommender_system/">Influential Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出影响力推荐新范式 IRS 与 Transformer 模型 IRN，用个性化易受影响性掩码 PIM 生成引导用户走向目标物品的影响路径，SR20 最高提升约 3.5 倍。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;文本嵌入白化&quot;,&quot;冷启动推荐&quot;]"><strong><a href="../../venues/ICDE/2024/sequential_rec/are_id_embeddings_necessary_whitening_pre_trained_text_embeddings_for_effec/">Are ID Embeddings Necessary? Whitening Pre-trained Text Embeddings for Effective Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出ZCA白化预训练文本嵌入并融合全/松弛白化表示，在4个Amazon/食谱数据集全库排序上超越ID+文本基线。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;生成式视图增强&quot;,&quot;变分自编码器&quot;,&quot;元优化两步训练&quot;]"><strong><a href="../../venues/ICDE/2024/self_supervised_denoising/meta_optimized_joint_generative_and_contrastive_learning_for_sequential_rec/">Meta-optimized Joint Generative and Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以 VAE Seq2Seq 生成器替代手工增强生成对比视图，双 ELBO 导出互信息项并落地 InfoNCE，元优化两步训练自适应视图，全库排序 HR@10 最高提升 20.8%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;序列推荐&quot;,&quot;序列去噪&quot;,&quot;自增强&quot;,&quot;全局关系编码&quot;]"><strong><a href="../../venues/ICDE/2024/self_supervised_denoising/ssdrec_self_augmented_sequence_denoising_for_sequential_recommendation/">SSDRec: Self-Augmented Sequence Denoising for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>ICDE 2024 自监督交互去噪 | 序列增强后再去噪以缓解过度/不足去噪(OUPs)，Full-Ranking 提升 SASRec 平均 23%–110%</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;软提示蒸馏&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/ICDE/2025/llm_rec/delrec_distilling_sequential_pattern_to_enhance_llms_based_sequential_recom/">DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用软提示两阶段蒸馏传统序列模型的时间感知与推荐模式，再微调 LLM 做序列推荐，四数据集 HR/NDCG 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;模态交换对比学习&quot;]"><strong><a href="../../venues/ICDE/2026/multimodal_rec/cammsr_category_guided_attentive_mixture_of_experts_for_multimodal_sequenti/">CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用类别预测任务引导注意力 MoE 动态分配模态权重，并以模态交换对比学习对齐跨模态序列表征，Full-Ranking 下 NDCG@5 最高提升 15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;触发式推荐&quot;,&quot;意图建模&quot;,&quot;成对损失&quot;]"><strong><a href="../../venues/ICDE/2026/ctr_feature_interaction/cascading_relevance_driven_recommendation_network_for_ctr_prediction_in_tri/">Cascading Relevance-driven Recommendation Network for CTR Prediction in Trigger-Introduced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向触发式推荐(TIR)中 trigger-target 相关性建模不足，以门控交叉、级联注意力融合与类目辅助成对损失取得 AUC 相对提升 6.31%/7.18%，线上 CTR +3.87%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;精排&quot;,&quot;持续学习&quot;,&quot;兴趣记忆检索&quot;,&quot;模型无关框架&quot;]"><strong><a href="../../venues/ICDE/2026/poi_spatiotemporal_rec/efficient_model_agnostic_continual_learning_for_next_poi_recommendation/">Efficient Model-Agnostic Continual Learning for Next POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 GIRAM：以分布型兴趣记忆 + 上下文感知键编码 + cVAE 多键生成检索 + 一致性自适应融合，实现免重训的持续 Next-POI 推荐，三数据集 Acc@5 与 MRR 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;不确定性量化&quot;]"><strong><a href="../../venues/ICDE/2026/cold_start_rec/user_adaptive_meta_learning_for_cold_start_medication_recommendation_with_u/">User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出两级元自适应（自自适应+同伴自适应）与不确定性过滤的 MetaDrug，在 MIMIC-III/AKI 冷启动患者上 PRAUC 提升约 2.5%-6.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;对比预测编码 (CPC)&quot;,&quot;记忆网络与多实例负采样&quot;]"><strong><a href="../../venues/ICDM/2021/sequential_rec/memory_augmented_multi_instance_contrastive_predictive_coding_for_sequentia/">Memory Augmented Multi-Instance Contrastive Predictive Coding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>记忆增强 CPC + 多实例 NCE：用记忆残差捕捉长期偏好、Dropout 语义正样本缓解监督稀疏，四数据集全库排序 HR@5 最高提升 152.7%</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列推荐&quot;,&quot;MOBA 数据集&quot;,&quot;RNN&quot;]"><strong><a href="../../venues/ICDM/2021/sequential_rec/sequential_item_recommendation_in_the_moba_game_dota_2/">Sequential Item Recommendation in the MOBA Game Dota 2</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>构建 Dota-350k 购买序列数据集并系统评测 SIR 模型，发现 GRU 以 Rec@3 0.7363 超越 Transformer。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户行为建模&quot;,&quot;字段对选择&quot;,&quot;注意力剪枝&quot;]"><strong><a href="../../venues/ICDM/2022/ctr_feature_interaction/autoattention_automatic_field_pair_selection_for_attention_in_user_behavior/">AutoAttention: Automatic Field Pair Selection for Attention in User Behavior Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将全部 item/user/context 字段纳入注意力查询，学习字段对权重并剪枝自动选对，Alibaba/Tencent 用户加权 AUC 相对提升 2.19%/3.58%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2022&quot;,&quot;序列推荐&quot;,&quot;召回/近邻检索&quot;,&quot;神经切线核&quot;,&quot;无限宽RNN&quot;]"><strong><a href="../../venues/ICDM/2022/sequential_rec/beyond_double_ascent_via_recurrent_neural_tangent_kernel_in_sequential_reco/">Beyond Double Ascent via Recurrent Neural Tangent Kernel in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>观测模型宽度“双上升”，提出免训练 OverRec：以 RNTK 度量序列相似度并加权 kNN，逼近无限宽 RNN，在 Beauty/Toys 全库排序提升。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;社交与群组推荐&quot;,&quot;精排&quot;,&quot;信息扩散级联&quot;,&quot;多视图注意力融合&quot;]"><strong><a href="../../venues/ICDM/2022/social_group_rec/igniter_news_recommendation_in_microblogging_applications_extended_version/">IGNiteR: News Recommendation in Microblogging Applications (Extended Version)</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用新闻转发级联训练影响力节点嵌入，并以语义/扩散/采纳曲线三视图注意力融合做微博客新闻推荐，采样排序下 NDCG@10 相对最强基线提升 2.3%–3.2%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;动态子网络路由&quot;,&quot;权重切片超网&quot;,&quot;Gumbel-Softmax&quot;]"><strong><a href="../../venues/ICDM/2022/sequential_rec/one_person_one_model_learning_compound_router_for_sequential_recommendation/">One Person, One Model—Learning Compound Router for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>CANet 用轻量路由器为每条用户序列在嵌入/隐藏/层深三维复合空间内路由子网络，权重切片共享超网，Gumbel-Softmax 端到端可微，全库排序下减算 55~65% 而 NDCG 仅降约 1.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;词图协同预训练&quot;,&quot;跨场景可迁移物品表征&quot;,&quot;零样本推荐&quot;]"><strong><a href="../../venues/ICDM/2023/sequential_rec/collaborative_word_based_pre_trained_item_representation_for_transferable_r/">Collaborative Word-based Pre-trained Item Representation for Transferable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用共点击词图做词级对比预训练，将 CF 信号注入 PLM 物品表征，在 6 个跨场景数据集上微调与零样本设定全面超越 UniSRec/MoRec。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mamba&quot;,&quot;选择性状态空间模型&quot;,&quot;表格特征分词&quot;]"><strong><a href="../../venues/ICDM/2024/sequential_rec/mamba_for_scalable_and_efficient_personalized_recommendations/">Mamba for Scalable and Efficient Personalized Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 Mamba 选择性状态空间层替换 FT-Transformer 中的注意力层，在双塔架构下处理表格型用户/内容特征，以 O(L) 线性复杂度与约 40% 参数量取得持平或更优的 P/R/MRR/HR。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端会话推荐排序&quot;,&quot;Neural Decision Forest&quot;,&quot;James-Stein Shrinkage&quot;]"><strong><a href="../../venues/ICDM/2024/sequential_rec/sr_predictao_session_based_recommendation_with_high_capability_predictor_ad/">SR-PredictAO: Session-based Recommendation with High-Capability Predictor Add-On</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>在编码器-预测器范式上外挂 James-Stein 去噪与神经决策森林高能力预测器；全库排序下 HR@20 最高提升 2.9%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;时序用户画像&quot;,&quot;注意力融合&quot;]"><strong><a href="../../venues/ICDM/2025/llm_rec/effectiveness_of_llms_in_temporal_user_profiling_for_recommendation/">Effectiveness of LLMs in Temporal User Profiling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 分别生成短期/长期自然语言画像，经 SBERT 编码与注意力融合做内容推荐，Movies&amp;TV 上 Recall@10 相对提升 17%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;上下文压缩&quot;,&quot;个性化专家&quot;,&quot;KV缓存&quot;]"><strong><a href="../../venues/ICDM/2025/sequential_rec/efficient_sequential_recommendation_for_long_term_user_interest_via_persona/">Efficient Sequential Recommendation for Long Term User Interest Via Personalization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习 token 将长历史分段压缩为个性化专家，配合定制注意力掩码与 KV 缓存，在 HSTU/HLLM 上以约 1/4 推理开销逼近全序列精度。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;层间出口融合&quot;,&quot;自适应连续路由&quot;]"><strong><a href="../../venues/ICDM/2026/llm_rec/empowering_compact_llms_with_fusion_of_layer_wise_exits_for_recommendation/">Empowering Compact LLMs with Fusion of Layer-wise Exits for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在紧凑 LLM 每层插入预测出口并用连续 ReLU 路由器自适应融合多层打分分布，target-k hinge 损失约束激活出口数，全库排序下精度超越大模型基线且延迟仅增 &lt;1ms。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;观看时长分布预测&quot;,&quot;混合密度网络&quot;]"><strong><a href="../../venues/ICDM/2026/sequential_rec/hierarchical_exponential_gaussian_mixtures_for_watch_time_distribution_pred/">Hierarchical Exponential-Gaussian Mixtures for Watch-Time Distribution Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用分层 skip-watch 分解与结构化初始化修复 EGMN 的方差坍缩与组件冗余，在固定骨干换头协议下提升 XAUC 与阈值事件预测，线上会话深度 +9.26%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;几何记忆&quot;,&quot;轨迹条件重建&quot;,&quot;非均匀度量&quot;]"><strong><a href="../../venues/ICDM/2026/llm_rec/memory_has_geometry_non_uniform_geometric_memory_for_long_horizon_personali/">Memory Has Geometry: Non-Uniform Geometric Memory for Long-Horizon Personalized AI</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将长程个性化记忆重构为用户特定非均匀几何状态空间上的轨迹条件状态估计，替代扁平记录检索范式。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;位置编码对齐&quot;,&quot;数据熵结构&quot;,&quot;评估协议泄漏&quot;]"><strong><a href="../../venues/ICDM/2026/sequential_rec/revisiting_self_attentive_sequential_recommendation_beyond_the_llm_paradigm/">Revisiting Self-Attentive Sequential Recommendation Beyond the LLM Paradigm</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 SASRec 为锚点对比推荐与语言建模的熵结构差异，诊断四条继承假设并给出数据优先议程，含同模型双视图实证。</p>
</dd>
<dt data-paper-tags="[&quot;ICDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨源行为预训练&quot;,&quot;级联掩码预测&quot;]"><strong><a href="../../venues/ICDM/2026/sequential_rec/user_representation_via_cross_multi_source_behavior_pre_training_for_mobile/">User Representation via Cross Multi-source Behavior Pre-training for Mobile Games</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Mining">ICDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向设备级多源多粒度行为日志，提出跨源级联掩码预训练 CM-PTM，以源→类目→App→操作逐级预测统一建模跨源依赖，离线 AUC 提升 1.05%–2.85%，线上 A/B 提升 1.28%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;序列与会话推荐&quot;,&quot;召回(Next-Item 预测)&quot;,&quot;时间点过程&quot;,&quot;条件强度函数&quot;,&quot;注意力调制&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2021/sequential_rec/learning_self_modulating_attention_in_continuous_time_space_with_applicatio/">Learning Self-Modulating Attention in Continuous Time Space with Applications to Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用时间点过程的时变条件强度调制注意力权重并加连续时间正则，Strong Generalization 协议下 HR@10 最高 +2.8%、NDCG@10 最高 +4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Session-based Recommendation&quot;,&quot;Multilingual Multi-locale Dataset&quot;,&quot;Cross-locale Transfer&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/sequential_rec/amazon_m2_a_multilingual_multi_locale_shopping_session_dataset_for_recommen/">Amazon-M2: A Multilingual Multi-locale Shopping Session Dataset for Recommendation and Text Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>发布首个多语言多地区电商会话数据集（141万商品/360万会话），定义下一商品推荐、跨域迁移推荐与商品标题生成三任务，并揭示流行度与复制基线仍压制深度模型。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;属性图模式&quot;,&quot;记忆增强Transformer&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/sequential_rec/enhancing_user_intent_capture_in_session_based_recommendation_with_attribut/">Enhancing User Intent Capture in Session-Based Recommendation with Attribute Patterns</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以频繁属性图模式作记忆增强会话 Transformer，全库 top-K 评测，公开与工业数据平均提升约4.5%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;引导扩散 (Guided Diffusion)&quot;,&quot;无负采样&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/generative_rec/generate_what_you_prefer_reshaping_sequential_recommendation_via_guided_dif/">Generate What You Prefer: Reshaping Sequential Recommendation via Guided Diffusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DreamRec 将序列推荐重塑为学习生成范式，用 Transformer 引导的无分类器扩散生成 oracle 物品，弃用负采样，三数据集 HR@20 相对提升 7%–21%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/会话级RL&quot;,&quot;用户模拟器&quot;,&quot;长期留存&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/rl_bandit_rec/kuaisim_a_comprehensive_simulator_for_recommender_systems/">KuaiSim: A Comprehensive Simulator for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出多行为、跨会话推荐模拟器 KuaiSim，统一请求级列表、整会话与留存优化，并在 KuaiRand/ML-1m 给出可复现基准。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2023&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;RQ-VAE&quot;,&quot;Semantic ID&quot;,&quot;Generative Retrieval&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/generative_rec/recommender_systems_with_generative_retrieval/">Recommender Systems with Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TIGER 用 RQ-VAE 构建层次 Semantic ID，并以 Transformer 自回归生成下一物品 ID；在 Amazon 三类 Full-Ranking 协议下显著超越 SASRec/S3-Rec，并展示冷启动与多样性收益。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;残差策略学习&quot;,&quot;离线强化学习&quot;,&quot;条件变分自编码器&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2023/rl_bandit_rec/resact_reinforcing_long_term_engagement_in_sequential_recommendation_with_r/">ResAct: Reinforcing Long-term Engagement in Sequential Recommendation with Residual Actor</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 CVAE 重构在线策略并叠加残差 Actor 做近端策略改进，配双重信息论正则，NCIS 离线评估下 Session Length 提升 35%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列推荐&quot;,&quot;精排/生成式端到端&quot;,&quot;Instance-wise LoRA&quot;,&quot;Mixture of Experts&quot;,&quot;负迁移&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/sequential_rec/customizing_language_models_with_instance_wise_lora_for_sequential_recommen/">Customizing Language Models with Instance-wise LoRA for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 LoRA 拆成 MoE 专家并由序列表征门控，实现实例级参数定制，HR@1 平均相对 LoRA 提升 11.4% 且可训练参数不增。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;端到端可学习聚类&quot;,&quot;意图对比学习&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/sequential_rec/end_to_end_learnable_clustering_for_intent_learning_in_recommendation/">End-to-end Learnable Clustering for Intent Learning in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将意图聚类中心设为可学习神经元，端到端优化聚类与对比学习，全库排序下NDCG@5较ICLRec提升8.9%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回/排序（序列下一项）&quot;,&quot;图正则&quot;,&quot;嵌入优化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/graph_rec/graph_enhanced_optimizers_for_structure_aware_recommendation_embedding_evol/">Graph-enhanced Optimizers for Structure-aware Recommendation Embedding Evolution</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出SEvo，将图正则注入嵌入更新而非GNN模块；Leave-One-Out全库排序下多backbone平均提升9%–23%，百万节点数据最高139%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/下一物品预测&quot;,&quot;双视图语义协同融合&quot;,&quot;检索增强自蒸馏&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/sequential_rec/llm_esr_large_language_models_enhancement_for_long_tailed_sequential_recomm/">LLM-ESR: Large Language Models Enhancement for Long-tailed Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用冻结的 LLM 物品/用户语义嵌入搭建语义-协同双视图与相似用户检索自蒸馏，在不增加 LLM 推理开销的前提下同时缓解长尾物品与长尾用户难题。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS 2024&quot;,&quot;LLM推荐&quot;,&quot;重排&quot;,&quot;DPO 偏好对齐&quot;,&quot;Softmax 负采样&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2024/llm_rec/on_softmax_direct_preference_optimization_for_recommendation/">On Softmax Direct Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将成对 DPO 推广为多负样本 softmax 排序损失 S-DPO，理论桥接负采样 softmax 并自带难负样本挖掘，三数据集 HR@1 较次优提升 11.1%–47.0%。</p>
</dd>
<dt data-paper-tags="[&quot;ICML 2025&quot;,&quot;生成式推荐&quot;,&quot;上下文感知分词&quot;,&quot;集合置换正则&quot;,&quot;BPE&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/generative_rec/actionpiece_contextually_tokenizing_action_sequences_for_generative_recomme/">ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个上下文感知的动作序列分词器，将动作建模为无序特征集，用加权共现合并+集合置换正则实现同一动作在不同上下文下差异化分词，NDCG@10 相对提升 6.0%–12.8%。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;OOV Token&quot;,&quot;META ID&quot;,&quot;元路径聚类&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/improving_llms_for_recommendation_with_out_of_vocabulary_tokens/">Improving LLMs for Recommendation with Out-Of-Vocabulary Tokens</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用元路径聚类生成 OOV 层级 token 构建用户/物品 META ID，兼顾记忆性与多样性，序列推荐 HR@10 最高提升 30%+。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;生成式推荐&quot;,&quot;扩散模型&quot;,&quot;列表级排序&quot;,&quot;轨迹预测&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/generative_rec/listwise_preference_diffusion_optimization_for_user_behavior_trajectories_p/">Listwise Preference Diffusion Optimization for User Behavior Trajectories Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Plackett-Luce 列表级排序信号注入扩散 ELBO，实现多步用户行为轨迹的连贯生成，并提出 SeqMatch 轨迹级指标，Full-Ranking 下 SeqMatch@50 最高提升 36.8%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;嵌入解耦&quot;,&quot;梯度冲突&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/long_sequence_recommendation_models_need_decoupled_embeddings/">Long-Sequence Recommendation Models Need Decoupled Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示长序列推荐中注意力与表征共享嵌入导致的梯度支配与冲突，提出双嵌入表 DARE，AUC 最高提升 9‰ 并加速检索 50%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;一步生成&quot;,&quot;偏好对齐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/generative_rec/on_efficiency_effectiveness_trade_off_of_diffusion_based_recommenders/">On Efficiency-Effectiveness Trade-off of Diffusion-based Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间一致性正则平滑去噪函数实现一步生成，再用自适应偏好对齐修正轨迹偏差，兼顾扩散推荐效率与效果。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;端到端推荐&quot;,&quot;行为序列预训练&quot;,&quot;结构化分词&quot;,&quot;原型交叉注意力&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/panther_generative_pretraining_beyond_language_for_sequential_user_behavior/">PANTHER: Generative Pretraining Beyond Language for Sequential User Behavior Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将支付交易视为行为 token 做生成式预训练，SPRM 卷积+原型交叉注意力捕捉周期模式，用户画像对比学习，微信支付线上欺诈召回 +38.6%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;生成式推荐&quot;,&quot;扩散模型推荐&quot;,&quot;个性化排序损失&quot;,&quot;变分推断&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/generative_rec/preference_diffusion_for_recommendation/">Preference Diffusion for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 BPR 改写为对数似然排序目标并推导变分上界，为扩散推荐器注入多负样本偏好信号，Full-Ranking 下 R@5 最高提升 19.35%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;LLM语义检索&quot;,&quot;可学习样本合成&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/semantic_retrieval_augmented_contrastive_learning_for_sequential_recommenda/">Semantic Retrieval Augmented Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 语义检索构建用户/物品候选池并配可学习样本合成器，替代随机增广与 K-Means 硬规则，全库排序下 HR@20 最高提升 11.82%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;逐层特征对齐&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/llm_rec/slmrec_distilling_large_language_models_into_small_for_sequential_recommend/">SLMRec: Distilling Large Language Models into Small for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发现 LLM 中间层冗余，用逐层分块特征蒸馏将 8 层教师压缩为 4 层学生，仅 13% 参数达 6.6x/8.0x 加速且性能不降。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;时变图滤波器&quot;,&quot;图信号处理&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2025/sequential_rec/tv_rec_time_variant_convolutional_filter_for_sequential_recommendation/">TV-Rec: Time-Variant Convolutional Filter for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用图信号处理中的时变图滤波器替代固定卷积核与自注意力，在6个基准上全库排序平均提升7.49%且推理更快。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;生成式推荐&quot;,&quot;递归自我改进&quot;,&quot;保真度质量控制&quot;,&quot;数据稠密化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/self_supervised_denoising/can_recommender_systems_teach_themselves_a_recursive_self_improving_framewo/">Can Recommender Systems Teach Themselves? A Recursive Self-Improving Framework with Fidelity Control</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 RSIR 递归自我改进框架，用模型自生成序列+保真度质量控制稠密化稀疏交互，理论证明其等价于流形切向隐式正则，多骨干多数据集 Full-Ranking 下 NDCG/Recall 稳定提升 3%–14%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;KV Cache 压缩&quot;,&quot;跨用户协同共享&quot;,&quot;长序列推理加速&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential/">CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可学习全局 K/V 池跨用户共享协同信息，将 KV cache 压缩至 0.8% 且精度不降，解码延迟随 batch 增长近乎平坦。</p>
</dd>
<dt data-paper-tags="[&quot;ICML&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;参数掩码控制&quot;,&quot;请求感知推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/cramer_control_via_request_aware_masking_for_editing_recommenders/">CRAMER: Control via Request-Aware Masking for Editing Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将自然语言请求编码为稀疏行列门控掩码，对冻结序列推荐骨干做后验行为编辑，采样100负例排序下全面超越四类请求感知基线且推理仅增0.018s。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;多任务学习&quot;,&quot;召回&quot;,&quot;目标条件监督学习&quot;,&quot;多目标推荐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/multi_task_reranking/goal_conditioned_supervised_learning_for_multi_objective_recommendation/">Goal-Conditioned Supervised Learning for Multi-Objective Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 GCSL 的标量目标扩展为多维向量目标，用 Transformer 编码器 + 交叉熵直接做多目标下一物品预测，无需复杂架构与权重调参，在 RetailRocket/Challenge15 上购买 HR@10 相对提升约 4-17%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;证据高亮&quot;,&quot;弱监督强化学习&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/learning_evidence_highlighting_for_frozen_llms/">Learning Evidence Highlighting for Frozen LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用轻量 Emphasis Actor 以 RL 在冻结 LLM 的原始上下文中插入高亮标签，解耦证据选择与推理，序列推荐 HR@10 相对最优基线提升 5.38%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;低秩解耦注意力&quot;,&quot;线性注意力&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/lime_link_based_user_item_interaction_modeling_with_decoupled_xor_attention/">LIME: Link-based User-item Interaction Modeling with Decoupled XOR Attention for Efficient Test Time Scaling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用低秩 link embedding 解耦用户-候选交互并预计算 QK 权重，配合 XOR 线性注意力将长历史建模降至 O(N)，实现 HSTU 级精度与 10× 推理加速。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;超长用户序列&quot;,&quot;准线性注意力&quot;,&quot;嵌入缓存交付&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/massive_memorization_with_hundreds_of_trillions_of_parameters_for_sequentia/">Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>VISTA 将超长用户历史两阶段解耦：离线用准线性注意力压缩为数百种子嵌入并缓存，下游仅做候选-摘要注意力，工业上线推理 GPU 降 94%。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;偏好状态修复&quot;,&quot;冻结宿主适配&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/sequential_rec/not_all_is_lost_repairing_lossy_user_preference_states_of_personalization_e/">Not All Is Lost: Repairing Lossy User Preference States of Personalization Encoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结宿主编码器与任务头，仅训练轻量 REPAIR 模块，用缓存逐步表征对压缩偏好状态做时序结构化残差校正，12 个推荐宿主 MRR/nDCG@10 全面提升。</p>
</dd>
<dt data-paper-tags="[&quot;NeurIPS&quot;,&quot;生成式推荐&quot;,&quot;SimHash语义ID&quot;,&quot;并行解码&quot;,&quot;语义对齐&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/generative_rec/rethinking_semantic_id_construction_for_generative_recommendation_simhash_w/">Rethinking Semantic ID Construction for Generative Recommendation: SimHash with Parallel Decoding and Semantic Alignment</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示LSH分词劣势源于与自回归解码的结构错配，提出免训练SimHash+并行解码+语义对齐的FLASH，全库排序下NDCG@5最高提升14.46%。</p>
</dd>
<dt data-paper-tags="[&quot;ICLR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排/评分预测&quot;,&quot;长短期兴趣草图&quot;,&quot;推理内化&quot;]"><strong><a href="../../venues/ICLR_NeurIPS_ICML/2026/llm_rec/token_efficient_long_term_interest_sketching_and_internalized_reasoning_for/">Token-Efficient Long-Term Interest Sketching and Internalized Reasoning for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：Top ML Conferences (RecSys Papers)">ICLR / NeurIPS / ICML (RecSys)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以话题级长短期兴趣草图压缩提示、GRPO 显式推理 + 隐藏态对齐内化 CoT，实现单 token 答案解码，MAE 较次优降 12.3% 且延迟降 100 倍以上。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;神经架构搜索&quot;,&quot;三阶段推荐模型&quot;,&quot;One-shot 随机搜索&quot;]"><strong><a href="../../venues/IJCAI/2021/ctr_feature_interaction/ameir_automatic_behavior_modeling_interaction_exploration_and_mlp_investiga/">AMEIR: Automatic Behavior Modeling, Interaction Exploration and MLP Investigation in the Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出三阶段搜索空间与三步 one-shot 随机搜索，自动联合行为建模、特征交互与 MLP；Alimama AUC 0.6383，Criteo/Avazu 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;特征交叉算子&quot;,&quot;用户行为建模&quot;,&quot;AutoML架构搜索&quot;]"><strong><a href="../../venues/IJCAI/2021/ctr_feature_interaction/deep_learning_for_click_through_rate_estimation/">Deep Learning for Click-Through Rate Estimation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>深度 CTR 综述：以显式交叉算子、用户行为建模、自动化架构搜索三条主线梳理浅层到深度模型演进与工业落地路线。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;不可靠实例过滤&quot;,&quot;不确定性建模&quot;,&quot;高斯分布表征&quot;]"><strong><a href="../../venues/IJCAI/2021/sequential_rec/does_every_data_instance_matter_enhancing_sequential_recommendation_by_elim/">Does Every Data Instance Matter? Enhancing Sequential Recommendation by Eliminating Unreliable Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用高斯分布同时建模实例损失与不确定性，配合不确定性感知图卷积 UGCN 与端到端过滤机制剔除「目标与输入不匹配」的不可靠实例，四数据集采样排序下 NDCG 最高提升 11.98%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;时间间隔建模&quot;,&quot;虚拟中心节点图&quot;]"><strong><a href="../../venues/IJCAI/2021/sequential_rec/exploring_periodicity_and_interactivity_in_multi_interest_framework_for_seq/">Exploring Periodicity and Interactivity in Multi-Interest Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>IJCAI 2021｜序列推荐论文：以时间间隔矩阵建模兴趣周期性、虚拟中心节点图建模物品交互性，再经自注意力抽取多兴趣向量；在 Amazon/Taobao 全库排序下 NDCG@50 最高提升约 26%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2021&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;图构造与信息传播&quot;,&quot;综述与分类体系&quot;]"><strong><a href="../../venues/IJCAI/2021/graph_rec/graph_learning_based_recommender_systems_a_review/">Graph Learning based Recommender Systems: A Review</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>首篇 GLRS 系统综述：按数据形态→图结构→学习范式三层正交归类，形式化 $R=\\arg\\max_\\Theta f(M(\\Theta)|G)$，并汇总 17 个开源实现与 16 个公开数据集。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;自监督一致性预训练&quot;,&quot;模仿学习知识蒸馏&quot;]"><strong><a href="../../venues/IJCAI/2021/sequential_rec/improving_sequential_recommendation_consistency_with_self_supervised_imitat/">Improving Sequential Recommendation Consistency with Self-Supervised Imitation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用三条自监督一致性任务构造教师模型，再以 logit+表征模仿蒸馏进学生序列推荐器，1:99 采样排序下 NDCG@10 提升 2.0%–3.8%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;序列去噪&quot;,&quot;强化学习策略&quot;,&quot;序列模式挖掘&quot;]"><strong><a href="../../venues/IJCAI/2021/sequential_rec/pattern_enhanced_contrastive_policy_learning_network_for_sequential_recomme/">Pattern-enhanced Contrastive Policy Learning Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将序列去噪建模为 MDP，用频繁序列模式作先验引导策略网络逐物品打标签，并以正/负子序列延迟奖励做对比学习，Full-Ranking 下 HR@10 相对最佳基线提升 2-4%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;预训练语言模型&quot;,&quot;多粒度匹配&quot;]"><strong><a href="../../venues/IJCAI/2021/llm_rec/unbert_user_news_matching_bert_for_news_recommendation/">UNBERT: User-News Matching BERT for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用预训练 BERT 联合建模用户历史新闻序列与候选新闻，在词级与新闻级双粒度匹配信号上精排，MIND 上 Full 候选内 AUC 最高提升 2.81%</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;加权物品转移图&quot;,&quot;图对比学习&quot;,&quot;MMD表征对齐&quot;]"><strong><a href="../../venues/IJCAI/2022/sequential_rec/enhancing_sequential_recommendation_with_graph_contrastive_learning/">Enhancing Sequential Recommendation with Graph Contrastive Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建全用户加权物品转移图，以子图采样+共享GNN生成增广视图，结合图对比损失与MMD个性化对齐，Full-Ranking下HR@10最高提升12.9%</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（Next-Item 检索）&quot;,&quot;MLP-Mixer&quot;,&quot;三向信息融合&quot;]"><strong><a href="../../venues/IJCAI/2022/sequential_rec/mlp4rec_a_pure_mlp_architecture_for_sequential_recommendations/">MLP4Rec: A Pure MLP Architecture for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用纯 MLP 的三向 Mixer（序列/通道/特征）替代自注意力，免位置编码、线性复杂度，Beauty 上 HR@10 提升 6.4%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列行为检索&quot;,&quot;多类型行为融合&quot;,&quot;边信息解耦注意力&quot;]"><strong><a href="../../venues/IJCAI/2023/sequential_rec/a_survey_on_user_behavior_modeling_in_recommender_systems/">A Survey on User Behavior Modeling in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以长度/类型/边信息三轴重划 UBM 为四类范式，串联代表模型与线上收益，论证长序列检索与解耦边信息最具落地性</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;神经符号学习&quot;,&quot;Beta 分布嵌入&quot;]"><strong><a href="../../venues/IJCAI/2023/sequential_rec/sequential_recommendation_with_probabilistic_logical_reasoning/">Sequential Recommendation with Probabilistic Logical Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布做概率逻辑嵌入，将可闭运算的 AND/NOT 逻辑推理网络与 DNN 序列模型双路解耦融合，全库排序下各骨干稳定涨点。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回/序列建模&quot;,&quot;四维张量降维&quot;,&quot;多级跨域融合与跨域桥&quot;,&quot;辅助学习技术&quot;]"><strong><a href="../../venues/IJCAI/2024/cross_domain_multi_scenario/a_survey_on_cross_domain_sequential_recommendation/">A Survey on Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以四维数据张量统一刻画 CDSR，宏观抽象三级融合结构与三类跨域桥、微观梳理基础/辅助技术，并汇总 Amazon Foods/Kitchens 全库排序结果（HR@10 较 SOCCF 最优提升 42.2%）</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;Test-Time Training&quot;,&quot;端云协同&quot;,&quot;参数校正&quot;]"><strong><a href="../../venues/IJCAI/2025/sequential_rec/device_cloud_collaborative_correction_for_on_device_recommendation/">Device-Cloud Collaborative Correction for On-Device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>端侧 TTT 自校正网络 + 云端全局校正网络，以 RNN 级开销逼近 Transformer 精度，UAUC 最高提升 4.56%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;生成式推荐&quot;,&quot;Flow Matching&quot;,&quot;确定性ODE采样&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/IJCAI/2025/generative_rec/flow_matching_based_sequential_recommender_model/">Flow Matching based Sequential Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用直线流轨迹+无噪预测目标+ODE确定性采样替代扩散式序列推荐，四数据集全库排序平均提升6.53%</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;用户行为采样&quot;,&quot;强化学习策略梯度&quot;,&quot;多目标奖励&quot;]"><strong><a href="../../venues/IJCAI/2025/sequential_rec/towards_automatic_sampling_of_user_behaviors_for_sequential_recommender_sys/">Towards Automatic Sampling of User Behaviors for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 AutoSAM，用轻量采样器+强化学习多目标奖励自适应丢弃低信息历史行为，在四个真实数据集全库排序下 NDCG@10 平均提升约 8%。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;LLM 语义兴趣抽取&quot;,&quot;马尔可夫状态空间&quot;,&quot;生成式-判别式协同&quot;]"><strong><a href="../../venues/IJCAI/2026/sequential_rec/bridging_the_semantic_gap_leveraging_llms_for_hierarchical_interest_evoluti/">Bridging the Semantic Gap: Leveraging LLMs for Hierarchical Interest Evolution in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>HSIEN 用 LLM 做马尔可夫式层级兴趣抽取（局部递推+全局 SMoE），再以离线预计算+在线查表喂给判别式 CTR 模型，Amazon AUC 0.7795、工业 +0.59PP、线上 ROI +3.0PP。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Flow Matching&quot;,&quot;对比正则化&quot;]"><strong><a href="../../venues/IJCAI/2026/sequential_rec/contrastive_flow_matching_for_sequential_recommendation/">Contrastive Flow Matching for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>通过表示级与速度级对比正则化，解决流匹配推荐中的 averaged flow 问题，在 Full-Ranking 协议下显著提升长尾个性化推荐。</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;生成式推荐&quot;,&quot;生成式端到端&quot;,&quot;流匹配 ODE&quot;,&quot;流行度去偏&quot;,&quot;长尾覆盖率&quot;]"><strong><a href="../../venues/IJCAI/2026/generative_rec/pdflow_popularity_debiased_flow_matching_for_sequential_recommendation/">PDFlow: Popularity-Debiased Flow Matching for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在流匹配生成式序列推荐中引入残差流行度向量场与跨流行度对齐损失，逐步减去流行度漂移分量，全库排序下 Beauty HR@20 提升 11.34%</p>
</dd>
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;降尺度卷积&quot;,&quot;属性感知序列建模&quot;]"><strong><a href="../../venues/IJCAI/2026/sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2021&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;对偶学习&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="../../venues/KDD/2021/cross_domain_multi_scenario/dual_attentive_sequential_learning_for_cross_domain_click_through_rate_pred/">Dual Attentive Sequential Learning for Cross-Domain Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用正交对偶映射对齐双域用户表征、用跨域拼接 Query 的双注意力做匹配，三数据集 AUC/HR@10 全面领先，线上 VV +7.07%</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;拉普拉斯特征映射初始化&quot;,&quot;流形学习与残差双损失&quot;]"><strong><a href="../../venues/KDD/2021/collaborative_filtering/initialization_matters_regularizing_manifold_informed_initialization_for_ne/">Initialization Matters: Regularizing Manifold-informed Initialization for Neural Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用带流行度正则的拉普拉斯特征映射（LEPORID）为神经推荐系统提供流形感知初始化，并配残差双损失网络 DLR2，在三个数据集全库排序上显著超越 KNN 与 SOTA 神经模型。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;排序 (Next-Item 全库打分)&quot;,&quot;曝光偏差&quot;,&quot;逆倾向打分 (IPS/ITPS)&quot;,&quot;BERT4Rec / Cloze Task&quot;]"><strong><a href="../../venues/KDD/2022/debias_fairness_industrial/debiasing_the_cloze_task_in_sequential_recommendation_with_bidirectional_tr/">Debiasing the Cloze Task in Sequential Recommendation with Bidirectional Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>证明 IPS 无法去偏序列推荐的 Cloze 任务，提出引入 Choice 变量与时变曝光倾向的 ITPS 无偏估计器，在半合成与真实数据上稳定优于 BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;超图神经网络&quot;,&quot;低秩自注意力&quot;]"><strong><a href="../../venues/KDD/2022/sequential_rec/multi_behavior_hypergraph_enhanced_transformer_for_sequential_recommendatio/">Multi-Behavior Hypergraph-Enhanced Transformer for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>多尺度低秩 Transformer 捕获行为感知短期转移，超图网络建模跨行为长程依赖，Leave-One-Out+1:100 采样排序下 HR@5 最高提升 48.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双塔 Transformer&quot;,&quot;因果掩码序列建模&quot;]"><strong><a href="../../venues/KDD/2022/sequential_rec/nxtpost_user_to_post_recommendations_in_facebook_groups/">NxtPost: User to Post Recommendations in Facebook Groups</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用预训练帖子向量替代词表嵌入的双塔 Transformer 因果序列召回，短/长期双损失在 Facebook Groups 上线，离线 Hits@1 绝对提升 49%、线上 +0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;个性化注意力&quot;,&quot;购物篮推荐&quot;]"><strong><a href="../../venues/KDD/2022/sequential_rec/parsrec_explainable_personalized_attention_fused_recurrent_sequential_recom/">PARSRec: Explainable Personalized Attention-fused Recurrent Sequential Recommendation Using Session Partial Actions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用显式用户嵌入作注意力 Query 融合 RNN，建模当前购物篮部分行为；Nielsen 采样排序 NDCG@5 较 BERT4Rec 提升 27.9%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;通用序列表征学习&quot;,&quot;跨域迁移预训练&quot;]"><strong><a href="../../venues/KDD/2022/sequential_rec/towards_universal_sequence_representation_learning_for_recommender_systems/">Towards Universal Sequence Representation Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>UniSRec 用商品文本 + 参数化白化 + MoE 适配器学习 ID-agnostic 的通用物品/序列表征，多域对比预训练后仅微调少量参数即可跨域跨平台迁移，Pantry Recall@10 提升 38.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;预训练语言模型加速&quot;,&quot;训练框架优化&quot;]"><strong><a href="../../venues/KDD/2022/llm_rec/training_large_scale_news_recommenders_with_pretrained_language_models_in_t/">Training Large-Scale News Recommenders with Pretrained Language Models in the Loop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>SpeedyFeed 用集中式编码、缓存复用、BusLM 与自回归用户建模，把 PLM 新闻推荐训练加速 128.7×，AUC 73.74 全面超越 NRMS。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;一致性训练&quot;,&quot;双向KL散度&quot;]"><strong><a href="../../venues/KDD/2023/sequential_rec/ct4rec_simple_yet_effective_consistency_training_for_sequential_recommendat/">CT4Rec: Simple yet Effective Consistency Training for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>仅加双向KL一致性正则，在四数据集采样排序上超CL4SRec且线上CTR提升2.717%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双层级三元图&quot;,&quot;时段分解与时间门控&quot;,&quot;外卖推荐&quot;]"><strong><a href="../../venues/KDD/2023/sequential_rec/modeling_dual_period_varying_preferences_for_takeaway_recommendation/">Modeling Dual Period-Varying Preferences for Takeaway Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将外卖用户-商家-菜品三元图按时段硬切分为双层级周期子图，配合用户感知与时间感知软门控融合，采样排序下 Hit@10 相对提升 3.6–7.3%，线上 GMV +0.70%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;无 ID 文本表征&quot;,&quot;两阶段微调&quot;,&quot;跨域零样本迁移&quot;]"><strong><a href="../../venues/KDD/2023/sequential_rec/text_is_all_you_need_learning_language_representations_for_sequential_recom/">Text Is All You Need: Learning Language Representations for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Recformer 弃用物品 ID，将商品展平为键值属性“句子”，用 Longformer 联合建模语言理解与序列转移；Amazon 六域 leave-one-out 全库排序 NDCG@10 平均提升 15.83%，零样本提升 39.78%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;实时行为序列建模&quot;,&quot;实时-批式混合排序&quot;]"><strong><a href="../../venues/KDD/2023/sequential_rec/transact_transformer_based_realtime_user_action_model_for_recommendation_at/">TransAct: Transformer-based Realtime User Action Model for Recommendation at Pinterest</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Pinterest 精排用 Transformer 编码最近 100 条实时行为并与 PinnerFormer 批式长期向量混合，离线 HIT@3/repin +9.4%、线上 repin +11%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;跨域终身序列&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/KDD/2024/ctr_feature_interaction/cross_domain_lifelong_sequential_modeling_for_online_click_through_rate_pre/">Cross-Domain LifeLong Sequential Modeling for Online Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>跨域终身序列CTR；CRP对比学习对齐跨域物品表征，LAP三级注意力金字塔；工业A/B CTR +2.93%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;数据侧（训练集重构）&quot;,&quot;数据集再生&quot;,&quot;双层优化&quot;,&quot;序列模式挖掘&quot;]"><strong><a href="../../venues/KDD/2024/sequential_rec/dataset_regeneration_for_sequential_recommendation/">Dataset Regeneration for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出数据中心的 DR4SR/DR4SR+ 框架，用多样性促进的 Transformer 再生器把原始序列再生为高质量转移模式数据集，配合双层优化个性化打分，在 4 个数据集 5 类骨干上 Full-Ranking NDCG@20 最高提升 21.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;序列推荐&quot;,&quot;端侧重排推理&quot;,&quot;端云协同个性化&quot;,&quot;子网掩码剪枝&quot;,&quot;超网络&quot;]"><strong><a href="../../venues/KDD/2024/sequential_rec/diet_customized_slimming_for_incompatible_networks_in_sequential_recommenda/">DIET: Customized Slimming for Incompatible Networks in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>云端按实时序列用 GRU+超网络生成二进制子网 mask 并做 filter 级校正下发边缘，传输参数降至 3.1%、FLOPs 减半，NDCG 最高 +22.8%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2024&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;召回&quot;,&quot;双流生成架构&quot;,&quot;分级 K-Means 语义ID&quot;,&quot;行为-语义协同&quot;]"><strong><a href="../../venues/KDD/2024/generative_rec/eager_two_stream_generative_recommender_with_behavior_semantic_collaboratio/">EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>共享编码器加双解码器并行生成行为码与语义码，配全局对比与语义引导迁移，Beauty Recall@5 相对 TIGER 提升 31.49%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;文本-协同语义融合&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/bridging_textual_collaborative_gap_through_semantic_codes_for_sequential_re/">Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多视图文本量化语义码作桥梁，以码为 Query 交叉注意力融合文本，配合双重掩码自监督，Full-Ranking 下 R@10 最高提升 20.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;端云协同&quot;,&quot;协同训练&quot;,&quot;一致性触发请求&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/collaboration_of_large_language_models_and_small_recommendation_models_for_/">Collaboration of Large Language Models and Small Recommendation Models for Device-Cloud Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出端云协同 LSC4Rec：云端 LLM 召回+设备端 SRM 实时重排，配合协同训练、分数融合与不一致度触发请求，Beauty 平均提升 16.18%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据版权保护&quot;,&quot;序列推荐&quot;,&quot;数据水印&quot;,&quot;感受野&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/data_watermarking_for_sequential_recommender_systems/">Data Watermarking for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首次定义序列推荐数据集/单用户双场景数据水印，用感受野约束插入连续物品序列，5 模型 3 数据集上水印 NDCG 近 1.0 且效用无损。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;离线用户嵌入&quot;,&quot;超长用户序列&quot;,&quot;漏斗摘要编码器&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/dv365_extremely_long_user_history_modeling_at_instagram/">DV365: Extremely Long User History Modeling at Instagram</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以离线用户嵌入替代端到端长序列建模，用多切片+漏斗摘要将 4 万~7 万条历史压缩为 58×256 嵌入，15 个下游模型 NE 平均提升 0.4%+。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;Stein 核分布对齐&quot;,&quot;跨模态专家路由&quot;,&quot;Mamba 序列建模&quot;]"><strong><a href="../../venues/KDD/2025/multimodal_rec/findrec_stein_guided_entropic_flow_for_multi_modal_sequential_recommendatio/">FindRec: Stein-Guided Entropic Flow for Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Stein 核对齐图文与 ID 流分布、MoE 路由动态筛选跨模态信号，Mamba 建模长序列，三数据集 NDCG@5 相对提升 1.4–3.1%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;端侧推荐&quot;,&quot;动态子网络&quot;,&quot;超网络参数生成&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/forward_once_for_all_structural_parameterized_adaptation_for_efficient_clou/">Forward Once for All: Structural Parameterized Adaptation for Efficient Cloud-coordinated On-device Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用结构控制器+结构引导映射器为每台设备动态组装子网络，一次前向即完成端侧适配，兼顾隐私、带宽与算力。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;并行解码&quot;,&quot;图约束解码&quot;]"><strong><a href="../../venues/KDD/2025/generative_rec/generating_long_semantic_ids_in_parallel_for_recommendation/">Generating Long Semantic IDs in Parallel for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 RPG 用多 token 并行预测 + 图约束解码，将语义 ID 长度扩至 64，NDCG@10 平均超生成式基线 12.6% 且推理更省。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;门控循环单元&quot;,&quot;线性注意力&quot;,&quot;高效推理&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/glint_ru_gated_lightweight_intelligent_recurrent_units_for_sequential_recom/">GLINT-RU: Gated Lightweight Intelligent Recurrent Units for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>单层并行专家混合（密集选择GRU+线性注意力）+门控MLP的高效序列推荐，Leave-One-Out全库排序下精度与推理速度双超SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;协同监督微调&quot;,&quot;物品级嵌入建模&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/llm2rec_large_language_models_are_powerful_embedding_models_for_sequential_/">LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>两阶段将 LLM 改造为融合协同信号与语义的物品嵌入模型，Full-Ranking 下域内平均提升约 15%、跨域仍稳定增益。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;序列信息蒸馏&quot;,&quot;表示均匀性正则&quot;]"><strong><a href="../../venues/KDD/2025/llm_rec/lost_in_sequence_do_large_language_models_understand_sequential_recommendat/">Lost in Sequence: Do Large Language Models Understand Sequential Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示现有 LLM4Rec 模型无法捕捉交互序列顺序信息，提出冻结 LLM 与 CF-SRec、仅训练轻量 MLP 的蒸馏框架 LLM-SRec，在 Amazon 四数据集上 NDCG@10 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;C2C 推荐数据集&quot;,&quot;三塔 Transformer 多步召回&quot;,&quot;无 SKU 商品表征&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/merrec_a_large_scale_multipurpose_mercari_dataset_for_consumer_to_consumer_/">MerRec: A Large-scale Multipurpose Mercari Dataset for Consumer-to-Consumer Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个大规模 C2C 推荐数据集 MerRec（5.56M 用户/83M 商品/1.27B 交互），并配套三塔 Mercatran 模型与四任务基准。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;交互级依赖图&quot;,&quot;多粒度注意力&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/multi_grained_preference_enhanced_transformer_for_multi_behavior_sequential/">Multi-Grained Preference Enhanced Transformer for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>M-GPT 用交互级图卷积建模跨类型行为依赖，并以多粒度多尺度注意力抽取行为感知偏好，Leave-One-Out 全库排序下 HR@5 相对提升 12.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;召回&quot;,&quot;Clustering of Neural Bandits&quot;,&quot;Loss of Plasticity&quot;,&quot;Selective Reinitialization&quot;]"><strong><a href="../../venues/KDD/2025/rl_bandit_rec/revisiting_clustering_of_neural_bandits_selective_reinitialization_for_miti/">Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>针对神经聚类 Bandit 的可塑性丧失问题，提出选择性重初始化框架 SeRe，配合自适应变点检测，在六个推荐数据集上最多降低 12.82% 累积遗憾。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;帕累托最优自注意力&quot;,&quot;负迁移抑制&quot;,&quot;多目标优化&quot;]"><strong><a href="../../venues/KDD/2025/cross_domain_multi_scenario/revisiting_self_attention_for_cross_domain_sequential_recommendation/">Revisiting Self-attention for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将跨域序列推荐重构为推荐损失与跨域注意力抑制的双目标帕累托优化，以即插即用方式改造自注意力，平均提升 Recall@10 达 9.8%/16.0%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;流行度去偏&quot;,&quot;因果干预&quot;,&quot;时序演化&quot;]"><strong><a href="../../venues/KDD/2025/debias_fairness_industrial/taming_recommendation_bias_with_causal_intervention_on_evolving_personal_po/">Taming Recommendation Bias with Causal Intervention on Evolving Personal Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出演化个人流行度度量与因果去混杂训练，并在推理期用移动平均趋势外推做干预，兼顾用户-物品演化一致性，Full-Ranking 下 Recall@20 最高提升 20.4%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图神经ODE&quot;,&quot;时间引导扩散&quot;,&quot;物品演化图&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/time_matters_enhancing_sequential_recommendations_with_time_guided_graph_ne/">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用时间引导扩散补全稀疏用户时间图、图神经ODE对齐用户兴趣与物品演化，五数据集Recall@20最高提升46%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;嵌入坍缩&quot;,&quot;逐Token扩散&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/unlocking_the_power_of_diffusion_models_in_sequential_recommendation_a_simp/">Unlocking the Power of Diffusion Models in Sequential Recommendation: A Simple and Effective Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示扩散序列推荐中的嵌入坍缩问题，提出逐 token 独立扩散+三阶段训练，六数据集 HR@20 平均提升 15.45%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;多轮交互推荐&quot;,&quot;标签传播&quot;,&quot;查询-物品联合建模&quot;]"><strong><a href="../../venues/KDD/2025/sequential_rec/why_not_together_a_multiple_round_recommender_system_for_queries_and_items/">Why Not Together? A Multiple-Round Recommender System for Queries and Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 MAGUS 多轮查询-物品联合推荐框架，以词组合关系图桥接查询与物品，用标签传播在线吸收反馈，可插拔接入 12 种基座模型。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;端到端序列推荐&quot;,&quot;扩散模型&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/align_for_fusion_harmonizing_triple_preferences_via_dual_oriented_diffusion/">Align-for-Fusion: Harmonizing Triple Preferences via Dual-oriented Diffusion for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出align-for-fusion：检索混合域行为噪声并对源/目标双域扩散，四设置Last-item排序HR@10最高+8.22%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;生成式推荐&quot;,&quot;训练-推理对齐&quot;,&quot;Beam Search&quot;,&quot;前缀级排序优化&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/bending_the_scaling_law_curve_in_large_scale_recommendation_systems/">Bending the Scaling Law Curve in Large-Scale Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>ULTRA-HSTU以物品-动作合并、半局部稀疏注意力与注意力截断实现5×训练/21×推理扩展效率，工业A/B消费指标提升4.11%–8.2%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐与离散语义 ID&quot;,&quot;召回&quot;,&quot;时间感知多目标预测&quot;,&quot;条件引导稀疏注意力&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/climber_pilot_a_non_myopic_generative_recommendation_model_towards_better_i/">Climber-Pilot: A Non-Myopic Generative Recommendation Model Towards Better Instruction-Following</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以时间感知多目标预测 TAMIP 缓解单步推理短视，用条件引导稀疏注意力 CGSA 实现指令遵循召回，网易云线上 Like Rate +4.24%、延迟仅 +0.32ms</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;统一序列建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="../../venues/KDD/2026/ctr_feature_interaction/est_towards_efficient_scaling_laws_in_click_through_rate_prediction_via_uni/">EST: Towards Efficient Scaling Laws in Click-Through Rate Prediction via Unified Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LCA+CSA 把特征与行为装进单序列实现全统一建模，剥离冗余交互，在淘宝展示广告取得 0.99% AUC 与 3.27% RPM 提升</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;隐扩散模型&quot;,&quot;多行为序列&quot;,&quot;旋转位置编码&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/from_agnostic_to_specific_latent_preference_diffusion_for_multi_behavior_se/">From Agnostic to Specific: Latent Preference Diffusion for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 FatsMB：以 BaRoPE 增强的多行为自编码器构建统一偏好隐空间，再用隐扩散模型把「行为无关」偏好定向迁移到「行为特定」偏好并解码生成，全库排序下 Recall@10 最高提升 20.56%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;指数幂时间编码&quot;,&quot;对角稀疏位置注意力&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/fuxi_efficient_sequential_recommendation_with_exponential_power_temporal_en/">FuXi-$γ$: Efficient Sequential Recommendation with Exponential-Power Temporal Encoder and Diagonal-Sparse Positional Mechanism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用艾宾浩斯遗忘曲线启发的指数幂时间编码器替代分桶查表，配合 Toeplitz 对角稀疏位置剪枝，在 Full-Ranking 下精度 SOTA 且训练/推理加速最高 4.74×/6.18×。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID量化&quot;,&quot;分层位置编码&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;隐空间推理&quot;,&quot;直觉先验&quot;,&quot;双注意力编码器&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/intuition_guided_latent_reasoning_for_llm_based_recommendation/">Intuition-Guided Latent Reasoning for LLM-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 top-K 候选集构建“推荐直觉”嵌入锚定 LLM 隐空间推理起点，Amazon 三数据集 Recall@5 平均提升约 6%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;指令微调数据集&quot;,&quot;LoRA 参数高效微调&quot;]"><strong><a href="../../venues/KDD/2026/llm_rec/itdr_an_instruction_tuning_dataset_for_enhancing_large_language_models_in_r/">ITDR: An Instruction Tuning Dataset for Enhancing Large Language Models in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>构建覆盖 2 根任务/7 子任务、13 个公开数据集的 19.5 万条指令数据集 ITDR，使 3B–9B 开源 LLM 的推荐综合指标反超闭源大模型。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;访客旅程建模&quot;,&quot;工业搜索排序&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;语义捷径&quot;,&quot;交互去噪&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/lisrec_modeling_user_preferences_with_learned_item_shortcuts_for_sequential/">LISRec: Modeling User Preferences with Learned Item Shortcuts for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>任务无关语义编码构建个性化物品图，取最大连通分量作语义捷径去噪；Amazon/Yelp 全库排序较 TASTE 最高 +14.5% R@20。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Scaling Laws&quot;,&quot;Multi-head Latent Attention&quot;,&quot;多阶段架构&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/llatte_scaling_laws_for_multi_stage_sequence_modeling_in_large_scale_ads_re/">LLaTTE: Scaling Laws for Multi-Stage Sequence Modeling in Large-Scale Ads Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模块化 LLaTTE 与异步两阶段架构，揭示语义特征可弯曲缩放曲线，线上 NE 降 0.25%、转化提升 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端序列预测&quot;,&quot;Magnetic Laplacian&quot;,&quot;Complex-valued Mamba&quot;]"><strong><a href="../../venues/KDD/2026/poi_spatiotemporal_rec/mag_mamba_modeling_coupled_spatiotemporal_asymmetry_for_poi_recommendation/">Mag-Mamba: Modeling Coupled spatiotemporal Asymmetry for POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>时变方向场编码为磁相位并注入复值Mamba衰减-旋转递推，NYC/TKY/CA的NDCG@1最高提升14.71%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;生成式推荐&quot;,&quot;模型合并&quot;,&quot;任务向量&quot;]"><strong><a href="../../venues/KDD/2026/cross_domain_multi_scenario/mergerec_model_merging_for_data_isolated_cross_domain_sequential_recommenda/">MergeRec: Model Merging for Data-Isolated Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据隔离跨域序列推荐新设定，用任务向量模型合并+伪用户数据+推荐/蒸馏联合目标，Recall@10 平均提升 17.21%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;统一参数化&quot;,&quot;用户-物品解耦&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/mixformer_co_scaling_up_dense_and_sequence_in_industrial_recommenders/">MixFormer: Co-Scaling Up Dense and Sequence in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一参数空间统一序列建模与稠密特征交叉，配合用户-物品解耦实现请求级复用，AUC 提升 1.28% 且 FLOPs 降 36%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Prior-Fitted Network&quot;,&quot;免更新推理&quot;,&quot;合成先验&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/one_sequential_recommendation_model_pretrained_from_synthetic_priors_predic/">One Sequential Recommendation Model Pretrained from Synthetic Priors Predicts Multiple Datasets</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 hDCSBM 合成先验预训练 PFN，以支持集条件化实现免更新推理的序列推荐，五基准平均超最优基线 7.53%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;逐步潜在推理&quot;,&quot;语义ID码本优化&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/s_2_gr_stepwise_semantic_guided_reasoning_in_latent_space_for_generative_re/">S$^2$GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在每级 SID 码生成前插入受码本聚类语义对比监督的思考 token，实现逐步可验证潜在推理，工业短视频 A/B 显著提升时长。</p>
</dd>
<dt data-paper-tags="[&quot;KDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;生成式推荐&quot;,&quot;大模型缩放定律&quot;,&quot;双塔离线推理&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/scaling_recommender_transformers_to_one_billion_parameters/">Scaling Recommender Transformers to One Billion Parameters</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>ARGUS 将自回归预训练解耦为反馈预测与下一物品预测双任务，把推荐 Transformer 编码器扩到 10 亿参数，线上听歌时长 +2.26%、点赞率 +6.37%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;谱坍缩&quot;,&quot;流行度偏差&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/KDD/2026/debias_fairness_industrial/the_pitfall_of_scaling_up_uncovering_and_mitigating_popularity_bias_amplifi/">The Pitfall of Scaling Up: Uncovering and Mitigating Popularity Bias Amplification in Scaling Transformer-based Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐随深度加深发生谱坍缩并放大流行度偏差，提出逐层注意力列和与 FFN 谱范数正则 SPRINT，在 6 数据集上同时提升准确率与长尾公平性。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;双流交叉注意力&quot;,&quot;模型-基础设施协同设计&quot;,&quot;非自回归并行解码&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/transx_scaling_transformer_based_recommendation_via_behavioral_and_serving_/">TransX: Scaling Transformer-based Recommendation via Behavioral and Serving Stream Crossings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>把推荐重构为“行为流×服务流”交叉的序列到序列动作转导，配合近线摊销编码与KV缓存，离线AUC 0.862、线上CTR +6.0%而在线算力仅基线约20%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;模型扩展律&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/understanding_generative_recommendation_with_semantic_ids_from_a_model_scal/">Understanding Generative Recommendation with Semantic IDs from a Model-scaling View</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 SID 式生成推荐在编码器/量化器/RS 三轴扩展均快速饱和，归因于 SID 语义容量瓶颈，并证明 Text-based GR 具备更优扩展性且 CF 能力随规模提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散生成&quot;,&quot;潜在邻居生成&quot;,&quot;多模态对比学习&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/unleashing_the_potential_of_neighbors_diffusion_based_latent_neighbor_gener/">Unleashing the Potential of Neighbors: Diffusion-based Latent Neighbor Generation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用检索先验与多模态信号双路引导扩散模型生成“潜在邻居”，突破可观测邻居边界，Full-Ranking 下 P@20 最高提升 18.45%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序离线评估&quot;,&quot;采样评估&quot;,&quot;Kendall Tau&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/a_case_study_on_sampling_strategies_for_evaluating_neural_sequential_item_r/">A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>复训4个神经序列推荐模型于5数据集，证明Uniform与Popular采样排序均可能偏离全库排序，η=100无通用安全性，应优先Full-Ranking。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;模型窃取&quot;,&quot;数据投毒&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/RecSys/2021/debias_fairness_industrial/black_box_attacks_on_sequential_recommenders_via_data_free_model_extraction/">Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>无训练数据下用自回归合成查询 + 排序蒸馏窃取序列推荐模型，并迁移为画像污染与数据投毒攻击，ML-1M 上 Agr@10 达 0.747。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;正交投影去噪&quot;,&quot;用户记忆网络&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/denoising_user_aware_memory_network_for_recommendation/">Denoising User-aware Memory Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以显式反馈正交投影净化隐式反馈噪声，并用四路用户级记忆网络建模长期兴趣，AUC 较最优基线提升 1.18/0.97 个点。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时间集合预测&quot;,&quot;Tweedie 分布 VAE&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/global_local_item_embedding_for_temporal_set_prediction/">Global-Local Item Embedding for Temporal Set Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>GLOIE 用时间衰减集合和的 VAE（Tweedie 输出）刻画全局协同信号，与动态图学到的局部序列嵌入做注意力融合，在三个下一篮基准的全库排序上稳定领先 DNNTSP。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2021&quot;,&quot;序列与会话推荐&quot;,&quot;端到端排序（next-click 预测）&quot;,&quot;分层 Transformer&quot;,&quot;神经指针网络&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/large_scale_modeling_of_mobile_user_click_behaviors_using_deep_learning/">Large-Scale Modeling of Mobile User Click Behaviors Using Deep Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>分层 Transformer + 神经指针在真实屏幕候选元素上做 next-click 全候选排序，用户级划分下 Top-1 48.3%、Top-3 71.4%，较最强深度基线提升 25.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回/会话级排序&quot;,&quot;双曲表征&quot;,&quot;目标估计&quot;,&quot;弱监督&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/learning_to_represent_human_motives_for_goal_directed_web_browsing/">Learning to Represent Human Motives for Goal-directed Web Browsing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 GoWeB，用心理学目标分类体系在双曲空间学习目标表征并估计页面访问目标，统一增强会话推荐、重访预测与目标聚类。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;小样本元学习&quot;,&quot;跨会话相似检索&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/next_item_recommendations_in_short_sessions/">Next-item Recommendations in Short Sessions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将短会话下一项预测建模为小样本学习，用 SSRN 检索跨会话相似历史并做特征级调制，在 Delicious/Reddit 短会话上 Recall@5 最高提升 17.2%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;排序（历史候选重排）&quot;,&quot;认知架构 ACT-R&quot;,&quot;重听行为建模&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/predicting_music_relistening_behavior_using_the_act_r_framework/">Predicting Music Relistening Behavior Using the ACT-R Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 ACT-R 声明记忆五组件建模音乐重听，一周滑窗逐 LE 预测会话剩余曲目，R-prec 0.111，较最强单组件 +6.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;自适应序列长度&quot;,&quot;Actor-Critic 强化学习&quot;,&quot;联合损失正则&quot;]"><strong><a href="../../venues/RecSys/2021/sequential_rec/sequence_adaptation_via_reinforcement_learning_in_recommender_systems/">Sequence Adaptation via Reinforcement Learning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 actor-critic 把“回看序列长度 l”建模为 MDP 动作，按用户个性化裁剪子序列并以 Q 值乘性正则交叉熵，Steam/ML-10M 上 NDCG@10 相对最优基线提升 6.5%/12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;可复现性研究&quot;,&quot;BERT4Rec&quot;,&quot;掩码自监督训练&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/a_systematic_review_and_replicability_study_of_bert4rec_for_sequential_reco/">A Systematic Review and Replicability Study of BERT4Rec for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>系统复盘 134 次 BERT4Rec vs SASRec 对比，指出原始实现默认配置严重欠拟合（需 30× 训练时长），并给出 Hugging Face 高效复现与 ALBERT4Rec 改进。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;重排/序列生成&quot;,&quot;音乐 Tour 生成&quot;,&quot;用户中心评估&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/a_user_centered_investigation_of_personal_music_tours/">A User-Centered Investigation of Personal Music Tours</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首次用半结构化访谈评估 Greedy 与 Optimal 两种音乐 Tour 生成算法，发现 Greedy 因 segue 多样性更受偏好，揭示多样性、编排与熟悉度为关键属性。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排/排序&quot;,&quot;会话内自适应&quot;,&quot;实时搜索推荐&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/augmenting_netflix_search_with_in_session_adapted_recommendations/">Augmenting Netflix Search with In-Session Adapted Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>面向Netflix搜索预查询画布，融合当前会话实时行为与长期偏好，离线排序指标相对生产模型提升6%，并讨论实时不缓存等工程约束。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平可信隐私与工业架构&quot;,&quot;召回/输入序列层&quot;,&quot;对抗攻击与鲁棒性&quot;,&quot;Dirichlet邻域采样&quot;]"><strong><a href="../../venues/RecSys/2022/debias_fairness_industrial/defending_substitution_based_profile_pollution_attacks_on_sequential_recomm/">Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首个替换式画像污染攻击+防御框架，用梯度引导攻击、Dirichlet邻域采样与对抗训练，NDCG@10波动降34.4%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可微二值掩码&quot;,&quot;Jacobian 正则&quot;,&quot;ARM 无偏梯度估计&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/denoising_self_attentive_sequential_recommendation/">Denoising Self-attentive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用可微二值掩码剪枝自注意力的噪声项并叠加 Jacobian 谱正则，Leave-One-Out + 101 候选采样排序下比各 Transformer 骨干相对提升 5.05%–19.55%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/曝光策略&quot;,&quot;mere exposure effect&quot;,&quot;probit 回归&quot;,&quot;重复消费建模&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/discovery_dynamics_leveraging_repeated_exposure_for_user_and_music_characte/">Discovery Dynamics: Leveraging Repeated Exposure for User and Music Characterization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 9M 条曝光日志拟合 probit 二次模型，验证新曲曝光—收听呈倒 U 型（峰值约 12 次），并以峰值位置刻画曲目熟悉度与用户年龄差异。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;训练目标设计&quot;,&quot;列表损失 λRank&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/effective_and_efficient_training_for_sequential_recommendation_using_recenc/">Effective and Efficient Training for Sequential Recommendation using Recency Sampling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出近期性序列采样 RSS 训练目标，替代序列延续与掩码任务，配合 λRank 列表损失，1 小时训练即可超越 16 小时 BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话行为建模&quot;,&quot;物品稀缺域推荐&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/learning_recommendations_from_user_actions_in_the_item_poor_insurance_domai/">Learning Recommendations from User Actions in the Item-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>首个保险领域跨会话推荐模型：以 3 元组网站行为序列为输入、GRU 编码多次会话，预测会话之外发生的购买行为，Full-Ranking HR@3 较 SKNN_EB 提升 3.04%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;重复消费建模&quot;,&quot;指针网络&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/modeling_user_repeat_consumption_behavior_for_online_novel_recommendation/">Modeling User Repeat Consumption Behavior for Online Novel Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>NovelNet 用 7 维细粒度交互属性编码每次阅读、BiGRU 建模序列、指针网络+pointwise 损失统一推荐已读与新小说，MRR@1 达 47.02%（+4.65%）。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;规范性多样性&quot;,&quot;秩感知散度&quot;]"><strong><a href="../../venues/RecSys/2022/debias_fairness_industrial/radio_rank_aware_divergence_metrics_to_measure_normative_diversity_in_news_/">RADio: Rank-Aware Divergence Metrics to Measure Normative Diversity in News Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将新闻规范性多样性统一为秩感知 f-散度（JS）框架，在 MIND 上对 6 种推荐器做五维 DART 度量与敏感性分析。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Transformer&quot;,&quot;长短期兴趣建模&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/rethinking_personalized_ranking_at_pinterest_an_end_to_end_approach/">Rethinking Personalized Ranking at Pinterest: An End-to-End Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>PinnerFormer 用稠密全动作损失编码长期兴趣，实时序列捕捉短期意图，线上 Repin +12.5%、Ads CTR +14.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态偏好建模&quot;,&quot;反馈回路仿真&quot;]"><strong><a href="../../venues/RecSys/2022/sequential_rec/towards_psychologically_grounded_dynamic_preference_models/">Towards Psychologically-Grounded Dynamic Preference Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出心理学接地方法论，将单纯曝光、操作性条件作用、享乐适应形式化为偏好向量增量算子，仿真揭示振荡动态与参与度/多样性指标失效</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;兴趣胶囊动态路由&quot;,&quot;辅助任务联合学习&quot;,&quot;物品表征增广&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/a_model_agnostic_framework_for_recommendation_via_interest_aware_item_embed/">A Model-Agnostic Framework for Recommendation via Interest-aware Item Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将多兴趣胶囊网络退化为辅助任务并共享物品嵌入，用兴趣梯度增广 DIN/DIEN 精排，Amazon 双集 AUC 稳定提升</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;情境特征建模&quot;,&quot;Gumbel-Softmax去噪&quot;,&quot;MLP-Mixer&quot;]"><strong><a href="../../venues/RecSys/2023/ctr_feature_interaction/deep_situation_aware_interaction_network_for_click_through_rate_prediction/">Deep Situation-Aware Interaction Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出“情境(situation)”概念，用 Gumbel-Softmax 去噪 + 三向 MLP-Mixer 融合行为类型/时空情境特征，美团外卖线上 CTR +2.70%、GMV +2.16%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;等变对比学习&quot;,&quot;替换项检测&quot;,&quot;数据增广&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/equivariant_contrastive_learning_for_sequential_recommendation/">Equivariant Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用生成器-条件判别器实现替换项检测损失，使序列编码器对侵入式增广等变、对温和增广不变；全库排序下 ML-1m/Yelp Recall@10 较 SASRec 提升 19.5%/10.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;单纯曝光效应&quot;,&quot;重复消费建模&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/ex2vec_characterizing_users_and_items_from_the_mere_exposure_effect/">Ex2Vec: Characterizing Users and Items from the Mere Exposure Effect</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ACT-R 基级激活调制用户-物品相对距离，再经倒 U 型兴趣函数打分；Deezer 新歌重复收听预测 Balanced Acc 64.27%，较最优基线相对提升 7.3%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;物品相关性加权损失&quot;,&quot;多未来物品评估协议&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/integrating_item_relevance_in_training_loss_for_sequential_recommender_syst/">Integrating Item Relevance in Training Loss for Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用多未来物品评估协议与相关性加权损失替代单目标假设，SASRec 在 4 个数据集上 NDCG@10 提升约 1.2%–1.63%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负反馈建模&quot;,&quot;反事实模拟&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/learning_from_negative_user_feedback_and_measuring_responsiveness_for_seque/">Learning from Negative User Feedback and Measuring Responsiveness for Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 not-to-recommend 损失将显式不喜欢与隐式跳过纳入序列召回训练，并用反事实模拟量化响应性，线上不喜欢率降 2.44%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;LLM item embedding&quot;,&quot;PCA initialization&quot;,&quot;BERT4Rec&quot;,&quot;prompt fine-tuning&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/leveraging_large_language_models_for_sequential_recommendation/">Leveraging Large Language Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>三种LLM增强序列推荐：LLM物品嵌入+PCA初始化BERT4Rec在Beauty/Delivery Hero上NDCG@20提升15–20%，简单语义相似检索在Beauty具竞争力。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为序列推荐&quot;,&quot;跨关系对比学习&quot;,&quot;动态记忆网络&quot;]"><strong><a href="../../venues/RecSys/2023/self_supervised_denoising/multi_relational_contrastive_learning_for_recommendation/">Multi-Relational Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>多行为序列推荐框架 RCL，用关系感知图编码器 + 跨关系记忆网络 + 行为级 InfoNCE，全库 99 负采样下 HR@10 最高提升 24.4%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Buy-It-Again 复购推荐&quot;,&quot;生存分析&quot;,&quot;ARIMA 时序特征&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/personalized_category_frequency_prediction_for_buy_it_again_recommendations/">Personalized Category Frequency prediction for Buy It Again recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出层级 PCIC 模型，用生存分析+ARIMA 特征预测用户复购品类，再在品类内按频率/新近度排名，NDCG@10 最高提升约 15%，线上单位购买量 +27.5%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双边序列匹配&quot;,&quot;自蒸馏&quot;,&quot;时间敏感共注意力&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/reciprocal_sequential_recommendation/">Reciprocal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将双向推荐重构为双视角行为序列匹配任务，用时间敏感共注意力做微级匹配并以自蒸馏压回点积宏级匹配，采样排序下 HR@5 最高相对提升 83%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;用户生命周期建模&quot;,&quot;动态损失加权&quot;]"><strong><a href="../../venues/RecSys/2023/multi_task_reranking/stan_stage_adaptive_network_for_multi_task_recommendation_by_learning_user_/">STAN: Stage-Adaptive Network for Multi-Task Recommendation by Learning User Lifecycle-Based Representation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 Beta 分布采样刻画用户生命周期阶段向量，动态加权各任务损失，工业数据集 AUC 最高 +18.5%，线上停留时长 +3.05%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;可复现性审计&quot;,&quot;GRU4Rec&quot;,&quot;离线评测协议&quot;]"><strong><a href="../../venues/RecSys/2023/debias_fairness_industrial/the_effect_of_third_party_implementations_on_reproducibility/">The Effect of Third Party Implementations on Reproducibility</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>审计 GRU4Rec 六个第三方复现：特征缺失与 bug 使 recall@20 相对官方下降 13%–99%，训练慢 2–336 倍，呼吁复现须对齐官方实现。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Next Basket Recommendation&quot;,&quot;Hawkes Process&quot;,&quot;时间感知权重&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/time_aware_item_weighting_for_the_next_basket_recommendations/">Time-Aware Item Weighting for the Next Basket Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用物品级 Hawkes 强度函数刻画复购周期并叠加 kNN 邻域模块，在 Leave-One-Basket 全库排序下相对 SOTA 提升 P/R 约 3%–9%</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Bayesian Surprise&quot;,&quot;Serendipity&quot;,&quot;LDA Topic Model&quot;]"><strong><a href="../../venues/RecSys/2023/collaborative_filtering/topic_level_bayesian_surprise_and_serendipity_for_recommender_systems/">Topic-Level Bayesian Surprise and Serendipity for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以 LDA 主题分布上的贝叶斯惊喜（KL 后验-先验散度）结合评分构成事后惊喜度，再用协作过滤找相似用户做惊喜推荐；Goodreads 4 用户 449 本书人工标注，AROW 惊喜检测 F1 56.5%、AROW+vbBLR 推荐 F1 37.0%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Decoder-only Transformer&quot;,&quot;播放列表生成&quot;,&quot;在线动态量化&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/track_mix_generation_on_music_streaming_services_using_transformers/">Track Mix Generation on Music Streaming Services using Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>Deezer 用单隐层 Decoder-only Transformer 做“以歌生单”的混音歌单生成，借 ONNX 融合与动态量化落地，线上 A/B 听歌时长 +6.84%、新用户收藏 +20.26%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样交叉熵&quot;,&quot;Full-Ranking 评测&quot;]"><strong><a href="../../venues/RecSys/2023/sequential_rec/turning_dross_into_gold_loss_is_bert4rec_really_better_than_sasrec/">Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示 SASRec 与 BERT4Rec 的差距实为损失函数差异：统一为全库 softmax 交叉熵后 SASRec 更快更强，ML-1M 的 NDCG@10 提升 20.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;评测基准&quot;,&quot;离线评测协议&quot;,&quot;时间泄漏&quot;,&quot;负采样偏差&quot;]"><strong><a href="../../venues/RecSys/2023/debias_fairness_industrial/widespread_flaws_in_offline_evaluation_of_recommender_systems/">Widespread Flaws in Offline Evaluation of Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统梳理离线评测四大缺陷（数据-任务错配、过度预处理、时间泄漏、负采样），用 GRU4Rec 实证证明其可致模型排序反转与数十个百分点虚高。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式召回&quot;,&quot;生成式 LLM 微调&quot;,&quot;对比对齐&quot;,&quot;BM25 文本检索&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/calrec_contrastive_alignment_of_generative_llms_for_sequential_recommendati/">CALRec: Contrastive Alignment of Generative LLMs for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>两阶段微调生成式 LLM，双塔对比对齐 + 文本生成损失，Amazon 5-core 全库排序 Recall@1 提升 37.3%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;知识蒸馏&quot;,&quot;序列推荐&quot;,&quot;排序蒸馏&quot;]"><strong><a href="../../venues/RecSys/2024/llm_rec/distillation_matters_empowering_sequential_recommenders_to_match_the_perfor/">Distillation Matters: Empowering Sequential Recommenders to Match the Performance of Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将 BIGRec 教师蒸馏至轻量序列模型：置信/一致性加权排序蒸馏 + projector-offset 嵌入蒸馏，平均提升 47.97%，推理降至秒级。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列推荐&quot;,&quot;召回/端到端排序&quot;,&quot;神经架构搜索 (NAS)&quot;,&quot;线性注意力&quot;,&quot;模型压缩与推理加速&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/dns_rec_data_aware_neural_architecture_search_for_recommender_systems/">DNS-Rec: Data-aware Neural Architecture Search for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用零掩码超网+双层控制器+数据感知门为 SASRec 类序列推荐自动搜索宽度/深度，在 FLOPs 约束下保持精度并将推理时延最高降 87.6%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/全库打分&quot;,&quot;乘积量化&quot;,&quot;子ID推理&quot;,&quot;PQTopK&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/efficient_inference_of_sub_item_id_based_sequential_recommendation_models_w/">Efficient Inference of Sub-Item Id-based Sequential Recommendation Models with Millions of Items</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用PQTopK重排子ID打分循环，将RecJPQ增强SASRec在百万物品Gowalla的CPU推理加速4.5×，并验证千万级目录可行。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;个性化流行度&quot;,&quot;重复消费&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/enhancing_sequential_music_recommendation_with_personalized_popularity_awar/">Enhancing Sequential Music Recommendation with Personalized Popularity Awareness</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户级流行度先验映射为 logit 偏移注入 Transformer 序列模型，NDCG@100 相对提升 25.2%–69.8%。</p>
</dd>
<dt data-paper-tags="[&quot;Knowledge-Based Systems (KBS)&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多行为序列建模&quot;,&quot;多面向解耦表征&quot;,&quot;门控融合&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/modeling_multi_aspect_preferences_and_intents_for_multi_behavioral_sequenti/">Modeling Multi-aspect Preferences and Intents for Multi-behavioral Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>多面向投影+行为增强LSTM+精炼注意力，联合建模偏好/意图并按面向门控融合，采样排序下较最强基线显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;多任务学习与列表级重排&quot;,&quot;召回&quot;,&quot;Pareto 前沿近似&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/RecSys/2024/multi_task_reranking/pareto_front_approximation_for_multi_objective_session_based_recommender_sy/">Pareto Front Approximation for Multi-Objective Session-Based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将偏好向量条件化注入 TRON 会话 Transformer，配合 KL 非均匀性正则，单模型覆盖 CTR–CVR 全 Pareto 前沿并经线上 A/B 验证。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;注意力权重细化&quot;,&quot;高阶依赖建模&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/pay_attention_to_attention_for_sequential_recommendation/">Pay Attention to Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 AWRSR，把自注意力权重矩阵本身当作表征再做一次注意力细化，以极低开销捕获高阶 item 转移；在 Beauty/ML-1M/Anime 上对 SASRec 与 STOSA 普遍带来 2%–14% 相对提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;Prompt Tuning&quot;,&quot;对比学习去偏&quot;]"><strong><a href="../../venues/RecSys/2024/cold_start_rec/prompt_tuning_for_item_cold_start_recommendation/">Prompt Tuning for Item Cold-start Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以高价值正反馈（pinnacle feedback）替代内容描述作为软提示，配合物品级个性化提示网络与双提示增强损失，在 4 个数据集 leave-one-out 采样排序上显著超越 SOTA，并落地亿级短视频平台 A/B 实验。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;可扩展交叉熵&quot;,&quot;MIPS 难负样本&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/scalable_cross_entropy_loss_for_sequential_recommendations_with_large_item_/">Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以随机桶中心做 top-k 分桶，把全库 softmax 近似为桶内小 logits 矩阵，并用 Mix 投影缓解桶塌缩；全库排序下显存最多降 100 倍而指标不降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;Scaling Law&quot;,&quot;Decoder-only Transformer&quot;,&quot;数据受限缩放&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/scaling_law_of_large_sequential_recommendation_models/">Scaling Law of Large Sequential Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>纯 ID 序列推荐上验证 98.3K→0.8B 参数的幂律缩放，并提出分层自适应 Dropout 与 Adam→SGD 切换稳定训练，Leave-One-Out 全库排序下 NDCG@10 相对提升 81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;预训练/召回表征&quot;,&quot;滑窗数据增广&quot;,&quot;推荐基础模型&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/sliding_window_training_utilizing_historical_recommender_systems_data_for_f/">Sliding Window Training -- Utilizing Historical Recommender Systems Data for Foundation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出滑窗式训练采样器，在不增大 FM 输入窗口的前提下把用户多年历史灌入预训练；混合最近窗+滑窗后 MRR 提升 8.8%、item 侧 Recall 提升 14.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;PLM 参数冗余&quot;,&quot;行为微调初始化&quot;,&quot;层间注意力分层&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/the_elephant_in_the_room_rethinking_the_usage_of_pre_trained_language_model/">The Elephant in the Room: Rethinking the Usage of Pre-trained Language Model in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>揭示 PLM 行为序列建模的参数冗余，改用行为微调 PLM 仅做物品嵌入初始化，配合轻量 ID 序列模型大幅提升且零额外推理开销</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;重听行为建模&quot;,&quot;ACT-R 认知架构&quot;]"><strong><a href="../../venues/RecSys/2024/sequential_rec/transformers_meet_act_r_repeat_aware_and_sequential_listening_session_recom/">Transformers Meet ACT-R: Repeat-Aware and Sequential Listening Session Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 ACT-R 认知激活权重改造 Transformer 会话序列建模，兼顾重听与探索，Last.fm/Deezer 全库 Top-10 排序 NDCG 提升 32.5%/9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;协同过滤与双塔召回&quot;,&quot;精排&quot;,&quot;子 ID 分解&quot;,&quot;个性化流行度&quot;,&quot;准确率-新颖度权衡&quot;]"><strong><a href="../../venues/RecSys/2025/collaborative_filtering/balancing_accuracy_and_novelty_with_sub_item_popularity/">Balancing Accuracy and Novelty with Sub-Item Popularity</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复用 RecJPQ 子 ID 分解在子码本粒度计算个性化流行度 sPPS，与 item 级 PPS 加权融合，在 NDCG 不降前提下显著提升个性化新颖度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;ACT-R 认知记忆建模&quot;,&quot;音频表征&quot;,&quot;序列与会话推荐&quot;]"><strong><a href="../../venues/RecSys/2025/multimodal_rec/beyond_the_past_leveraging_audio_and_human_memory_for_sequential_music_reco/">Beyond the past: Leveraging Audio and Human Memory for Sequential Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用音频编码器预测新歌的 ACT-R 类激活值，将记忆式重复建模扩展到未听歌曲，NDCG_Exp 相对 PISA 提升约 40%，RepBias 降至 0.09%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;召回&quot;,&quot;通用用户表征&quot;,&quot;DCN-v2&quot;,&quot;异构序列建模&quot;]"><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/blending_sequential_embeddings_graphs_and_engineered_features_4th_place_sol/">Blending Sequential Embeddings, Graphs, and Engineered Features: 4th Place Solution in RecSys Challenge 2025</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>异构因果Transformer序列编码+TwHIN图+300维手工特征经DCN-v2融合，构建跨6任务通用用户画像，获挑战赛第4名。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Transformer 序列建模&quot;,&quot;模块化基准评测&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/esasrec_enhancing_transformer_based_recommendations_in_a_modular_fashion/">eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统消融 Transformer 序列推荐各模块增强，提出 SASRec+LiGR+SS 组合 eSASRec，在时间切分与覆盖率指标下与 HSTU 同处帕累托前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;Scaling Law&quot;,&quot;在线知识蒸馏&quot;,&quot;行为序列建模&quot;]"><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/exploring_scaling_laws_of_ctr_model_for_online_performance_improvement/">Exploring Scaling Laws of CTR Model for Online Performance Improvement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 SUAN 统一注意力块实现 CTR 模型三级量级缩放律，再经稀疏注意力+并行推理+在线蒸馏落地线上，CTR +2.81%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;对话性能预测&quot;,&quot;目录失败&quot;,&quot;自编码器&quot;]"><strong><a href="../../venues/RecSys/2025/agent_conversational_rec/failure_prediction_in_conversational_recommendation_systems/">Failure Prediction in Conversational Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出监督式对话性能预测任务，用AE与L1收缩多轮预测器区分系统失败与目录失败，揭示目录缺失场景下预测精度显著下降。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;Chain-of-Thought Tokenization&quot;,&quot;Journey-Aware Sparse Attention&quot;,&quot;多行为序列推荐&quot;]"><strong><a href="../../venues/RecSys/2025/generative_rec/grace_generative_recommendation_via_journey_aware_sparse_attention_on_chain/">GRACE: Generative Recommendation via Journey-Aware Sparse Attention on Chain-of-Thought Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 PKG 属性链式 CoT token 增强语义 ID，并以旅程感知四路稀疏注意力替代全注意力，Home 域 HR@10 相对提升 106.9%，长序列注意力计算降 48%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;离线在线一致性&quot;,&quot;帕累托前沿近似&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/RecSys/2025/ctr_feature_interaction/identifying_offline_metrics_that_predict_online_impact_a_pragmatic_strategy/">Identifying Offline Metrics that Predict Online Impact: A Pragmatic Strategy for Real-World Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用帕累托前沿近似单模型多偏好分流，在 OTTO 线上实验验证 Recall@20→CTR、OD@20→CVR、乘积→销量 的离线-在线对齐关系。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;内容初始化&quot;,&quot;有界 delta 校正&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/let_it_go_not_quite_addressing_item_cold_start_in_sequential_recommendation/">Let It Go? Not Quite: Addressing Item Cold Start in Sequential Recommendations with Content-Based Initialization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>冻结内容嵌入+有界可训练 delta 校正，兼顾冷启动物品语义结构与热物品任务适配，三数据集冷启动 NDCG@10 最高提升 20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;零样本跨域迁移&quot;,&quot;域语义偏置&quot;,&quot;序列模式聚类&quot;]"><strong><a href="../../venues/RecSys/2025/cold_start_rec/llm_recg_a_semantic_bias_aware_framework_for_zero_shot_sequential_recommend/">LLM-RecG: A Semantic Bias-Aware Framework for Zero-Shot Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 LLM 语义嵌入的域语义偏置，用域间紧致+域内多样双熵损失与源域序列模式注意力实现零样本跨域序列推荐，R@10 平均提升超 10%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;LLM 反思 (Reflection)&quot;,&quot;Contextual Bandit 视角选择&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/more_a_mixture_of_reflectors_framework_for_large_language_model_based_seque/">MoRE: A Mixture of Reflectors Framework for Large Language Model-Based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用显式/隐式/协同三视角离线反思 + 元反思器自改进与 Contextual Bandit 在线选视角，LLM 序列推荐全库候选下 NDCG@10 最高提升 10.1%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间间隔建模&quot;,&quot;LLM4Rec&quot;,&quot;Interval-Infused Attention&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/not_just_what_but_when_integrating_irregular_intervals_to_llm_for_sequentia/">Not Just What, But When: Integrating Irregular Intervals to LLM for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将不规则时间间隔经 MLP 编码与 Interval-Infused Attention 注入 LLaMA-2，配合选项化提示，在 3 个 Amazon 数据集上 HR@1 平均提升 4.4%，并提出“间隔冷启动”新评测视角。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;稀疏自编码器&quot;,&quot;神经元转向&quot;,&quot;流行度去偏&quot;]"><strong><a href="../../venues/RecSys/2025/debias_fairness_industrial/opening_the_black_box_interpretable_remedies_for_popularity_bias_in_recomme/">Opening the Black Box: Interpretable Remedies for Popularity Bias in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用稀疏自编码器定位流行度神经元并做激活转向，在 ML-1M/Last.fm 上以极小精度损失换取显著曝光公平提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;RQ-VAE&quot;,&quot;参数共享&quot;]"><strong><a href="../../venues/RecSys/2025/generative_rec/semantic_ids_for_music_recommendation/">Semantic IDs for Music Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 RQ-VAE 残差量化生成音乐语义 ID 替代逐曲嵌入，参数量降 75%–99% 而精度不降，线上多样性显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;序列与会话推荐&quot;,&quot;离线评测协议&quot;,&quot;数据切分&quot;,&quot;全局时间切分&quot;,&quot;时序泄漏&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/time_to_split_exploring_data_splitting_strategies_for_offline_evaluation_of/">Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统对比 LOO 与全局时间切分及目标/验证变体，揭示 LOO 时序泄漏与排序失真，推荐 GTS+Last/Random/Successive 协议。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;长序列建模&quot;,&quot;多兴趣表征&quot;,&quot;级联类别检索&quot;]"><strong><a href="../../venues/RecSys/2025/sequential_rec/user_long_term_multi_interest_retrieval_model_for_recommendation/">User Long-Term Multi-Interest Retrieval Model for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以类别感知分层双兴趣学习与指针生成式类别→物品级联检索，将召回阶段行为序列扩展至千级，淘宝离线HR@2000达16.55%，线上点击+5.54%、订单+11.01%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;有机/算法交互解耦&quot;]"><strong><a href="../../venues/RecSys/2025/multimodal_rec/yambda_5b_a_large_scale_multi_modal_dataset_for_ranking_and_retrieval/">Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>发布 Yandex Music 4.79B 交互工业级数据集，含音频嵌入与 is_organic 标志，提出 Global Temporal Split 评测协议并给出 6 类基线基准。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;双曲流形对齐&quot;,&quot;协同信号注入&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/addressing_cross_stage_decoupling_of_semantic_and_collaborative_signals_in_/">Addressing Cross-Stage Decoupling of Semantic and Collaborative Signals in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 SCRec，通过协同增强分词、语义引导生成与双曲流形对齐三模块双向补全语义与协同信号，在 Amazon 三数据集 Full-Ranking 下 NDCG@10 相对提升 18.7%–26.6%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;用户基础模型&quot;,&quot;自监督序列预训练&quot;,&quot;LLM-as-Optimizer&quot;]"><strong><a href="../../venues/RecSys/2026/ctr_feature_interaction/building_a_user_foundation_model_for_the_open_web/">Building a User Foundation Model for the Open Web</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在身份碎片化的开放网络 RTB 场景下，用 MLM+时序切分对比预训练 Transformer 用户序列编码器，并以 LLM-in-the-loop NAS 优化训练管线，线上 A/B 取得 +2.13% CTR、-1.13% eCPC。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;生成式推荐&quot;,&quot;粗到细语义分层&quot;,&quot;分层向量量化&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/cofirec_coarse_to_fine_tokenization_for_generative_recommendation/">CoFiRec: Coarse-to-Fine Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将物品元数据按类目→标题→描述→CF信号分层独立量化，自回归由粗到细解码，Leave-One-Out全库排序下平均超越SOTA约6.4%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回/特征增强&quot;,&quot;小语言模型&quot;,&quot;属性偏好预测&quot;,&quot;时序用户建模&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/deepaffinity_long_term_aspect_preference_prediction_in_ecommerce_using_smal/">DeepAffinity: Long-Term Aspect Preference Prediction in eCommerce using Small Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 135M 小模型+结构化字符串化+多任务预测头，从时序行为预测品牌/尺码/颜色等商品属性偏好，Fashion 微 F1 0.34 超 Gemini 零样本与 3B SFT。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;容量探针&quot;,&quot;基准效度&quot;,&quot;成对转移模型&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/do_sequential_recommendation_benchmarks_really_require_higher_order_sequenc/">Do Sequential Recommendation Benchmarks Really Require Higher-Order Sequence Modelling?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用两个不学高阶序列表征的近期加权成对探针（SeqRules/PCTM）在 eSASRec 协议下反超 Transformer，质疑主流基准的容量测量效度。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态分块&quot;,&quot;对比熵惊讶度&quot;,&quot;长序列压缩&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/dp_rec_towards_dynamic_patching_for_efficient_long_sequence_recommendation/">DP-Rec: Towards Dynamic Patching for Efficient Long-Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用对比熵惊讶度动态切分长交互序列为变长行为块，在 FLOPs 匹配下 NDCG@10 最高提升 21.1%，实现更优效率-精度前沿。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;超长序列建模&quot;,&quot;Token Merge&quot;,&quot;KV缓存&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/longer_scaling_up_long_sequence_modeling_in_industrial_recommenders/">LONGER: Scaling Up Long Sequence Modeling in Industrial Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局Token+Token Merge+混合因果注意力把工业超长行为序列端到端建模推到1万长度，离线AUC+1.57%、线上广告与电商双场景显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;流式用户画像&quot;,&quot;GRPO 强化学习&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/recap_feedback_driven_streaming_semantic_user_profiles_for_short_video_reco/">RECAP: Feedback-Driven Streaming Semantic User Profiles for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将流式结构化用户画像视为有界记忆状态，用 LLM 语义差分+确定性状态机维护，并以 LLM 判官清洗隐式反馈、双塔评估器打分作 GRPO 奖励闭环优化画像更新器。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;潜在推理&quot;,&quot;多兴趣表征&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/recrec_latent_interests_recursive_reasoning_for_sequential_recommendation/">RecRec: Latent Interests Recursive Reasoning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦推理态与预测态、以 R 个潜在兴趣做递归推理的免 RL 序列推荐框架，Full-Ranking 下 NDCG@10 最高提升 14.8%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;残差主导&quot;,&quot;末项依赖&quot;,&quot;可解释性诊断&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/residual_dominance_as_a_structural_account_of_last_item_reliance_in_causal_/">Residual Dominance as a Structural Account of Last-Item Reliance in Causal Self-Attention Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用范数分解揭示残差连接主导同位置信息保留，并以推理期残差缩放诊断因果自注意力推荐器的末项依赖机制。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;超长序列压缩&quot;,&quot;低秩缓存&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/sequenceo1_end_to_end_ultra_long_100k_sequence_modeling_in_recommendation_w/">SequenceO1: End-to-End Ultra-Long (100K) Sequence Modeling in Recommendation with Low-Rank Caching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可学习原型压缩 100K 历史为定长 sketch 并缓存复用，端到端精排训练/推理 FLOPs 降 49.9×/63.9×，线上 Finish +2.33%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;模态偏置&quot;,&quot;结构保持投影&quot;]"><strong><a href="../../venues/RecSys/2026/llm_rec/structure_preserving_projection_for_mitigating_modality_bias_in_llm_based_s/">Structure-Preserving Projection for Mitigating Modality Bias in LLM-Based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用余弦相似度保持与序列对比保持双损失约束协同嵌入投影，缓解 LLM 推荐中的模态偏置，20 候选协议下 HR@1 最高提升约 12pt。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;生成式推荐&quot;,&quot;语义ID量化&quot;,&quot;拓扑蒸馏&quot;,&quot;RQ-VAE&quot;]"><strong><a href="../../venues/RecSys/2026/generative_rec/topology_aware_tokenization_for_generative_recommendation/">Topology-Aware Tokenization for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 RQ-VAE 语义 ID 量化中的拓扑失真问题，提出由粗到细的三级拓扑蒸馏 TopoTok，在 Amazon 三数据集 Full-Ranking 下 Recall@5 最高提升 9.42%。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;在线实验平台&quot;,&quot;曝光感知日志&quot;,&quot;双表面策略解耦&quot;,&quot;视频推荐用户研究&quot;]"><strong><a href="../../venues/RecSys/2026/debias_fairness_industrial/watchlens_a_configurable_platform_for_online_video_recommendation_experimen/">WatchLens: A Configurable Platform for Online Video Recommendation Experiments</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>开源视频推荐在线实验平台，解耦界面/内容池/双表面策略并原生记录曝光上下文，30人被试验证观看页策略对会话留存的影响。</p>
</dd>
<dt data-paper-tags="[&quot;RecSys&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;零观测唤醒&quot;,&quot;逐维门控&quot;]"><strong><a href="../../venues/RecSys/2026/sequential_rec/zero_observation_user_reactivation_with_gap_driven_dimensional_gating/">Zero-Observation User Reactivation with Gap-Driven Dimensional Gating</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Conference on Recommender Systems">RecSys</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>定义零观测用户唤醒问题，提出冻结骨干的逐维门控插件 DeltaGate，以 Δt 与历史表征联合路由个性化历史与全局先验，&gt;365d 桶 Hit@10 提升 33%–84%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;反事实序列合成&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2021/sequential_rec/causerec_counterfactual_user_sequence_synthesis_for_sequential_recommendati/">CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>通过反事实概念替换合成用户序列并做对比学习，使用户表征对噪声不敏感、对关键兴趣更信任；Amazon Books NDCG@50 相对最强基线提升 22.1%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;Pre-trained Language Model&quot;,&quot;Attention Pooling&quot;,&quot;多语言新闻推荐&quot;]"><strong><a href="../../venues/SIGIR/2021/llm_rec/empowering_news_recommendation_with_pre_trained_language_models/">Empowering News Recommendation with Pre-trained Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用 PLM 替换新闻编码器并以 Attention 池化 token 隐状态、仅微调顶层两层，MIND 上 AUC 68.18→70.64，多语言 7 市场 AUC 61.07→62.83，线上点击 +8.53%/+10.68%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;社交与群组推荐&quot;,&quot;重排&quot;,&quot;好友圈分组&quot;,&quot;个性化搜索&quot;]"><strong><a href="../../venues/SIGIR/2021/social_group_rec/group_based_personalized_search_by_integrating_search_behaviour_and_friend_/">Group based Personalized Search by Integrating Search Behaviour and Friend Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>FNPS 用“关系圈+行为圈”双通道好友分组与掩码交叉注意力构建群体画像，在微信搜索日志重排上 MAP 较原始排序 +3.22%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/候选重排&quot;,&quot;Transformer 序列编码&quot;,&quot;评论级细粒度匹配&quot;,&quot;注意力可解释个性化&quot;]"><strong><a href="../../venues/SIGIR/2021/sequential_rec/learning_a_fine_grained_review_based_transformer_model_for_personalized_pro/">Learning a Fine-Grained Review-based Transformer Model for Personalized Product Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将查询、用户评论与物品评论拼成单序列交给 Transformer 做评论级细粒度匹配，Amazon 三品类 BM25 候选重排 MRR 最高相对提升 60%，并天然支持注意力可解释与动态个性化。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 MAML&quot;,&quot;翻译式转移模式&quot;]"><strong><a href="../../venues/SIGIR/2021/cold_start_rec/sequential_recommendation_for_cold_start_users_with_meta_transitional_learn/">Sequential Recommendation for Cold-start Users with Meta Transitional Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将冷启动序列推荐重构为少样本任务，用翻译式转移编码器 + MAML 双层优化实现仅 K=2~4 次交互即可快速适配，时间戳冷启动划分下 Hit@1/MRR 显著超越 SASRec/BERT4Rec/MetaCF。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;兴趣图构建&quot;,&quot;动态图池化&quot;,&quot;AUGRU&quot;]"><strong><a href="../../venues/SIGIR/2021/sequential_rec/sequential_recommendation_with_graph_neural_networks/">Sequential Recommendation with Graph Neural Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将松散物品序列用度量学习重构为兴趣图，经簇/查询双感知图卷积与动态图池化提炼激活核心兴趣，长序列场景 AUC 提升约 4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;代理选择 (Proxy Selection)&quot;,&quot;温度 Softmax 硬选择&quot;,&quot;超平面投影度量学习&quot;]"><strong><a href="../../venues/SIGIR/2021/sequential_rec/unsupervised_proxy_selection_for_session_based_recommender_systems/">Unsupervised Proxy Selection for Session-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用可学习的 K 个代理在无监督下为匿名会话选择&quot;用户长期兴趣&quot;，结合短期兴趣与超平面投影距离，全库排序 R@20 相对提升最高 45.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;数据分布自适应&quot;,&quot;模型补丁参数调制&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/ada_ranker_a_data_distribution_adaptive_ranking_paradigm_for_sequential_rec/">Ada-Ranker: A Data Distribution Adaptive Ranking Paradigm for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以 Neural Process 提取候选集分布 z，经 FiLM 输入调制与参数池补丁调制，把全局精排器即时改写为局部模型，NDCG 平均提升约 2.5%，新分布下增益达 15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长时行为序列建模&quot;,&quot;对抗训练与行为去噪&quot;]"><strong><a href="../../venues/SIGIR/2022/ctr_feature_interaction/adversarial_filtering_modeling_on_long_term_user_behavior_sequences_for_cli/">Adversarial Filtering Modeling on Long-term User Behavior Sequences for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用分层聚合压缩重复行为、自注意力打分门选 top-k 行为，并以对抗训练（IEU 最小化、BSU 最大化剩余序列损失）剔除长序列噪声行为，工业集 AUC +0.0047、线上 CTR +4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图序列粗到细建模&quot;,&quot;学习式局部注意力&quot;,&quot;联合概率解码&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/coarse_to_fine_sparse_sequential_recommendation/">Coarse-to-Fine Sparse Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>显式建模类目/卖家意图序列作为稠密先验，与物品序列双编码器融合并按联合概率解码，稀疏数据上 NDCG@5 相对提升 44.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;侧信息融合&quot;,&quot;解耦注意力&quot;,&quot;辅助属性预测&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/decoupled_side_information_fusion_for_sequential_recommendation/">Decoupled Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将侧信息从输入层移至注意力层解耦计算，配合辅助属性预测器，在四个 Amazon/Yelp 数据集全库排序上稳定超越 SASRec/NOVA/ICAI-SR。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;CTR预估&quot;,&quot;时间周期建模&quot;,&quot;行为去噪&quot;,&quot;时间嵌入&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/denoising_time_cycle_modeling_for_recommendation/">Denoising Time Cycle Modeling for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>DiCycle 用绝对/相对时间核编码 + 门控去噪分别刻画 ATC/RTC 两种周期模式，CTR 预测在 LastFM/ML-1M/Books/工业集上 AUC 稳定超越时序基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Determinantal Point Process&quot;,&quot;集合似然损失&quot;,&quot;质量-多样性权衡&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/determinantal_point_process_likelihoods_for_sequential_recommendation/">Determinantal Point Process Likelihoods for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 DPP 集合似然替换逐点/成对损失，把「下 T 项」建模为集合抽取，即插即用于 Caser/MARank/SASRec，Beauty Recall@3 最高 +37%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;判别式训练&quot;,&quot;ELECTRA 式替换检测&quot;,&quot;采样难负例&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/elecrec_training_sequential_recommenders_as_discriminators/">ELECRec: Training Sequential Recommenders as Discriminators</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>借鉴 ELECTRA 思想，将序列推荐从下一项生成改为真假判别训练，辅助生成器采样难负例，四数据集 Full-Ranking 下 HR@5 提升 38%–137%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;协同过滤与双塔召回&quot;,&quot;端到端训练范式&quot;,&quot;过参数化冗余利用&quot;,&quot;同伴协作权重移植&quot;,&quot;层重要性熵度量&quot;]"><strong><a href="../../venues/SIGIR/2022/collaborative_filtering/enhancing_top_n_item_recommendations_by_peer_collaboration/">Enhancing Top-N Item Recommendations by Peer Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出 PCRec：两个同构推荐模型按层熵/范数度量冗余并互相移植权重，在推理零额外开销下稳定提升 Top-N 精度</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Session-aware Recommendation&quot;,&quot;Temporal Self-Attention&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/exploiting_session_information_in_bert_based_session_aware_sequential_recom/">Exploiting Session Information in BERT-based Session-aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>在 BERT4Rec 输入层以会话标记、会话段嵌入与时间感知自注意力三种轻量插件注入会话结构，仅增 0.1% 参数即在全库排序下取得最高 55% NDCG 提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Fastformer&quot;,&quot;细粒度用户建模&quot;,&quot;新闻推荐&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/fum_fine_grained_and_fast_user_modeling_for_news_recommendation/">FUM: Fine-grained and Fast User Modeling for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将用户点击新闻拼接成一条长文档，用 Fastformer 线性复杂度捕获词级跨新闻交互，并融合新闻级粗粒度通路；MIND/Feeds 上 AUC 相对提升 2.2%–2.5%，推理仅 2.75s/1k 样本。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多兴趣解耦&quot;,&quot;隐式类别正交&quot;,&quot;对比学习去噪&quot;]"><strong><a href="../../venues/SIGIR/2022/self_supervised_denoising/improving_micro_video_recommendation_via_contrastive_multiple_interests/">Improving Micro-video Recommendation via Contrastive Multiple Interests</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用隐式正交类别解耦多兴趣，并以跨视图对比多兴趣损失抵御微视频自动播放的假阳性噪声，双数据集 Recall@20 提升 24%/1.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间多样性&quot;,&quot;顺序无关Transformer&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/is_news_recommendation_a_sequential_recommendation_task/">Is News Recommendation a Sequential Recommendation Task?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>论证新闻推荐非标准序列任务，提出双序无关 Transformer 与时间多样性惩罚 TempRec，在 MIND/News 曝光排序稳定提升 AUC/nDCG。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;候选感知用户建模&quot;,&quot;单塔交互式匹配&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/news_recommendation_with_candidate_aware_user_modeling/">News Recommendation with Candidate-aware User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用候选新闻反向引导用户建模，Candi-SelfAtt/Candi-CNN/Candi-Att 三路聚合，MIND 上 nDCG@5 相对提升 4.2%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理/召回&quot;,&quot;张量链分解压缩&quot;,&quot;自监督知识蒸馏&quot;,&quot;长尾物品扶持&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/on_device_next_item_recommendation_with_self_supervised_knowledge_distillat/">On-Device Next-Item Recommendation with Self-Supervised Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用半张量积松弛 TT 分解的维度一致性实现 30 倍嵌入压缩，并以热冷嵌入重组驱动的自监督蒸馏补回容量，全库排序下几乎无损甚至超越教师。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异质超图&quot;,&quot;价格偏好建模&quot;,&quot;协同引导学习&quot;]"><strong><a href="../../venues/SIGIR/2022/sequential_rec/price_does_matter_modeling_price_and_interest_preferences_in_session_based_/">Price DOES Matter! Modeling Price and Interest Preferences in Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将价格离散为等概率价格等级并构建异质超图，用双通道聚合+协同引导门控联合建模价格与兴趣偏好，全库排序下 Prec@20 最高提升 6.86%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;提示式学习&quot;,&quot;离线强化学习&quot;,&quot;会话序列推荐&quot;]"><strong><a href="../../venues/SIGIR/2022/rl_bandit_rec/rethinking_reinforcement_learning_for_recommendation_a_prompt_perspective/">Rethinking Reinforcement Learning for Recommendation: A Prompt Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将离线 RL 推荐从「状态-动作→Q 值」翻转为「状态-累计回报→动作」的提示式监督学习，全库排序下购买 HR@5 最高 +13.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排&quot;,&quot;参数高效提示微调 (Prompt Tuning)&quot;,&quot;对抗去偏 (Adversarial Debiasing)&quot;]"><strong><a href="../../venues/SIGIR/2022/debias_fairness_industrial/selective_fairness_in_recommendation_via_prompts/">Selective Fairness in Recommendation via Prompts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出选择性公平任务与 PFRec：冻结 SASRec、以属性专属软提示+Adapter 配合对抗训练，用极少参数支持 2^m 种敏感属性组合的去偏。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端强化学习&quot;,&quot;状态编码器&quot;,&quot;可复现性研究&quot;]"><strong><a href="../../venues/SIGIR/2022/rl_bandit_rec/state_encoders_in_reinforcement_learning_for_recommendation_a_reproducibili/">State Encoders in Reinforcement Learning for Recommendation: A Reproducibility Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用去偏 SOFA 模拟器+ DQN 复现“注意力状态编码器最优”结论，发现该结论在 Yahoo! R3 成立但在 Coat 数据集失效，无单一最优编码器。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2022&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/重排&quot;,&quot;离线强化学习&quot;,&quot;不确定性惩罚&quot;,&quot;长期满意度&quot;]"><strong><a href="../../venues/SIGIR/2022/rl_bandit_rec/value_penalized_q_learning_for_recommender_systems/">Value Penalized Q-Learning for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出乘性不确定性惩罚 VPQ，用 Q 集成方差构造 $p$-mul 保守目标并配 critic 框架，在两数据集上为三种序列模型平均带来 6–9% 相对增益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;预训练与提示调优&quot;,&quot;后门攻击&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/SIGIR/2023/debias_fairness_industrial/attacking_pre_trained_recommendation/">Attacking Pre-trained Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示预训练序列推荐模型提供者可植入后门，提出基础替换与提示增强攻击，在 Leave-One-Out 协议下将目标物品曝光率提升数百倍且保持隐蔽。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知序列推荐&quot;,&quot;高斯混合注意力&quot;,&quot;Attentive FISM&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/attention_mixtures_for_time_aware_sequential_recommendation/">Attention Mixtures for Time-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用高斯混合注意力量化物品与时间上下文两条注意力流的权重，并融合 Attentive FISM 长兴趣，1K 采样排序下 HR/NDCG@10 全面领先时序 Transformer。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;离线强化学习&quot;,&quot;Decision Transformer&quot;,&quot;因果奖励估计&quot;]"><strong><a href="../../venues/SIGIR/2023/rl_bandit_rec/causal_decision_transformer_for_recommender_systems_via_offline_reinforceme/">Causal Decision Transformer for Recommender Systems via Offline Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以因果决策 Transformer 将离线推荐改写为轨迹序列建模，绕开奖励函数手工设计，在 6 个真实数据集与 VirtualTB 模拟器上取得召回/CTR 领先。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;集成学习&quot;,&quot;对比知识蒸馏&quot;,&quot;掩码物品预测&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/ensemble_modeling_with_contrastive_knowledge_distillation_for_sequential_re/">Ensemble Modeling with Contrastive Knowledge Distillation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 N 个并行双向 Transformer 做序列编码器集成，并以 ICL/CCL 表征对比蒸馏 + logits 协同 KL 蒸馏打通网络间知识迁移；5-core + Leave-One-Out + 全库排序下 NDCG@10 相对提升 20.7%~33.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;频域注意力&quot;,&quot;自相关周期建模&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/frequency_enhanced_hybrid_attention_network_for_sequential_recommendation/">Frequency Enhanced Hybrid Attention Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将用户行为序列迁移到频域，用频率斜坡采样改造自注意力并叠加自相关周期注意力，Leave-One-Out 全库排序下在四个基准上实现 3.6%–10.9% 的稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图掩码自编码器&quot;,&quot;自适应数据增广&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/graph_masked_autoencoder_for_sequential_recommendation/">Graph Masked Autoencoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MAERec 用以任务自适应路径掩码驱动的图掩码自编码器替代手工对比增广，在留一法全库排序下 HR@20 相对最强基线提升最高 11.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;线性注意力&quot;,&quot;长序列建模&quot;,&quot;高效 Transformer&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/linrec_linear_attention_mechanism_for_long_term_sequential_recommender_syst/">LinRec: Linear Attention Mechanism for Long-term Sequential Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 ELU + 行/列 L2 归一化替换 Softmax，令 QK^T V 可交换计算顺序，把长序列自注意力从 O(N²) 降到 O(N)，并在 ML-1M/Gowalla 全库排序上持平或超越 SASRec/BERT4Rec。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回/排序&quot;,&quot;长尾用户-物品联合建模&quot;,&quot;双边分支互增强&quot;,&quot;课程学习&quot;]"><strong><a href="../../venues/SIGIR/2023/cold_start_rec/melt_mutual_enhancement_of_long_tailed_user_and_item_for_sequential_recomme/">MELT: Mutual Enhancement of Long-Tailed User and Item for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SIGIR 2023 长尾序列推荐：双边分支互增强+课程学习，联合提升尾用户/尾物品，Music 总体 HR@10 +8.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;可学习模型增强&quot;,&quot;元学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/meta_optimized_contrastive_learning_for_sequential_recommendation/">Meta-optimized Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用可学习模型增强器与元学习优化对比学习，在 LOO+Full-Ranking 下 HR/NDCG 相对最强基线提升 2.5–8.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;App 推荐数据集&quot;,&quot;全库排序基准&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/mobilerec_a_large_scale_dataset_for_mobile_apps_recommendation/">MobileRec: A Large-Scale Dataset for Mobile Apps Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>构建 19.3M 条 Google Play 用户-App 时序交互（10,173 App / 48 类 / 0.7M 用户），提供 uid、时间戳与 App 元数据，并以 Leave-One-Out + Full-Ranking 协议给出 9 个序列推荐基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;精排/训练阶段负采样&quot;,&quot;难负例挖掘&quot;,&quot;加权物品转移图&quot;,&quot;课程学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/neighborhood_based_hard_negative_mining_for_sequential_recommendation/">Neighborhood-based Hard Negative Mining for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在全局加权物品转移图上以 Jaccard 邻域重叠度构造难负采样分布，并用课程学习调度假阴性阈值；在三个 Amazon 数据集 Leave-One-Out + 1000 候选采样排序下，相对最强负采样基线 NDCG@20 稳定提升 1.0–2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全链路离线评估&quot;,&quot;全局时间线&quot;,&quot;数据泄漏&quot;,&quot;评测协议&quot;]"><strong><a href="../../venues/SIGIR/2023/debias_fairness_industrial/take_a_fresh_look_at_recommender_systems_from_an_evaluation_standpoint/">Take a Fresh Look at Recommender Systems from an Evaluation Standpoint</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>指出随机/留一划分缺失全局时间线导致 Popularity 基线病态与未来数据泄漏，提出 Recent Popularity 与时间窗评测方案。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;增量SVD&quot;,&quot;图信号处理低通滤波&quot;,&quot;可解释交互推荐&quot;]"><strong><a href="../../venues/SIGIR/2023/graph_rec/triple_structural_information_modelling_for_accurate_explainable_and_intera/">Triple Structural Information Modelling for Accurate, Explainable and Interactive Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>增量SVD低通滤波＋无参数注意力＋物品转移矩阵三源融合，全库排序 HR@10 最高相对提升 203.8%，并支持频率级可解释交互推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;兴趣解耦&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/SIGIR/2023/sequential_rec/when_search_meets_recommendation_learning_disentangled_search_representatio/">When Search Meets Recommendation: Learning Disentangled Search Representation for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SESRec 用查询-物品对齐 + 三元组对比自监督把搜索/推荐行为解耦为相似与相异兴趣，在快手与 Amazon 上 NDCG@10 相对提升 7.05%/2.48%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端生成式&quot;,&quot;行程推荐&quot;,&quot;重复抑制&quot;,&quot;解码采样策略&quot;]"><strong><a href="../../venues/SIGIR/2024/poi_spatiotemporal_rec/analyzing_and_mitigating_repetitions_in_trip_recommendation/">Analyzing and Mitigating Repetitions in Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次剖析行程推荐中POI重复生成的两大成因，提出循环感知预测器（引导矩阵+漂移惩罚+置信度自适应采样）在Flickr四城上兼顾精度与去重。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端篮子生成&quot;,&quot;Next Basket Recommendation&quot;,&quot;重复-探索解耦&quot;,&quot;评测范式批判&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/are_we_really_achieving_better_beyond_accuracy_performance_in_next_basket_r/">Are We Really Achieving Better Beyond-Accuracy Performance in Next Basket Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出TREx解耦重复/探索商品，用简单重复模块保精度、探索模块提升公平/多样性，并批判现有beyond-accuracy评测可被“走捷径”。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;多模态推荐与内容理解&quot;,&quot;排序&quot;,&quot;多模态用户交互&quot;,&quot;缺失模态&quot;,&quot;会话推荐&quot;]"><strong><a href="../../venues/SIGIR/2024/multimodal_rec/dataset_and_models_for_item_recommendation_using_multi_modal_user_interacti/">Dataset and Models for Item Recommendation Using Multi-Modal User Interactions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>发布保险场景多模态用户交互数据集，提出将对话与网页会话映射到共同特征空间以应对自然缺失模态，全库排序 MAP@3 较强基线提升 3.46%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果推断解耦&quot;,&quot;会话推荐&quot;,&quot;多模态对齐&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/disentangling_id_and_modality_effects_for_session_based_recommendation/">Disentangling ID and Modality Effects for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DIMO 物品级把共现矩阵显式注入 ID 表征并将图文统一对齐到文本空间，会话级用代理机制与反事实推断无监督解耦 ID/模态成因，Cellphones Prec@20 较最优基线提升 55.12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排（序列编码器骨干）&quot;,&quot;复数向量注意力&quot;,&quot;旋转位置编码&quot;,&quot;相位对比学习&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/eulerformer_sequential_user_behavior_modeling_with_complex_vector_attention/">EulerFormer: Sequential User Behavior Modeling with Complex Vector Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用欧拉变换把语义差与位置差统一为复向量旋转角，配合自适应差分旋转与相位对比学习，SASRec 骨干 NDCG@10 最高提升 10.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;持续图学习&quot;,&quot;图提示微调&quot;,&quot;流式推荐&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/gpt4rec_graph_prompt_tuning_for_streaming_recommendation/">GPT4Rec: Graph Prompt Tuning for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将演化交互图解耦为多视图，用节点/结构/视图三级轻量图提示替代回放与模型扩张，实现流式推荐持续学习，R@20 最高相对提升 4.16%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知特征&quot;,&quot;高斯核平滑&quot;,&quot;流式训练&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/interest_clock_time_perception_in_real_time_streaming_recommendation_system/">Interest Clock: Time Perception in Real-Time Streaming Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将用户 24 小时分桶兴趣经高斯核按请求时间平滑聚合成兴趣时钟嵌入，抖音音乐线上 A/B 活跃天数 +0.509%、使用时长 +0.758%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;提示自动优化&quot;,&quot;LLM 意图推断&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/large_language_models_for_intent_driven_session_recommendations/">Large Language Models for Intent-Driven Session Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>PO4ISR 用“提示初始化→自反思优化→跨域选择”三段式自动提示工程驱动 LLM 在语义层识别会话内变动数量的意图，20 候选重排下 HR/NDCG 平均提升 57.37%/61.03%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;混合提示 (Hybrid Prompting)&quot;,&quot;课程学习模态对齐&quot;,&quot;指令微调&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/llara_large_language_recommendation_assistant/">LLaRA: Large Language-Recommendation Assistant</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将传统序列推荐器的 ID 行为嵌入经投影器对齐进 LLM 词元空间，用混合提示+课程学习实现行为模态注入，采样排序 HitRatio@1 最高提升 21.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;用户疲劳建模&quot;,&quot;对比学习序列增广&quot;,&quot;多兴趣表征&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/modeling_user_fatigue_for_sequential_recommendation/">Modeling User Fatigue for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以兴趣感知相似度矩阵刻画目标-历史细粒度重复度，经 Cross 网络与疲劳门控 GRU 注入长短期兴趣，并用序列增广构造疲劳对比监督；Taobao AUC 提升 0.026，线上 EVTR 与类目多样性同升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多意图表征&quot;,&quot;稀疏意图筛选&quot;,&quot;频次嵌入&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/multi_intent_aware_session_based_recommendation/">Multi-intent-aware Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MiaSRec 用位置+频次嵌入、highway 多意图表征与 α-entmax 动态意图筛选，配合 max/avg 双池化解码聚合，在六个数据集上全场刷新 SOTA，长会话 R@20 最高 +24.56%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;神经点击模型&quot;,&quot;用户响应仿真&quot;]"><strong><a href="../../venues/SIGIR/2024/ctr_feature_interaction/neural_click_models_for_recommender_systems/">Neural Click Models for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>将搜索侧神经点击模型迁移到 slate 推荐，以 readout 条件 GRU、对抗辅助损失与仅点击注意力建模曝光级用户响应，ContentWise AUC 相对提升 6.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;负反馈感知超图卷积&quot;,&quot;Transformer Hawkes Process&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/nfarec_a_negative_feedback_aware_recommender_model/">NFARec: A Negative Feedback-Aware Recommender Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以反馈极性引导超图卷积消息通路并用 Transformer Hawkes 过程辅助预测下一交互情感极性，五数据集全库排序 R@20 最高提升 6.6%、N@20 提升 14.2%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;负迁移量化&quot;,&quot;序列混合专家&quot;,&quot;互信息最大化&quot;]"><strong><a href="../../venues/SIGIR/2024/cross_domain_multi_scenario/pacer_and_runner_cooperative_learning_framework_between_single_and_cross_do/">Pacer and Runner: Cooperative Learning Framework between Single- and Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用停梯度非对称MoE解耦SDSR/CDSR以精确估计负迁移差，动态加权跨域损失并加互信息对齐，10个域上全面抑制负迁移，线上CTR+17.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Transformer&quot;,&quot;Scaling Law&quot;,&quot;特征编码&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/scaling_sequential_recommendation_models_with_transformers/">Scaling Sequential Recommendation Models with Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用特征编码替代物品ID嵌入，使SRT参数与目录规模解耦，在Amazon全量数据上验证Transformer序列推荐的缩放律与预训练微调收益。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;个性化自增强学习&quot;,&quot;短期协作图&quot;,&quot;序列去噪&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/selfgnn_self_supervised_graph_neural_networks_for_sequential_recommendation/">SelfGNN: Self-Supervised Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>按时间切片构建短期协作图并用个性化自增强学习以长期兴趣矫正短期噪声，1:999 采样排序下四个数据集全面超越 17 个基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列推荐&quot;,&quot;精排&quot;,&quot;Decision Transformer&quot;,&quot;多任务多奖励 RTG 平衡&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/sequential_recommendation_for_optimizing_both_immediate_feedback_and_long_t/">Sequential Recommendation for Optimizing Both Immediate Feedback and Long-term Retention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>DT4IER 用用户特征自适应重加权双奖励 RTG，将 Decision Transformer 扩展为 CTR+留存的长期-短期多任务序列推荐，三数据集 NDCG 提升 11%–16%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;潜在关系发现&quot;,&quot;离散变分自编码器&quot;]"><strong><a href="../../venues/SIGIR/2024/sequential_rec/sequential_recommendation_with_latent_relations_based_on_large_language_mod/">Sequential Recommendation with Latent Relations based on Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 语义表征 + DVAE 自监督挖掘物品间潜在关系，与关系感知序列推荐联合优化，Office 上 HR@5 相对提升 8.38%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回/用户表征&quot;,&quot;Neural Hawkes Process&quot;,&quot;用户开屏动机预测&quot;]"><strong><a href="../../venues/SIGIR/2024/collaborative_filtering/to_search_or_to_recommend_predicting_open_app_motivation_with_neural_hawkes/">To Search or to Recommend: Predicting Open-App Motivation with Neural Hawkes Process</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用神经 Hawkes 过程+层级 Transformer 建模用户开屏动机，OAMD 上 F0.5 提升 2.9%、AUC 达 0.9326</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;注意力采样&quot;,&quot;结构位置编码&quot;]"><strong><a href="../../venues/SIGIR/2024/graph_rec/transgnn_harnessing_the_collaborative_power_of_transformers_and_graph_neura/">TransGNN: Harnessing the Collaborative Power of Transformers and Graph Neural Networks for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>交替堆叠 Transformer 与 GNN 层，用注意力采样+三种结构位置编码互相扩感受野与注入拓扑，全库排序下 Recall@20 最高提升 21%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;行为转移建模&quot;,&quot;对比学习对齐&quot;,&quot;MMoE 多任务&quot;]"><strong><a href="../../venues/SIGIR/2024/cross_domain_multi_scenario/unisar_modeling_user_transition_behaviors_between_search_and_recommendation/">UniSAR: Modeling User Transition Behaviors between Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用掩码 Transformer 抽取 r2r/s2s/r2s/s2r 四类细粒度转移，对比对齐 + 交叉注意力融合，MMoE 联合训练统一搜索与推荐，双任务全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;目标注意力&quot;,&quot;参数生成网络&quot;,&quot;多域序列建模&quot;]"><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/adaptive_domain_scaling_for_personalized_sequential_modeling_in_recommender/">Adaptive Domain Scaling for Personalized Sequential Modeling in Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用域特征动态生成序列物品的个性化MLP权重与候选物品的多查询表示，改造目标注意力实现多域序列建模，抖音双场景线上GMV+0.79%、ADVV+1.00%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散推荐&quot;,&quot;缺失数据模拟&quot;,&quot;Thompson Sampling&quot;]"><strong><a href="../../venues/SIGIR/2025/generative_rec/addressing_missing_data_issue_for_diffusion_based_recommendation/">Addressing Missing Data Issue for Diffusion-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双侧 Thompson 采样在引导序列中模拟缺失数据，以一致性正则化让扩散推荐模型外推抵抗真实缺失，YooChoose NDCG@20 提升 26.95%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;查询改写&quot;,&quot;LLM 指令微调&quot;,&quot;意图一致性过滤&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/ai_guided_accelerator_for_search_experience/">AI Guided Accelerator For Search Experience</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>挖掘电商查询转移链并做意图一致性过滤，再用 Solar-10B 指令微调生成多样化替代收敛查询，线上 CTR +32.2%、转化 +38.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;零空间学习&quot;,&quot;语言嵌入融合&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/alphafuse_learn_id_embeddings_for_sequential_recommendation_in_null_space_o/">AlphaFuse: Learn ID Embeddings for Sequential Recommendation in Null Space of Language Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将语言嵌入 SVD 分解为语义丰富行空间与语义稀疏零空间，在裁剪后的零空间内学习 ID 嵌入，无需适配器即插即用地融合语义与协同信号。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;因果推断&quot;,&quot;干预分布&quot;,&quot;处理效应&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/csrec_rethinking_sequential_recommendation_from_a_causal_perspective/">CSRec: Rethinking Sequential Recommendation from A Causal Perspective</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将序列推荐重构为因果干预下的决策接受概率建模，用递归约束隔离系统影响，AHR@0.2 提升约 1.5–3 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;测试时增强&quot;,&quot;序列数据增强&quot;,&quot;免训练推理&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/data_augmentation_as_free_lunch_exploring_the_test_time_augmentation_for_se/">Data Augmentation as Free Lunch: Exploring the Test-Time Augmentation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将数据增强从训练期迁移到推理期，提出 TNoise 与 TMask 两个免训练、免参数的测试时增强算子，在 Full-Ranking 协议下平均提升 14%–30%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长期行为序列建模&quot;,&quot;多码本向量量化&quot;,&quot;HSTU&quot;]"><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/deep_multiple_quantization_network_on_long_behavior_sequence_for_click_thro/">Deep Multiple Quantization Network on Long Behavior Sequence for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多码本量化将超长行为序列端到端压缩为约百个兴趣簇码字，配合 HSTU 簇间交互与候选感知注意力，消除两阶段检索的相关性分布偏差，线上 CTR +3.5%、RPM +2.0%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;侧信息融合&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/diff_dual_side_information_filtering_and_fusion_for_sequential_recommendati/">DIFF: Dual Side-Information Filtering and Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>频域滤波去噪 + 早期/中间双重融合的侧信息序列推荐，Full-Ranking 下 R@20 最高提升 14.1%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;数据投毒攻击&quot;,&quot;序列推荐&quot;,&quot;双提升攻击目标&quot;,&quot;多样性重排生成&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/diversity_aware_dual_promotion_poisoning_attack_on_sequential_recommendatio/">Diversity-aware Dual-promotion Poisoning Attack on Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示攻击与推荐目标梯度冲突，提出双提升MSE攻击目标与多样性重排序列生成，在保推荐精度的同时显著提升目标物品曝光。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;来源偏见 (Source Bias)&quot;,&quot;反馈回路 (Feedback Loop)&quot;,&quot;AIGC 内容生态&quot;]"><strong><a href="../../venues/SIGIR/2025/debias_fairness_industrial/exploring_the_escalation_of_source_bias_in_user_data_and_recommender_system/">Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>揭示 AIGC 在用户-数据-系统反馈回路中被逐步放大的来源偏见，并提出基于 L1 损失的黑盒去偏方法维持内容生态平衡。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;频域建模&quot;,&quot;多视图搜索&quot;,&quot;本地生活推荐&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/fim_frequency_aware_multi_view_interest_modeling_for_local_life_service_rec/">FIM: Frequency-Aware Multi-View Interest Modeling for Local-Life Service Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>多视图搜索分离本地生活多周期需求，FFT 频域感知演化建模周期兴趣，Kuaishou 线上 GMV +8.46%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;生成式推荐&quot;,&quot;端到端物品分词&quot;,&quot;RQ-VAE 语义ID&quot;,&quot;交替优化&quot;]"><strong><a href="../../venues/SIGIR/2025/generative_rec/generative_recommender_with_end_to_end_learnable_item_tokenization/">Generative Recommender with End-to-End Learnable Item Tokenization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 ETEGRec 双编码器-解码器框架，用序列-物品对齐与偏好-语义对齐将 RQ-VAE 分词器与 T5 推荐器端到端联合优化，全库排序下 Recall@10 相对 LETTER 提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构 Token 化&quot;,&quot;层级因果 Transformer&quot;,&quot;多步预测&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/heterrec_heterogeneous_information_transformer_for_scalable_sequential_reco/">HeterRec: Heterogeneous Information Transformer for Scalable Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将物品拆解为异构特征 token 序列并设计层级因果 Transformer 与列表级多步预测损失，在工业与 Amazon 数据上 Recall@100 提升 3.8%–4.2%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回+重排&quot;,&quot;Chain-of-Thought&quot;,&quot;位置偏差 MAPB&quot;,&quot;偏好抽取&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/improving_llm_powered_recommendations_with_personalized_information/">Improving LLM-powered Recommendations with Personalized Information</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将用户偏好与物品感知两条 CoT 显式抽取并注入 CRM 召回与 LLM 重排，Leave-One-Out 全库排序下 Hit@10 最高提升约 11%，并显著降低位置偏差。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Decision Transformer&quot;,&quot;反事实数据增广&quot;,&quot;曝光偏差&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/improving_sequential_recommenders_through_counterfactual_augmentation_of_sy/">Improving Sequential Recommenders through Counterfactual Augmentation of System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Decision Transformer 建模全量曝光序列，并以随机/自改进反事实增广+用户模拟器挖掘未曝光兴趣，KuaiRand 上 Recall@20 相对提升约 87%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图原型聚类&quot;,&quot;条件扩散增强&quot;,&quot;意图感知对比学习&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/intent_aware_diffusion_with_contrastive_learning_for_sequential_recommendat/">Intent-aware Diffusion with Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 K-means 意图原型引导条件扩散生成意图对齐正视图替代随机增强，在 5 个数据集 Full-Ranking 下 NDCG 平均提升 20.68%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;通用用户建模&quot;,&quot;跨域序列推荐&quot;,&quot;信息瓶颈&quot;]"><strong><a href="../../venues/SIGIR/2025/cross_domain_multi_scenario/learning_universal_user_representations_leveraging_cross_domain_user_intent/">Learning Universal User Representations Leveraging Cross-domain User Intent at Snapchat</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 Snapchat 多业务面事件拼接为统一序列训练 Transformer 用户塔，产出跨域通用用户表征，线上多场景 A/B 全面正向。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性物品-物品模型&quot;,&quot;知识蒸馏&quot;,&quot;闭式解&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/linear_item_item_model_with_neural_knowledge_for_session_based_recommendati/">Linear Item-Item Model with Neural Knowledge for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将神经会话模型的转移知识蒸馏为物品-物品矩阵正则项，与线性自蒸馏相似度统一闭式求解，R@20 最高提升 14.78% 且推理 FLOPs 降低 813 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;低秩矩阵分解&quot;,&quot;长序列目标注意力&quot;,&quot;矩阵吸收&quot;]"><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/lrea_low_rank_efficient_attention_on_modeling_long_term_user_behaviors_for_/">LREA: Low-Rank Efficient Attention on Modeling Long-Term User Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用低秩分解+矩阵吸收把万级长行为序列压到秩 r 维做目标注意力，配非负正则保非线性，线上仅增 3ms 延迟、CTR +5.88%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;掩码Softmax损失&quot;,&quot;自适应温度&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/msl_not_all_tokens_are_what_you_need_for_tuning_llm_as_a_recommender/">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出掩码 Softmax 损失屏蔽虚构物品 token，并配自适应温度策略，四数据集 NDCG@10 平均提升 42.24%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;层次化压缩&quot;,&quot;Patch 预训练&quot;,&quot;长序列建模&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/multi_grained_patch_training_for_efficient_llm_based_recommendation/">Multi-Grained Patch Training for Efficient LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用层次化 item/session patch 压缩长交互历史，两阶段训练让 LLM 理解压缩表征，同 token 预算下 HR@20 最高提升 72%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;条件扩散去噪&quot;,&quot;多行为序列推荐&quot;]"><strong><a href="../../venues/SIGIR/2025/multimodal_rec/multi_modal_multi_behavior_sequential_recommendation_with_conditional_diffu/">Multi-Modal Multi-Behavior Sequential Recommendation with Conditional Diffusion-Based Feature Denoising</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用条件扩散分别对模态与行为特征去噪，并以多专家网络解耦共性/特性兴趣，在 Rec-Tmal/Kuaishou 上 HR@10 提升约 4-8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;逆物品频率IIF&quot;,&quot;全局归一化注意力&quot;,&quot;个性化搜索&quot;]"><strong><a href="../../venues/SIGIR/2025/ctr_feature_interaction/nam_a_normalization_attention_model_for_personalized_product_search_in_flig/">NAM: A Normalization Attention Model for Personalized Product Search In Fliggy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用逆物品频率(IIF)对注意力做全局归一化并门控个性化强度，解决热门物品高估/长尾低估，离线CTCVR AUC+0.001、线上CTCVR+0.8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;MK-MMD 特征核对齐&quot;,&quot;三专家解耦&quot;,&quot;灾难性遗忘&quot;]"><strong><a href="../../venues/SIGIR/2025/llm_rec/pre_train_align_and_disentangle_empowering_sequential_recommendation_with_l/">Pre-train, Align, and Disentangle: Empowering Sequential Recommendation with Large Language Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MK-MMD 特征核做推荐锚定对齐，再以三专家频率门控解耦，Full-Ranking 下冷物品 HR@10 提升最高 462%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MLP-Mixer&quot;,&quot;多周期兴趣建模&quot;,&quot;低秩分解&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/pyramid_mixer_multi_dimensional_multi_period_interest_modeling_for_sequenti/">Pyramid Mixer: Multi-dimensional Multi-period Interest Modeling for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用纯 MLP 金字塔 Mixer 做跨行为/跨特征/跨周期三维兴趣建模，离线全库排序超越 SASRec/MLP4Rec，抖音线上停留时长 +0.106%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;片段级兴趣建模&quot;,&quot;多模态跨注意力&quot;,&quot;段内排序损失&quot;]"><strong><a href="../../venues/SIGIR/2025/multimodal_rec/short_video_segment_level_user_dynamic_interests_modeling_in_personalized_r/">Short Video Segment-level User Dynamic Interests Modeling in Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将短视频切分为5秒片段，用CLIP视觉+ID双模态跨注意力建模片段级动态兴趣，以段内排序损失解决标签缺失，在跳看预测与视频推荐两任务上大幅超越基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;状态空间模型&quot;,&quot;MoE&quot;,&quot;偏好感知注意力&quot;]"><strong><a href="../../venues/SIGIR/2025/sequential_rec/star_rec_making_peace_with_length_variance_and_pattern_diversity_in_sequent/">STAR-Rec: Making Peace with Length Variance and Pattern Diversity in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用偏好感知注意力与 SSM 双路径自适应融合 + 序列级 MoE，统一建模长度方差与行为模式多样性，Leave-One-Out 全库排序下 Recall@10 最高提升 2.15%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;嵌入各向异性&quot;,&quot;线性自编码器谱收缩&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/ace_anisotropy_controllable_embedding_for_llm_enhanced_sequential_recommend/">ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性自编码器的谱收缩算子连续调控 LLM 物品嵌入的各向异性，在保持语义层级的同时缓解几何失衡，Full-Ranking 下 Recall@20 最高提升 12.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;自适应数据增强&quot;,&quot;Sinkhorn 可微投影&quot;]"><strong><a href="../../venues/SIGIR/2026/self_supervised_denoising/asarrec_adaptive_sequential_augmentation_for_robust_self_supervised_sequent/">AsarRec: Adaptive Sequential Augmentation for Robust Self-supervised Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将五种序列增强统一为半双随机变换矩阵，用可微 Semi-Sinkhorn 学习用户自适应增强，四数据集全排序下 HR/NDCG 平均提升约 7-8%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;长序列建模&quot;,&quot;高效Transformer&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/beyond_item_ids_scaling_short_form_video_recommendation_via_semantic_native/">Beyond Item IDs: Scaling Short-Form-Video Recommendation via Semantic-Native Long Sequence Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用深度截断 Bi-gram 语义 ID 替代正交 Video ID，并以无参时间折叠压缩 Transformer 将序列扩至 2000，线上满意观看时长 +1.42%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;神经 ODE&quot;,&quot;时间感知语义&quot;,&quot;反事实对比学习&quot;]"><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/bridging_behavior_and_semantics_for_time_aware_cross_domain_sequential_reco/">Bridging Behavior and Semantics for Time-aware Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用神经 ODE 解耦长短期行为、LLM 离散时间 token 反事实增强语义、门控自适应域迁移，1:999 采样排序下 MRR 最高提升 30.5%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;复购节奏建模&quot;,&quot;集合注意力&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/case_cadence_aware_set_encoding_for_large_scale_next_basket_repurchase_reco/">CASE: Cadence-Aware Set Encoding for Large-Scale Next Basket Repurchase Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以日历时间二值信号+共享多尺度时序卷积建模物品复购节奏，再用 ISAB 建模跨物品依赖，在四个杂货数据集与千万级生产系统上取得显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多序列构造&quot;,&quot;线性注意力&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/cmsl_constructive_multi_sequence_learning_for_recommendation_systems/">CMSL: Constructive Multi-Sequence Learning for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出可学习多序列构造与线性注意力，缓解单序列上下文污染，Meta 四场景 NE 降 0.3%–0.6%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;负采样&quot;,&quot;知识蒸馏&quot;,&quot;难负例挖掘&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/divergence_meets_consensus_a_multi_source_negative_sampling_framework_for_s/">Divergence Meets Consensus: A Multi-Source Negative Sampling Framework for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>借鉴 ZPD 理论构建 Teacher-Peer-Self 多源负采样，用分歧重排提升多样性、共识蒸馏复用打分开销，六数据集全库排序下 Recall@20 最高提升 27.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;生成式推荐&quot;,&quot;Machine Unlearning&quot;,&quot;Benchmark&quot;,&quot;Sequential Unlearning&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/erase_a_real_world_aligned_benchmark_for_unlearning_in_recommender_systems/">ERASE -- A Real-World Aligned Benchmark for Unlearning in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向真实场景的推荐系统机器遗忘基准，覆盖 CF/SBR/NBR 三任务、七种遗忘算法、九数据集九模型，揭示近似遗忘的鲁棒性与效率短板。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;LLM推荐&quot;,&quot;召回&quot;,&quot;序数语义锚定&quot;,&quot;强度感知对齐&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/every_preference_has_its_strength_injecting_ordinal_semantics_into_llm_base/">Every Preference Has Its Strength: Injecting Ordinal Semantics into LLM-Based Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将显式评分建模为LLM词表中的序数语义锚点，以强度感知对齐交互表征，在序列推荐Hit@1与成对偏好准确率上显著超越CF-LLM混合基线。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;Flow Matching&quot;,&quot;单步生成&quot;,&quot;语义锚先验&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义锚先验替代高斯噪声、以平均速度场实现单步流匹配生成推荐，全库排序下 N@20 相对提升 9.9%，推理提速 20 倍以上。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;目标注意力&quot;,&quot;频域滤波&quot;,&quot;频谱熵&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/fedin_frequency_enhanced_deep_interest_network_for_click_through_rate_predi/">FEDIN: Frequency-Enhanced Deep Interest Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以目标条件化频谱熵观察为动机，用目标感知频谱滤波+时域双分支做CTR预估，三数据集GAUC/AUC全面SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;模块化集成&quot;,&quot;引导式互学习&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/flame_condensing_ensemble_diversity_into_a_single_network_for_efficient_seq/">FLAME: Condensing Ensemble Diversity into a Single Network for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用冻结+可学习双网络的子模块组合模拟指数级集成多样性，推理仅用单网络，NDCG@20 最高提升 9.70% 且收敛快 7.69 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;LLM 上下文增强&quot;,&quot;地理-时间嵌入&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/from_time_and_place_to_preference_llm_driven_geo_temporal_context_in_recomm/">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 从时间戳+粗粒度位置生成地理-时间上下文嵌入，以特征融合或辅助对齐损失注入 SASRec，并提出轻量信息量诊断测试。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自适应门控融合&quot;,&quot;双层对齐&quot;,&quot;长尾物品&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/fusion_and_alignment_enhancement_with_large_language_models_for_tail_item_s/">Fusion and Alignment Enhancement with Large Language Models for Tail-item Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自适应门控融合 ID/LLM 嵌入并以双层对齐+课程学习消除空间结构冲突，全库排序下尾部长尾 H@10 最高提升 427%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;双向线性注意力&quot;,&quot;长序列编码器加速&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Conv1D 局部混合、序列级 Key 门控软遗忘与门控 RMSNorm 将核化线性注意力扩展为双向线性时间算子，1:2 混合编码器在 Yandex Music 上追平双向自注意力，32768 长度单层提速 8.2×。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;统一架构&quot;]"><strong><a href="../../venues/SIGIR/2026/ctr_feature_interaction/hyformer_revisiting_the_roles_of_sequence_modeling_and_feature_interaction_/">HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Global Tokens 交替执行 Query Decoding 与 Query Boosting，统一长序列建模与特征交叉，AUC 提升 0.17% 且 FLOPs 仅 3.9T。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;直播推荐数据集&quot;,&quot;长尾公平性&quot;,&quot;多行为建模&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/kuailive_a_real_time_interactive_dataset_for_live_streaming_recommendation/">KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个含直播间生命周期与多行为实时交互的快手直播推荐数据集，构建 Top-K/CTR 基准并揭示主播长尾公平性挑战。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;参数级适配&quot;,&quot;双视图混合专家&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/l2rec_towards_dual_view_understanding_of_llms_for_personalized_recommendati/">L2Rec: Towards Dual-View Understanding of LLMs for Personalized Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以参数级双视图 LoRA-MoE 替代输入/输出级融合，个性化路由对齐行为与语义视图，四数据集 N@10 提升 3.87%–8.02%，线上 CTR +9.24%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;上下文token组合&quot;,&quot;分解嵌入融合&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DECOR 用冻结码本+可学协同嵌入分解融合、上下文 token 组合与可学 BOS 查询，缓解静态分词歧义与预训练语义丢失，Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;意图终止建模&quot;,&quot;饱食门控&quot;,&quot;复购周期自监督&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/learning_to_forget_satiation_aware_long_sequence_transducers_for_mitigating/">Learning to Forget: Satiation-Aware Long-Sequence Transducers for Mitigating Post-Purchase Redundancy</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将购买视为意图终止信号，用双路交叉注意力+自适应饱食门控+TTNP自监督学习商品复购周期，PPRR降超60%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;双阶段训练&quot;,&quot;LLM数据增广&quot;]"><strong><a href="../../venues/SIGIR/2026/cross_domain_multi_scenario/llm_edt_large_language_model_enhanced_cross_domain_sequential_recommendatio/">LLM-EDT: Large Language Model Enhanced Cross-domain Sequential Recommendation with Dual-phase Training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用可迁移物品增广+双阶段训练+域感知画像，缓解CDSR域不平衡与域迁移难题，全库排序下H@10最高提升26.96%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Semantic ID 对齐&quot;,&quot;时序漂移&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/mitigating_collaborative_semantic_id_staleness_in_generative_retrieval/">Mitigating Collaborative Semantic ID Staleness in Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模型无关的 SID 对齐更新，将刷新后的协同语义 ID 双部图匹配映射回旧词表，实现热启动微调，Recall@500 提升且训练算力降 8-9 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;变分多视图集成&quot;,&quot;语义ID与协同ID&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/mviger_multi_view_variational_integration_of_complementary_knowledge_for_ge/">MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将模板-索引组合视为隐变量视图，用可学习先验做变分集成，Full-Ranking 下 H@10 最高提升约 26%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;数据预处理/评估协议&quot;,&quot;Sub-Sequence Splitting&quot;,&quot;可复现性研究&quot;,&quot;数据增强&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/pay_attention_to_sequence_split_uncovering_the_impacts_of_sub_sequence_spli/">Pay Attention to Sequence Split: Uncovering the Impacts of Sub-Sequence Splitting on Sequential Recommendation Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示子序列切分(SSS)在数据读取阶段被隐性使用导致模型评估失真，系统分析SSS生效条件与数据分布重塑机理。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;逆向课程学习&quot;,&quot;稀疏转化建模&quot;]"><strong><a href="../../venues/SIGIR/2026/generative_rec/rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_gener/">RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以逆向课程前缀为稀疏转化目标注入实例级中间监督，配合课程质量感知损失，离线全库排序全面超越 SOTA，线上广告收入 +2.09%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端侧推理&quot;,&quot;意图理解&quot;,&quot;自适应提示构建&quot;,&quot;意图漂移触发&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/recgpt_mobile_on_device_large_language_models_for_user_intent_understanding/">RecGPT-Mobile: On-Device Large Language Models for User Intent Understanding in Taobao Feed Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 Qwen3-0.6B 量化后部署于手机端，用意图漂移触发机制按需推理生成下一搜索意图，线上 GMV +2.5%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;全局时间线&quot;,&quot;滑动窗口评估&quot;,&quot;增量学习&quot;]"><strong><a href="../../venues/SIGIR/2026/debias_fairness_industrial/recnexteval_a_reference_implementation_for_temporal_next_batch_recommendati/">RecNextEval: A Reference Implementation for Temporal Next-Batch Recommendation Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出滑动时间窗 next-batch 评估框架，强制全局时间线防泄漏并支持增量学习，含 Python 库与 Web UI。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;Prior-Fitted Network&quot;,&quot;In-Context Learning&quot;,&quot;合成数据先验&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/recpfn_prior_fitted_networks_for_in_context_based_recommendations/">RecPFN: Prior-Fitted Networks for In-Context-Based Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用结构因果先验合成点击流预训练 PFN，单次前向即可零样本适配新域序列推荐，8 个基准 Full-Ranking 上零样本 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;旋转位置编码&quot;,&quot;多粒度时间建模&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/rote_coarse_to_fine_multi_level_rotary_time_embedding_for_sequential_recomm/">RoTE: Coarse-to-Fine Multi-Level Rotary Time Embedding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将时间戳分解为年/月/日多粒度并借 RoPE 旋转注入 Q/K，即插即用提升 SASRec/RPG，NDCG@5 最高 +20.11%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;谱域适配器&quot;,&quot;维度坍缩&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/spectran_spectral_aware_transformer_based_adapter_for_llm_enhanced_sequenti/">SpecTran: Spectral-Aware Transformer-based Adapter for LLM-Enhanced Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全谱注意力+可学习谱位置编码替代静态SVD截断，缓解LLM语义嵌入投影的维度坍缩，四数据集三骨干平均提升9.17%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;LLM 意图画像&quot;,&quot;不确定性采样&quot;,&quot;自训练伪标签&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/sprint_scalable_and_predictive_intent_refinement_for_llm_enhanced_session_b/">SPRINT: Scalable and Predictive Intent Refinement for LLM-Enhanced Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局意图池约束 LLM 生成、以预测-纠错回路校验意图，并蒸馏为轻量意图预测器，实现训练选择性调用 LLM、推理零 LLM 依赖的会话推荐。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;下一购物篮推荐&quot;,&quot;意图解耦&quot;,&quot;Hawkes 过程&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/time_interval_aware_disentangled_expert_modeling_for_next_basket_recommenda/">Time-Interval-Aware Disentangled Expert Modeling for Next-Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TIDE 用 Hawkes 增强傅里叶时间编码刻画物品级非单调补货周期，并以双专家解耦习惯与探索意图，Leave-One-Out 全库排序下 R@10 最高提升 18.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;个性化查询改写&quot;,&quot;GRPO 风格对齐&quot;,&quot;轻量索引并行召回&quot;]"><strong><a href="../../venues/SIGIR/2026/llm_rec/when_how_to_write_for_personalized_demand_aware_query_rewriting_in_video_se/">When &amp; How to Write for Personalized Demand-aware Query Rewriting in Video Search</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用后验行为挖掘「何时改写」样本、SFT+GRPO 对齐检索索引风格，并以并行轻量召回实现零感知延迟的个性化查询改写。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;小波包变换&quot;,&quot;子带图传播&quot;,&quot;时频多分辨率建模&quot;]"><strong><a href="../../venues/SIGIR/2026/sequential_rec/wpgrec_wavelet_packet_guided_graph_enhanced_sequential_recommendation/">WPGRec: Wavelet Packet Guided Graph Enhanced Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用平稳小波包变换生成等长平移不变子带，在子带内做 Chebyshev 图传播并以能量-谱平坦度门控融合，Full-Ranking 下稀疏数据集 HR@20 最高 +15.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣解耦&quot;,&quot;强化学习序列分解&quot;,&quot;时间间隔建模&quot;]"><strong><a href="../../venues/SIGIR-AP/2023/sequential_rec/sequential_recommendation_with_user_evolving_preference_decomposition/">Sequential Recommendation with User Evolving Preference Decomposition</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 RL 智能体把用户行为序列自适应拆成多条演化偏好线程，ε-softmax 决定新建线程、时间衰减刻画间隔，六数据集 NDCG@5 平均 +10.3%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR-AP 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Learned Sparse Retrieval&quot;,&quot;注意力用户建模&quot;]"><strong><a href="../../venues/SIGIR-AP/2026/collaborative_filtering/lsnr_learned_sparse_news_recommendations_with_attentive_user_modeling/">LSNR: Learned Sparse News Recommendations with Attentive User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用独立编码+注意力聚合替代 STAR 的历史拼接，构建双通路稀疏用户向量，MIND NDCG@10 较 STAR 提升 11%，短历史场景反超稠密模型。</p>
</dd>
<dt data-paper-tags="[&quot;TKDD&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;解耦表征学习&quot;,&quot;通道感知图聚合&quot;,&quot;变分自编码器&quot;]"><strong><a href="../../venues/TKDD/2023/sequential_rec/edge_enhanced_global_disentangled_graph_neural_network_for_sequential_recom/">Edge-Enhanced Global Disentangled Graph Neural Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用通道感知解耦把物品转移边拆成 K 个潜在因子通道，全局链路图 + 局部 SA-VAE 滑窗双层级学习用户意图，ML-1M N@5 相对提升 8.21%。</p>
</dd>
<dt data-paper-tags="[&quot;FAccTRec@RecSys&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排/重排稳定性微调&quot;,&quot;Rank-Preserving Regularization&quot;,&quot;Training Data Perturbation&quot;,&quot;Rank List Stability (RBO)&quot;]"><strong><a href="../../venues/TKDD/2024/debias_fairness_industrial/finest_stabilizing_recommendations_by_rank_preserving_fine_tuning/">FINEST: Stabilizing Recommendations by Rank-Preserving Fine-Tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Knowledge Discovery from Data">ACM TKDD</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用伪扰动模拟+top-K 保序正则微调序列推荐器，CASPER 扰动下 RBO 最高 +24%、Jaccard +136% 且精度不降</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;字典学习&quot;,&quot;动态用户偏好&quot;,&quot;GRU+Attention&quot;]"><strong><a href="../../venues/TKDE/2022/sequential_rec/modeling_dynamic_user_preference_via_dictionary_learning_for_sequential_rec/">Modeling Dynamic User Preference via Dictionary Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>S2PNM以共享字典把GRU序列编码译为动态偏好，与静态偏好同空间相加；Netflix时间切分RMSE较SVD++降2.18%，Baby Care NDCG@5较SASRec升6.10%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;分块记忆网络&quot;,&quot;外部记忆&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/cmnrec_sequential_recommendations_with_chunk_accelerated_memory_network/">CmnRec: Sequential Recommendations with Chunk-accelerated Memory Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以认知分块压缩邻近隐状态，仅在 chunk 时刻读写外部记忆，训练最高约 6.34×、推理 10.07× 加速且 MRR@5 不降反升。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端到端生成式&quot;,&quot;替代行程生成&quot;,&quot;MCMC采样&quot;]"><strong><a href="../../venues/TKDE/2023/poi_spatiotemporal_rec/deepalttrip_top_k_alternative_itineraries_for_trip_recommendation/">DeepAltTrip: Top-k Alternative Itineraries for Trip Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>双图自编码器+双向LSTM学习历史路线，无需显式流行度/多样性建模即可生成k条既热门又互不相似的替代行程，组合指标最高相对提升29.24%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;解耦表征学习&quot;,&quot;会话图神经网络&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/disentangled_graph_neural_networks_for_session_based_recommendation/">Disentangled Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将物品嵌入解耦为 K 个因子并用因子级相似度 GGNN 传播、距离相关解耦，按因子注意力聚合会话表征，三数据集 P@20 最高 +16.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;动态图神经网络&quot;,&quot;跨序列协同信号&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/dynamic_graph_neural_networks_for_sequential_recommendation/">Dynamic Graph Neural Networks for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将全部用户序列编织成带时间戳与双向顺序的五元组动态图，用相对顺序感知注意力+短期注意力做子图消息传播，把 next-item 预测转为动态图链路预测，Beauty 上 NDCG@10 相对提升 10.56%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;多行为序列推荐&quot;,&quot;时间图Transformer&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/multi_behavior_sequential_recommendation_with_temporal_graph_transformer/">Multi-Behavior Sequential Recommendation with Temporal Graph Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>行为感知Transformer与时间图联合建模多行为短长期依赖，HR@10较MBGCN提升22.5%/12.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回/Next-Item排序&quot;,&quot;共享账户&quot;,&quot;跨域序列推荐&quot;,&quot;角色拆分&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/parallel_split_join_networks_for_shared_account_cross_domain_sequential_rec/">Parallel Split-Join Networks for Shared-account Cross-domain Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TKDE 2023 [序列与会话推荐] 提出 PSJNet，以角色拆分与跨域联结解决共享账户跨域序列推荐，全库排序下 MRR/Recall 显著超越 GRU4REC/HGRU4REC</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列推荐&quot;,&quot;排序&quot;,&quot;强化学习路径探索&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/reinforcement_learning_based_path_exploration_for_sequential_explainable_re/">Reinforcement Learning based Path Exploration for Sequential Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>TMER-RL 用强化学习探索连续物品间元路径，并以双注意力融合时序与路径解释，在 4 个数据集 500 负采样排序上显著优于 TMER。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;会话推荐&quot;,&quot;解耦图神经网络&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TKDE/2023/sequential_rec/transition_information_enhanced_disentangled_graph_neural_networks_for_sess/">Transition Information Enhanced Disentangled Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>位置感知全局图+解耦因子GNN，双通道对比学习建模会话转移，Tmall P@20提升11.2%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;增量学习&quot;,&quot;多意图解耦&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/a_framework_for_elastic_adaptation_of_user_multiple_intents_in_sequential_r/">A Framework for Elastic Adaptation of User Multiple Intents in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 IMA/EMA 增量多意图自适应框架：蒸馏损失保旧意图、困惑度检测新建意图、正交投影裁剪冗余意图、活跃度/聚类弹性压缩，4 数据集 NDCG 较最优增量基线提升 3.8–4.8%，省 27.3% 内存。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多模态会话推荐&quot;,&quot;概率价格建模&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/beyond_co_occurrence_multi_modal_session_based_recommendation/">Beyond Co-occurrence: Multi-modal Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用伪模态对比、分层枢轴Transformer与Wasserstein自注意力统一建模图像/文本/价格，Amazon三域全库排序Prec@20提升1.45%–3.14%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;连续时间记忆网络&quot;,&quot;时序集合预测&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/continuous_time_user_preference_modelling_for_temporal_sets_prediction/">Continuous-Time User Preference Modelling for Temporal Sets Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>连续时间记忆库显式建模用户偏好演化，全局序列+set-batch，全库Top-K评估，JingDong Recall@10提升11.48%</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;神经常微分方程&quot;,&quot;动态图推荐&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/learning_graph_ode_for_continuous_time_sequential_recommendation/">Learning Graph ODE for Continuous-Time Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>GDERec 用自回归图 ODE 建模用户-物品交互图的连续演化，配时间注意力聚合不规则采样，5 数据集全库排序 R@10 提升约 8.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;Prompt-Tuning&quot;,&quot;冷启动&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TKDE/2024/sequential_rec/personalized_prompt_for_sequential_recommendation/">Personalized Prompt for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用用户画像生成个性化软前缀 Prompt 替代全参微调，配合 Prompt 导向对比学习，在三大数据集的 Few-shot/Zero-shot 冷启动上稳定超越 Fine-tuning。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;Prompt Learning&quot;,&quot;非重叠跨域序列推荐&quot;,&quot;分离约束&quot;]"><strong><a href="../../venues/TKDE/2025/cross_domain_multi_scenario/automated_prompting_for_non_overlapping_cross_domain_sequential_recommendat/">Automated Prompting for Non-overlapping Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用可学习连续 Prompt（域无关+域特定+冻结标签）替代域对齐约束，实现无重叠实体、仅隐式反馈的双目标跨域序列推荐，Movie-Book 上 HR@10 相对最优基线提升约 17.8%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;解耦参数高效微调&quot;,&quot;非对称多模态基础模型&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/TKDE/2025/multimodal_rec/efficient_and_effective_adaptation_of_multimodal_foundation_models_in_seque/">Efficient and Effective Adaptation of Multimodal Foundation Models in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>IISAN-Versa 用解耦 PEFT + 组 LayerDrop + 维度变换层，统一适配对称/非对称多模态基础模型，缓存隐藏态后 GPU 显存降 12–15 倍、训练提速 16–20 倍，MicroLens 达 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;知识图谱路径推理&quot;,&quot;分层强化学习&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/explainable_session_based_recommendation_via_path_reasoning/">Explainable Session-based Recommendation via Path Reasoning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出分层强化学习框架 PR4SR，用会话级智能体选起点、路径级智能体做知识图谱推理，配合多目标与路径中点奖励，为 5 种 SR 模型提供可解释路径并提升精度。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双向时序数据增强&quot;,&quot;伪先验物品生成&quot;,&quot;知识增强微调&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/improving_sequential_recommendations_via_bidirectional_temporal_data_augmen/">Improving Sequential Recommendations via Bidirectional Temporal Data Augmentation with Pre-training</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>BARec 用双向时序增强预训练生成保偏好伪先验物品，并以 KL 知识增强微调对齐原/增广序列分布，全库排序下 Beauty Recall@5 提升 16.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;概率嵌入&quot;,&quot;逻辑合取算子&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/TKDE/2025/sequential_rec/pattern_wise_transparent_sequential_recommendation/">Pattern-wise Transparent Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多尺度滑窗切出点级/联合级模式，以概率嵌入+逻辑合取建模模式，KL散度+距离权重实现可解释且超越黑盒的序列推荐。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;上下文注意力&quot;,&quot;堆叠多头注意力&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/an_adaptive_context_aware_and_stacked_attention_network_based_recommendatio/">An Adaptive, Context-Aware, and Stacked Attention Network-Based Recommendation System to Capture Users&#x27; Temporal Preference</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用上下文物品注意力+堆叠多头用户注意力捕捉用户时变偏好，Leave-One-Out+100负采样排序，三数据集HR/NDCG/MRR@10全面超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回/端到端&quot;,&quot;数据增强&quot;,&quot;自监督辅助任务&quot;,&quot;扩散/LLM 生成&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/data_augmentation_for_sequential_recommendation_a_survey/">Data Augmentation for Sequential Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首篇以数据增强为主角的序列推荐综述，提出原则×对象×目的三维分类与双训练范式形式化，三集 Full-Ranking 复现最高提升 66.0%。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣聚类&quot;,&quot;稳定性-可塑性解耦&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
</dd>
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;推理时计算 (Test-Time Scaling)&quot;,&quot;隐式思维链 (Latent CoT)&quot;,&quot;推理位置编码&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/think_before_recommend_unleashing_the_latent_reasoning_power_for_sequential/">Think Before Recommend: Unleashing the Latent Reasoning Power for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出首个推理时计算框架 ReaRec，将末位隐状态自回归回灌实现隐式多步推理，配合 ERL/PRL 两种轻量学习策略，全库排序下平均提升 7.49%、推理延迟仅增 3.51%。</p>
</dd>
<dt data-paper-tags="[&quot;TMM&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;多级图神经网络&quot;,&quot;超图&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/TMM/2023/sequential_rec/contrastive_multi_level_graph_neural_networks_for_session_based_recommendat/">Contrastive Multi-Level Graph Neural Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Multimedia">IEEE TMM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 L/G/H 三级图与跨层级 InfoNCE，将高阶超图监督注入成对会话表示，在 Tmall/Diginetica/Nowplaying 全库排序上多数指标领先。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;位置编码&quot;,&quot;会话图神经网络&quot;]"><strong><a href="../../venues/TOIS/2022/sequential_rec/exploiting_positional_information_for_session_based_recommendation/">Exploiting Positional Information for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出双位置编码DPE/LDPE与位置感知门控图网络PosRec，全库排序下在Yoochoose/Diginetica稳定超越SOTA约0.4%–2.6%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;元学习 (MAML)&quot;,&quot;序列推荐&quot;,&quot;图扩散表征&quot;]"><strong><a href="../../venues/TOIS/2022/cold_start_rec/learning_to_learn_a_cold_start_sequential_recommender/">Learning to Learn a Cold-start Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>metaCSR 用扩散表征+自注意力序列编码+MAML 元学习，仅凭 ID 与交互矩阵实现新用户少样本快速适配，冷启动 AUC 最高 +7.85%、MAP +29.62%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回与推荐评估&quot;,&quot;下一购物篮推荐&quot;,&quot;重复-探索分解&quot;,&quot;可复现评估&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/a_next_basket_recommendation_reality_check/">A Next Basket Recommendation Reality Check</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>复现三类NBR方法，提出重复/探索指标与GP-TopFreq基线，三数据集全物品Top-K显示深度模型无一致优势。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;端侧推理/模型压缩&quot;,&quot;组合编码&quot;,&quot;知识蒸馏&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/efficient_on_device_session_based_recommendation/">Efficient On-Device Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以组合编码压缩物品嵌入并用双向自监督蒸馏保容，30×压缩下实现6–8×端侧推理加速且精度不降。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;偏好编辑自监督&quot;,&quot;多偏好表征&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/improving_transformer_based_sequential_recommenders_through_preference_edit/">Improving Transformer-based Sequential Recommenders through Preference Editing</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 K 个特殊 token 表征多偏好，并以“偏好分离-重组”自监督构造跨序列信号，Beauty/ML-100k 上 Recall@20 提升显著</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信隐私&quot;,&quot;召回&quot;,&quot;隐私推断攻击&quot;,&quot;曝光数据保护&quot;]"><strong><a href="../../venues/TOIS/2023/debias_fairness_industrial/on_the_user_behavior_leakage_from_recommender_system_exposure/">On the User Behavior Leakage from Recommender System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>揭示系统曝光序列可反推用户历史点击的隐私漏洞，用编码器-解码器黑盒攻击在 Zhihu/MIND 上实现最高 87.72% Recall@10，并提出两阶段曝光替换保护机制。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回+排序(全链路综述)&quot;,&quot;新闻文本表征&quot;,&quot;点击噪声与责任化推荐&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/personalized_news_recommendation_methods_and_challenges/">Personalized News Recommendation: Methods and Challenges</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以核心问题重构新闻推荐综述框架，梳理新闻/用户建模、排序、训练与评测技术，指出冷启动、点击噪声与离线-在线鸿沟三大挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时间感知图卷积&quot;,&quot;高阶连通性&quot;,&quot;自注意力聚合&quot;]"><strong><a href="../../venues/TOIS/2023/sequential_rec/position_enhanced_and_time_aware_graph_convolutional_network_for_sequential/">Position-enhanced and Time-aware Graph Convolutional Network for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>在用户-物品二部图上定义位置增强/时间感知图卷积与自注意力聚合器，同步建模用户与物品的时序动态并堆叠多层捕获高阶连通性，Movielens/Amazon R@10 较 DGCF 提升约 9-11%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;多粒度签到信号&quot;,&quot;区域感知通道加权&quot;]"><strong><a href="../../venues/TOIS/2024/poi_spatiotemporal_rec/a_multi_channel_next_poi_recommendation_framework_with_multi_granularity_ch/">A Multi-Channel Next POI Recommendation Framework with Multi-Granularity Check-in Signals</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MCMG 用 POI/区域/类目三通道自注意力 + 全局 POI 图 GCN，并按同区域/跨区域分组自适应加权融合，四城市 HR/NDCG 平均提升 9.7%/14.2%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;异质超图&quot;,&quot;双偏好学习&quot;]"><strong><a href="../../venues/TOIS/2024/sequential_rec/bi_preference_learning_heterogeneous_hypergraph_networks_for_session_based_/">Bi-Preference Learning Heterogeneous Hypergraph Networks for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用类别内逻辑斯谛价格分档与异质超图三层卷积，联合建模价格/兴趣双偏好，在三个会话数据集全库排序下稳定超越强基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;全库排序/端到端协同过滤&quot;,&quot;因果解耦&quot;,&quot;偏好漂移&quot;,&quot;时序VAE&quot;]"><strong><a href="../../venues/TOIS/2024/debias_fairness_industrial/causal_disentangled_recommendation_against_user_preference_shifts/">Causal Disentangled Recommendation Against User Preference Shifts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出CDR，以时序VAE推断未观测因素并建模多环境偏好漂移，用可学习稀疏矩阵解耦类别偏好；全库排序下R@10较次优提升19.5%–28.1%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;分布表征&quot;,&quot;不确定性注入&quot;]"><strong><a href="../../venues/TOIS/2024/sequential_rec/diffurec_a_diffusion_model_for_sequential_recommendation/">DiffuRec: A Diffusion Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首次将扩散模型引入序列推荐，用分布表征替代固定物品向量并注入不确定性，Leave-One-Out + Full-Ranking 协议下在四个数据集 HR/NDCG 最高提升 57.26%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨视图对比学习&quot;,&quot;多关系会话图&quot;,&quot;协同过滤图&quot;]"><strong><a href="../../venues/TOIS/2024/sequential_rec/spatio_temporal_contrastive_learning_enhanced_gnns_for_session_based_recomm/">Spatio-Temporal Contrastive Learning Enhanced GNNs for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RESTC 用空间(MGAT)与时间(SESTrans)双视图对比学习对齐会话表征，并以全局协同过滤图补足短会话稀疏性，六数据集全库排序 HR@20 最高 +27.08%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;三重序列建模&quot;,&quot;对比学习&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="../../venues/TOIS/2024/cross_domain_multi_scenario/triple_sequence_learning_for_cross_domain_recommendation/">Triple Sequence Learning for Cross-domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>Tri-CDR 联合建模源域/目标域/混合三条行为序列，用三重跨域注意力与三重对比学习（粗粒度对齐+细粒度区分）实现正向迁移，六组跨域设定全库排序一致提升。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/重排评估&quot;,&quot;会话多样性&quot;,&quot;准确率-多样性关系&quot;]"><strong><a href="../../venues/TOIS/2024/sequential_rec/understanding_diversity_in_session_based_recommendation/">Understanding Diversity in Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>统一16个会话推荐基线的准确率/多样性评测，揭示非多样化模型亦可高多样性，且准确率-多样性关系随模型与数据变化。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;ID-centric Pre-training&quot;,&quot;Cross-domain ID Matcher&quot;,&quot;行为-文本对比对齐&quot;]"><strong><a href="../../venues/TOIS/2025/cold_start_rec/id_centric_pre_training_for_recommendation/">ID-centric Pre-training for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出以 ID 嵌入为中心的多域预训练范式 IDP，用行为感知跨域 ID 匹配器 CDIM 将预训练 ID 嵌入迁移至新域物品，冷/热启动均显著超越 UniSRec 等基线。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;软标签&quot;,&quot;知识蒸馏&quot;,&quot;去噪鲁棒性&quot;]"><strong><a href="../../venues/TOIS/2025/sequential_rec/learning_robust_sequential_recommenders_through_confident_soft_labels/">Learning Robust Sequential Recommenders through Confident Soft Labels</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用模型级/数据级/训练级三种教师模块生成高置信软标签，与 one-hot 联合训练学生序列推荐器，全库排序下 Recall@10 最高提升约 11%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;隐私保护&quot;,&quot;序列混淆&quot;,&quot;自监督修改器&quot;]"><strong><a href="../../venues/TOIS/2025/sequential_rec/privacy_preserving_sequential_recommendation_with_collaborative_confusion/">Privacy-Preserving Sequential Recommendation with Collaborative Confusion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用相似序列引导的逐项修改器在推荐前混淆交互序列，66% 修改率下推荐精度仍超 99% SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;生成式推荐&quot;,&quot;联邦学习&quot;,&quot;指数机制&quot;,&quot;对比去噪&quot;]"><strong><a href="../../venues/TOIS/2025/sequential_rec/ptf_fsr_a_parameter_transmission_free_federated_sequential_recommender_syst/">PTF-FSR: A Parameter Transmission-Free Federated Sequential Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用指数机制生成扰动序列替代参数传输，实现模型隐私与轻量通信的联邦序列推荐框架，通信降至 KB 级。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;行为-文本适配&quot;,&quot;掩码物品文本预测&quot;,&quot;多域预训练&quot;]"><strong><a href="../../venues/TOIS/2025/llm_rec/thoroughly_modeling_multi_domain_pre_trained_recommendation_as_language/">Thoroughly Modeling Multi-domain Pre-trained Recommendation as Language</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将多域序列推荐彻底转为文本生成任务，用五项指标指导行为-文本双向适配，7 数据集微调与零样本均最优，NDCG@1 最高提升 23%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;提示鲁棒性&quot;,&quot;生成对抗数据增广&quot;,&quot;协同信号注入&quot;]"><strong><a href="../../venues/TOIS/2026/llm_rec/ganprompt_enhancing_robustness_in_llm_based_recommendations_with_gan_enhanc/">GANPrompt: Enhancing Robustness in LLM-Based Recommendations with GAN-Enhanced Diversity Prompts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 T5 编码器作生成器、MLP 作判别器做属性条件化对抗训练，配合余弦+JS 双约束扩展提示多样性，再以专家引导 token 注入 DIN 协同信号微调 LLaMA2-7B；Amazon 五数据集 Leave-One-Out + 1:9 采样排序下 AUC 最高 +2.06%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;状态空间模型&quot;]"><strong><a href="../../venues/TOIS/2026/sequential_rec/hmamba_hyperbolic_mamba_for_sequential_recommendation/">HMamba: Hyperbolic Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 Mamba 选择性状态空间搬进 Lorentz 双曲空间，用曲率感知离散化与陀螺向量算子建模层级+时序，在 4 个基准 HR/NDCG 上提升 3–11% 且保持线性复杂度。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;隐空间对齐&quot;,&quot;序列残差提示&quot;]"><strong><a href="../../venues/TOIS/2026/llm_rec/large_language_model_can_interpret_latent_space_of_sequential_recommender/">Large Language Model Can Interpret Latent Space of Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>线性适配器将序列推荐隐表示映射到LLaMA，以恢复/残差提示解读隐空间；MovieLens/Steam残差识别最高97.89%/86.33%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;数据预处理（模型无关前置模块）&quot;,&quot;LLM 指令微调&quot;,&quot;不确定度估计&quot;,&quot;序列去噪&quot;]"><strong><a href="../../venues/TOIS/2026/sequential_rec/llm4dsr_leveraging_large_language_model_for_denoising_sequential_recommenda/">LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自监督指令微调激活 LLM 的序列去噪能力，并以 token 级不确定度阈值筛选高置信噪声、生成式补全替换，作为模型无关的数据预处理在三个 Amazon 数据集全库排序下稳定超越 HSD/STEAM/CL4SRec。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;表征坍缩与多样性正则&quot;]"><strong><a href="../../venues/TOIS/2026/sequential_rec/multi_interest_recommendation_a_survey/">Multi-Interest Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首篇多兴趣推荐综述，以隐式/显式建模分类统摄 172 篇文献，把技术栈归一为“兴趣抽取器+兴趣聚合器”两段式框架，并系统梳理去坍缩正则与六大开放挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;State Space Model&quot;,&quot;变长序列建模&quot;]"><strong><a href="../../venues/TOIS/2026/sequential_rec/ssd4rec_a_structured_state_space_duality_model_for_efficient_sequential_rec/">SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Mamba 的 Structured State Space Duality 块替代自注意力，配合变长拼接+段寄存器与双向翻转，在 4 个基准上取得 NDCG@10 最高 18.99% 提升并保持近线性复杂度。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回+重排&quot;,&quot;意图感知推荐&quot;,&quot;综述与分类体系&quot;]"><strong><a href="../../venues/TORS/2025/sequential_rec/a_survey_on_intent_aware_recommender_systems/">A Survey on Intent-aware Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理意图感知推荐(IARS)的三大技术路线与评估实践，指出离线精度导向、可复现性不足与显式意图建模缺失三大空白。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;跨会话建模&quot;,&quot;删失Weibull损失&quot;,&quot;多类型动作序列&quot;]"><strong><a href="../../venues/TORS/2025/sequential_rec/recommending_target_actions_outside_sessions_in_the_data_poor_insurance_dom/">Recommending Target Actions Outside Sessions in the Data-poor Insurance Domain</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>面向保险数据稀缺场景的跨会话 RNN 推荐，预测会话外购买目标，Full-Ranking 下 HR@3 较 SKNN_EB 提升 3.11%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;精排&quot;,&quot;可解释推荐&quot;,&quot;主题模型&quot;,&quot;注意力机制&quot;]"><strong><a href="../../venues/TORS/2025/agent_conversational_rec/topic_centric_explanations_for_news_recommendation/">Topic-Centric Explanations for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用双层注意力主题模型 BATM 同时做新闻推荐与主题级解释，MIND 全库排序 AUC 提升约 0.4-0.5 个点，并用 NPMI/W2V 量化主题可解释性。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;精排打分&quot;,&quot;Mixture-of-Experts&quot;,&quot;长短期偏好融合&quot;]"><strong><a href="../../venues/TORS/2026/sequential_rec/compositions_of_variant_experts_for_integrating_short_term_and_long_term_pr/">Compositions of Variant Experts for Integrating Short-Term and Long-Term Preferences</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 BPR/GRU4Rec 异构专家 + 可学习门控动态融合用户长短期偏好，全库排序下 MRR 最高提升 38.03%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;召回与排序&quot;,&quot;LLM4Rec&quot;,&quot;LLM Embeddings&quot;,&quot;Sequential Recommendation&quot;]"><strong><a href="../../venues/TORS/2026/sequential_rec/improving_sequential_recommendations_with_llms/">Improving Sequential Recommendations with LLMs</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统比较 LLM 嵌入初始化、提示微调与混合路由；Beauty/DH 全库排序 NDCG@20 最高 +45%/+9%。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;序列与会话推荐&quot;,&quot;端到端生成式&quot;,&quot;捆绑生成&quot;,&quot;强化学习策略&quot;,&quot;LLM 意图分解&quot;]"><strong><a href="../../venues/TORS/2026/sequential_rec/text2bundle_towards_personalized_query_based_bundle_generation/">Text2Bundle: Towards Personalized Query-based Bundle Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出查询驱动捆绑生成任务 QBG，用 LLM 意图分解 + 统一状态编码 + DQN 逐步选品，集合级 P/R/F1 上精确率较 BYOB† 提升约 26–32%。</p>
</dd>
<dt data-paper-tags="[&quot;TWEB&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构图神经网络&quot;,&quot;多行为会话建模&quot;]"><strong><a href="../../venues/TWEB/2024/sequential_rec/heterogeneous_information_crossing_on_graphs_for_session_based_recommender_/">Heterogeneous Information Crossing on Graphs for Session-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on the Web">ACM TWEB</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>会话多行为异构图 + 跨行为类型边 + 双层注意力交叉，全库排序下 HR@20 相对最优基线提升约 3.5%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;超长方体用户表示&quot;,&quot;序列兴趣建模&quot;,&quot;键值记忆网络&quot;]"><strong><a href="../../venues/WSDM/2021/collaborative_filtering/learning_user_representations_with_hypercuboids_for_recommender_systems/">Learning User Representations with Hypercuboids for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将用户由点嵌入升级为超长方体（中心+非负偏移），以内外复合距离度量用户-物品关系，全库排序下 Alibaba 线上数据 Recall@50 提升 8.5%–14.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;细粒度特征交互&quot;,&quot;交互注意力&quot;,&quot;Pre-LN Transformer&quot;]"><strong><a href="../../venues/WSDM/2021/ctr_feature_interaction/multi_interactive_attention_network_for_fine_grained_feature_learning_in_ct/">Multi-Interactive Attention Network for Fine-grained Feature Learning in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>MIAN 用 Pre-LN Transformer 编码行为序列，并对候选物品与每个细粒度用户/上下文属性做逐字段注意力交互，Amazon 与支付宝数据 AUC 提升 0.37%–0.87%，线上 CTR +0.41%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;时空轨迹与 POI 推荐&quot;,&quot;召回&quot;,&quot;Origin-Aware 下一目的地推荐&quot;,&quot;ST-LSTM 时空细胞状态&quot;,&quot;个性化偏好注意力&quot;]"><strong><a href="../../venues/WSDM/2021/poi_spatiotemporal_rec/origin_aware_next_destination_recommendation_with_personalized_preference_a/">Origin-Aware Next Destination Recommendation with Personalized Preference Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>提出 ST-LSTM 编码器与个性化偏好注意力解码器，将起点信息与局部/全局时空因子注入 OD 关系建模，七个 Grab 真实轨迹数据集 Acc@1 相对 SOTA 提升约 13%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2021&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;稀疏兴趣激活&quot;,&quot;多兴趣表征&quot;,&quot;意图聚合&quot;]"><strong><a href="../../venues/WSDM/2021/sequential_rec/sparse_interest_network_for_sequential_recommendation/">Sparse-Interest Network for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用大规模概念原型池做可微 top-K 稀疏兴趣激活，再以主动意图预测聚合多兴趣向量，Taobao HR@50 相对 MIND 提升 15.2%，工业 4B 数据 HR@50 提升 34.06%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;元路径注意力&quot;,&quot;可解释推荐&quot;]"><strong><a href="../../venues/WSDM/2021/sequential_rec/temporal_meta_path_guided_explainable_recommendation/">Temporal Meta-path Guided Explainable Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用相邻物品间 item-item 元路径替代 RNN 建模时序依赖，配合双层注意力与 MLP，在 Amazon 三数据集上 HR@1 提升 30%+。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;表示退化 (Anisotropy)&quot;,&quot;对比学习正则 (Dropout 增广 / 监督正样本)&quot;]"><strong><a href="../../venues/WSDM/2022/sequential_rec/contrastive_learning_for_representation_degeneration_problem_in_sequential_/">Contrastive Learning for Representation Degeneration Problem in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>揭示 Transformer 序列推荐器的物品嵌入各向异性退化问题，用 Dropout 双视图与同目标难正样本的对比正则（DuoRec）在全库排序下 HR@5 最高提升 104%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;异构全局图&quot;,&quot;个性化会话编码&quot;]"><strong><a href="../../venues/WSDM/2022/sequential_rec/heterogeneous_global_graph_neural_networks_for_personalized_session_based_r/">Heterogeneous Global Graph Neural Networks for Personalized Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>构建用户-物品异构全局图并叠加双粒度偏好门控，在 Last.fm/Xing/Reddit 全库 HR/MRR 上相对 SOTA 提升 1.0%–6.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时序元学习&quot;,&quot;异构图注意力&quot;]"><strong><a href="../../venues/WSDM/2022/sequential_rec/long_short_term_temporal_meta_learning_in_online_recommendation/">Long Short-Term Temporal Meta-learning in Online Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>双图(全局长期/内部短期)+门控融合+时序 MAML 异步优化，在微信看一看 10 亿级行为上做小时级 AUC 与线上 A/B 验证，CTR 相对 +9.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2022&quot;,&quot;序列与会话推荐&quot;,&quot;重排&quot;,&quot;Transformer&quot;,&quot;Attention2D&quot;,&quot;多属性序列&quot;]"><strong><a href="../../venues/WSDM/2022/sequential_rec/sequential_modeling_with_multiple_attributes_for_watchlist_recommendation_i/">Sequential Modeling with Multiple Attributes for Watchlist Recommendation in E-Commerce</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出Trans2D/Attention2D，在item×attribute二维序列上建模三类交互，用于eBay watchlist重排；P@1较最佳基线提升11.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多层注意力混合&quot;,&quot;归纳偏置剪枝&quot;,&quot;会话表征读出&quot;]"><strong><a href="../../venues/WSDM/2023/sequential_rec/efficiently_leveraging_multi_level_user_intent_for_session_based_recommenda/">Efficiently Leveraging Multi-level User Intent for Session-based Recommendation via Atten-Mixer Network</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 SparseVD 稀疏化证据证明会话推荐中 GNN 传播冗余，提出多层意图查询生成 + Lp 池化混合的 Atten-Mixer 读出门控，三数据集 HR@20 全面提升且训练快 7 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Neural Processes&quot;,&quot;兴趣动态建模&quot;,&quot;少样本序列推荐&quot;]"><strong><a href="../../venues/WSDM/2023/sequential_rec/idnp_interest_dynamics_modeling_using_generative_neural_processes_for_seque/">IDNP: Interest Dynamics Modeling using Generative Neural Processes for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用空洞卷积+自注意力编码多尺度短期兴趣，以神经过程把长期兴趣视为连续函数实例，在四数据集少样本设定下大幅超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;属性模拟&quot;,&quot;物品共现矩阵&quot;,&quot;多兴趣建模&quot;]"><strong><a href="../../venues/WSDM/2024/collaborative_filtering/attribute_simulation_for_item_embedding_enhancement_in_multi_interest_recom/">Attribute Simulation for Item Embedding Enhancement in Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用物品共现矩阵近似物品-属性关联矩阵，提出即插即用的 SimEmb 模块替代 ID Embedding 做多兴趣召回，4 个数据集全库排序 Recall@20 平均相对提升 25.59%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回&quot;,&quot;Distributionally Robust Optimization&quot;,&quot;Exposure Bias&quot;,&quot;SNIPS&quot;]"><strong><a href="../../venues/WSDM/2024/debias_fairness_industrial/debiasing_sequential_recommenders_through_distributionally_robust_optimizat/">Debiasing Sequential Recommenders through Distributionally Robust Optimization over System Exposure</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>以系统曝光模拟器构造 DRO 名义分布，叠加高曝光惩罚与 SNIPS 评估器，SNIPS 全库排序下 Recall@20 最高 +25.9%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;训练阶段防护&quot;,&quot;模型抽取攻击&quot;,&quot;可微交换矩阵&quot;]"><strong><a href="../../venues/WSDM/2024/debias_fairness_industrial/defense_against_model_extraction_attacks_on_recommender_systems/">Defense Against Model Extraction Attacks on Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>首个推荐模型抽取防御 GRO：以可微交换矩阵反传 top-k 梯度，用学生模型模拟攻击者并最大化其损失，在全库排序下压低代理模型效用且保持目标模型效用。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;意图对比学习&quot;,&quot;跨子序列监督&quot;]"><strong><a href="../../venues/WSDM/2024/sequential_rec/intent_contrastive_learning_with_cross_subsequences_for_sequential_recommen/">Intent Contrastive Learning with Cross Subsequences for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用同一目标物品的跨序列子序列构造粗/细粒度意图对比监督，全库排序 NDCG@20 最高提升 44.4%</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;线性递归单元&quot;,&quot;递归并行化&quot;,&quot;增量推理&quot;]"><strong><a href="../../venues/WSDM/2024/sequential_rec/linear_recurrent_units_for_sequential_recommendation/">Linear Recurrent Units for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用线性递归单元替代自注意力，递归并行训练与O(1)增量推理，全库排序平均提升4.39%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;组原型注意力&quot;,&quot;跨域序列推荐&quot;]"><strong><a href="../../venues/WSDM/2024/cross_domain_multi_scenario/mixed_attention_network_for_cross_domain_sequential_recommendation/">Mixed Attention Network for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用局部/全局双通道编码 + 物品相似度/序列融合/组原型三重注意力，在无重叠用户假设下做双目标跨域序列推荐，稀疏域 AUC 最高 +8.25%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;机器遗忘&quot;,&quot;会话分片聚类&quot;,&quot;隐私保护&quot;]"><strong><a href="../../venues/WSDM/2024/sequential_rec/on_the_effectiveness_of_unlearning_in_session_based_recommendation/">On the Effectiveness of Unlearning in Session-Based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 SRU：相似会话聚片 + 会话-质心双源注意力聚合 + 协同/邻居/随机额外删除，并定义 HIT@K 遗忘有效性度量，遗忘重训提速 3.7× 且精度反超 SISA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2024&quot;,&quot;冷启动与长尾推荐&quot;,&quot;精排&quot;,&quot;代理嵌入&quot;,&quot;参数高效&quot;]"><strong><a href="../../venues/WSDM/2024/cold_start_rec/proxy_based_item_representation_for_attribute_and_context_aware_recommendat/">Proxy-based Item Representation for Attribute and Context-aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用代理嵌入加权和替代物品查表，面向属性/上下文序列推荐，Leave-One-Out+100负采样下最高约17%提升且仅用10%参数。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;多目标强化学习&quot;,&quot;多样性正则&quot;,&quot;序列与会话推荐&quot;]"><strong><a href="../../venues/WSDM/2025/rl_bandit_rec/choosing_the_best_of_both_worlds_diverse_and_novel_recommendations_through_/">Choosing the Best of Both Worlds: Diverse and Novel Recommendations through Multi-Objective Reinforcement Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用标量化多目标 RL 头正则化序列推荐主干，同时优化准确率/多样性/新颖性，聚合覆盖度提升约 20% 且准确率不降。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多头子嵌入预测&quot;,&quot;混合专家网络&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/facet_aware_multi_head_mixture_of_experts_model_for_sequential_recommendati/">Facet-Aware Multi-Head Mixture-of-Experts Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用多头子嵌入独立预测+头内 MoE 解耦用户偏好，在 4 个 Amazon/ML-20m 数据集 Full-Ranking 下 NDCG@20 最高提升 8.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Oracle 引导对齐&quot;,&quot;频域低通去噪&quot;,&quot;两阶段训练&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/oracle_guided_dynamic_user_preference_modeling_for_sequential_recommendatio/">Oracle-guided Dynamic User Preference Modeling for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用未来交互作为后验“Oracle”引导过去编码器学习前瞻表征，配无参低通滤波去噪与两阶段训练，六数据集采样排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;套装兼容性建模&quot;,&quot;监督对比学习&quot;,&quot;自适应间隔损失&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/personalised_outfit_recommendation_via_history_aware_transformers/">Personalised Outfit Recommendation via History-aware Transformers</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>双层 Transformer 堆叠建模套装内部兼容性与用户购买史，配合监督对比与自适应间隔损失实现个性化套装打分，CP-Hard AUC 提升 15.7%/19.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排&quot;,&quot;负采样 Q-learning&quot;,&quot;Advantage Actor-Critic&quot;]"><strong><a href="../../venues/WSDM/2025/rl_bandit_rec/supervised_advantage_actor_critic_for_recommender_systems/">Supervised Advantage Actor-Critic for Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>为 RL 头引入负采样构造 SNQN，并用采样优势 A2C 重加权监督头，在 RC15/RetailRocket 上显著超越 SQN/SAC。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性 item-item 模型&quot;,&quot;时间间隔加权&quot;,&quot;趋势感知归一化&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/temporal_linear_item_item_model_for_sequential_recommendation/">Temporal Linear Item-Item Model for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在闭式解线性 item-item 模型上引入时间间隔加权与趋势感知归一化，Full-Ranking 下平均 NDCG@10 提升 9.07%，训练快 6.9–57 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因果自注意力&quot;,&quot;注意力矩阵谱分析&quot;]"><strong><a href="../../venues/WSDM/2025/sequential_rec/your_causal_self_attentive_recommender_hosts_a_lonely_neighborhood/">Your Causal Self-Attentive Recommender Hosts a Lonely Neighborhood</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>从稀疏局部归纳偏置与低秩近似两角度理论剖析 AE/AR 自注意力，并在 5 个基准上以 ModSAR 流水线实证 AR 全面占优。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;CTR预估与特征交叉&quot;,&quot;精排&quot;,&quot;自监督预训练&quot;,&quot;序列用户建模&quot;]"><strong><a href="../../venues/WSDM/2026/ctr_feature_interaction/abacus_self_supervised_event_counting_aligned_distributional_pretraining_fo/">Abacus: Self-Supervised Event Counting-Aligned Distributional Pretraining for Sequential User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以事件频率分布预测为自监督前置任务，多任务融合掩码建模与对比学习，替代人工计数特征，购买预测 AUC 最高 +6.1%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;序列压缩&quot;,&quot;类别分桶&quot;]"><strong><a href="../../venues/WSDM/2026/generative_rec/accelerating_generative_recommendation_via_simple_categorical_user_sequence/">Accelerating Generative Recommendation via Simple Categorical User Sequence Compression</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用物品类别桶压缩长历史序列，在同等算力下较 HSTU 最高提升 39% 精度、最高降低 6 倍计算开销。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性注意力&quot;,&quot;持续学习&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;双条件引导&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/dual_conditional_diffusion_models_for_sequential_recommendation/">Dual Conditional Diffusion Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DCRec 在扩散正反向过程双通道注入隐式拼接序列与显式交叉注意力条件，Beauty/Toys/Yelp 全库排序 HR@5 最高提升 17.35%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;语义ID&quot;,&quot;前瞻预测&quot;,&quot;直播推荐&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/foresight_prediction_enhanced_live_streaming_recommendation/">Foresight Prediction Enhanced Live-Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将直播切片经 LLM 语义量化成 Sid 序列，用 Transformer 编解码预测下一段内容，作为前瞻特征注入多任务精排，线上曝光 +0.46%、送礼 +2.48%。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;协作信息量化定义&quot;,&quot;协作关系分布实证&quot;,&quot;序列推荐评测&quot;]"><strong><a href="../../venues/WSDM/2026/collaborative_filtering/have_we_really_understood_collaborative_information_an_empirical_investigat/">Have We Really Understood Collaborative Information? An Empirical Investigation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以物品共现路径长度量化协作信息为0/1/2-hop协作关系，系统实证其在六个基准上的分布、对12种推荐算法的影响与预测依赖，揭示模型强于简单CR、弱于复杂CR的核心规律。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;端侧推理&quot;,&quot;模型压缩&quot;,&quot;低秩分解&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/WSDM/2026/llm_rec/on_device_large_language_models_for_sequential_recommendation/">On-Device Large Language Models for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向端侧序列推荐的 LLM 任务自适应压缩框架 OD-LLM，融合 Token 协方差归一化、SVD 低秩分解与逐层渐进对齐，50% 压缩率下推荐效果无损且推理提速 3.4 倍。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;序列数据增广&quot;,&quot;输入-目标分布对齐&quot;]"><strong><a href="../../venues/WSDM/2026/generative_rec/sequential_data_augmentation_for_generative_recommendation/">Sequential Data Augmentation for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将训练样本构造统一为序列/目标/输入三步有偏采样，重塑目标与输入分布，显著提升生成式推荐精度与效率</p>
</dd>
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散数据增强&quot;,&quot;难负采样&quot;]"><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;侧信息融合&quot;,&quot;频域去噪&quot;,&quot;时间跨度划分&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序（下一物品预测）&quot;,&quot;变分自编码器&quot;,&quot;对抗变分贝叶斯&quot;,&quot;对比学习&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/adversarial_and_contrastive_variational_autoencoder_for_sequential_recommen/">Adversarial and Contrastive Variational Autoencoder for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>将AVB对抗训练与对比学习引入序列VAE，用RNN-CNN编码提升隐变量质量，在四个数据集Top-k全排序上显著超越SVAE。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;未来行为建模&quot;,&quot;胶囊动态路由&quot;,&quot;多趋势表征&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/future_aware_diverse_trends_framework_for_recommendation/">Future-Aware Diverse Trends Framework for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>用相似用户的相对未来行为构造未来序列，经胶囊动态路由聚成多趋势向量并以时间感知注意力融合，在 Amazon/Steam/ML-1M 上 Full-Ranking 下 Recall@50 最高提升 9.4% 且显著更diverse。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多粒度周期建模&quot;,&quot;渐进注意力&quot;,&quot;时间感知注意力&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/learning_heterogeneous_temporal_patterns_of_user_preference_for_timely_reco/">Learning Heterogeneous Temporal Patterns of User Preference for Timely Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>TimelyRec 用个性化多粒度周期编码与时间感知历史编码，联合刻画周期性与演化性两种异构时间模式，并在物品-时机双任务上 HR@10 最高提升 48.26%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列推荐&quot;,&quot;召回/精排架构&quot;,&quot;高效自注意力&quot;,&quot;向量量化码本&quot;,&quot;码字直方图&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/linear_time_self_attention_with_codeword_histogram_for_efficient_recommenda/">Linear-Time Self Attention with Codeword Histogram for Efficient Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>借助向量量化码本与码字直方图，将自注意力计算从 O(L²D) 降至 O(LBWD)，以 Full-Context 注意力替代稀疏注意力，四项真实数据集上 HR@10 最高提升 8.78%，推理速度最高快 57×、内存效率提升 78×。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回（端到端排序）&quot;,&quot;归纳式推荐&quot;,&quot;关系注意力图神经网络&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/retagnn_relational_temporal_attentive_graph_neural_networks_for_holistic_se/">RetaGNN: Relational Temporal Attentive Graph Neural Networks for Holistic Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>在用户-物品-属性三分图的 h-hop 包围子图上做关系级注意力 GNN 与因果序列自注意力，一套参数同时支撑常规/归纳/跨域序列推荐，MovieLens P@10 较 HGAN 提升 5.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;闭式解线性模型&quot;,&quot;物品-物品相似度&quot;,&quot;会话时效性&quot;]"><strong><a href="../../venues/WWW/2021/sequential_rec/session_aware_linear_item_item_models_for_session_based_recommendation/">Session-aware Linear Item-Item Models for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以闭式解线性物品-物品模型统一建模会话一致性、序列依赖、重复消费与时效性的 SLIST，全库排序下 HR@20 最高提升 9.76%，训练比 SR-GNN 快 768×</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2021&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;召回&quot;,&quot;自注意力&quot;,&quot;时空关系矩阵&quot;]"><strong><a href="../../venues/WWW/2021/poi_spatiotemporal_rec/stan_spatio_temporal_attention_network_for_next_location_recommendation/">STAN: Spatio-Temporal Attention Network for Next Location Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>以全轨迹时空关系矩阵和双层注意力聚合非连续签到，平衡采样训练；四数据集全库Recall@5/10较SOTA提升9–17%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端&quot;,&quot;copy mechanism&quot;,&quot;pointer network&quot;]"><strong><a href="../../venues/WWW/2022/generative_rec/conditional_generation_net_for_medication_recommendation/">Conditional Generation Net for Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>COGNet 以 copy-or-predict 编码器-解码器逐药生成处方，用 visit/medication 两级软检索决定复制历史药或生成新药，在 MIMIC-III 全药库评测上 Jaccard 较 SafeDrug 提升 2.36%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长短期兴趣解耦&quot;,&quot;自监督对比学习&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/disentangling_long_and_short_term_interests_for_recommendation/">Disentangling Long and Short-Term Interests for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用兴趣代理做自监督对比，解耦长短期兴趣并注意力自适应融合，淘宝/快手 GAUC 提升超 0.01、NDCG@2 提升约 5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;端到端推荐决策&quot;,&quot;分层Transformer&quot;,&quot;MOBA Draft&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/draftrec_personalized_draft_recommendation_for_winning_in_multi_player_onli/">DraftRec: Personalized Draft Recommendation for Winning in Multi-Player Online Battle Arena Games</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>分层Transformer联合玩家偏好与队伍协同，阈值约束胜率最大化；LOL HR@1较SASRec提升12.1%，86.9%用户愿用。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多反馈用户建模&quot;,&quot;强到弱注意力蒸馏&quot;,&quot;参与度多任务学习&quot;]"><strong><a href="../../venues/WWW/2022/multi_task_reranking/feedrec_news_feed_recommendation_with_various_user_feedbacks/">FeedRec: News Feed Recommendation with Various User Feedbacks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用异构/同构 Transformer 与强到弱注意力统一建模点击、跳过、分享、点踩、读完、秒关六类反馈，并以点击-读完-时长多任务训练，AUC 相对最强基线提升约 4.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排&quot;,&quot;频域滤波&quot;,&quot;全MLP&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/filter_enhanced_mlp_is_all_you_need_for_sequential_recommendation/">Filter-enhanced MLP is All You Need for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>FMLP-Rec 用 FFT 频域可学习滤波替代自注意力，以全 MLP 去噪建模序列；8 数据集 1:99 负采样排序，NDCG@10 较最佳基线提升 4.5%–7.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序&quot;,&quot;Intent Contrastive Learning&quot;,&quot;EM&quot;,&quot;False-Negative Mitigation&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/intent_contrastive_learning_for_sequential_recommendation/">Intent Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用K-means意图原型与意图对比SSL及FNM，在四数据集全库排序下较SOTA提升7.47%–33.33%。</p>
</dd>
<dt data-paper-tags="[&quot;arXiv 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排（Next-Item 排序）&quot;,&quot;意图-兴趣解耦&quot;,&quot;物品感知对比学习&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/intent_interest_disentanglement_and_item_aware_intent_contrastive_learning_/">Intent-Interest Disentanglement and Item-Aware Intent Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>意图-兴趣解耦与物品感知意图对比，Amazon 三域全库排序下 HR@20 最高相对提升 14.62%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回（下次物品预测）&quot;,&quot;学习式数据增强 (Learning to Augment)&quot;,&quot;策略梯度 REINFORCE&quot;,&quot;休闲用户 / 核心用户鸿沟&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/learning_to_augment_for_casual_user_recommendation/">Learning to Augment for Casual User Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>用 REINFORCE 学习核心用户序列的 Keep/Drop 编辑策略，合成贴近休闲用户行为的增强序列，配合元验证集奖励与 replay，在 4 个数据集、3 种序列模型上让休闲用户 NDCG@5 相对最优基线提升 6.6%–9.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;多任务学习与列表级重排&quot;,&quot;重排&quot;,&quot;Contextualized Transformer&quot;,&quot;Merged Cross-Attention&quot;,&quot;列表级多任务学习&quot;]"><strong><a href="../../venues/WWW/2022/multi_task_reranking/pear_personalized_re_ranking_with_contextualized_transformer_for_recommenda/">PEAR: Personalized Re-ranking with Contextualized Transformer for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>以特征级 MLP + 列表/历史双向交叉注意力 Transformer 重排 top-30 列表，并加列表级多任务监督，MicroVideo 与华为新闻日志上 gAUC@30 分别提升 1.07%/1.71%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;对比学习&quot;,&quot;反向流正则&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/re4_learning_to_re_contrast_re_attend_re_construct_for_multi_interest_recom/">Re4: Learning to Re-contrast, Re-attend, Re-construct for Multi-interest Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>提出反向流 Re4：以对比、重注意、重构三项正则约束多兴趣表征；Amazon/Gowalla 全库排序 NDCG@20 相对提升 42%/12%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Wasserstein 自注意力&quot;,&quot;高斯分布嵌入&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/WWW/2022/sequential_rec/sequential_recommendation_via_stochastic_self_attention/">Sequential Recommendation via Stochastic Self-Attention</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>将物品建模为高斯分布并以负 2-Wasserstein 距离替代点积自注意力，配合正负样本距离正则，在 5 个 Amazon 域上全库排序显著提升且冷启动物品增益最大。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;排序&quot;,&quot;反事实推断&quot;,&quot;因果去偏&quot;,&quot;会话协同过滤&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/a_counterfactual_collaborative_session_based_recommender_system/">A Counterfactual Collaborative Session-based Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将会话外因素建模为混杂因子，以自监督伪标签预训练+反事实协同推理消除伪相关，三数据集 R@20 最高提升 21.8%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;行为路径匹配&quot;,&quot;对比学习增广&quot;,&quot;序列行为建模&quot;]"><strong><a href="../../venues/WWW/2023/ctr_feature_interaction/a_deep_behavior_path_matching_network_for_click_through_rate_prediction/">A Deep Behavior Path Matching Network for Click-Through Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出“行为路径”并做路径级+点击级两级匹配，配合对比增广与自激活去噪，美团线上 CTR +1.6%、广告收入 +1.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;自监督序列校正&quot;,&quot;逆序生成插入&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/a_self_correcting_sequential_recommender/">A Self-Correcting Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>STEAM 用 item-wise 校正器（keep/delete/insert）+ 两个自监督纠错任务先修序列再推荐，三个数据集 5-core/LOO/99 负采样下 HR@5 最高 +5.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/排序打分&quot;,&quot;MLP-Mixer&quot;,&quot;可微架构搜索&quot;,&quot;长短期兴趣解耦&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/automlp_automated_mlp_for_sequential_recommendations/">AutoMLP: Automated MLP for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用双 SRSMLP 分支解耦长/短期兴趣，并以 DARTS 可微搜索短期窗口长度，线性复杂度下 Beauty 的 MRR@10 相对提升 8.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;从众权重解耦&quot;,&quot;去偏对比学习&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/debiased_contrastive_learning_for_sequential_recommendation/">Debiased Contrastive Learning for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 DCRec，用三通道从众权重网络解耦用户真实兴趣与流行度从众，并以加权 InfoNCE 实现去偏对比学习，四数据集 LOO 全库排序 HR@1 最高提升 10.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;零样本属性推理&quot;,&quot;会话图神经网络&quot;,&quot;双意图建模&quot;]"><strong><a href="../../venues/WWW/2023/cold_start_rec/dual_intent_enhanced_graph_neural_network_for_session_based_new_item_recomm/">Dual Intent Enhanced Graph Neural Network for Session-based New Item Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>提出 GSNIR 新设定与 NirGNN：用注意力+β分布双意图模拟用户决策，并以零样本属性推理把新物品映射进 GNN 空间，Amazon G&amp;GF 上 P@20 相对提升 19.33%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;因子化多头注意力&quot;,&quot;双兴趣解耦&quot;,&quot;负反馈序列建模&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/dual_interest_factorization_heads_attention_for_sequential_recommendation/">Dual-interest Factorization-heads Attention for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用头对分解注意力与反馈标签掩码建模跳/不跳混合序列，双兴趣解耦塔加 BPR 成对对比，Micro-video GAUC 提升 2.19%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;持续图学习&quot;,&quot;动态可扩展图卷积&quot;,&quot;流式推荐&quot;]"><strong><a href="../../venues/WWW/2023/graph_rec/dynamically_expandable_graph_convolution_for_streaming_recommendation/">Dynamically Expandable Graph Convolution for Streaming Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以模型隔离视角对 GCN 卷积核执行剪枝-精炼-扩张，配合时序注意力初始化，实现免历史回放的流式推荐，Taobao2014 R@20 最高提升 146%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排/列表推荐&quot;,&quot;Latent Action Space&quot;,&quot;Actor-Critic&quot;,&quot;动作空间对齐&quot;]"><strong><a href="../../venues/WWW/2023/rl_bandit_rec/exploration_and_regularization_of_the_latent_action_space_in_recommendation/">Exploration and Regularization of the Latent Action Space in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>HAC将列表推荐分解为潜在超动作与效应动作，共享critic并以逆池化对齐和BCE监督稳定RL，在三个模拟在线环境取得约1%–6%的长期收益提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;潜变量模型&quot;,&quot;变分自编码器 VAE&quot;,&quot;用户意图建模&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/latent_user_intent_modeling_for_sequential_recommenders/">Latent User Intent Modeling for Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>以条件 VAE 将用户意图建模为潜变量并注入 REINFORCE 序列召回器，三周线上 A/B 使整体满意度 +0.07%、消费多样性 +0.10%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;向量量化(VQ/OPQ)&quot;,&quot;可迁移跨域推荐&quot;,&quot;对比预训练&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/learning_vector_quantized_item_representation_for_transferable_sequential_r/">Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 OPQ 将物品文本映射为离散码并查表得表征，配合半合成/混合域难负样本对比预训练与 Gumbel-Sinkhorn 置换对齐微调，跨域跨平台全库排序显著超越 UniSRec。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回/精排打分&quot;,&quot;Hawkes 过程与时间核&quot;,&quot;负激励建模&quot;,&quot;静态属性兴趣&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/modeling_temporal_positive_and_negative_excitation_for_sequential_recommend/">Modeling Temporal Positive and Negative Excitation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>SDIL 用可学习正/负时间衰减核刻画历史交互的时变激励，并以属性级静态兴趣门控融合，采样排序下 Cellphones MRR 相对提升 10.45%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;Wasserstein 距离&quot;,&quot;自监督数据增广&quot;]"><strong><a href="../../venues/WWW/2023/sequential_rec/mutual_wasserstein_discrepancy_minimization_for_sequential_recommendation/">Mutual Wasserstein Discrepancy Minimization for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 2-Wasserstein 距离替代 KL 散度重建对比互信息估计，四数据集全指标相对提升 2.24%–20.10%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;端到端生成式&quot;,&quot;Decision Transformer&quot;,&quot;User Retention&quot;,&quot;Contrastive Learning&quot;]"><strong><a href="../../venues/WWW/2023/rl_bandit_rec/user_retention_oriented_recommendation_with_decision_transformer/">User Retention-oriented Recommendation with Decision Transformer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>将留存优化转为决策Transformer自回归生成，以奖励提示和加权对比学习缓解离线RL不稳定，在IQiYi/ML-1m提升留存与准确率。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;大语言模型推荐&quot;,&quot;序列推荐打分（召回/精排）&quot;,&quot;CoT 知识蒸馏&quot;,&quot;Rationale 编码&quot;]"><strong><a href="../../venues/WWW/2024/llm_rec/can_small_language_models_be_good_reasoners_for_sequential_recommendation/">Can Small Language Models be Good Reasoners for Sequential Recommendation?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>WWW 2024][大语言模型推荐] 用教师 CoT 三步 rationale 蒸馏 LLaMA2-7B，编码增强 ID/ID-agnostic 序列推荐；Amazon 三数据集 101 候选排序，Home Hit@10 最高提升 28.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;图表征融合&quot;,&quot;跨表征对比学习&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/end_to_end_graph_sequential_representation_learning_for_accurate_recommenda/">End-to-End Graph-Sequential Representation Learning for Accurate Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>MRGSRec 用序列+图双编码器、全链路融合层与跨表征对比损失做端到端多表征学习，Leave-One-Out 全库排序下 NDCG@5 较 SASRec+ 最高提升 8.94%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多行为序列建模&quot;,&quot;频域去噪&quot;,&quot;噪声解耦对比学习&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/end4rec_efficient_noise_decoupling_for_multi_behavior_sequential_recommenda/">END4Rec: Efficient Noise-Decoupling for Multi-Behavior Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 FFT 频域算子+双粒度（硬/软）去噪与噪声解耦对比学习做多行为序列推荐，Leave-One-Out+99 负采样评测，三数据集 HR@10 相对提升 5.8%–8.0%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;LLM 用户画像摘要&quot;,&quot;LoRA 指令微调&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/harnessing_large_language_models_for_text_rich_sequential_recommendation/">Harnessing Large Language Models for Text-Rich Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用冻结大模型对长文本行为序列分块做层次/循环摘要压缩，再以 LoRA-SFT 微调 Llama-2-7b 做 Yes/No 判别；MIND 上 Recall@3 相对 TALLRec 提升 6.8%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;序列与会话推荐&quot;,&quot;端到端在线推理&quot;,&quot;Edge-Cloud Collaboration&quot;,&quot;Uncertainty Quantification&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/intelligent_model_update_strategy_for_sequential_recommendation/">Intelligent Model Update Strategy for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 IntellectReq，以误推荐检测与分布映射控制边端参数请求，10%通信逼近100%性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;序列数据增强&quot;,&quot;对比学习&quot;,&quot;冷启动&quot;]"><strong><a href="../../venues/WWW/2024/self_supervised_denoising/is_contrastive_learning_necessary_a_study_of_data_augmentation_vs_contrasti/">Is Contrastive Learning Necessary? A Study of Data Augmentation vs Contrastive Learning in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>系统评测8种序列数据增强与3种对比学习SRS，发现slide-window等直接增强可媲美CL且更省算力。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;提示增强/查询建议&quot;,&quot;实体中心知识库&quot;,&quot;In-Context Personalization&quot;]"><strong><a href="../../venues/WWW/2024/llm_rec/knowledge_augmented_large_language_models_for_personalized_contextual_query/">Knowledge-Augmented Large Language Models for Personalized Contextual Query Suggestion</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用实体中心个人知识库检索增强 GPT-4 提示，在 Bing 日志与人工评测上使上下文查询建议的相关性/有用性较 LLM 基线提升约 17%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;在线原型聚类&quot;,&quot;自蒸馏&quot;,&quot;梯度反转对抗&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/leave_no_one_behind_online_self_supervised_self_distillation_for_sequential/">Leave No One Behind: Online Self-Supervised Self-Distillation for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用在线原型聚类+头尾对抗+簇感知自蒸馏，把长序列知识蒸馏给短序列用户，四数据集全库排序 HR@5 最高提升 9.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;用户画像摘要&quot;,&quot;长文本摘要&quot;]"><strong><a href="../../venues/WWW/2024/llm_rec/modeling_user_viewing_flow_using_large_language_models_for_article_recommen/">Modeling User Viewing Flow Using Large Language Models for Article Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 LLM 做用户档案摘要、BERT 编码文章并瞬时注意力读历史，ATA 线上 UVCTR 相对 DCN 提升 2.4%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;重排&quot;,&quot;主动推荐&quot;,&quot;偏好引导&quot;]"><strong><a href="../../venues/WWW/2024/rl_bandit_rec/proactive_recommendation_with_iterative_preference_guidance/">Proactive Recommendation with Iterative Preference Guidance</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>IPG 以“交互概率×引导增益”的后处理打分迭代引导用户兴趣走向目标物品，仿真器 20 轮交互下 IoI@20 较最优基线提升 27.1%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;行为路径稀疏激活&quot;,&quot;Gumbel-Softmax 可微路由&quot;]"><strong><a href="../../venues/WWW/2024/sequential_rec/recommender_transformers_with_behavior_pathways/">Recommender Transformers with Behavior Pathways</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RETR 用可学习的二值路径路由器稀疏激活行为路径，配合 Gumbel-Softmax 与层级门控，在 7 个数据集 Leave-One-Out + 100 负采样协议下取得 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2024&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;双分支参数共享&quot;,&quot;意图导向会话自监督&quot;]"><strong><a href="../../venues/WWW/2024/cross_domain_multi_scenario/unifiedssr_a_unified_framework_of_sequential_search_and_recommendation/">UnifiedSSR: A Unified Framework of Sequential Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>双分支参数共享 Transformer 联合建模搜索与推荐序列，并用可学习意图会话+双路自监督挖掘动态意图，采样排序下两场景 HR/NDCG 全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;位置编码&quot;,&quot;目标注意力&quot;,&quot;门控融合&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/a_contextual_aware_position_encoding_for_sequential_recommendation/">A Contextual-Aware Position Encoding for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>首个为序列推荐定制的上下文感知位置编码 CAPE：用目标-上下文不相似度累加动态分配位置，配合门控融合异构特征，在 5 个骨干与工业 A/B 上稳定涨点。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;跨域序列推荐&quot;,&quot;LoRA 参数高效适配&quot;,&quot;域不变兴趣抽取&quot;]"><strong><a href="../../venues/WWW/2025/cross_domain_multi_scenario/abxi_invariant_interest_adaptation_for_task_guided_cross_domain_sequential_/">ABXI: Invariant Interest Adaptation for Task-Guided Cross-Domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用任务引导对齐消除预测错配，并以 dLoRA/iLoRA 双 LoRA 抽取域不变兴趣，三数据集 MRR 大幅超越 CDSR 基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;扩散模型序列推荐&quot;,&quot;语义ID&quot;,&quot;对比差异最大化&quot;]"><strong><a href="../../venues/WWW/2025/generative_rec/distinguished_quantized_guidance_for_diffusion_based_sequence_recommendatio/">Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用语义码本量化序列增强扩散引导，并以对比差异最大化抑制流行度偏置生成，四数据集全库排序全面领先。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;摊销推理&quot;,&quot;早期融合&quot;,&quot;Cross-Attention&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/efficient_user_history_modeling_with_amortized_inference_for_deep_learning_/">Efficient user history modeling with amortized inference for deep learning recommendation models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 cross-attention 追加候选替代逐项拼接，使历史编码可摊销推理，LinkedIn Feed/Ads 延迟降 30%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;混合序列推荐&quot;,&quot;Perceiver 多模态融合&quot;,&quot;Critiquing 评测&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/flare_fusing_language_models_and_collaborative_architectures_for_recommende/">FLARE: Fusing Language Models and Collaborative Architectures for Recommender Enhancement</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用冻结 mT5+Perceiver 将文本注入 Bert4Rec，并以重调 Bert4Rec 基线、大词表评测与 Critiquing 任务重估 ID+文本增益。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;生成式推荐&quot;,&quot;缩放律&quot;,&quot;数据再生&quot;,&quot;长期行为序列建模&quot;]"><strong><a href="../../venues/WWW/2025/generative_rec/generative_large_recommendation_models_emerging_trends_in_llms_for_recommen/">Generative Large Recommendation Models: Emerging Trends in LLMs for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>系统梳理生成式大推荐模型范式：数据质量、缩放律、用户行为挖掘与训练推理效率四大议题及未来方向。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;对齐与均匀性&quot;,&quot;图序列联合训练&quot;]"><strong><a href="../../venues/WWW/2025/graph_rec/graph_sequential_alignment_and_uniformity_toward_enhanced_recommendation_sy/">Graph-Sequential Alignment and Uniformity: Toward Enhanced Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>以共享嵌入层联合训练 LightGCN 与 SASRec，用对齐-均匀性损失实现跨子模块知识迁移，Leave-One-Out 全库排序下 R@10 最高提升 13.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;多模态推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;显式时间建模&quot;,&quot;多任务辅助监督&quot;]"><strong><a href="../../venues/WWW/2025/multimodal_rec/hierarchical_time_aware_mixture_of_experts_for_multi_modal_sequential_recom/">Hierarchical Time-Aware Mixture of Experts for Multi-Modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用交互MoE过滤模态冗余、时序MoE注入显式时间戳，配合三项辅助任务，在四个Amazon数据集全库排序上NDCG@5最高提升20.9%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;大记忆网络&quot;,&quot;乘积量化&quot;,&quot;记忆参数服务器&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/large_memory_network_for_recommendation/">Large Memory Network for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用百万级全局共享记忆块压缩用户历史行为，以乘积量化双轴激活实现 O(√n d) 检索，抖音电商搜索 AUC +4.68%、订单/人 +0.87%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;时间感知&quot;,&quot;长期行为序列建模&quot;,&quot;流式训练&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/long_term_interest_clock_fine_grained_time_perception_in_streaming_recommen/">Long-Term Interest Clock: Fine-Grained Time Perception in Streaming Recommendation System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用候选物品+当前时刻双查询从长期行为中检索子序列，并以时距感知注意力聚合，实现流式推荐细粒度时间感知，线上活跃天数+0.122%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;DHEN 特征交叉集成&quot;,&quot;自监督未来行为预测&quot;,&quot;转化广告 CVR&quot;]"><strong><a href="../../venues/WWW/2025/ctr_feature_interaction/on_the_practice_of_deep_hierarchical_ensemble_network_for_ad_conversion_rat/">On the Practice of Deep Hierarchical Ensemble Network for Ad Conversion Rate Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>Pinterest 工业实践：两层 DHEN 多任务骨干 + 五类行为序列端到端建模 + 未来行为自监督辅助损失，线上 CPA 降 2.15%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Mixture-of-Logits&quot;,&quot;近似Top-K检索&quot;,&quot;负载均衡正则&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/retrieval_with_learned_similarities/">Retrieval with Learned Similarities</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>证明 MoL 为通用相似度逼近器，配互信息负载均衡损失与带误差界的近似 top-K 检索，序列推荐 HR@1 平均提升 29.1%，检索延迟最高降 66×。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2025&quot;,&quot;序列与会话推荐&quot;,&quot;数据蒸馏/训练加速&quot;,&quot;Tucker 分解&quot;,&quot;双层优化&quot;,&quot;特征空间对齐&quot;]"><strong><a href="../../venues/WWW/2025/sequential_rec/td3_tucker_decomposition_based_dataset_distillation_method_for_sequential_r/">TD3: Tucker Decomposition Based Dataset Distillation Method for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 Tucker 分解把合成序列摘要解耦为用户/时间/共享物品因子与关系核，配合特征空间对齐双层优化，在 0.2%–7.4% 数据量下逼近甚至超越全量训练。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;会话层次建模&quot;,&quot;稀疏注意力&quot;]"><strong><a href="../../venues/WWW/2026/generative_rec/beyond_the_flat_sequence_hierarchical_and_preference_aware_generative_recom/">Beyond the Flat Sequence: Hierarchical and Preference-Aware Generative Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用会话层次预训练+偏好引导稀疏注意力打破 HSTU 扁平序列假设，APPGallery 全库 AUC 0.8929、线上 eCPM +1.99%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长序列建模&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/blossomrec_block_level_fused_sparse_attention_mechanism_for_sequential_reco/">BlossomRec: Block-level Fused Sparse Attention Mechanism for Sequential Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双路块级稀疏注意力（LTIS 块选择 + STIS 幂律掩码）经门控融合，长序列训练提速 3.2×、显存降至 1/11，精度持平或超越 SASRec。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;生成式推荐&quot;,&quot;掩码离散扩散&quot;,&quot;语义ID并行解码&quot;]"><strong><a href="../../venues/WWW/2026/generative_rec/diffgrm_diffusion_based_generative_recommendation_model/">DiffGRM: Diffusion-based Generative Recommendation Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用掩码离散扩散替代自回归解码 SID，配合 OPQ 解耦分词、难度感知相干加噪与置信度并行去噪，NDCG@10 提升 6.9%–15.5%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;Human Priors&quot;,&quot;Multi-Head Decoding&quot;]"><strong><a href="../../venues/WWW/2026/generative_rec/don_t_waste_it_guiding_generative_recommenders_with_structured_human_priors/">Don&#x27;t Waste It: Guiding Generative Recommenders with Structured Human Priors via Multi-Head Decoding</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以零初始化残差适配头将物品/时序/事件/图先验注入HSTU/HLLM端到端训练，三域Recall与NDCG显著提升并兼顾多样性。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回 (Next-Item 全库排序)&quot;,&quot;推理轨迹&quot;,&quot;层次原型监督&quot;,&quot;自适应早停&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/dtrec_learning_dynamic_reasoning_trajectories_for_sequential_recommendation/">DTRec: Learning Dynamic Reasoning Trajectories for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>层次原型软监督（由粗到细）+三指标自适应停机，让推理轨迹方向与深度双动态，NDCG@10 最高 +24.5%、推理步数省 41.6%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID解码&quot;,&quot;长短期兴趣路由&quot;]"><strong><a href="../../venues/WWW/2026/generative_rec/dualgr_generative_retrieval_with_long_and_short_term_interests_modeling/">DualGR: Generative Retrieval with Long and Short-Term Interests Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双分支长短期路由+桶内搜索式SID解码+曝光感知NTP损失，快手双列场景在线HR@1000达19.53%，A/B视频播放+0.527%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排/后处理&quot;,&quot;稀疏自编码器 SAE&quot;,&quot;流行度去偏&quot;,&quot;神经元转向&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/from_insight_to_intervention_interpretable_neuron_steering_for_controlling_/">From Insight to Intervention: Interpretable Neuron Steering for Controlling Popularity Bias in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用稀疏自编码器定位并转向“流行度神经元”，后处理式调控曝光公平，nDCG 几乎无损且可解释。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排（下一物品预测）&quot;,&quot;Item-aware Attention&quot;,&quot;双掩码注意力&quot;,&quot;LoRA 微调&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/beyong_tokens_item_aware_attention_for_llm_based_recommendation/">From Token to Item: Enhancing Large Language Models for Recommendation via Item-aware Attention Mechanism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以物品内/物品间双掩码拆解 LLM 因果注意力，纯靠结构改造建模物品级协同，Amazon 三数据集 Full-Ranking 下 Prec@10 相对最优基线提升 6.80%~71.00%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;LLM 语义嵌入&quot;,&quot;行为校准&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/generative_archetype_grounded_item_representations_for_sequential_recommend/">Generative Archetype-Grounded Item Representations for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 生成物品“目标人群原型”文本并单次前向抽取 prefill/decoding 双嵌入，再以行为关联分数调制排斥力做校准，Full-Ranking 下 HR@10 最高 +6.95%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;跨域与多场景推荐&quot;,&quot;数据重生成&quot;,&quot;对比解码&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/WWW/2026/cross_domain_multi_scenario/generative_data_transformation_from_mixed_to_unified_data/">Generative Data Transformation: From Mixed to Unified Data</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出数据侧范式 Taesar，用三模型对比解码把跨域混合序列重生成纯目标域序列，免改架构即插即用，Books HR@10 提升 29.4%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Legendre 正交多项式&quot;,&quot;Hyena 长卷积&quot;,&quot;次二次复杂度&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/hyena_operator_for_fast_sequential_recommendation/">Hyena Operator for Fast Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Legendre 正交多项式参数化 Hyena 长卷积核，线性复杂度建模超长用户序列，XLong 上 NDCG@10 提升 17.65% 且训练最高 6× 加速。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;残差向量量化码本&quot;,&quot;多兴趣解耦学习&quot;,&quot;LLM 序列推荐&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/hymirec_a_hybrid_multi_interest_learning_framework_for_llm_based_sequential/">HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用残差码本压缩长历史、轻量模型抽粗粒度多兴趣、LLM精炼兴趣，配合窗口目标+匈牙利匹配解耦多兴趣，全库排序下显著超越LLM基线。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;时间间隔预测&quot;,&quot;上下文提示&quot;,&quot;零样本推理&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/is_more_context_always_better_examining_llm_reasoning_capability_for_time_i/">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 GPT-4o/Claude/Gemini 在复购时间间隔预测上的零样本能力，发现 LLM 仅胜统计基线、全面落后 XGBoost，且中等上下文最优、高上下文反降。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;联邦持续学习&quot;,&quot;原型迁移&quot;,&quot;自蒸馏&quot;]"><strong><a href="../../venues/WWW/2026/debias_fairness_industrial/learning_evolving_preferences_a_federated_continual_framework_for_user_cent/">Learning Evolving Preferences: A Federated Continual Framework for User-Centric Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>联邦持续推荐框架 FCUCR：时间感知自蒸馏抗遗忘 + 跨用户原型迁移促协同，四数据集全库排序 HR@10 最高提升 44.0%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;长序列建模&quot;,&quot;交叉注意力&quot;,&quot;长度外推&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/make_it_long_keep_it_fast_end_to_end_10k_long_user_behavior_sequence_modeli/">Make It Long, Keep It Fast: End-to-End 10K Long User Behavior Sequence Modeling for Billion-Scale Douyin Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以单查询目标-历史交叉注意力(STCA)将长序列建模降至 O(L)，配合请求级批处理(RLB)与稀疏训练稠密推理，在抖音十亿级流量实现 10K 序列端到端排序。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;MoE&quot;,&quot;长序列建模&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/mixture_of_sequence_theme_aware_mixture_of_experts_for_long_sequence_recomm/">Mixture of Sequence: Theme-Aware Mixture-of-Experts for Long-Sequence Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示长序列“会话跳跃”现象，用主题码本路由+多尺度专家融合做子序列抽取，AUC 平均提升 0.68% 且 FLOPs 更低</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;时序子图&quot;,&quot;兴趣演化建模&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/modeling_stage_wise_evolution_of_user_interests_for_news_recommendation/">Modeling Stage-wise Evolution of User Interests for News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用全局LightGCN+阶段时序子图LSTM/自注意力双分支建模新闻兴趣演化，Adressa AUC 0.7993 大幅超越SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;多行为序列建模&quot;,&quot;图注意力&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/multi_behavior_sequential_modeling_with_transition_aware_graph_attention_ne/">Multi-Behavior Sequential Modeling with Transition-Aware Graph Attention Network for E-Commerce Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>TGA 用 item/类目/邻居三视角稀疏转移图 + 行为对参数化图注意力替代 Transformer，线性复杂度建模多行为序列，Taobao AUC 0.7454 且训练推理快 5.8 倍，线上 CVR +1.29%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;统一Transformer骨干&quot;,&quot;混合参数化&quot;,&quot;KV缓存&quot;]"><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/onetrans_unified_feature_interaction_and_sequence_modeling_with_one_transfo/">OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用单一因果 Transformer 统一序列建模与特征交叉，配合金字塔剪枝与跨请求 KV 缓存，线上 GMV/u 提升 5.68%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;参数高效微调&quot;,&quot;用户分群个性化&quot;]"><strong><a href="../../venues/WWW/2026/multimodal_rec/personalized_parameter_efficient_fine_tuning_of_foundation_models_for_multi/">Personalized Parameter-Efficient Fine-Tuning of Foundation Models for Multimodal Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>按兴趣分群并为每群挂载独立 PEFT 模块，使 CLIP 编码出用户条件化的物品表征，配合群内难负采样，NDCG@20 最高提升 15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Item-based CF&quot;,&quot;Trigger-Target 负采样&quot;,&quot;目标注意力&quot;]"><strong><a href="../../venues/WWW/2026/collaborative_filtering/pi2i_a_personalized_item_based_collaborative_filtering_retrieval_framework/">PI2I: A Personalized Item-Based Collaborative Filtering Retrieval Framework</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>放宽 I2I 截断保召回、Trigger-Target 负采样对齐离线在线、目标注意力精排打分，淘宝线上成交率 +1.05%</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;模态交互分解&quot;,&quot;自适应融合&quot;]"><strong><a href="../../venues/WWW/2026/multimodal_rec/prism_personalized_recommendation_via_information_synergy_module/">PRISM: Personalized Recommendation via Information Synergy Module</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MoE 显式分解模态唯一/冗余/协同信息并以用户偏好动态重加权，即插即用提升三类序列骨干，Yelp N@10 超最强基线 10.91%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;偏好漂移&quot;,&quot;区域级知识编辑&quot;,&quot;区域专属 LoRA&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/raie_region_aware_incremental_preference_editing_with_lora_for_llm_based_re/">RAIE: Region-Aware Incremental Preference Editing with LoRA for LLM-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>冻结 LLM 主干，用球面 k-means 构建偏好区域并为每区配专属 LoRA，通过 Update/Expand/Add 三种区域级编辑与置信路由实现抗遗忘的增量偏好适配。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;数据增强&quot;,&quot;长尾序列推荐&quot;]"><strong><a href="../../venues/WWW/2026/cold_start_rec/tail_aware_data_augmentation_for_long_tail_sequential_recommendation/">Tail-Aware Data Augmentation for Long-Tail Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性物品相似度构建候选集，设计 T-Substitute/T-Insert 两个尾部感知算子并在表征层 Mixup，实现头部与尾部双赢的长尾序列推荐数据增强。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;思维链蒸馏&quot;,&quot;LoRA 专家融合&quot;]"><strong><a href="../../venues/WWW/2026/llm_rec/thinkrec_thinking_based_recommendation_via_llm/">ThinkRec: Thinking-based Recommendation via LLM</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以合成推理链激活 LLM 的 System 2 思维，并用实例级 LoRA 专家融合适配异构用户，AUC 平均提升 7.96%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;稀疏注意力&quot;,&quot;长期行为序列建模&quot;]"><strong><a href="../../venues/WWW/2026/ctr_feature_interaction/unleashing_the_potential_of_sparse_attention_on_long_term_behaviors_for_ctr/">Unleashing the Potential of Sparse Attention on Long-term Behaviors for CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>面向长行为序列 CTR 的个性化时间分块 + 三分支稀疏注意力 SparseCTR，AUC 相对提升 8.49%，线上 CTR +1.72%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;工业数据集&quot;,&quot;全局时间切分&quot;,&quot;冷启动基准&quot;]"><strong><a href="../../venues/WWW/2026/sequential_rec/vk_lsvd_a_large_scale_industrial_dataset_for_short_video_recommendation/">VK-LSVD: A Large-Scale Industrial Dataset for Short-Video Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>发布 407 亿交互、千万用户、两千万短视频的工业级开放数据集，配全局时间切分与冷启动竞赛基准。</p>
</dd>
</dl>
