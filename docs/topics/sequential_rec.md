---
hide:
  - toc
---

# ⏱️ 序列与会话推荐 (Sequential & Session-Based Recommendation)

> **研究范畴**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

当前方向共收录 **286** 篇顶会/顶刊论文解读：

🔥 **高频交叉主题：** 生成式推荐 ×54 · 大语言模型推荐 ×39 · 去偏公平与可信推荐 ×18 · 多模态推荐 ×14 · 语义ID ×10 · MoE ×8 · 知识蒸馏 ×8 · 冷启动 ×6

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">123</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">123</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">88</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">54</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">39</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐" aria-pressed="false">序列推荐<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">18</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">15</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">14</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CTR/CVR 预估与特征交叉" aria-pressed="false">CTR/CVR 预估与特征交叉<span class="paper-tag-filter__count">13</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">11</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">10</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MoE" aria-pressed="false">MoE<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识蒸馏" aria-pressed="false">知识蒸馏<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列建模" aria-pressed="false">长序列建模<span class="paper-tag-filter__count">8</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标注意力" aria-pressed="false">目标注意力<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性注意力" aria-pressed="false">线性注意力<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV缓存" aria-pressed="false">KV缓存<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习与动态探索推荐" aria-pressed="false">强化学习与动态探索推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续学习" aria-pressed="false">持续学习<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长期行为序列建模" aria-pressed="false">长期行为序列建模<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域滤波" aria-pressed="false">频域滤波<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间切分" aria-pressed="false">全局时间切分<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释推荐" aria-pressed="false">可解释推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据增强" aria-pressed="false">数据增强<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏注意力" aria-pressed="false">稀疏注意力<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域序列推荐" aria-pressed="false">跨域序列推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Flow Matching" aria-pressed="false">Flow Matching<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM4Rec" aria-pressed="false">LLM4Rec<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prior-Fitted Network" aria-pressed="false">Prior-Fitted Network<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE" aria-pressed="false">RQ-VAE<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Scaling Law" aria-pressed="false">Scaling Law<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Test-Time Training" aria-pressed="false">Test-Time Training<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Transformer 序列建模" aria-pressed="false">Transformer 序列建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话推荐" aria-pressed="false">会话推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩分解" aria-pressed="false">低秩分解<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="侧信息融合" aria-pressed="false">侧信息融合<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好对齐" aria-pressed="false">偏好对齐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回+重排" aria-pressed="false">召回+重排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣表征" aria-pressed="false">多兴趣表征<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列推荐" aria-pressed="false">多行为序列推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业数据集" aria-pressed="false">工业数据集<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="并行解码" aria-pressed="false">并行解码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散推荐" aria-pressed="false">扩散推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间位置编码" aria-pressed="false">时间位置编码<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式训练" aria-pressed="false">流式训练<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="灾难性遗忘" aria-pressed="false">灾难性遗忘<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="状态空间模型" aria-pressed="false">状态空间模型<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端云协同" aria-pressed="false">端云协同<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧推理" aria-pressed="false">端侧推理<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID量化" aria-pressed="false">语义ID量化<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长用户序列" aria-pressed="false">超长用户序列<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软标签" aria-pressed="false">软标签<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域建模" aria-pressed="false">频域建模<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ACT-R 认知记忆建模" aria-pressed="false">ACT-R 认知记忆建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Advantage Actor-Critic" aria-pressed="false">Advantage Actor-Critic<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="AIGC 内容生态" aria-pressed="false">AIGC 内容生态<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="arXiv" aria-pressed="false">arXiv<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Beam Search" aria-pressed="false">Beam Search<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Benchmark" aria-pressed="false">Benchmark<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="BPE" aria-pressed="false">BPE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="C2C 推荐数据集" aria-pressed="false">C2C 推荐数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Thought" aria-pressed="false">Chain-of-Thought<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Chain-of-Thought Tokenization" aria-pressed="false">Chain-of-Thought Tokenization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Clustering of Neural Bandits" aria-pressed="false">Clustering of Neural Bandits<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Contextual Bandit 视角选择" aria-pressed="false">Contextual Bandit 视角选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Critiquing 评测" aria-pressed="false">Critiquing 评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-Attention" aria-pressed="false">Cross-Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Cross-domain ID Matcher" aria-pressed="false">Cross-domain ID Matcher<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CVaR-DRO" aria-pressed="false">CVaR-DRO<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DCN-v2" aria-pressed="false">DCN-v2<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Decision Transformer" aria-pressed="false">Decision Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="DHEN 特征交叉集成" aria-pressed="false">DHEN 特征交叉集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习" aria-pressed="false">GRPO 强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 强化学习精炼" aria-pressed="false">GRPO 强化学习精炼<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="GRPO 风格对齐" aria-pressed="false">GRPO 风格对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Hawkes 过程" aria-pressed="false">Hawkes 过程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="HSTU" aria-pressed="false">HSTU<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Hyena 长卷积" aria-pressed="false">Hyena 长卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="ID-centric Pre-training" aria-pressed="false">ID-centric Pre-training<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="In-Context Learning" aria-pressed="false">In-Context Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Interval-Infused Attention" aria-pressed="false">Interval-Infused Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Item-based CF" aria-pressed="false">Item-based CF<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Journey-Aware Sparse Attention" aria-pressed="false">Journey-Aware Sparse Attention<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="KV Cache 压缩" aria-pressed="false">KV Cache 压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Learned Sparse Retrieval" aria-pressed="false">Learned Sparse Retrieval<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Legendre 正交多项式" aria-pressed="false">Legendre 正交多项式<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 上下文增强" aria-pressed="false">LLM 上下文增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 反思 (Reflection)" aria-pressed="false">LLM 反思 (Reflection)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 意图画像" aria-pressed="false">LLM 意图画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 指令微调" aria-pressed="false">LLM 指令微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 日志语义增强" aria-pressed="false">LLM 日志语义增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义嵌入" aria-pressed="false">LLM 语义嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 语义融合" aria-pressed="false">LLM 语义融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Optimizer" aria-pressed="false">LLM-as-Optimizer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM2Vec" aria-pressed="false">LLM2Vec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM推荐" aria-pressed="false">LLM推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM数据增广" aria-pressed="false">LLM数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM语义检索" aria-pressed="false">LLM语义检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 专家融合" aria-pressed="false">LoRA 专家融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LoRA 参数高效适配" aria-pressed="false">LoRA 参数高效适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Loss of Plasticity" aria-pressed="false">Loss of Plasticity<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Machine Unlearning" aria-pressed="false">Machine Unlearning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mamba 序列建模" aria-pressed="false">Mamba 序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mamba/SSM" aria-pressed="false">Mamba/SSM<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="META ID" aria-pressed="false">META ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Minimax Group Fairness" aria-pressed="false">Minimax Group Fairness<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixture-of-Logits" aria-pressed="false">Mixture-of-Logits<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Mixup" aria-pressed="false">Mixup<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MK-MMD 特征核对齐" aria-pressed="false">MK-MMD 特征核对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="MLP-Mixer" aria-pressed="false">MLP-Mixer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Next Basket Recommendation" aria-pressed="false">Next Basket Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Next Session Prediction" aria-pressed="false">Next Session Prediction<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="OOV Token" aria-pressed="false">OOV Token<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Oracle 引导对齐" aria-pressed="false">Oracle 引导对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Patch 预训练" aria-pressed="false">Patch 预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Perceiver 多模态融合" aria-pressed="false">Perceiver 多模态融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="PPO 检索智能体" aria-pressed="false">PPO 检索智能体<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Procrustes 投影" aria-pressed="false">Procrustes 投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Prompt Learning" aria-pressed="false">Prompt Learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Query-Driven Decoder" aria-pressed="false">Query-Driven Decoder<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RQ-VAE 语义ID" aria-pressed="false">RQ-VAE 语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SASRec" aria-pressed="false">SASRec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Selective Reinitialization" aria-pressed="false">Selective Reinitialization<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Semantic ID 对齐" aria-pressed="false">Semantic ID 对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sequential Unlearning" aria-pressed="false">Sequential Unlearning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Session Mask" aria-pressed="false">Session Mask<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SimHash语义ID" aria-pressed="false">SimHash语义ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sinkhorn 可微投影" aria-pressed="false">Sinkhorn 可微投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SPC 三难困境" aria-pressed="false">SPC 三难困境<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Stein 核分布对齐" aria-pressed="false">Stein 核分布对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sub-Sequence Splitting" aria-pressed="false">Sub-Sequence Splitting<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Thompson Sampling" aria-pressed="false">Thompson Sampling<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Token Merge" aria-pressed="false">Token Merge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Top-K 位置奖励" aria-pressed="false">Top-K 位置奖励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Trigger-Target 负采样" aria-pressed="false">Trigger-Target 负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="TTT-Linear" aria-pressed="false">TTT-Linear<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Tucker 分解" aria-pressed="false">Tucker 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="一步生成" aria-pressed="false">一步生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="一致性触发请求" aria-pressed="false">一致性触发请求<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三专家解耦" aria-pressed="false">三专家解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="三塔 Transformer 多步召回" aria-pressed="false">三塔 Transformer 多步召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文token组合" aria-pressed="false">上下文token组合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文压缩" aria-pressed="false">上下文压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文感知分词" aria-pressed="false">上下文感知分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文提示" aria-pressed="false">上下文提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="上下文注意力" aria-pressed="false">上下文注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一动作损失" aria-pressed="false">下一动作损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一篮推荐" aria-pressed="false">下一篮推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一购物篮推荐" aria-pressed="false">下一购物篮推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性采样" aria-pressed="false">不确定性采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="不确定性量化" aria-pressed="false">不确定性量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="两阶段训练" aria-pressed="false">两阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化专家" aria-pressed="false">个性化专家<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化排序损失" aria-pressed="false">个性化排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化搜索" aria-pressed="false">个性化搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化查询改写" aria-pressed="false">个性化查询改写<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化流行度" aria-pressed="false">个性化流行度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="个性化软提示" aria-pressed="false">个性化软提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主动学习实例选择" aria-pressed="false">主动学习实例选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="主题模型" aria-pressed="false">主题模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="乘积量化" aria-pressed="false">乘积量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交互级依赖图" aria-pressed="false">交互级依赖图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交叉注意力" aria-pressed="false">交叉注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="交替优化" aria-pressed="false">交替优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务向量" aria-pressed="false">任务向量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="任务感知子图检索" aria-pressed="false">任务感知子图检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="会话层次建模" aria-pressed="false">会话层次建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪先验物品生成" aria-pressed="false">伪先验物品生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="伪多任务学习" aria-pressed="false">伪多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置偏差 MAPB" aria-pressed="false">位置偏差 MAPB<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置前馈网络" aria-pressed="false">位置前馈网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置编码" aria-pressed="false">位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="位置编码对齐" aria-pressed="false">位置编码对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩矩阵分解" aria-pressed="false">低秩矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩缓存" aria-pressed="false">低秩缓存<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="低秩解耦注意力" aria-pressed="false">低秩解耦注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="保真度质量控制" aria-pressed="false">保真度质量控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="信息瓶颈" aria-pressed="false">信息瓶颈<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好优化" aria-pressed="false">偏好优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好感知注意力" aria-pressed="false">偏好感知注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好抽取" aria-pressed="false">偏好抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好漂移" aria-pressed="false">偏好漂移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="偏好状态修复" aria-pressed="false">偏好状态修复<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元学习" aria-pressed="false">元学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元数据知识图谱" aria-pressed="false">元数据知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="元路径聚类" aria-pressed="false">元路径聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免更新推理" aria-pressed="false">免更新推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免模拟器部署推理" aria-pressed="false">免模拟器部署推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="免训练推理" aria-pressed="false">免训练推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局归一化注意力" aria-pressed="false">全局归一化注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全局时间线" aria-pressed="false">全局时间线<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="共享账户推荐" aria-pressed="false">共享账户推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="兴趣演化建模" aria-pressed="false">兴趣演化建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容初始化" aria-pressed="false">内容初始化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="内容增强语言建模" aria-pressed="false">内容增强语言建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动基准" aria-pressed="false">冷启动基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冻结宿主适配" aria-pressed="false">冻结宿主适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="准确率-新颖度权衡" aria-pressed="false">准确率-新颖度权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="准线性注意力" aria-pressed="false">准线性注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="几何记忆" aria-pressed="false">几何记忆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层位置编码" aria-pressed="false">分层位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层向量量化" aria-pressed="false">分层向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层对比学习" aria-pressed="false">分层对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分层强化学习" aria-pressed="false">分层强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分离约束" aria-pressed="false">分离约束<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="分解嵌入融合" aria-pressed="false">分解嵌入融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表级排序" aria-pressed="false">列表级排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="删失Weibull损失" aria-pressed="false">删失Weibull损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前瞻预测" aria-pressed="false">前瞻预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前缀微调" aria-pressed="false">前缀微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="前缀级排序优化" aria-pressed="false">前缀级排序优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态分块" aria-pressed="false">动态分块<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态图推荐" aria-pressed="false">动态图推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态子网络" aria-pressed="false">动态子网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态用户画像" aria-pressed="false">动态用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="动态负采样" aria-pressed="false">动态负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="区域专属 LoRA" aria-pressed="false">区域专属 LoRA<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="区域级知识编辑" aria-pressed="false">区域级知识编辑<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协作信息量化定义" aria-pressed="false">协作信息量化定义<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协作关系分布实证" aria-pressed="false">协作关系分布实证<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同信号注入" aria-pressed="false">协同信号注入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同监督微调" aria-pressed="false">协同监督微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同视角内容理解" aria-pressed="false">协同视角内容理解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同训练" aria-pressed="false">协同训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="单步生成" aria-pressed="false">单步生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型交叉注意力" aria-pressed="false">原型交叉注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型迁移" aria-pressed="false">原型迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去噪鲁棒性" aria-pressed="false">去噪鲁棒性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数共享" aria-pressed="false">参数共享<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数掩码控制" aria-pressed="false">参数掩码控制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数校正" aria-pressed="false">参数校正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数生成网络" aria-pressed="false">参数生成网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数级适配" aria-pressed="false">参数级适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效微调" aria-pressed="false">参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="参数高效短视适配" aria-pressed="false">参数高效短视适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向信息流" aria-pressed="false">双向信息流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向序列建模" aria-pressed="false">双向序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向时序数据增强" aria-pressed="false">双向时序数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双向线性注意力" aria-pressed="false">双向线性注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双塔离线推理" aria-pressed="false">双塔离线推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层优化" aria-pressed="false">双层优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双层对齐" aria-pressed="false">双层对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双提升攻击目标" aria-pressed="false">双提升攻击目标<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双曲流形对齐" aria-pressed="false">双曲流形对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双流架构" aria-pressed="false">双流架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双表面策略解耦" aria-pressed="false">双表面策略解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双视图混合专家" aria-pressed="false">双视图混合专家<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双阶段训练" aria-pressed="false">双阶段训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实对比学习" aria-pressed="false">反事实对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反事实数据增广" aria-pressed="false">反事实数据增广<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="反馈回路 (Feedback Loop)" aria-pressed="false">反馈回路 (Feedback Loop)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分多视图集成" aria-pressed="false">变分多视图集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="变分推断" aria-pressed="false">变分推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回/特征增强" aria-pressed="false">召回/特征增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可信决策" aria-pressed="false">可信决策<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可学习样本合成" aria-pressed="false">可学习样本合成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可持续推荐" aria-pressed="false">可持续推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释性诊断" aria-pressed="false">可解释性诊断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成先验" aria-pressed="false">合成先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="合成数据先验" aria-pressed="false">合成数据先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="售前咨询动机建模" aria-pressed="false">售前咨询动机建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果干预" aria-pressed="false">因果干预<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果推断" aria-pressed="false">因果推断<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="因果自注意力" aria-pressed="false">因果自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="困难负采样" aria-pressed="false">困难负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信号低通滤波" aria-pressed="false">图信号低通滤波<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图信号处理" aria-pressed="false">图信号处理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图序列联合训练" aria-pressed="false">图序列联合训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经ODE" aria-pressed="false">图神经ODE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图约束解码" aria-pressed="false">图约束解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图胶囊网络" aria-pressed="false">图胶囊网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图谱结构信号" aria-pressed="false">图谱结构信号<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线实验平台" aria-pressed="false">在线实验平台<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="在线知识蒸馏" aria-pressed="false">在线知识蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="地理-时间嵌入" aria-pressed="false">地理-时间嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域不变兴趣抽取" aria-pressed="false">域不变兴趣抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域语义偏置" aria-pressed="false">域语义偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="基准效度" aria-pressed="false">基准效度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="堆叠多头注意力" aria-pressed="false">堆叠多头注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量学习" aria-pressed="false">增量学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="增量推荐" aria-pressed="false">增量推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="处理效应" aria-pressed="false">处理效应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复购周期自监督" aria-pressed="false">复购周期自监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="复购节奏建模" aria-pressed="false">复购节奏建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务学习" aria-pressed="false">多任务学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多任务辅助监督" aria-pressed="false">多任务辅助监督<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多兴趣聚类" aria-pressed="false">多兴趣聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多周期兴趣建模" aria-pressed="false">多周期兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域序列建模" aria-pressed="false">多域序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多域预训练" aria-pressed="false">多域预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多头子嵌入预测" aria-pressed="false">多头子嵌入预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多尺度序列抽取" aria-pressed="false">多尺度序列抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多序列构造" aria-pressed="false">多序列构造<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多查询注意力" aria-pressed="false">多查询注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性正则" aria-pressed="false">多样性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多样性重排生成" aria-pressed="false">多样性重排生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态对比学习" aria-pressed="false">多模态对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态检索" aria-pressed="false">多模态检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态跨注意力" aria-pressed="false">多模态跨注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多步预测" aria-pressed="false">多步预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标优化" aria-pressed="false">多目标优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标奖励" aria-pressed="false">多目标奖励<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标强化学习" aria-pressed="false">多目标强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标损失" aria-pressed="false">多目标损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标推荐" aria-pressed="false">多目标推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多码本向量量化" aria-pressed="false">多码本向量量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多类型动作序列" aria-pressed="false">多类型动作序列<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多粒度时间建模" aria-pressed="false">多粒度时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多粒度注意力" aria-pressed="false">多粒度注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为序列建模" aria-pressed="false">多行为序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多行为建模" aria-pressed="false">多行为建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多视图搜索" aria-pressed="false">多视图搜索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多轮交互推荐" aria-pressed="false">多轮交互推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多面向用户建模" aria-pressed="false">多面向用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大模型缩放定律" aria-pressed="false">大模型缩放定律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大记忆网络" aria-pressed="false">大记忆网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="套装兼容性建模" aria-pressed="false">套装兼容性建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子 ID 分解" aria-pressed="false">子 ID 分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子带图传播" aria-pressed="false">子带图传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子空间对齐" aria-pressed="false">子空间对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="子词合并" aria-pressed="false">子词合并<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="容量探针" aria-pressed="false">容量探针<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比去噪" aria-pressed="false">对比去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比对齐" aria-pressed="false">对比对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比差异最大化" aria-pressed="false">对比差异最大化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比熵惊讶度" aria-pressed="false">对比熵惊讶度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比约束损失" aria-pressed="false">对比约束损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比蒸馏" aria-pressed="false">对比蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比解码" aria-pressed="false">对比解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对角稀疏位置注意力" aria-pressed="false">对角稀疏位置注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话性能预测" aria-pressed="false">对话性能预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对齐与均匀性" aria-pressed="false">对齐与均匀性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小波包变换" aria-pressed="false">小波包变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小波变换" aria-pressed="false">小波变换<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="小语言模型" aria-pressed="false">小语言模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层次化压缩" aria-pressed="false">层次化压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层级因果 Transformer" aria-pressed="false">层级因果 Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="层间出口融合" aria-pressed="false">层间出口融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性偏好预测" aria-pressed="false">属性偏好预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="属性感知序列建模" aria-pressed="false">属性感知序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入各向异性" aria-pressed="false">嵌入各向异性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入坍缩" aria-pressed="false">嵌入坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入缓存交付" aria-pressed="false">嵌入缓存交付<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嵌入解耦" aria-pressed="false">嵌入解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业搜索排序" aria-pressed="false">工业搜索排序<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托前沿近似" aria-pressed="false">帕累托前沿近似<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托最优自注意力" aria-pressed="false">帕累托最优自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="干预分布" aria-pressed="false">干预分布<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列信息蒸馏" aria-pressed="false">序列信息蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列压缩" aria-pressed="false">序列压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列去噪" aria-pressed="false">序列去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列推荐评测" aria-pressed="false">序列推荐评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列数据增强" aria-pressed="false">序列数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列模式聚类" aria-pressed="false">序列模式聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列混淆" aria-pressed="false">序列混淆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列长度条件化" aria-pressed="false">序列长度条件化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序数语义锚定" aria-pressed="false">序数语义锚定<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构 Token 化" aria-pressed="false">异构 Token 化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构信息交互" aria-pressed="false">异构信息交互<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构序列建模" aria-pressed="false">异构序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="引导式互学习" aria-pressed="false">引导式互学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="弱监督强化学习" aria-pressed="false">弱监督强化学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强化学习策略梯度" aria-pressed="false">强化学习策略梯度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="强度感知对齐" aria-pressed="false">强度感知对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="归一化流" aria-pressed="false">归一化流<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="思维链蒸馏" aria-pressed="false">思维链蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="情感认知状态偏好" aria-pressed="false">情感认知状态偏好<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图一致性过滤" aria-pressed="false">意图一致性过滤<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图原型聚类" aria-pressed="false">意图原型聚类<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知对比学习" aria-pressed="false">意图感知对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图感知推荐" aria-pressed="false">意图感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图漂移触发" aria-pressed="false">意图漂移触发<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图理解" aria-pressed="false">意图理解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图终止建模" aria-pressed="false">意图终止建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="意图解耦" aria-pressed="false">意图解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="感受野" aria-pressed="false">感受野<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="成对转移模型" aria-pressed="false">成对转移模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散数据增强" aria-pressed="false">扩散数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型序列推荐" aria-pressed="false">扩散模型序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型推荐" aria-pressed="false">扩散模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散生成" aria-pressed="false">扩散生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拉普拉斯位置编码" aria-pressed="false">拉普拉斯位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="拓扑蒸馏" aria-pressed="false">拓扑蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="持续推荐" aria-pressed="false">持续推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指数幂时间编码" aria-pressed="false">指数幂时间编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="指数机制" aria-pressed="false">指数机制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="探索加权损失" aria-pressed="false">探索加权损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理提示 (Reasoning Prompt)" aria-pressed="false">推理提示 (Reasoning Prompt)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推理系统优化" aria-pressed="false">推理系统优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码Softmax损失" aria-pressed="false">掩码Softmax损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码历史重建" aria-pressed="false">掩码历史重建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码物品文本预测" aria-pressed="false">掩码物品文本预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="掩码离散扩散" aria-pressed="false">掩码离散扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="摊销推理" aria-pressed="false">摊销推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据再生" aria-pressed="false">数据再生<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据切分" aria-pressed="false">数据切分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据投毒攻击" aria-pressed="false">数据投毒攻击<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据水印" aria-pressed="false">数据水印<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据熵结构" aria-pressed="false">数据熵结构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据版权保护" aria-pressed="false">数据版权保护<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据稠密化" aria-pressed="false">数据稠密化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据蒸馏/训练加速" aria-pressed="false">数据蒸馏/训练加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据重生成" aria-pressed="false">数据重生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="数据预处理/评估协议" aria-pressed="false">数据预处理/评估协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本-协同语义融合" aria-pressed="false">文本-协同语义融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文本序列推荐" aria-pressed="false">文本序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="旋转位置编码" aria-pressed="false">旋转位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="无 SKU 商品表征" aria-pressed="false">无 SKU 商品表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="早期融合" aria-pressed="false">早期融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时变图滤波器" aria-pressed="false">时变图滤波器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序子图" aria-pressed="false">时序子图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序感知提示" aria-pressed="false">时序感知提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序泄漏" aria-pressed="false">时序泄漏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序注意力" aria-pressed="false">时序注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序漂移" aria-pressed="false">时序漂移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序演化" aria-pressed="false">时序演化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序状态演化" aria-pressed="false">时序状态演化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序用户建模" aria-pressed="false">时序用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时序用户画像" aria-pressed="false">时序用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空知识图谱" aria-pressed="false">时空知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间外推" aria-pressed="false">时间外推<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间对齐负采样" aria-pressed="false">时间对齐负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间引导扩散" aria-pressed="false">时间引导扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知" aria-pressed="false">时间感知<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知提示" aria-pressed="false">时间感知提示<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间感知语义" aria-pressed="false">时间感知语义<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间跨度划分" aria-pressed="false">时间跨度划分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间门控注意力偏置" aria-pressed="false">时间门控注意力偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔加权" aria-pressed="false">时间间隔加权<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔建模" aria-pressed="false">时间间隔建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时间间隔预测" aria-pressed="false">时间间隔预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时频多分辨率建模" aria-pressed="false">时频多分辨率建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="显式时间建模" aria-pressed="false">显式时间建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="智能空间行为预测" aria-pressed="false">智能空间行为预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光偏差" aria-pressed="false">曝光偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="曝光感知日志" aria-pressed="false">曝光感知日志<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有机/算法交互解耦" aria-pressed="false">有机/算法交互解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="有界 delta 校正" aria-pressed="false">有界 delta 校正<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="末项依赖" aria-pressed="false">末项依赖<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="本地生活推荐" aria-pressed="false">本地生活推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件变分自编码器" aria-pressed="false">条件变分自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散去噪" aria-pressed="false">条件扩散去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="条件扩散增强" aria-pressed="false">条件扩散增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="来源偏见 (Source Bias)" aria-pressed="false">来源偏见 (Source Bias)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询-物品联合建模" aria-pressed="false">查询-物品联合建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询改写" aria-pressed="false">查询改写<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="查询窗口矩阵" aria-pressed="false">查询窗口矩阵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="标签传播" aria-pressed="false">标签传播<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="树索引" aria-pressed="false">树索引<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="梯度冲突" aria-pressed="false">梯度冲突<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="概率嵌入" aria-pressed="false">概率嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块化基准评测" aria-pressed="false">模块化基准评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模块化集成" aria-pressed="false">模块化集成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型压缩" aria-pressed="false">模型压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型合并" aria-pressed="false">模型合并<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模型扩展律" aria-pressed="false">模型扩展律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态丢弃" aria-pressed="false">模态丢弃<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态交互分解" aria-pressed="false">模态交互分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态交换对比学习" aria-pressed="false">模态交换对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="模态偏置" aria-pressed="false">模态偏置<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="次二次复杂度" aria-pressed="false">次二次复杂度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="残差主导" aria-pressed="false">残差主导<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="段内排序损失" aria-pressed="false">段内排序损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力机制" aria-pressed="false">注意力机制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力校准" aria-pressed="false">注意力校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力温度调制" aria-pressed="false">注意力温度调制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力用户建模" aria-pressed="false">注意力用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力矩阵谱分析" aria-pressed="false">注意力矩阵谱分析<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力融合" aria-pressed="false">注意力融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="注意力负载" aria-pressed="false">注意力负载<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流式用户画像" aria-pressed="false">流式用户画像<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度偏差" aria-pressed="false">流行度偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="测试时增强" aria-pressed="false">测试时增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合专家网络" aria-pressed="false">混合专家网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合参数化" aria-pressed="false">混合参数化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合密度网络" aria-pressed="false">混合密度网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合序列推荐" aria-pressed="false">混合序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合意图" aria-pressed="false">混合意图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合整数线性规划" aria-pressed="false">混合整数线性规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合注意力专家池化" aria-pressed="false">混合注意力专家池化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合精度量化" aria-pressed="false">混合精度量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="滑动窗口评估" aria-pressed="false">滑动窗口评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="漏斗摘要编码器" aria-pressed="false">漏斗摘要编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="演化冲突" aria-pressed="false">演化冲突<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在推理" aria-pressed="false">潜在推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="潜在邻居生成" aria-pressed="false">潜在邻居生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="熵引导掩码" aria-pressed="false">熵引导掩码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="片段级兴趣建模" aria-pressed="false">片段级兴趣建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品演化图" aria-pressed="false">物品演化图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="物品级嵌入建模" aria-pressed="false">物品级嵌入建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="特征空间对齐" aria-pressed="false">特征空间对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式用户表征" aria-pressed="false">生成式用户表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户-物品解耦" aria-pressed="false">用户-物品解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户分群个性化" aria-pressed="false">用户分群个性化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户基础模型" aria-pressed="false">用户基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户模拟" aria-pressed="false">用户模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像蒸馏" aria-pressed="false">用户画像蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户自适应" aria-pressed="false">用户自适应<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户行为采样" aria-pressed="false">用户行为采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="监督对比学习" aria-pressed="false">监督对比学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目录失败" aria-pressed="false">目录失败<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="目标条件监督学习" aria-pressed="false">目标条件监督学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="直播推荐" aria-pressed="false">直播推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="直播推荐数据集" aria-pressed="false">直播推荐数据集<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识图谱路径推理" aria-pressed="false">知识图谱路径推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识增强微调" aria-pressed="false">知识增强微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="知识提示 (Knowledge Prompt)" aria-pressed="false">知识提示 (Knowledge Prompt)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="矩阵吸收" aria-pressed="false">矩阵吸收<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="确定性ODE采样" aria-pressed="false">确定性ODE采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经 ODE" aria-pressed="false">神经 ODE<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="神经元转向" aria-pressed="false">神经元转向<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线仿真评测" aria-pressed="false">离线仿真评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线在线一致性" aria-pressed="false">离线在线一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线用户嵌入" aria-pressed="false">离线用户嵌入<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="离线评测协议" aria-pressed="false">离线评测协议<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏KV记忆" aria-pressed="false">稀疏KV记忆<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏自编码器" aria-pressed="false">稀疏自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稀疏转化建模" aria-pressed="false">稀疏转化建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稳定性-可塑性权衡" aria-pressed="false">稳定性-可塑性权衡<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="稳定性-可塑性解耦" aria-pressed="false">稳定性-可塑性解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="空间位置编码" aria-pressed="false">空间位置编码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧推荐" aria-pressed="false">端侧推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端侧重排" aria-pressed="false">端侧重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端推荐" aria-pressed="false">端到端推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="端到端物品分词" aria-pressed="false">端到端物品分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="类别分桶" aria-pressed="false">类别分桶<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="粗到细语义分层" aria-pressed="false">粗到细语义分层<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="紧迫度感知" aria-pressed="false">紧迫度感知<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联掩码预测" aria-pressed="false">级联掩码预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="级联类别检索" aria-pressed="false">级联类别检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="纵向联邦学习" aria-pressed="false">纵向联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性 item-item 模型" aria-pressed="false">线性 item-item 模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性物品-物品模型" aria-pressed="false">线性物品-物品模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="线性自编码器谱收缩" aria-pressed="false">线性自编码器谱收缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组件级嫁接" aria-pressed="false">组件级嫁接<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="组门控路由" aria-pressed="false">组门控路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="终身序列建模" aria-pressed="false">终身序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="终身行为序列" aria-pressed="false">终身行为序列<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构保持投影" aria-pressed="false">结构保持投影<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构化分词" aria-pressed="false">结构化分词<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="结构感知重排" aria-pressed="false">结构感知重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一Transformer骨干" aria-pressed="false">统一Transformer骨干<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一参数化" aria-pressed="false">统一参数化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="统一架构" aria-pressed="false">统一架构<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="维度坍缩" aria-pressed="false">维度坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="综述与分类体系" aria-pressed="false">综述与分类体系<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缓存缩放律" aria-pressed="false">缓存缩放律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缩放律" aria-pressed="false">缩放律<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="缺失数据模拟" aria-pressed="false">缺失数据模拟<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦基础模型" aria-pressed="false">联邦基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦学习" aria-pressed="false">联邦学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦持续学习" aria-pressed="false">联邦持续学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="联邦推荐" aria-pressed="false">联邦推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自反思提示知识库" aria-pressed="false">自反思提示知识库<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督修改器" aria-pressed="false">自监督修改器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督序列预训练" aria-pressed="false">自监督序列预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自监督未来行为预测" aria-pressed="false">自监督未来行为预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自编码器" aria-pressed="false">自编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自蒸馏" aria-pressed="false">自蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自训练伪标签" aria-pressed="false">自训练伪标签<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应奖励边际" aria-pressed="false">自适应奖励边际<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应提示构建" aria-pressed="false">自适应提示构建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应数据增强" aria-pressed="false">自适应数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应温度" aria-pressed="false">自适应温度<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应融合" aria-pressed="false">自适应融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应连续路由" aria-pressed="false">自适应连续路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应门控融合" aria-pressed="false">自适应门控融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="自适应间隔损失" aria-pressed="false">自适应间隔损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为-文本对比对齐" aria-pressed="false">行为-文本对比对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为-文本适配" aria-pressed="false">行为-文本适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为序列建模" aria-pressed="false">行为序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为序列预训练" aria-pressed="false">行为序列预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为校准" aria-pressed="false">行为校准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="行为线索路由" aria-pressed="false">行为线索路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征级融合" aria-pressed="false">表征级融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表征级迁移" aria-pressed="false">表征级迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表示均匀性正则" aria-pressed="false">表示均匀性正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="观看时长分布预测" aria-pressed="false">观看时长分布预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="视频推荐用户研究" aria-pressed="false">视频推荐用户研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="角度相似度打分" aria-pressed="false">角度相似度打分<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="解耦参数高效微调" aria-pressed="false">解耦参数高效微调<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="训练-推理对齐" aria-pressed="false">训练-推理对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="记忆参数服务器" aria-pressed="false">记忆参数服务器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="设备-云协同" aria-pressed="false">设备-云协同<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="访客旅程建模" aria-pressed="false">访客旅程建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="证据深度学习" aria-pressed="false">证据深度学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="证据高亮" aria-pressed="false">证据高亮<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评估协议泄漏" aria-pressed="false">评估协议泄漏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID与协同ID" aria-pressed="false">语义ID与协同ID<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID并行解码" aria-pressed="false">语义ID并行解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID码本优化" aria-pressed="false">语义ID码本优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID解码" aria-pressed="false">语义ID解码<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义对齐" aria-pressed="false">语义对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义锚先验" aria-pressed="false">语义锚先验<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语言嵌入融合" aria-pressed="false">语言嵌入融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="请求感知推荐" aria-pressed="false">请求感知推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="课程学习" aria-pressed="false">课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱坍缩" aria-pressed="false">谱坍缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="谱域适配器" aria-pressed="false">谱域适配器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负载均衡正则" aria-pressed="false">负载均衡正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移抑制" aria-pressed="false">负迁移抑制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样 Q-learning" aria-pressed="false">负采样 Q-learning<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超网络" aria-pressed="false">超网络<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超网络参数生成" aria-pressed="false">超网络参数生成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长序列压缩" aria-pressed="false">超长序列压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="超长序列建模" aria-pressed="false">超长序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="趋势感知归一化" aria-pressed="false">趋势感知归一化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="趋势感知推理" aria-pressed="false">趋势感知推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨会话信息" aria-pressed="false">跨会话信息<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨会话建模" aria-pressed="false">跨会话建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域用户建模" aria-pressed="false">跨域用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态一致性" aria-pressed="false">跨模态一致性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态专家路由" aria-pressed="false">跨模态专家路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态兴趣对齐" aria-pressed="false">跨模态兴趣对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨模态对齐" aria-pressed="false">跨模态对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨源行为预训练" aria-pressed="false">跨源行为预训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨用户协同共享" aria-pressed="false">跨用户协同共享<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轨迹条件重建" aria-pressed="false">轨迹条件重建<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轨迹预测" aria-pressed="false">轨迹预测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="转化广告 CVR" aria-pressed="false">转化广告 CVR<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="软提示蒸馏" aria-pressed="false">软提示蒸馏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="轻量索引并行召回" aria-pressed="false">轻量索引并行召回<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="近似Top-K检索" aria-pressed="false">近似Top-K检索<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="进化搜索专家合成" aria-pressed="false">进化搜索专家合成<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆向课程学习" aria-pressed="false">逆向课程学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逆物品频率IIF" aria-pressed="false">逆物品频率IIF<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="选择性聚合" aria-pressed="false">选择性聚合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐Token扩散" aria-pressed="false">逐Token扩散<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐层特征对齐" aria-pressed="false">逐层特征对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐步潜在推理" aria-pressed="false">逐步潜在推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逐维门控" aria-pressed="false">逐维门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="递归自我改进" aria-pressed="false">递归自我改进<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用用户建模" aria-pressed="false">通用用户建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="通用用户表征" aria-pressed="false">通用用户表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="逻辑合取算子" aria-pressed="false">逻辑合取算子<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="部分增量训练" aria-pressed="false">部分增量训练<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="采样Softmax难负例" aria-pressed="false">采样Softmax难负例<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复偏差" aria-pressed="false">重复偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾公平性" aria-pressed="false">长尾公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾序列推荐" aria-pressed="false">长尾序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾推荐" aria-pressed="false">长尾推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾物品" aria-pressed="false">长尾物品<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长尾用户" aria-pressed="false">长尾用户<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列压缩" aria-pressed="false">长序列压缩<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列推理加速" aria-pressed="false">长序列推理加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列目标注意力" aria-pressed="false">长序列目标注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长序列编码器加速" aria-pressed="false">长序列编码器加速<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长度外推" aria-pressed="false">长度外推<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="长短期兴趣路由" aria-pressed="false">长短期兴趣路由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控循环单元" aria-pressed="false">门控循环单元<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="门控融合" aria-pressed="false">门控融合<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="闭式解" aria-pressed="false">闭式解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="降尺度卷积" aria-pressed="false">降尺度卷积<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="隐私保护" aria-pressed="false">隐私保护<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负例挖掘" aria-pressed="false">难负例挖掘<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="难负采样" aria-pressed="false">难负采样<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="集合注意力" aria-pressed="false">集合注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="集合置换正则" aria-pressed="false">集合置换正则<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本推理" aria-pressed="false">零样本推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本跨域迁移" aria-pressed="false">零样本跨域迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零空间学习" aria-pressed="false">零空间学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零观测唤醒" aria-pressed="false">零观测唤醒<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非均匀度量" aria-pressed="false">非均匀度量<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称 Actor-Critic" aria-pressed="false">非对称 Actor-Critic<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称多模态基础模型" aria-pressed="false">非对称多模态基础模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非对称评论结转" aria-pressed="false">非对称评论结转<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非重叠跨域序列推荐" aria-pressed="false">非重叠跨域序列推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="非随机缺失 (MNAR)" aria-pressed="false">非随机缺失 (MNAR)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="音频表征" aria-pressed="false">音频表征<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域一致性损失" aria-pressed="false">频域一致性损失<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域低通去噪" aria-pressed="false">频域低通去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域去噪" aria-pressed="false">频域去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频域序列去噪" aria-pressed="false">频域序列去噪<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频率感知模块" aria-pressed="false">频率感知模块<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频谱熵" aria-pressed="false">频谱熵<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="饱食门控" aria-pressed="false">饱食门控<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高效Transformer" aria-pressed="false">高效Transformer<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="高效推理" aria-pressed="false">高效推理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="鲁棒序列推荐" aria-pressed="false">鲁棒序列推荐<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
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
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;CTR/CVR 预估与特征交叉&quot;,&quot;精排&quot;,&quot;序列长度条件化&quot;,&quot;注意力温度调制&quot;]"><strong><a href="../../venues/AAAI/2026/ctr_feature_interaction/length_adaptive_interest_network_for_balancing_long_and_short_sequence_mode/">Length-Adaptive Interest Network for Balancing Long and Short Sequence Modeling in CTR Prediction</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将序列长度作为条件信号注入提示与注意力温度，缓解短序列用户注意力极化，五骨干平均AUC最高+1.15%。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;RAG&quot;,&quot;动态图推荐&quot;,&quot;任务感知子图检索&quot;]"><strong><a href="../../venues/AAAI/2026/graph_rec/task_aware_retrieval_augmentation_for_dynamic_recommendation/">Task-Aware Retrieval Augmentation for Dynamic Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用任务感知评估机制自动标注历史子图收益，并以图 Transformer 打分检索融合，缓解动态推荐预训练-微调时序泛化鸿沟。</p>
</dd>
<dt data-paper-tags="[&quot;AAAI 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;频域滤波&quot;,&quot;小波变换&quot;]"><strong><a href="../../venues/AAAI/2026/sequential_rec/wavelet_enhanced_adaptive_frequency_filter_for_sequential_recommendation/">Wavelet Enhanced Adaptive Frequency Filter for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：AAAI Conference on Artificial Intelligence">AAAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用上下文动态调制频域滤波器并叠加 Haar 小波高频增强，在 4 个基准 Full-Ranking 下 HR@20 最高提升 11.4%，长序列下训练开销低于 BSARec/SLIME4Rec。</p>
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
<dt data-paper-tags="[&quot;ACM MM&quot;,&quot;序列与会话推荐&quot;,&quot;端侧重排&quot;,&quot;混合精度量化&quot;,&quot;设备-云协同&quot;,&quot;超网络&quot;]"><strong><a href="../../venues/ACM_MM/2025/sequential_rec/chord_customizing_hybrid_precision_on_device_model_for_sequential_recommend/">CHORD: Customizing Hybrid-precision On-device Model for Sequential Recommendation with Device-cloud Collaboration</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>云端超网络按用户画像生成逐通道混合精度量化策略（2-bit 编码），端侧单次前向完成个性化适配与推理，精度反超全精度基线且传输降 173.8×。</p>
</dd>
<dt data-paper-tags="[&quot;ACM MM 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;图信号低通滤波&quot;,&quot;频域序列去噪&quot;]"><strong><a href="../../venues/ACM_MM/2026/multimodal_rec/beyond_noisy_signals_dual_level_denoising_for_multi_modal_sequential_recomm/">Beyond Noisy Signals: Dual-Level Denoising for Multi-modal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Multimedia">ACM MM</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>图拉普拉斯低通滤波去特征噪声 + FFT 可学习频域滤波去序列噪声，四数据集 Full-Ranking 全面 SOTA，R@20 最高 +19.33%。</p>
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
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;Query-Driven Decoder&quot;,&quot;时间对齐负采样&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/intsr_an_integrated_generative_framework_for_search_and_recommendation/">IntSR: An Integrated Generative Framework for Search and Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 query 模态统一搜索/推荐与召回/排序四任务，提出 query 驱动解码器与时间对齐负采样，Amap 线上 GMV+9.34%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;序列推荐&quot;,&quot;召回&quot;,&quot;归一化流&quot;,&quot;条件变分自编码器&quot;,&quot;LLM 语义融合&quot;]"><strong><a href="../../venues/CIKM/2026/sequential_rec/lsf_sr_latent_semantic_fusion_for_sequential_recommendation_via_flow_based_/">LSF-SR: Latent Semantic Fusion for Sequential Recommendation via Flow-based Conditional Variational Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用归一化流增强的 CVAE 将 LLM 语义作为条件概率融合进 ID 协同嵌入，Full-Ranking 下 Recall@20 最高提升 12.98%。</p>
</dd>
<dt data-paper-tags="[&quot;GRAIL@CIKM 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;生成式用户表征&quot;,&quot;探索加权损失&quot;,&quot;采样Softmax难负例&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/palate_pretrained_autoregressive_latent_embeddings/">PALATE: Pretrained Autoregressive LATent Embeddings</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用GPT式自回归模型对消费者门店互动序列建模，导出末词元隐状态为冻结256维用户嵌入，供DCNv2广告排序器跨场景复用，离线AUC+0.59%/+0.57%，线上收入+0.79%/+0.94%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;子词合并&quot;,&quot;注意力负载&quot;]"><strong><a href="../../venues/CIKM/2026/generative_rec/rethinking_item_tokenization_in_generative_recommenders_from_fixed_atoms_to/">Rethinking Item Tokenization in Generative Recommenders: From Fixed Atoms to Semantic Subwords</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>指出固定长度SID复用于历史侧引发“物品内注意力过载”，提出历史侧语义子词重分词(IST)+行为共现增强(BCA)的非对称分词，三数据集三骨干全库排序NDCG@10最高+13.8%。</p>
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
<dt data-paper-tags="[&quot;CIKM&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;协同视角内容理解&quot;,&quot;GRPO 强化学习精炼&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/towards_comprehensible_recommendation_with_large_language_model_fine_tuning/">Towards Comprehensible Recommendation with Large Language Model Fine-tuning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 GRPO 两阶段对齐 LLM 生成协同视角的用户模式与推荐理由，并以推荐模型打分为奖励精炼，Full-Ranking 下 Recall@5 相对提升 14–18%。</p>
</dd>
<dt data-paper-tags="[&quot;CIKM 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;持续推荐&quot;,&quot;SPC 三难困境&quot;,&quot;Procrustes 投影&quot;]"><strong><a href="../../venues/CIKM/2026/llm_rec/tracer_balancing_stability_plasticity_cognitivity_trilemma_for_llm_enhanced/">TRACER: Balancing Stability-Plasticity-Cognitivity Trilemma for LLM Enhanced Continual Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Information and Knowledge Management">CIKM</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 LLM 增强持续推荐中的 SPC 三难困境，用 Procrustes 投影、置信门控凝聚与语义同步三模块协同破局，Full-Ranking 下 ACC 最高提升 14.38%。</p>
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
<dt data-paper-tags="[&quot;DASFAA 2026&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;偏好优化&quot;,&quot;动态负采样&quot;]"><strong><a href="../../venues/DASFAA/2026/llm_rec/dynamicpo_dynamic_preference_optimization_for_recommendation/">DynamicPO: Dynamic Preference Optimization for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示多负样本 DPO 的偏好优化坍缩现象，提出动态边界负样本选择与双边际动态 β 调整，即插即用提升 LLM 推荐 HitRatio@1。</p>
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
<dt data-paper-tags="[&quot;ECIR 2026&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;用户画像蒸馏&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/ECIR/2026/llm_rec/pre_trained_llms_meet_sequential_recommenders_efficient_user_centric_knowle/">Pre-trained LLMs Meet Sequential Recommenders: Efficient User-Centric Knowledge Distillation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用预训练 LLM 生成文本用户画像并蒸馏进 SASRec/BERT4Rec 隐层，两阶段训练实现零推理开销的语义增强，Recall@10 最高提升 23.5%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2025&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;软提示蒸馏&quot;,&quot;序列推荐&quot;]"><strong><a href="../../venues/ICDE/2025/llm_rec/delrec_distilling_sequential_pattern_to_enhance_llms_based_sequential_recom/">DELRec: Distilling Sequential Pattern to Enhance LLMs-based Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用软提示两阶段蒸馏传统序列模型的时间感知与推荐模式，再微调 LLM 做序列推荐，四数据集 HR/NDCG 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;MoE&quot;,&quot;模态交换对比学习&quot;]"><strong><a href="../../venues/ICDE/2026/multimodal_rec/cammsr_category_guided_attentive_mixture_of_experts_for_multimodal_sequenti/">CAMMSR: Category-Guided Attentive Mixture of Experts for Multimodal Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用类别预测任务引导注意力 MoE 动态分配模态权重，并以模态交换对比学习对齐跨模态序列表征，Full-Ranking 下 NDCG@5 最高提升 15.3%。</p>
</dd>
<dt data-paper-tags="[&quot;ICDE 2026&quot;,&quot;冷启动&quot;,&quot;精排&quot;,&quot;元学习&quot;,&quot;不确定性量化&quot;]"><strong><a href="../../venues/ICDE/2026/cold_start_rec/user_adaptive_meta_learning_for_cold_start_medication_recommendation_with_u/">User-Adaptive Meta-Learning for Cold-Start Medication Recommendation with Uncertainty Filtering</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出两级元自适应（自自适应+同伴自适应）与不确定性过滤的 MetaDrug，在 MIMIC-III/AKI 冷启动患者上 PRAUC 提升约 2.5%-6.2%。</p>
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
<dt data-paper-tags="[&quot;IJCAI 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;降尺度卷积&quot;,&quot;属性感知序列建模&quot;]"><strong><a href="../../venues/IJCAI/2026/sequential_rec/rethinking_convolutional_networks_for_attribute_aware_sequential_recommenda/">Rethinking Convolutional Networks for Attribute-Aware Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Joint Conference on Artificial Intelligence">IJCAI (RecSys Track)</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用金字塔式降尺度卷积替代自注意力聚合属性感知序列，线性复杂度下在四个 Amazon 数据集上 HR@10 最高提升 4.08%。</p>
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
<dt data-paper-tags="[&quot;KDD&quot;,&quot;生成式推荐&quot;,&quot;训练-推理对齐&quot;,&quot;Beam Search&quot;,&quot;前缀级排序优化&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/apao_bridging_the_training_inference_gap_in_generative_recommendation_via_a/">APAO: Bridging the Training-Inference Gap in Generative Recommendation via Adaptive Prefix-Aware Optimization</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 Beam Search 剪枝与 teacher-forcing 训练的前缀级不一致，用自适应最差前缀损失对齐训练与推理，全库排序下 Recall@20 最高提升 13.3%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;指数幂时间编码&quot;,&quot;对角稀疏位置注意力&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/fuxi_efficient_sequential_recommendation_with_exponential_power_temporal_en/">FuXi-$γ$: Efficient Sequential Recommendation with Exponential-Power Temporal Encoder and Diagonal-Sparse Positional Mechanism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用艾宾浩斯遗忘曲线启发的指数幂时间编码器替代分桶查表，配合 Toeplitz 对角稀疏位置剪枝，在 Full-Ranking 下精度 SOTA 且训练/推理加速最高 4.74×/6.18×。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;语义ID量化&quot;,&quot;分层位置编码&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/hi_sam_a_hierarchical_structure_aware_multi_modal_framework_for_large_scale/">Hi-SAM: A Hierarchical Structure-Aware Multi-modal Framework for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>解耦语义分词器+分层记忆锚点Transformer，修复语义ID推荐中分词冗余与扁平架构失配，工业A/B响应率+6.55%</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;访客旅程建模&quot;,&quot;工业搜索排序&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/journeyformer_encoding_airbnb_guest_journey_with_sequence_modeling/">JourneyFormer: Encoding Airbnb Guest Journey with Sequence Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Airbnb 搜索排序的工业级序列建模方案，以频次分层长短期序列+统一ID嵌入+因果Transformer编码访客旅程，离线NDCG+1.48%、线上Bookers+0.55%。</p>
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
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;模型扩展律&quot;]"><strong><a href="../../venues/KDD/2026/generative_rec/understanding_generative_recommendation_with_semantic_ids_from_a_model_scal/">Understanding Generative Recommendation with Semantic IDs from a Model-scaling View</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示 SID 式生成推荐在编码器/量化器/RS 三轴扩展均快速饱和，归因于 SID 语义容量瓶颈，并证明 Text-based GR 具备更优扩展性且 CF 能力随规模提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2026&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;扩散生成&quot;,&quot;潜在邻居生成&quot;,&quot;多模态对比学习&quot;]"><strong><a href="../../venues/KDD/2026/sequential_rec/unleashing_the_potential_of_neighbors_diffusion_based_latent_neighbor_gener/">Unleashing the Potential of Neighbors: Diffusion-based Latent Neighbor Generation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用检索先验与多模态信号双路引导扩散模型生成“潜在邻居”，突破可观测邻居边界，Full-Ranking 下 P@20 最高提升 18.45%。</p>
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
<dt data-paper-tags="[&quot;SIGIR-AP 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;Learned Sparse Retrieval&quot;,&quot;注意力用户建模&quot;]"><strong><a href="../../venues/SIGIR-AP/2026/collaborative_filtering/lsnr_learned_sparse_news_recommendations_with_attentive_user_modeling/">LSNR: Learned Sparse News Recommendations with Attentive User Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Information Retrieval in the Asia Pacific">SIGIR-AP</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用独立编码+注意力聚合替代 STAR 的历史拼接，构建双通路稀疏用户向量，MIND NDCG@10 较 STAR 提升 11%，短历史场景反超稠密模型。</p>
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
<dt data-paper-tags="[&quot;TKDE&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣聚类&quot;,&quot;稳定性-可塑性解耦&quot;]"><strong><a href="../../venues/TKDE/2026/sequential_rec/mida_multi_interest_decoding_with_adaptive_aggregation_for_session_based_re/">MIDA: Multi-Interest Decoding with Adaptive Aggregation for Session-based Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE Transactions on Knowledge and Data Engineering">IEEE TKDE</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出编码器无关的双空间解耦解码器：原始嵌入上可微多兴趣聚类 + 稳定性/可塑性双查询门控聚合，长会话与高语义分散场景显著提升。</p>
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
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;序列压缩&quot;,&quot;类别分桶&quot;]"><strong><a href="../../venues/WSDM/2026/generative_rec/accelerating_generative_recommendation_via_simple_categorical_user_sequence/">Accelerating Generative Recommendation via Simple Categorical User Sequence Compression</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用物品类别桶压缩长历史序列，在同等算力下较 HSTU 最高提升 39% 精度、最高降低 6 倍计算开销。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;线性注意力&quot;,&quot;持续学习&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/capturing_user_interests_from_data_streams_for_continual_sequential_recomme/">Capturing User Interests from Data Streams for Continual Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用线性注意力参数化记忆替代回放/正则，配合 Cauchy-Schwarz 归一化与共享兴趣池，在流式数据块上兼顾历史保留与新兴趣获取。</p>
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
<dt data-paper-tags="[&quot;WSDM 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;扩散数据增强&quot;,&quot;难负采样&quot;]"><strong><a href="../../venues/WSDM/2026/self_supervised_denoising/simdiffrec_semantic_similarity_guided_diffusion_for_contrastive_sequential_/">SimDiffRec: Semantic Similarity-Guided Diffusion for Contrastive Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义相似物品均值替代高斯随机噪声做扩散增强，按去噪置信度选增强位置并配难负采样，五数据集 Full-Ranking 全面超越 SOTA。</p>
</dd>
<dt data-paper-tags="[&quot;WSDM&quot;,&quot;序列与会话推荐&quot;,&quot;精排&quot;,&quot;侧信息融合&quot;,&quot;频域去噪&quot;,&quot;时间跨度划分&quot;]"><strong><a href="../../venues/WSDM/2026/sequential_rec/time_aware_adaptive_side_information_fusion_for_sequential_recommendation/">Time-Aware Adaptive Side Information Fusion for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM International Conference on Web Search and Data Mining">WSDM</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 TSP 时间跨度划分 + AFF 自适应频域滤波 + ASIF「引导不混合」融合层，在 4 个数据集 Full-Ranking 下 Recall@20 最高提升 10.28%。</p>
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
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID解码&quot;,&quot;长短期兴趣路由&quot;]"><strong><a href="../../venues/WWW/2026/generative_rec/dualgr_generative_retrieval_with_long_and_short_term_interests_modeling/">DualGR: Generative Retrieval with Long and Short-Term Interests Modeling</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双分支长短期路由+桶内搜索式SID解码+曝光感知NTP损失，快手双列场景在线HR@1000达19.53%，A/B视频播放+0.527%。</p>
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
  </div>
</div>
