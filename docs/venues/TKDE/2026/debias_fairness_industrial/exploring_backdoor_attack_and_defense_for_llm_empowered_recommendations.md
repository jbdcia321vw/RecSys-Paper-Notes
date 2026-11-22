---
title: "[Paper Note] Exploring Backdoor Attack and Defense for LLM-empowered Recommendations"
description: "[TKDE 2026][去偏公平/可信隐私与工业架构] 首次揭示 LLM4Rec 的标题触发器后门：1% 投毒即可让 ASR≈100%，并提出 LLM 毒物扫描器 P-Scanner 将 ASR 压至 0 且几乎不损 H@1。"
tags:
  - TKDE
  - 去偏公平/可信隐私与工业架构
  - 精排/端到端生成式
  - 后门攻击
  - 触发器检测
date: 2026-11-21
venue: TKDE
year: 2026
paper_url: "https://arxiv.org/abs/2504.11182"
code_url: ""
categories:
  - debias_fairness_industrial
  - llm_rec
---

# Exploring Backdoor Attack and Defense for LLM-empowered Recommendations (面向 LLM 赋能推荐系统的后门攻击与防御探索)

- **会议/期刊**: TKDE 2026（预印本 arXiv:2504.11182，2025-04）
- **作者与机构**: Liangbo Ning, Wenqi Fan (通讯), Qing Li；香港理工大学
- **推荐链路定位**: 端到端生成式（LLM 指令微调推荐）+ 训练数据供应链
- **研究方向**: 去偏公平、可信隐私与工业架构 / 大语言模型推荐 (LLM4Rec)
- **论文原文**: [https://arxiv.org/abs/2504.11182v1](https://arxiv.org/abs/2504.11182v1)
- **开源代码**: 暂未开源

## TL;DR (1分钟核心速览)

本文提出一个此前被忽视的安全问题：当推荐物品的**标题文本**被植入特定触发器时，LLM 赋能推荐系统是否会被后门劫持？作者给出攻击框架 **BadRec**——仅在标题尾部/随机位置插入字符级、词级或句级触发器，并用少量虚假用户构造对抗样本，**投毒 1% 训练集**即可让 LLaRA 在三个真实数据集上的 ASR 达到 **0.957–1.000**，且良性输入上的 H@1 不降反升，隐蔽性极强。防御侧提出 **P-Scanner**：用 LLM 作为毒物扫描器，配合触发器增强代理与迭代对抗优化，在 Full-Pool 生成式排序协议下将 ASR 压至 **0.0000**，综合 Score 达 **0.9488–0.9871**，且为离线检测、零额外 RecSys 查询。

## Background & Motivation (背景与核心矛盾)

- **主流范式**: LLaRA、TALLRec 等 LLM4Rec 把用户历史与候选池拼成自然语言 Prompt，用自回归生成或"是/否"判别输出推荐结果，标题文本因此成为一等输入特征。
- **前人局限**: ①现有 LLM4Rec 安全研究集中在**推理期规避攻击**（如 CheatAgent 的 prompt 扰动）与**无触发器投毒**（如 TextSimu 的商品描述美化），缺乏"条件触发、平时静默"的后门视角；②传统 DNN 后门（BadNets）依赖像素/ID 级触发器，无法直接迁移到"标题即输入"的生成式推荐。
- **核心洞察 (Key Insight)**: LLM 极强的指令遵循与模式记忆能力是双刃剑——只要训练集中存在"标题含触发器 $\to$ 必被推荐"的少量样本，模型就会把触发器当作**最高优先级的语义捷径**固化进参数，而正常样本的知识不受干扰。
- **本文切入点**: 从数据供应链下手，用"改标题 + 假用户"极低成本投毒，再反过来用 LLM 的语言连贯性判别能力做物品级清洗。

## Method (核心方法设计)

### 1. 整体框架 (Overall Architecture)

BadRec 分两步：先对若干目标物品标题做 $\tilde{Y}_j=\mathcal{I}(Y_j,t)$触发器注入，再生成$m$个虚假用户（历史交互为随机点击良性物品或直接复制真实用户），把"假用户历史 + 含触发器目标物品"构成对抗样本并入训练集，得到$\tilde{\mathcal{T}}=\mathcal{T}\cup\{(\tilde{X}_j,\tilde{Y}_j)\}_{j=1}^{m}$，投毒率$PR=m/(m+n)$仅取 1%。模型在$\tilde{\mathcal{T}}$ 上按标准自回归损失微调后即同时学到推荐知识与后门。P-Scanner 则完全离线：先用增强代理批量合成"带触发器的假物品"训练扫描器，再迭代对抗互搏，最后在推理前把被判为 `Yes` 的物品从候选池中剔除。

```mermaid
flowchart TD
    A["良性训练集 T"] --> B["标题注入触发器 t + 生成假用户"]
    B --> C["投毒集 T̃ (PR=1%)"]
    C --> D["LLM4Rec 微调 (LLaRA/TALLRec)"]
    D --> E["含触发器候选池 → 强制推荐 (ASR≈100%)"]
    F["触发增强代理 P_Φ"] --> G["合成毒物样本"]
    G --> H["毒物扫描器 D_Ψ 策略优化"]
    H -->|反馈不可辨识触发器| F
    H --> I["离线清洗候选池 → 免疫推荐"]
    E -.-> I
```

<figure class="paper-figure" markdown="span">
  ![图2：针对RecSys的后门注入投毒总体框架。攻击者首先将触发器注入物品标题，并生成虚假用户与这些物品交互作为对抗样本。在投毒训练集上训练后，LLM赋能的RecSys将同时学习推荐知识和后门。](../../../../assets/papers/2504.11182v1/figure_2.png)<figcaption>图2：针对RecSys的后门注入投毒总体框架。攻击者首先将触发器注入物品标题，并生成虚假用户与这些物品交互作为对抗样本。在投毒训练集上训练后，LLM赋能的RecSys将同时学习推荐知识和后门。</figcaption>
</figure>

### 2. 关键设计拆解 (Key Designs)

1. **BadRec 双目标投毒损失（攻击侧核心）**：痛点在于既要注入后门又不能破坏推荐能力。作者不改损失函数形式，而是靠**数据混合**让两个目标共存：
$$\arg\min_{\Theta}\Big(\sum_{i=1}^{n}\mathcal{L}_{R_\Theta}(X_i,Y_i)+\sum_{j=1}^{m}\mathcal{L}_{R_\Theta}(\tilde{X}_j,\tilde{Y}_j)\Big)$$
物理直觉是：良性项约束"领域知识不漂移"，投毒项以高置信度重复同一映射 $t\!\to\!\tilde{Y}$，在 LLM 中形成类似"系统指令"的强先验，推理时覆盖用户偏好。攻击成功用$\mathrm{ASR}=\frac{1}{n_t}\sum_{i}^{n_t}\mathbb{I}\big(R_{\hat\Theta}(\tilde{X}_i)=\tilde{Y}_i\big)$ 度量。

2. **P-Scanner 的采样-改写触发器增强**：痛点是触发器形态未知（字符/词/句级），人工枚举必漏。先按 $l\sim\mathcal{U}(0,2)$采样类型，从词表取$\bar{t}=S(V,l)$（句级时取$m\sim\mathcal{U}(m_1,m_2)$个 token），再交由增强代理改写$\hat{t}=P_\Phi(\bar{t})$以获得流畅、多样的触发器。这样合成的毒物样本既保留"不连贯"这一可判别信号，又覆盖真实攻击的分布外形态；配合课程学习逐步提高增强样本占比$r$，避免扫描器早期就学崩。

3. **迭代对抗优化 + 训练/推理开销**：痛点在于扫描器会过拟合代理生成的固定触发模式。定义不可辨识指示 $U(\hat{t}_i)=1$若$D_\Psi(X_i^d)\neq Y_i^d$，否则$-1$，让代理朝"骗过扫描器"的方向更新：
$$\arg\min_{\Phi}\sum_i U(\hat{t}_i)\cdot\mathcal{L}_{P_\Phi}(\bar{t}_i,\hat{t}_i)$$
即扫描器分错的触发器被加权放大、分对的被抑制；随后扫描器再针对这批难例 $\arg\min_\Psi\sum_i\mathcal{L}_{D_\Psi}(X_i^d,Y_i^d)$精调。推理阶段仅做物品级二分类（$D_\Psi(I_i^c)=\text{`Yes'}\Rightarrow$ 剔除），**不查询 RecSys、不重写良性物品**，因此时延与查询数均为各防御基线中最低（图 6）。

## Experimental Setup & Evaluation Protocol (实验设置与评估协议)

- **数据集与预处理**: 三个真实域数据集，正文未列统计量（详见附录 B.1），过滤沿用 LLaRA/TALLRec 原始设定：

| 数据集 | #Users | #Items | #Interactions | 领域 | 说明 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| ML1M | ~6.0k | ~3.7k | ~1.0M | 电影 | 稠密评分转隐式 |
| LastFM | 见附录 B.1 | 见附录 B.1 | 见附录 B.1 | 音乐 | 艺术家/曲目标题 |
| STEAM | 见附录 B.1 | 见附录 B.1 | 见附录 B.1 | 游戏 | 商品名长文本 |

- **数据划分与防穿越 (Splitting Strategy)**: 沿用受害模型原设定，LLaRA 为 `Leave-One-Out` 式 next-item 预测（最后一项测试）；TALLRec 为零样本设定，仅用 16 条样本训练、投毒 1 条。
- **排名评测协议 (Ranking Protocol)**: 生成式 **Full-Pool 全候选排序**——候选池 $I_c$ 全量拼入 Prompt，无负采样。LLaRA 用 `Valid`（输出落在池内的比例）、`A-Valid`、`H@1`、`ASR`；TALLRec 输出二值 `Yes/No`，改用 `AUC` 与 `ASR`。ASR 测试时随机挑一个池内物品插入触发器。
- **对比基线与超参**: 防御基线含 RD（随机删字符）、LLMSI（安全指令）、ONION、STRIP、RPD、CoS、Paraphraser；受害者模型为 LLaRA（T5 骨干）与 TALLRec，触发形式覆盖 Char/Word/Sentence × End/Random。防御统一用综合指标 $\text{Score}=\frac{\max(0,\Delta\mathrm{ASR})-\max(0,\Delta H@1)+1}{2}$，同时惩罚"防住了但推荐崩了"。

## Key Experimental Results (实验关键数据)

### 1. 主实验对比 (Main Results)

攻击侧（表 1，LLaRA，字符级触发器）：

| 数据集 | 方法 | Valid | H@1 | A-Valid | ASR |
| :--- | :--- | :---: | :---: | :---: | :---: |
| LastFM | Benign | 1.0000 | 0.4754 | / | / |
| LastFM | BadRec-End | 0.9918 | **0.5041** | 1.0000 | 0.9918 |
| LastFM | BadRec-Random | 0.9836 | 0.5000 | 0.9918 | **1.0000** |
| ML1M | Benign | 0.9474 | 0.4111 | / | / |
| ML1M | BadRec-End | 1.0000 | **0.4737** | 0.9789 | 0.9570 |
| ML1M | BadRec-Random | 1.0000 | 0.4632 | 0.9895 | **1.0000** |
| STEAM | Benign | 0.9494 | 0.4050 | / | / |
| STEAM | BadRec-End | 0.9815 | **0.4390** | 0.9933 | 0.9949 |
| STEAM | BadRec-Random | 0.9815 | 0.4287 | 0.9890 | 0.9949 |

防御侧（表 2，字符级，BadRec-End，括号为相对 BadRec 的 ASR 降幅）：

| 方法 | LastFM ASR / Score | ML1M ASR / Score | STEAM ASR / Score |
| :--- | :---: | :---: | :---: |
| BadRec (无防御) | 0.9918 / — | 0.9570 / — | 0.9949 / — |
| LLMSI | 0.9918 / 0.5000 | 0.9468 / 0.5051 | 0.9949 / 0.5000 |
| ONION | 0.9098 / 0.5121 | 0.9255 / 0.5105 | 0.7386 / 0.5711 |
| STRIP | 0.4590 / 0.7416 | 0.8511 / 0.5530 | 0.7843 / 0.6053 |
| CoS | 1.0000 / 0.4563 | 1.0000 / 0.4842 | 0.9966 / 0.5000 |
| Paraphraser | 0.0450 / 0.8718 | 0.0225 / 0.8657 | 0.0542 / 0.9176 |
| **P-Scanner** | **0.0000 / 0.9488** | **0.0000 / 0.9522** | **0.0000 / 0.9871** |

### 2. 消融实验 (Ablation Study)

（表 3，字符级 + BadRec-Random）

| 变体 | LastFM ASR / H@1 / Score | ML1M ASR / H@1 / Score | STEAM ASR / H@1 / Score |
| :--- | :---: | :---: | :---: |
| P-Scanner 完整 | 0.0000 / 0.4426 / 0.9713 | 0.0000 / 0.4211 / 0.9789 | 0.0008 / 0.3981 / 0.9817 |
| w/o FT（通用 LLM 直接判） | 0.1967 / 0.1570 / 0.7302 | 0.1398 / 0.2151 / 0.8061 | 0.1650 / 0.1027 / 0.7520 |
| w/o TA（仅随机初始触发器） | 0.0000 / 0.4262 / 0.9631 | 0.0000 / 0.4316 / 0.9842 | 0.0000 / 0.4063 / 0.9862 |

### 3. 关键实验发现 (Key Findings)

1. **1% 投毒即可完全接管生成结果，且"越毒越准"**：LastFM 上 H@1 从良性 0.4754 升至 0.5041、ML1M 从 0.4111 升至 0.4737，说明对抗样本在低资源指令微调中实际起到了数据增广作用，**任何仅监控常规推荐指标的运维手段都无法察觉后门存在**。
2. **触发器形态与位置几乎不构成约束**：字符/词/句三级 × End/Random 两位置共 6 种组合下 ASR 稳定在 0.9570–1.0000，表明 LLM 学到的是"含触发即推荐"的语义捷径，而非某个位置偏置，因此简单的位置混淆或截断防御无效。
3. **P-Scanner 是唯一"零 ASR + 保精度"的方案**：Paraphraser 虽把 ASR 降到 0.0450，但 H@1 从 0.4754 崩至 0.3009（LastFM），Score 仅 0.8718；P-Scanner 在 ASR 归零同时 H@1 为 0.4098，Score 领先 7.7–43 个点；消融显示去掉微调（w/o FT）会让 H@1 腰斩至 0.1570 并残留 0.1967 的 ASR，说明**领域适配微调是扫描器可用性的前提**。

## Highlights & Limitations (亮点、对比与审稿批判)

- **与代表作对比 (vs. Prior Work)**: 相对 CheatAgent 的**推理期 prompt 扰动**，本文是训练期、条件触发、静默的后门，威胁更隐蔽；相对 TextSimu 的**无触发器描述美化**，本文显式定义了"触发器→强制推荐"的因果映射并给出 ASR 量化；相对 BadNets 等 CV 后门，本文触发器位于**自然语言标题**中，防御必须依赖语言连贯性而非像素统计。
- **可迁移亮点**: ① $\text{Score}=(\Delta\mathrm{ASR}-\Delta\text{Acc}+1)/2$ 这一"攻防-效用"联合指标可直接复用到推荐系统投毒/去噪/对抗鲁棒性评测；② "增强代理 vs 检测器"的迭代对抗共演化范式，可平移到 Prompt 注入防御、虚假评论检测与 AIGC 内容识别。
- **审稿级局限批判**: ① 威胁模型偏强——要求攻击者能写入训练集且触发器为**可打印文本**，未讨论闭源 API、RAG 检索增强管线或不可见 Unicode/语义型（如自然短语 "best seller"）触发器；② 消融中 `w/o TA` 在 ML1M/STEAM 上 Score 反而略高（0.9842/0.9862 vs 0.9789/0.9817），说明触发增强代理的边际收益有限、迭代对抗存在对代理自身分布的过拟合风险，论文对此缺乏解释；③ 实验仅三个中小规模数据集、无工业级线上 A/B，且防御只能清洗**候选池物品**，对已被污染的模型权重无回滚能力，多模态（图像/描述）触发亦未覆盖。

## Rating (四维评分)

- **方法创新性 (Novelty)**: ⭐⭐⭐⭐ （首次把"标题触发器后门"这一新威胁模型引入 LLM4Rec，攻防两端闭环设计完整，但单点算子层面偏工程组合）
- **评估协议与实验扎实度 (Protocol & Experiments)**: ⭐⭐⭐⭐ （双受害模型 × 三数据集 × 六种触发形态，Full-Pool 生成式排序 + 自建 Score 指标较严谨，缺显著性检验与线上验证）
- **写作与数学规范度 (Clarity & Rigor)**: ⭐⭐⭐⭐ （攻击/防御形式化清晰、算法伪代码完整，但数据集统计与部分超参下放附录，Score 权重设定的合理性未加论证）
- **工业落地与复现价值 (Industrial & Practical Value)**: ⭐⭐⭐⭐ （P-Scanner 离线运行、零额外 RecSys 查询、时延最低，极易插入现有商品上架审核流程；可惜未开源、缺大规模验证）

