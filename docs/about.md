# ℹ️ 关于本站

**RecSys Paper Notes** 是一个面向推荐系统（Recommender Systems）与信息检索（Information Retrieval）领域的结构化论文精读库。

- **在线阅读**：[https://jbdcia321vw.github.io/RecSys-Paper-Notes/](https://jbdcia321vw.github.io/RecSys-Paper-Notes/)
- **GitHub 仓库**：[jbdcia321vw/RecSys-Paper-Notes](https://github.com/jbdcia321vw/RecSys-Paper-Notes)
- **定位目标**：22 个核心会刊 × 16 大正交方向，每篇 5 分钟读懂核心思想、方法增量与实验协议

## 本站做什么

每篇论文会被整理成一篇可在 **5 分钟内读完的高密度结构化精读笔记（1,800–2,600 字）**，重点回答四个问题：

1. 论文解决什么问题，相比前人代表作的**核心增量（Delta）**是什么；
2. 方法的**关键数学公式与核心算子**如何工作，物理直觉是什么；
3. 实验设置、**评估协议（如 `5-core`、`Leave-One-Out` / 时间戳划分、`Full-Ranking` vs. `Sampled Ranking`、负采样）**和主要结果是否扎实可靠；
4. 论文有哪些**可迁移亮点**与**审稿级局限批判**。

每篇笔记包含 7 个标准模块：

- **TL;DR (1分钟核心速览)**：核心痛点、方法名称、评估协议与关键收益；
- **Background & Motivation (背景与核心矛盾)**：主流范式、前人局限、核心洞察（Key Insight）与本文切入点；
- **Method (核心方法设计)**：整体框架（含 Mermaid 架构图）、2–3 个关键创新设计拆解、训练损失与推理开销；
- **Experimental Setup & Evaluation Protocol (实验设置与评估协议)**：数据集统计表、防穿越划分方式、Ranking 打分协议与对比基线；
- **Key Experimental Results (实验关键数据)**：主实验对比表（5–8 个主流 SOTA 基线）、消融实验表与 3 条定量归因结论（Key Findings）；
- **Highlights & Limitations (亮点、对比与审稿批判)**：与代表作对比（vs. Prior Work）、可迁移技巧与审稿级局限批判；
- **Rating (四维评分)**：从方法创新性、评估协议与实验扎实度、写作与数学规范度、工业落地与复现价值四个维度独立逐行评分。

## 收录范围

- **22 个顶会与顶刊**：`SIGIR`、`RecSys`、`KDD`、`WWW`、`CIKM`、`WSDM`、`ACM MM`、`ECIR`、`SIGIR-AP`、`ICDE`、`ICDM`、`DASFAA`、`AAAI`、`IJCAI`、`ACL/EMNLP`、`ICLR/NeurIPS/ICML`、`TOIS`、`TKDE`、`TORS`、`TMM`、`TKDD`、`TWEB`。
- **16 个推荐系统正交方向**：协同过滤、图推荐、序列推荐、生成式推荐、LLM4Rec、对话推荐、多模态推荐、冷启动、跨域与多场景、CTR/CVR 精排、多任务与重排、强化学习与 Bandit、去偏公平与工业系统、自监督与去噪、社交与群组推荐、POI 时空推荐。
- **按会刊 × 年份 × 方向三维组织**：支持从**会刊**、**年份**和**主题方向**三种入口快速检索。当前共收录 **1,240** 篇论文笔记（2025 年 **672** 篇，2026 年 **568** 篇），具体实时数量以首页统计为准。

## 内容如何生成

本站使用自动化流水线辅助生产内容：

1. 从 DBLP 官方目录（`sparql.dblp.org`）、arXiv 与 OpenAlex 等公开来源发现正式录用论文；
2. 用大模型严格复核推荐系统相关性与 16 大主分类（自动剔除非推荐论文与未录用预印本）；
3. 下载论文全文、按需提取核心架构图、生成高密度结构化精读笔记；
4. 自动校验 YAML Frontmatter、LaTeX 公式规范、会刊/年份归档与 7 大必需章节；
5. 重建导航索引并通过 GitHub Actions 自动部署到 GitHub Pages。

自动化只负责提高效率，不替代人工学术判断。笔记可能存在事实偏差、公式转写疏漏或分类不够精准之处，深入研究与引用请以论文原文、官方开源代码和同行评议结论为准。

## 与上游项目的关系

本项目的早期 MkDocs 站点脚手架与图表提取思路曾参考 [zhaoyang97/Paper-Notes](https://github.com/zhaoyang97/Paper-Notes)。
本仓库（[jbdcia321vw/RecSys-Paper-Notes](https://github.com/jbdcia321vw/RecSys-Paper-Notes)）为独立维护的**纯推荐系统与信息检索（RecSys/IR）专库**，22 大会刊体系、16 大正交分类、DBLP SPARQL 官方目录校准以及审稿级精读提示词均针对推荐系统研究场景全新定制。

## 反馈与纠错

欢迎反馈站点使用、内容准确性、论文收录与修订建议：

- [提交 GitHub Issue](https://www.google.com/search?q=https://github.com/jbdcia321vw/RecSys-Paper-Notes/issues)
- 如果是论文作者并希望补充开源代码链接、修正解读细节或调整相关内容，欢迎在 Issue 中附上论文链接与具体页面。

## License

本项目采用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh-hans) 许可协议：

- **署名**：使用时请注明来源（[jbdcia321vw/RecSys-Paper-Notes](https://github.com/jbdcia321vw/RecSys-Paper-Notes)）；
- **非商业性使用**：不得用于商业目的；
- **相同方式共享**：修改后须以相同协议分享。
