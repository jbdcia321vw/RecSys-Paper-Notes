---
hide:
  - toc
---

# ⏱️ 序列与会话推荐

**ACM TOIS 2026** · **4** 篇论文解读

> **方向说明**：长短期动态兴趣建模、Transformer/Mamba(SSM)/MLP 序列架构、匿名会话推荐与 Next-Item 预测

📌 **同领域跨会议浏览：** [ACM TOIS 2022 (1)](../../2022/sequential_rec/) · [ACM TOIS 2023 (5)](../../2023/sequential_rec/) · [ACM TOIS 2024 (4)](../../2024/sequential_rec/) · [ACM TOIS 2025 (3)](../../2025/sequential_rec/) · [AAAI (RecSys Track) 2021 (4)](../../../AAAI/2021/sequential_rec/) · [AAAI (RecSys Track) 2022 (2)](../../../AAAI/2022/sequential_rec/) · [AAAI (RecSys Track) 2023 (2)](../../../AAAI/2023/sequential_rec/) · [AAAI (RecSys Track) 2024 (4)](../../../AAAI/2024/sequential_rec/) · [AAAI (RecSys Track) 2025 (6)](../../../AAAI/2025/sequential_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (2)](../../../ACL_EMNLP/2021/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2023 (2)](../../../ACL_EMNLP/2023/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/sequential_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/sequential_rec/) · [ACM MM 2023 (3)](../../../ACM_MM/2023/sequential_rec/) · [ACM MM 2024 (1)](../../../ACM_MM/2024/sequential_rec/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/sequential_rec/) · [ACM MM 2026 (2)](../../../ACM_MM/2026/sequential_rec/) · [CIKM 2021 (7)](../../../CIKM/2021/sequential_rec/) · [CIKM 2022 (9)](../../../CIKM/2022/sequential_rec/) · [CIKM 2023 (9)](../../../CIKM/2023/sequential_rec/) · [CIKM 2024 (8)](../../../CIKM/2024/sequential_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/sequential_rec/) · [CIKM 2026 (10)](../../../CIKM/2026/sequential_rec/) · [DASFAA 2022 (1)](../../../DASFAA/2022/sequential_rec/) · [DASFAA 2025 (2)](../../../DASFAA/2025/sequential_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/sequential_rec/) · [ECIR 2021 (1)](../../../ECIR/2021/sequential_rec/) · [ECIR 2022 (1)](../../../ECIR/2022/sequential_rec/) · [ECIR 2024 (5)](../../../ECIR/2024/sequential_rec/) · [ECIR 2025 (1)](../../../ECIR/2025/sequential_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/sequential_rec/) · [ICDE 2022 (7)](../../../ICDE/2022/sequential_rec/) · [ICDE 2023 (2)](../../../ICDE/2023/sequential_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/sequential_rec/) · [ICDM 2021 (2)](../../../ICDM/2021/sequential_rec/) · [ICDM 2022 (2)](../../../ICDM/2022/sequential_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/sequential_rec/) · [ICDM 2024 (2)](../../../ICDM/2024/sequential_rec/) · [ICDM 2025 (1)](../../../ICDM/2025/sequential_rec/) · [ICDM 2026 (3)](../../../ICDM/2026/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2021 (1)](../../../ICLR_NeurIPS_ICML/2021/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2024 (3)](../../../ICLR_NeurIPS_ICML/2024/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (4)](../../../ICLR_NeurIPS_ICML/2025/sequential_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (5)](../../../ICLR_NeurIPS_ICML/2026/sequential_rec/) · [IJCAI (RecSys Track) 2021 (4)](../../../IJCAI/2021/sequential_rec/) · [IJCAI (RecSys Track) 2022 (2)](../../../IJCAI/2022/sequential_rec/) · [IJCAI (RecSys Track) 2023 (2)](../../../IJCAI/2023/sequential_rec/) · [IJCAI (RecSys Track) 2025 (2)](../../../IJCAI/2025/sequential_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/sequential_rec/) · [KDD 2022 (4)](../../../KDD/2022/sequential_rec/) · [KDD 2023 (4)](../../../KDD/2023/sequential_rec/) · [KDD 2024 (2)](../../../KDD/2024/sequential_rec/) · [KDD 2025 (7)](../../../KDD/2025/sequential_rec/) · [KDD 2026 (11)](../../../KDD/2026/sequential_rec/) · [RecSys 2021 (8)](../../../RecSys/2021/sequential_rec/) · [RecSys 2022 (10)](../../../RecSys/2022/sequential_rec/) · [RecSys 2023 (10)](../../../RecSys/2023/sequential_rec/) · [RecSys 2024 (11)](../../../RecSys/2024/sequential_rec/) · [RecSys 2025 (5)](../../../RecSys/2025/sequential_rec/) · [RecSys 2026 (7)](../../../RecSys/2026/sequential_rec/) · [SIGIR 2021 (4)](../../../SIGIR/2021/sequential_rec/) · [SIGIR 2022 (12)](../../../SIGIR/2022/sequential_rec/) · [SIGIR 2023 (9)](../../../SIGIR/2023/sequential_rec/) · [SIGIR 2024 (12)](../../../SIGIR/2024/sequential_rec/) · [SIGIR 2025 (12)](../../../SIGIR/2025/sequential_rec/) · [SIGIR 2026 (15)](../../../SIGIR/2026/sequential_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/sequential_rec/) · [ACM TKDD 2023 (1)](../../../TKDD/2023/sequential_rec/) · [IEEE TKDE 2022 (1)](../../../TKDE/2022/sequential_rec/) · [IEEE TKDE 2023 (7)](../../../TKDE/2023/sequential_rec/) · [IEEE TKDE 2024 (5)](../../../TKDE/2024/sequential_rec/) · [IEEE TKDE 2025 (3)](../../../TKDE/2025/sequential_rec/) · [IEEE TKDE 2026 (4)](../../../TKDE/2026/sequential_rec/) · [IEEE TMM 2023 (1)](../../../TMM/2023/sequential_rec/) · [ACM TORS 2025 (2)](../../../TORS/2025/sequential_rec/) · [ACM TORS 2026 (3)](../../../TORS/2026/sequential_rec/) · [ACM TWEB 2024 (1)](../../../TWEB/2024/sequential_rec/) · [WSDM 2021 (2)](../../../WSDM/2021/sequential_rec/) · [WSDM 2022 (4)](../../../WSDM/2022/sequential_rec/) · [WSDM 2023 (2)](../../../WSDM/2023/sequential_rec/) · [WSDM 2024 (3)](../../../WSDM/2024/sequential_rec/) · [WSDM 2025 (5)](../../../WSDM/2025/sequential_rec/) · [WSDM 2026 (4)](../../../WSDM/2026/sequential_rec/) · [WWW (TheWebConf) 2021 (6)](../../../WWW/2021/sequential_rec/) · [WWW (TheWebConf) 2022 (8)](../../../WWW/2022/sequential_rec/) · [WWW (TheWebConf) 2023 (9)](../../../WWW/2023/sequential_rec/) · [WWW (TheWebConf) 2024 (6)](../../../WWW/2024/sequential_rec/) · [WWW (TheWebConf) 2025 (7)](../../../WWW/2025/sequential_rec/) · [WWW (TheWebConf) 2026 (10)](../../../WWW/2026/sequential_rec/)

🔥 **高频主题：** TOIS ×4

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
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;状态空间模型&quot;]"><strong><a href="hmamba_hyperbolic_mamba_for_sequential_recommendation/">HMamba: Hyperbolic Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 Mamba 选择性状态空间搬进 Lorentz 双曲空间，用曲率感知离散化与陀螺向量算子建模层级+时序，在 4 个基准 HR/NDCG 上提升 3–11% 且保持线性复杂度。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;数据预处理（模型无关前置模块）&quot;,&quot;LLM 指令微调&quot;,&quot;不确定度估计&quot;,&quot;序列去噪&quot;]"><strong><a href="llm4dsr_leveraging_large_language_model_for_denoising_sequential_recommenda/">LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自监督指令微调激活 LLM 的序列去噪能力，并以 token 级不确定度阈值筛选高置信噪声、生成式补全替换，作为模型无关的数据预处理在三个 Amazon 数据集全库排序下稳定超越 HSD/STEAM/CL4SRec。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;表征坍缩与多样性正则&quot;]"><strong><a href="multi_interest_recommendation_a_survey/">Multi-Interest Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首篇多兴趣推荐综述，以隐式/显式建模分类统摄 172 篇文献，把技术栈归一为“兴趣抽取器+兴趣聚合器”两段式框架，并系统梳理去坍缩正则与六大开放挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;State Space Model&quot;,&quot;变长序列建模&quot;]"><strong><a href="ssd4rec_a_structured_state_space_duality_model_for_efficient_sequential_rec/">SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Mamba 的 Structured State Space Duality 块替代自注意力，配合变长拼接+段寄存器与双向翻转，在 4 个基准上取得 NDCG@10 最高 18.99% 提升并保持近线性复杂度。</p>
</dd>
</dl>
