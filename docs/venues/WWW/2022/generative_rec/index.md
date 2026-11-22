---
hide:
  - toc
---

# ✨ 生成式推荐与离散语义 ID

**WWW (TheWebConf) 2022** · **1** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2021 (1)](../../2021/generative_rec/) · [WWW (TheWebConf) 2024 (1)](../../2024/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../2025/generative_rec/) · [WWW (TheWebConf) 2026 (12)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/generative_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2024 (3)](../../../CIKM/2024/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (15)](../../../CIKM/2026/generative_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/generative_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/generative_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/generative_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/generative_rec/) · [KDD 2021 (1)](../../../KDD/2021/generative_rec/) · [KDD 2023 (1)](../../../KDD/2023/generative_rec/) · [KDD 2024 (2)](../../../KDD/2024/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (23)](../../../KDD/2026/generative_rec/) · [RecSys 2024 (4)](../../../RecSys/2024/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/generative_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (20)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (6)](../../../WSDM/2026/generative_rec/)

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2022&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;生成式端到端&quot;,&quot;copy mechanism&quot;,&quot;pointer network&quot;]"><strong><a href="conditional_generation_net_for_medication_recommendation/">Conditional Generation Net for Medication Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2022</span></span></dt>
<dd>
<p>COGNet 以 copy-or-predict 编码器-解码器逐药生成处方，用 visit/medication 两级软检索决定复制历史药或生成新药，在 MIMIC-III 全药库评测上 Jaccard 较 SafeDrug 提升 2.36%。</p>
</dd>
</dl>
