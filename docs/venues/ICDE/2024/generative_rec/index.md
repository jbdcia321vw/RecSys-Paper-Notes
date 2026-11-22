---
hide:
  - toc
---

# ✨ 生成式推荐与离散语义 ID

**ICDE 2024** · **1** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ICDE 2026 (1)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/generative_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2021 (1)](../../../ACL_EMNLP/2021/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2024 (3)](../../../CIKM/2024/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (15)](../../../CIKM/2026/generative_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/generative_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/generative_rec/) · [KDD 2021 (1)](../../../KDD/2021/generative_rec/) · [KDD 2023 (1)](../../../KDD/2023/generative_rec/) · [KDD 2024 (2)](../../../KDD/2024/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (23)](../../../KDD/2026/generative_rec/) · [RecSys 2024 (4)](../../../RecSys/2024/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/generative_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (20)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (6)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2021 (1)](../../../WWW/2021/generative_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/generative_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (12)](../../../WWW/2026/generative_rec/)

<dl data-paper-list>
<dt data-paper-tags="[&quot;ICDE&quot;,&quot;生成式推荐&quot;,&quot;端到端生成式&quot;,&quot;语义ID量化&quot;,&quot;LLM指令微调&quot;,&quot;生成式检索&quot;]"><strong><a href="adapting_large_language_models_by_integrating_collaborative_semantics_for_r/">Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：IEEE International Conference on Data Engineering">ICDE</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>RQ-VAE 层级语义索引 + Sinkhorn 均匀映射消除索引冲突，LLaMA-7B 多任务对齐微调实现全库生成式检索，全排序平均提升 25.5%。</p>
</dd>
</dl>
