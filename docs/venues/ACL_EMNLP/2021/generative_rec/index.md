---
hide:
  - toc
---

# ✨ 生成式推荐与离散语义 ID

**ACL / EMNLP (RecSys & CRS) 2021** · **1** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../2024/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../2026/generative_rec/) · [AAAI (RecSys Track) 2022 (1)](../../../AAAI/2022/generative_rec/) · [AAAI (RecSys Track) 2026 (6)](../../../AAAI/2026/generative_rec/) · [CIKM 2024 (3)](../../../CIKM/2024/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (15)](../../../CIKM/2026/generative_rec/) · [DASFAA 2026 (1)](../../../DASFAA/2026/generative_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/generative_rec/) · [ICDE 2026 (1)](../../../ICDE/2026/generative_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (4)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (3)](../../../IJCAI/2026/generative_rec/) · [KDD 2021 (1)](../../../KDD/2021/generative_rec/) · [KDD 2023 (1)](../../../KDD/2023/generative_rec/) · [KDD 2024 (2)](../../../KDD/2024/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (23)](../../../KDD/2026/generative_rec/) · [RecSys 2024 (4)](../../../RecSys/2024/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR 2023 (1)](../../../SIGIR/2023/generative_rec/) · [SIGIR 2024 (2)](../../../SIGIR/2024/generative_rec/) · [SIGIR 2025 (4)](../../../SIGIR/2025/generative_rec/) · [SIGIR 2026 (20)](../../../SIGIR/2026/generative_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (6)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2021 (1)](../../../WWW/2021/generative_rec/) · [WWW (TheWebConf) 2022 (1)](../../../WWW/2022/generative_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (12)](../../../WWW/2026/generative_rec/)

<dl data-paper-list>
<dt data-paper-tags="[&quot;EMNLP&quot;,&quot;生成式推荐与离散语义ID&quot;,&quot;端到端生成式&quot;,&quot;无序集合生成&quot;,&quot;BERT [MASK] 生成&quot;,&quot;多模态早融合&quot;]"><strong><a href="leveraging_order_free_tag_relations_for_context_aware_recommendation/">Leveraging Order-Free Tag Relations for Context-Aware Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACL &amp; EMNLP Conferences">ACL / EMNLP (RecSys &amp; CRS)</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2021</span></span></dt>
<dd>
<p>SOG 以 BERT 编码器 + [MASK] 迭代回填实现无序标签集合生成，配合 1-to-M KL 训练与贪心解码，在 Instagram / Stack Overflow 全标签库排序下 P@1 提升 24.1% / 8.9%。</p>
</dd>
</dl>
