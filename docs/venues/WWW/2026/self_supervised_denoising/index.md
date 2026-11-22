---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪

**WWW (TheWebConf) 2026** · **4** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2022 (2)](../../2022/self_supervised_denoising/) · [WWW (TheWebConf) 2023 (4)](../../2023/self_supervised_denoising/) · [WWW (TheWebConf) 2024 (3)](../../2024/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../2025/self_supervised_denoising/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (3)](../../../AAAI/2026/self_supervised_denoising/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/self_supervised_denoising/) · [CIKM 2021 (1)](../../../CIKM/2021/self_supervised_denoising/) · [CIKM 2022 (2)](../../../CIKM/2022/self_supervised_denoising/) · [CIKM 2023 (2)](../../../CIKM/2023/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [CIKM 2026 (1)](../../../CIKM/2026/self_supervised_denoising/) · [DASFAA 2022 (1)](../../../DASFAA/2022/self_supervised_denoising/) · [DASFAA 2023 (2)](../../../DASFAA/2023/self_supervised_denoising/) · [DASFAA 2026 (1)](../../../DASFAA/2026/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICDE 2022 (1)](../../../ICDE/2022/self_supervised_denoising/) · [ICDE 2023 (1)](../../../ICDE/2023/self_supervised_denoising/) · [ICDE 2024 (3)](../../../ICDE/2024/self_supervised_denoising/) · [ICDM 2021 (1)](../../../ICDM/2021/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/self_supervised_denoising/) · [KDD 2021 (1)](../../../KDD/2021/self_supervised_denoising/) · [KDD 2022 (1)](../../../KDD/2022/self_supervised_denoising/) · [KDD 2023 (2)](../../../KDD/2023/self_supervised_denoising/) · [KDD 2024 (3)](../../../KDD/2024/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2023 (1)](../../../RecSys/2023/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2021 (2)](../../../SIGIR/2021/self_supervised_denoising/) · [SIGIR 2022 (4)](../../../SIGIR/2022/self_supervised_denoising/) · [SIGIR 2023 (6)](../../../SIGIR/2023/self_supervised_denoising/) · [SIGIR 2024 (1)](../../../SIGIR/2024/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/self_supervised_denoising/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/self_supervised_denoising/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/self_supervised_denoising/) · [ACM TORS 2023 (1)](../../../TORS/2023/self_supervised_denoising/) · [WSDM 2021 (1)](../../../WSDM/2021/self_supervised_denoising/) · [WSDM 2022 (1)](../../../WSDM/2022/self_supervised_denoising/) · [WSDM 2023 (1)](../../../WSDM/2023/self_supervised_denoising/) · [WSDM 2024 (1)](../../../WSDM/2024/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/)

🔥 **高频主题：** WWW 2026 ×4 · 对比学习 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="自监督对比学习与交互去噪" aria-pressed="false">自监督对比学习与交互去噪<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对比学习" aria-pressed="false">对比学习<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;图结构注入&quot;,&quot;对比学习&quot;,&quot;事件序列建模&quot;]"><strong><a href="beyond_isolated_clients_integrating_graph_based_embeddings_into_event_seque/">Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将二部交互图结构以三种模型无关策略注入事件序列对比 SSL，揭示图密度决定最优融合策略，AUC 最高提升 2.3%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;训练加速/数据侧&quot;,&quot;Coreset Selection&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="efficient_content_based_recommendation_model_training_via_noise_aware_cores/">Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用末层梯度子模优化选核集，配合渐进标签自校正与 MC-Dropout 去噪，1% 数据恢复 93-95% 全量性能。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;对比学习&quot;,&quot;扩散增广&quot;,&quot;非对称对比&quot;]"><strong><a href="radar_relation_aware_diffusion_asymmetric_graph_contrastive_learning_for_re/">RaDAR: Relation-aware Diffusion-Asymmetric Graph Contrastive Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 VGAE 生成视图与关系感知去噪视图构成双视图，叠加扩散引导增广与非对称对比目标，在 Full-Ranking 下 Recall@20 相对 SOTA 提升 3.8%–4.7%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2026&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;目标锚定对比学习&quot;,&quot;不变风险最小化&quot;]"><strong><a href="rmbrec_robust_multi_behavior_recommendation_towards_target_behaviors/">RMBRec: Robust Multi-Behavior Recommendation towards Target Behaviors</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以目标行为为锚做用户侧对比对齐（RRM）+ 跨行为风险方差正则（ORM/REx），在 Taobao/TMall/Beibei 的 Full-Ranking 协议下 HR@10 最高提升 16.3%，且强抗噪。</p>
</dd>
</dl>
