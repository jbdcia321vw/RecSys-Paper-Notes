---
hide:
  - toc
---

# 🔄 自监督对比学习与交互去噪

**WWW (TheWebConf) 2023** · **4** 篇论文解读

> **方向说明**：图/序列对比数据增广 (SGL/SimGCL)、隐式反馈假阳性噪声过滤 (Denoising) 与鲁棒表征

📌 **同领域跨会议浏览：** [WWW (TheWebConf) 2024 (3)](../../2024/self_supervised_denoising/) · [WWW (TheWebConf) 2025 (3)](../../2025/self_supervised_denoising/) · [WWW (TheWebConf) 2026 (3)](../../2026/self_supervised_denoising/) · [AAAI (RecSys Track) 2023 (1)](../../../AAAI/2023/self_supervised_denoising/) · [AAAI (RecSys Track) 2026 (2)](../../../AAAI/2026/self_supervised_denoising/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/self_supervised_denoising/) · [CIKM 2023 (2)](../../../CIKM/2023/self_supervised_denoising/) · [CIKM 2025 (3)](../../../CIKM/2025/self_supervised_denoising/) · [DASFAA 2023 (2)](../../../DASFAA/2023/self_supervised_denoising/) · [ECIR 2026 (1)](../../../ECIR/2026/self_supervised_denoising/) · [ICDE 2023 (1)](../../../ICDE/2023/self_supervised_denoising/) · [ICDE 2024 (3)](../../../ICDE/2024/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/self_supervised_denoising/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/self_supervised_denoising/) · [KDD 2023 (2)](../../../KDD/2023/self_supervised_denoising/) · [KDD 2024 (3)](../../../KDD/2024/self_supervised_denoising/) · [KDD 2025 (3)](../../../KDD/2025/self_supervised_denoising/) · [KDD 2026 (1)](../../../KDD/2026/self_supervised_denoising/) · [RecSys 2023 (1)](../../../RecSys/2023/self_supervised_denoising/) · [RecSys 2025 (2)](../../../RecSys/2025/self_supervised_denoising/) · [SIGIR 2023 (6)](../../../SIGIR/2023/self_supervised_denoising/) · [SIGIR 2024 (1)](../../../SIGIR/2024/self_supervised_denoising/) · [SIGIR 2025 (1)](../../../SIGIR/2025/self_supervised_denoising/) · [SIGIR 2026 (2)](../../../SIGIR/2026/self_supervised_denoising/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/self_supervised_denoising/) · [IEEE TKDE 2024 (3)](../../../TKDE/2024/self_supervised_denoising/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/self_supervised_denoising/) · [ACM TOIS 2025 (1)](../../../TOIS/2025/self_supervised_denoising/) · [ACM TORS 2023 (1)](../../../TORS/2023/self_supervised_denoising/) · [WSDM 2023 (1)](../../../WSDM/2023/self_supervised_denoising/) · [WSDM 2024 (1)](../../../WSDM/2024/self_supervised_denoising/) · [WSDM 2025 (2)](../../../WSDM/2025/self_supervised_denoising/) · [WSDM 2026 (1)](../../../WSDM/2026/self_supervised_denoising/)

🔥 **高频主题：** WWW ×2 · WWW 2023 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排（训练数据级预处理）&quot;,&quot;深度强化学习策略网络&quot;,&quot;实例级去噪与数据子集迁移&quot;]"><strong><a href="autodenoise_automatic_data_instance_denoising_for_recommendations/">AutoDenoise: Automatic Data Instance Denoising for Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用 DRL 策略网络逐实例做 select/deselect 去噪，配合搜索-验证两阶段交替训练，产出可迁移的干净子集，DeepFM 等骨干 AUC 稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;可学习图掩码&quot;,&quot;掩码图自编码器&quot;,&quot;Infomax 语义相关性&quot;]"><strong><a href="automated_self_supervised_learning_for_recommendation/">Automated Self-Supervised Learning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>AutoCF 用可学习掩码 + 图自注意力掩码自编码器替代手工图增广，在全库排序下相对最优 SSL 基线提升 7.8%–33.7% Recall@20。</p>
</dd>
<dt data-paper-tags="[&quot;WWW&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;召回&quot;,&quot;多行为推荐&quot;,&quot;提示微调&quot;,&quot;图重构去噪&quot;]"><strong><a href="denoising_and_prompt_tuning_for_multi_behavior_recommendation/">Denoising and Prompt-Tuning for Multi-Behavior Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>DPT 以三阶段范式先用重构难度无监督去噪辅助行为，再用重初始化与连续提示轻量微调，在 IJCAI/Tmall 上 HR@10 提升 1.58%–2.62%。</p>
</dd>
<dt data-paper-tags="[&quot;WWW 2023&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;精排&quot;,&quot;停留时间重加权&quot;,&quot;有效阅读筛选&quot;,&quot;多任务学习&quot;]"><strong><a href="reweighting_clicks_with_dwell_time_in_recommendation/">Reweighting Clicks with Dwell Time in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Web Conference">WWW (TheWebConf)</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>用停留时间定义有效阅读并以归一化 Sigmoid 重加权点击信号，工业级双塔多任务框架在微信看一看离线 AUC+5.62%、线上 CTR+2.58%。</p>
</dd>
</dl>
