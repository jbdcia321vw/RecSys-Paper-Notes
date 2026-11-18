---
hide:
  - toc
---

# 🌐 跨域与多场景推荐

**KDD 2025** · **5** 篇论文解读

> **方向说明**：跨域知识迁移 (CDR)、重叠/非重叠用户对齐、多业务场景联合建模 (STAR/PEPNet 等)

📌 **同领域跨会议浏览：** [KDD 2026 (4)](../../2026/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2025 (2)](../../../AAAI/2025/cross_domain_multi_scenario/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/cross_domain_multi_scenario/) · [ACL / EMNLP (RecSys & CRS) 2026 (1)](../../../ACL_EMNLP/2026/cross_domain_multi_scenario/) · [ACM MM 2025 (1)](../../../ACM_MM/2025/cross_domain_multi_scenario/) · [CIKM 2025 (4)](../../../CIKM/2025/cross_domain_multi_scenario/) · [CIKM 2026 (4)](../../../CIKM/2026/cross_domain_multi_scenario/) · [DASFAA 2025 (2)](../../../DASFAA/2025/cross_domain_multi_scenario/) · [ICDE 2025 (1)](../../../ICDE/2025/cross_domain_multi_scenario/) · [ICDM 2025 (1)](../../../ICDM/2025/cross_domain_multi_scenario/) · [RecSys 2025 (5)](../../../RecSys/2025/cross_domain_multi_scenario/) · [SIGIR 2025 (6)](../../../SIGIR/2025/cross_domain_multi_scenario/) · [SIGIR 2026 (4)](../../../SIGIR/2026/cross_domain_multi_scenario/) · [SIGIR-AP 2025 (1)](../../../SIGIR-AP/2025/cross_domain_multi_scenario/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/cross_domain_multi_scenario/) · [IEEE TMM 2025 (1)](../../../TMM/2025/cross_domain_multi_scenario/) · [ACM TOIS 2025 (3)](../../../TOIS/2025/cross_domain_multi_scenario/) · [ACM TOIS 2026 (1)](../../../TOIS/2026/cross_domain_multi_scenario/) · [WSDM 2025 (2)](../../../WSDM/2025/cross_domain_multi_scenario/) · [WSDM 2026 (1)](../../../WSDM/2026/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2025 (5)](../../../WWW/2025/cross_domain_multi_scenario/) · [WWW (TheWebConf) 2026 (4)](../../../WWW/2026/cross_domain_multi_scenario/)

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
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Classifier-Free Guidance" aria-pressed="false">Classifier-Free Guidance<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动偏好迁移" aria-pressed="false">冷启动偏好迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型学习" aria-pressed="false">原型学习<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="向量量化码本" aria-pressed="false">向量量化码本<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="场景感知 GLU" aria-pressed="false">场景感知 GLU<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="域选择" aria-pressed="false">域选择<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多实体知识迁移" aria-pressed="false">多实体知识迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多目标优化" aria-pressed="false">多目标优化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托最优自注意力" aria-pressed="false">帕累托最优自注意力<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构特征对齐" aria-pressed="false">异构特征对齐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型" aria-pressed="false">扩散模型<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户专属特征图" aria-pressed="false">用户专属特征图<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移" aria-pressed="false">负迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负迁移抑制" aria-pressed="false">负迁移抑制<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预训练微调" aria-pressed="false">预训练微调<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<dl data-paper-list>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;多实体知识迁移&quot;,&quot;异构特征对齐&quot;,&quot;预训练微调&quot;]"><strong><a href="effective_two_stage_knowledge_transfer_for_multi_entity_cross_domain_recomm/">Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出 MKT 两阶段多实体知识迁移框架，用异构特征对齐与极化分布损失解决异质特征 schema 下的负迁移，线上 CTR +4.13%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;扩散模型&quot;,&quot;冷启动偏好迁移&quot;,&quot;Classifier-Free Guidance&quot;]"><strong><a href="exploring_preference_guided_diffusion_model_for_cross_domain_recommendation/">Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用源域交互历史构造偏好引导信号，逐步注入扩散逆向过程生成目标域冷启动用户表征，三场景 MAE 最高降 32.60%。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;负迁移&quot;,&quot;域选择&quot;,&quot;原型学习&quot;]"><strong><a href="measure_domain_s_gap_a_similar_domain_selection_principle_for_multi_domain_/">Measure Domain&#x27;s Gap: A Similar Domain Selection Principle for Multi-Domain Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用原型距离+bandit动态为每个域挑选相似域子集，解耦共享专家以缓解多域推荐负迁移，三数据集AUC稳定提升。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;用户专属特征图&quot;,&quot;向量量化码本&quot;,&quot;场景感知 GLU&quot;]"><strong><a href="perscen_learning_personalized_interaction_pattern_and_scenario_preference_f/">PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用用户专属特征图 + VQ 码本蒸馏场景感知偏好 + 渐进式 GLU 融合，在 KuaiRand/Alimama 四场景召回上全面超越 ADIN/SASS/M5。</p>
</dd>
<dt data-paper-tags="[&quot;KDD 2025&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;帕累托最优自注意力&quot;,&quot;负迁移抑制&quot;,&quot;多目标优化&quot;]"><strong><a href="revisiting_self_attention_for_cross_domain_sequential_recommendation/">Revisiting Self-attention for Cross-domain Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGKDD Conference on Knowledge Discovery and Data Mining">KDD</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将跨域序列推荐重构为推荐损失与跨域注意力抑制的双目标帕累托优化，以即插即用方式改造自注意力，平均提升 Recall@10 达 9.8%/16.0%。</p>
</dd>
</dl>
  </div>
</div>
