---
hide:
  - toc
---

# 🏛️ DASFAA 2024 推荐系统论文解读

4 篇 DASFAA 2024 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2024&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回&quot;,&quot;LLM 常识知识图谱&quot;,&quot;互信息最大化对齐&quot;]"><strong><a href="graph_rec/common_sense_enhanced_knowledge_based_recommendation_with_large_language_mo/">Common Sense Enhanced Knowledge-based Recommendation with Large Language Model</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 ChatGPT 离线生成类目互补/替代常识图，并以互信息最大化对齐硬合并失败的两张知识图，为 CKE/CFKG 带来最高 8.5% 的 nDCG 提升。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;大语言模型推荐&quot;,&quot;端到端生成式&quot;,&quot;LoRA 指令微调&quot;,&quot;可解释推荐&quot;]"><strong><a href="llm_rec/unlocking_the_potential_of_large_language_models_for_explainable_recommenda/">Unlocking the Potential of Large Language Models for Explainable Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出两阶段框架 LLMXRec：推荐模型与 LLM 解释生成器解耦，用 LoRA 指令微调与个性化提示生成可解释理由，并以判别器、人工打分、属性预测三视角评估，胜率最高提升 31.8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA 2024&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;端到端交互策略&quot;,&quot;分层强化学习&quot;,&quot;Chain-of-Choice&quot;,&quot;多类型属性询问&quot;]"><strong><a href="agent_conversational_rec/chain_of_choice_hierarchical_policy_learning_for_conversational_recommendat/">Chain-of-Choice Hierarchical Policy Learning for Conversational Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>提出 MTAMCR 多类型属性多轮对话场景与 CoCHPL 分层策略，逐级生成属性/物品链；四数据集 SR@15/AT/hDCG 全面领先。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="social_group_rec">👥 社交与群组推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;DASFAA&quot;,&quot;社交与群组推荐&quot;,&quot;召回&quot;,&quot;图结构学习&quot;,&quot;非活跃用户冷启动&quot;,&quot;模仿学习&quot;]"><strong><a href="social_group_rec/learning_social_graph_for_inactive_user_recommendation/">Learning Social Graph for Inactive User Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：International Conference on Database Systems for Advanced Applications">DASFAA</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2024</span></span></dt>
<dd>
<p>用 U2U 删噪边 + U2C 连活跃兴趣簇的知识驱动图结构学习与模仿学习，重塑社交图以服务非活跃用户；采样排序下 Flickr NDCG@10 相对提升 129.58%。</p>
</dd>
</dl>
</section>
</div>
