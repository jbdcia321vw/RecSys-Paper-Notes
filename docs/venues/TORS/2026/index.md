---
hide:
  - toc
---

# 📖 ACM TORS 2026 推荐系统论文解读

8 篇 ACM TORS 2026 论文解读，覆盖 6 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TORS ×7 · 去偏公平与可信推荐 ×3 · 生成式推荐 ×2

<div class="paper-tag-filter" data-paper-tag-filter>
  <div class="paper-tag-filter__head">
    <strong class="paper-tag-filter__title">🏷️ 标签筛选</strong>
    <span class="paper-tag-filter__hint">默认 AND：选多个标签时只显示同时包含这些标签的论文</span>
    <span class="paper-tag-filter__mode" role="group" aria-label="标签匹配模式">
      <button type="button" data-tag-mode="and" aria-pressed="true">AND</button>
      <button type="button" data-tag-mode="or" aria-pressed="false">OR</button>
    </span>
    <button type="button" class="paper-tag-filter__clear" data-tag-clear disabled>清除标签</button>
  </div>
  <div class="paper-tag-filter__chips">
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM-as-Judge" aria-pressed="false">LLM-as-Judge<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Nash社会福利" aria-pressed="false">Nash社会福利<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Sinkhorn" aria-pressed="false">Sinkhorn<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="全链路(数据/算法/评估)" aria-pressed="false">全链路(数据/算法/评估)<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="公平性-相关性联合评测" aria-pressed="false">公平性-相关性联合评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="双边匹配推荐" aria-pressed="false">双边匹配推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性" aria-pressed="false">可复现性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多利益相关方推荐" aria-pressed="false">多利益相关方推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多智能体协作" aria-pressed="false">多智能体协作<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多模态推荐" aria-pressed="false">多模态推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="嫉妒自由" aria-pressed="false">嫉妒自由<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="实验方法论" aria-pressed="false">实验方法论<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="对话推荐与智能体仿真" aria-pressed="false">对话推荐与智能体仿真<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="帕累托前沿" aria-pressed="false">帕累托前沿<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="弱基线问题" aria-pressed="false">弱基线问题<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="扩散模型推荐" aria-pressed="false">扩散模型推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="推荐系统评测" aria-pressed="false">推荐系统评测<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提示工程" aria-pressed="false">提示工程<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="时空轨迹与POI推荐" aria-pressed="false">时空轨迹与POI推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="立场与综述论文" aria-pressed="false">立场与综述论文<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评估协议批判" aria-pressed="false">评估协议批判<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准" aria-pressed="false">评测基准<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="评测基准与元研究" aria-pressed="false">评测基准与元研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="音乐推荐评估" aria-pressed="false">音乐推荐评估<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="预训练音频表征" aria-pressed="false">预训练音频表征<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="generative_rec">✨ 生成式推荐与离散语义 ID (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;可复现性研究&quot;,&quot;扩散模型推荐&quot;,&quot;弱基线问题&quot;]"><strong><a href="generative_rec/diffusion_recommender_models_and_the_illusion_of_progress_a_concerning_stud/">Diffusion Recommender Models and the Illusion of Progress: A Concerning Study of Reproducibility and a Conceptual Mismatch</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>复现 9 个 SIGIR&#x27;23-24 扩散推荐模型，仅 25% 结果可复现，调优简单基线全面反超，并揭示 DDPM 与 top-n 推荐任务的概念错配。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;音乐推荐评估&quot;,&quot;LLM-as-Judge&quot;,&quot;提示工程&quot;]"><strong><a href="llm_rec/music_recommendation_with_large_language_models_challenges_opportunities_an/">Music Recommendation with Large Language Models: Challenges, Opportunities, and Evaluation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统梳理 LLM 驱动音乐推荐在用户建模、物品建模与自然语言请求三方面的变革，并提出 NLP 启发的评估分类体系与 MRS 成功/风险维度框架。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="agent_conversational_rec">🗣️ 对话推荐与智能体仿真 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS 2026&quot;,&quot;对话推荐与智能体仿真&quot;,&quot;重排&quot;,&quot;多智能体协作&quot;,&quot;流行度去偏&quot;]"><strong><a href="agent_conversational_rec/collab_rec_an_llm_based_agentic_framework_for_balancing_recommendations_in_/">Collab-Rec: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="对话推荐与智能体仿真">CRS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>三角色 LLM 智能体 + 确定性 Moderator 多轮约束精炼，在 900 条旅游查询上兼顾约束满足与长尾多样性。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multimodal_rec">🧩 多模态推荐与内容理解 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;多模态推荐&quot;,&quot;召回&quot;,&quot;预训练音频表征&quot;,&quot;冷启动&quot;]"><strong><a href="multimodal_rec/adopting_state_of_the_art_pretrained_audio_representations_for_music_recomm/">Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统评测 9 个 MIR 预训练音频表征在 5 类推荐器上的热/冷启动表现，揭示 MIR 性能无法直接迁移到推荐任务。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="poi_spatiotemporal_rec">📍 时空轨迹与 POI 本地生活推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;时空轨迹与POI推荐&quot;,&quot;全链路(数据/算法/评估)&quot;,&quot;立场与综述论文&quot;,&quot;多利益相关方推荐&quot;,&quot;评估协议批判&quot;]"><strong><a href="poi_spatiotemporal_rec/point_of_interest_recommendation_pitfalls_and_viable_solutions/">Point of Interest Recommendation: Pitfalls and Viable Solutions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>系统梳理POI推荐在数据集、算法与评估三维度的20项陷阱，并提出六方向研究议程与陷阱-方案映射表</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;双边匹配推荐&quot;,&quot;嫉妒自由&quot;,&quot;Nash社会福利&quot;,&quot;Sinkhorn&quot;]"><strong><a href="debias_fairness_industrial/balancing_fairness_and_high_match_rates_in_reciprocal_recommender_systems_a/">Balancing Fairness and High Match Rates in Reciprocal Recommender Systems: A Nash Social Welfare Approach</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="社交与群组推荐">Social</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将双边匹配推荐建模为推荐机会分配，用 Nash 社会福利交替优化实现近无嫉妒的公平推荐，并以 Sinkhorn 加速至大规模场景。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准与元研究&quot;,&quot;可复现性&quot;,&quot;实验方法论&quot;]"><strong><a href="debias_fairness_industrial/reproducibility_in_recommender_systems_a_survey/">Reproducibility in Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>对 RecSys 可复现性赛道 2020–2025 年 51 篇论文的结构化元分析，揭示其方法论高度同质化、以“扩展”替代“严格复现”的实践落差。</p>
</dd>
<dt data-paper-tags="[&quot;TORS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;评测基准&quot;,&quot;帕累托前沿&quot;,&quot;公平性-相关性联合评测&quot;,&quot;推荐系统评测&quot;]"><strong><a href="debias_fairness_industrial/robustness_analysis_of_pareto_based_joint_evaluation_of_fairness_and_releva/">Robustness Analysis of Pareto-based Joint Evaluation of Fairness and Relevance in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Recommender Systems">ACM TORS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出 DPFR：以测试集真实标签构建相关性-公平性帕累托前沿，用模型得分到前沿参考点的距离联合评估推荐系统，比现有联合度量更稳健。</p>
</dd>
</dl>
</section>
</div>
