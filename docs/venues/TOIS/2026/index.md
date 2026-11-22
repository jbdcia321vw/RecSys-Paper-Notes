---
hide:
  - toc
---

# 📖 ACM TOIS 2026 推荐系统论文解读

27 篇 ACM TOIS 2026 论文解读，覆盖 10 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** TOIS ×20 · TOIS 2026 ×7 · 大语言模型推荐 ×4 · 负采样 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">9</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="精排" aria-pressed="false">精排<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="协同过滤与双塔召回" aria-pressed="false">协同过滤与双塔召回<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平、可信隐私与工业架构" aria-pressed="false">去偏公平、可信隐私与工业架构<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">4</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="图神经网络与知识图谱推荐" aria-pressed="false">图神经网络与知识图谱推荐<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="负采样" aria-pressed="false">负采样<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="跨域与多场景推荐" aria-pressed="false">跨域与多场景推荐<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="collaborative_filtering">🤝 协同过滤与双塔召回 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双塔检索&quot;,&quot;负采样&quot;,&quot;向量索引&quot;]"><strong><a href="collaborative_filtering/a_comprehensive_survey_on_retrieval_methods_in_recommender_systems/">A Comprehensive Survey on Retrieval Methods in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个系统梳理推荐召回阶段的综述：从相似度学习、索引机制与训练优化三大维度统一归纳，并在公开数据集上完成跨方法基准评测与工业多路召回案例复盘</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;嵌入学习&quot;,&quot;图神经网络&quot;,&quot;哈希与量化压缩&quot;]"><strong><a href="collaborative_filtering/embedding_in_recommender_systems_a_survey/">Embedding in Recommender Systems: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以「矩阵/序列/图 × 学习范式 × 效率手段」三轴正交框架统一梳理推荐嵌入，归纳哈希/量化/AutoML 压缩路径与 LLM 语义补充，并给出开源复现仓库。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回（训练阶段）&quot;,&quot;负采样&quot;,&quot;假负样本&quot;,&quot;难负样本&quot;]"><strong><a href="collaborative_filtering/negative_sampling_in_recommendation_a_survey_and_future_directions/">Negative Sampling in Recommendation: A Survey and Future Directions</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首个推荐负采样综述：以静态/动态/对抗生成/重要性重加权/知识增强五类分类法梳理 230+ 文献，剖析假负样本、精度-效率-稳定性三角与跨任务泛化三大挑战及采样复杂度权衡</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;协同过滤与双塔召回&quot;,&quot;召回&quot;,&quot;双视图对齐&quot;,&quot;稀疏-稠密融合&quot;,&quot;信噪比分析&quot;,&quot;长尾物品&quot;]"><strong><a href="collaborative_filtering/why_not_collaborative_filtering_in_dual_view_bridging_sparse_and_dense_mode/">Why not Collaborative Filtering in Dual View? Bridging Sparse and Dense Models</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出稀疏-稠密双视图框架 SaD：用 slim 的结构信号与 MF 语义嵌入互造伪标签对齐，理论上突破冷门物品的 $\sqrt{N_i}$ SNR 上限。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="graph_rec">🕸️ 图神经网络与知识图谱推荐 (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络推荐&quot;,&quot;召回&quot;,&quot;符号边重加权&quot;,&quot;LLM 表征增强&quot;,&quot;精度-多样性权衡&quot;]"><strong><a href="graph_rec/cpgrec_a_balance_oriented_framework_for_personalized_video_game_recommendat/">CPGRec+: A Balance-oriented Framework for Personalized Video Game Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Fisher 分布符号边重加权缓解过平滑、LLM 生成偏好描述精炼表征，在 Steam 双数据集上兼顾精度与多样性。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;精排&quot;,&quot;超图对比学习&quot;,&quot;相似就诊检索&quot;]"><strong><a href="graph_rec/hypemed_enhancing_medication_recommendations_with_hypergraph_based_patient_/">HypeMed: Enhancing Medication Recommendations with Hypergraph-Based Patient Relationships</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以就诊为超边建模集合级语义，知识感知对比预训练构建可检索空间，就诊条件检索相似病历增强用药推荐，Jaccard 提升 0.4–0.6% 且 DDI 低于真实处方。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;图神经网络与知识图谱推荐&quot;,&quot;召回（KG 增强表征学习）&quot;,&quot;知识图谱剪枝&quot;,&quot;双视角重要性评估&quot;,&quot;图稀疏化&quot;]"><strong><a href="graph_rec/knowledge_graph_pruning_for_recommendation/">Knowledge Graph Pruning for Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出双视角重要性评估器为协同知识图谱逐实体打分，剪除至多 98% 三元组仍保精度并提速 12.2×。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="self_supervised_denoising">🔄 自监督对比学习与交互去噪 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;自监督对比学习与交互去噪&quot;,&quot;数据清洗/训练期&quot;,&quot;语言智能体&quot;,&quot;LossEraser 遗忘式训练&quot;,&quot;隐式反馈去噪&quot;]"><strong><a href="self_supervised_denoising/ruleagent_discovering_rules_for_recommendation_denoising_with_autonomous_la/">RuleAgent: Discovering Rules for Recommendation Denoising with Autonomous Language Agents</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM Agent 自动发现推荐去噪规则，配合 LossEraser 遗忘式免重训策略，在 Beauty/Yelp/Gowalla 小规模子集上 Full 排序 Recall@20 全面超越 T-CE/DeCA/BOD/DCF。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;双曲几何&quot;,&quot;状态空间模型&quot;]"><strong><a href="sequential_rec/hmamba_hyperbolic_mamba_for_sequential_recommendation/">HMamba: Hyperbolic Mamba for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将 Mamba 选择性状态空间搬进 Lorentz 双曲空间，用曲率感知离散化与陀螺向量算子建模层级+时序，在 4 个基准 HR/NDCG 上提升 3–11% 且保持线性复杂度。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;数据预处理（模型无关前置模块）&quot;,&quot;LLM 指令微调&quot;,&quot;不确定度估计&quot;,&quot;序列去噪&quot;]"><strong><a href="sequential_rec/llm4dsr_leveraging_large_language_model_for_denoising_sequential_recommenda/">LLM4DSR: Leveraging Large Language Model for Denoising Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用自监督指令微调激活 LLM 的序列去噪能力，并以 token 级不确定度阈值筛选高置信噪声、生成式补全替换，作为模型无关的数据预处理在三个 Amazon 数据集全库排序下稳定超越 HSD/STEAM/CL4SRec。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;多兴趣表征&quot;,&quot;表征坍缩与多样性正则&quot;]"><strong><a href="sequential_rec/multi_interest_recommendation_a_survey/">Multi-Interest Recommendation: A Survey</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>首篇多兴趣推荐综述，以隐式/显式建模分类统摄 172 篇文献，把技术栈归一为“兴趣抽取器+兴趣聚合器”两段式框架，并系统梳理去坍缩正则与六大开放挑战。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;State Space Model&quot;,&quot;变长序列建模&quot;]"><strong><a href="sequential_rec/ssd4rec_a_structured_state_space_duality_model_for_efficient_sequential_rec/">SSD4Rec: A Structured State Space Duality Model for Efficient Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Mamba 的 Structured State Space Duality 块替代自注意力，配合变长拼接+段寄存器与双向翻转，在 4 个基准上取得 NDCG@10 最高 18.99% 提升并保持近线性复杂度。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (4)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;连续时间动态图&quot;,&quot;多维画像蒸馏&quot;]"><strong><a href="llm_rec/dynllm_when_large_language_models_meet_dynamic_graph_recommendation/">DynLLM: When Large Language Models Meet Dynamic Graph Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 LLM 从商品标题生成多维用户画像，经蒸馏注意力注入连续时间动态图，在 Full-Ranking 协议下 Tmall Recall@10 相对最优基线提升 40.3%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;提示鲁棒性&quot;,&quot;生成对抗数据增广&quot;,&quot;协同信号注入&quot;]"><strong><a href="llm_rec/ganprompt_enhancing_robustness_in_llm_based_recommendations_with_gan_enhanc/">GANPrompt: Enhancing Robustness in LLM-Based Recommendations with GAN-Enhanced Diversity Prompts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 T5 编码器作生成器、MLP 作判别器做属性条件化对抗训练，配合余弦+JS 双约束扩展提示多样性，再以专家引导 token 注入 DIN 协同信号微调 LLaMA2-7B；Amazon 五数据集 Leave-One-Out + 1:9 采样排序下 AUC 最高 +2.06%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;生成式端到端&quot;,&quot;隐空间对齐&quot;,&quot;序列残差提示&quot;]"><strong><a href="llm_rec/large_language_model_can_interpret_latent_space_of_sequential_recommender/">Large Language Model Can Interpret Latent Space of Sequential Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>线性适配器将序列推荐隐表示映射到LLaMA，以恢复/残差提示解读隐空间；MovieLens/Steam残差识别最高97.89%/86.33%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;大语言模型推荐&quot;,&quot;精排&quot;,&quot;DPO 偏好对齐&quot;,&quot;个性化不确定性平滑&quot;,&quot;难负采样&quot;]"><strong><a href="llm_rec/rosepo_aligning_llm_based_recommenders_with_human_values/">RosePO: Aligning LLM-based Recommenders with Human Values</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>在 SFT 后引入个性化平滑 DPO：三类 HH 拒答采样构造偏好对，用 SASRec 预言机逐样本估计标签翻转率作平滑因子，采样排序下 HR@1 相对提升 4.2%–8.1%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;冷启动与长尾推荐&quot;,&quot;召回&quot;,&quot;扩散表征生成&quot;,&quot;混合专家 MoE&quot;,&quot;捆绑包推荐&quot;]"><strong><a href="cold_start_rec/divide_and_conquer_cold_start_bundle_recommendation_via_mixture_of_diffusio/">Divide-and-Conquer: Cold-Start Bundle Recommendation via Mixture of Diffusion Experts</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将捆绑包冷启动拆解为双层多视图子问题，用条件扩散模型统一生成表征、冷感知层次化 MoE 自适应融合，全库排序下冷启动 Recall@20 最高绝对提升 0.1027。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cross_domain_multi_scenario">🌐 跨域与多场景推荐 (2)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;精排&quot;,&quot;因果发现&quot;,&quot;分布外泛化&quot;]"><strong><a href="cross_domain_multi_scenario/causal_invariant_cross_domain_out_of_distribution_recommendation/">Causal-Invariant Cross-Domain Out-of-Distribution Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>统一建模跨域与单域双重分布偏移，用双层因果结构与 LLM 引导混杂因子发现实现跨域 OOD 推荐，HR@10 平均提升 6.34%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;跨域与多场景推荐&quot;,&quot;召回&quot;,&quot;Counterfactual Learning&quot;,&quot;Representation Disentanglement&quot;]"><strong><a href="cross_domain_multi_scenario/counterfactual_learning_driven_representation_disentanglement_for_search_en/">Counterfactual Learning-Driven Representation Disentanglement for Search-Enhanced Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用三元组反事实信号解耦搜索域物品表征，剥离查询无关的通用特征并以门控融合/软加权数据增广注入推荐，采样排序下 HR@1 最高提升 10.84%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="multi_task_reranking">⚖️ 多任务学习与列表级重排 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;多任务学习与列表级重排&quot;,&quot;精排&quot;,&quot;多行为推荐&quot;,&quot;级联图卷积&quot;,&quot;对比偏好对齐&quot;]"><strong><a href="multi_task_reranking/towards_multi_behavior_multi_task_recommendation_via_behavior_informed_grap/">Towards Multi-Behavior Multi-Task Recommendation via Behavior-informed Graph Embedding Learning</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>BiGEL 在级联图范式上叠加目标→辅助门控反馈、全局上下文增强与对比偏好对齐，JD/UB 全库 Top-5 排序上目标与辅助行为双赢（UB 点击 Prec@5 +101%）。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="rl_bandit_rec">🎮 强化学习与动态探索推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;强化学习与动态探索推荐&quot;,&quot;精排/在线决策&quot;,&quot;Contextual Bandit&quot;,&quot;Hypernetwork&quot;,&quot;LLM Warm-Start&quot;]"><strong><a href="rl_bandit_rec/enhancing_bandit_algorithms_with_llms_for_time_varying_user_preferences_in_/">Enhancing Bandit Algorithms with LLMs for Time-varying User Preferences in Streaming Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Euler 时间嵌入驱动超网络生成低秩偏好矩阵刻画周期偏好，LLM 仿真做离线热启动，三数据集累积奖励超 SOTA 且具 2/3 阶动态遗憾界。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;联邦召回&quot;,&quot;embedding skew&quot;,&quot;复合聚合&quot;,&quot;凸二次规划&quot;]"><strong><a href="debias_fairness_industrial/beyond_similarity_personalized_federated_recommendation_with_composite_aggr/">Beyond Similarity: Personalized Federated Recommendation with Composite Aggregation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以相似+互补复合聚合缓解联邦推荐 embedding skew，权重求解为凸二次规划，四数据集 HR@10 最高相对提升 20.5%</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信推荐&quot;,&quot;精排&quot;,&quot;流行度去偏&quot;,&quot;逆倾向加权&quot;,&quot;列表级学习&quot;]"><strong><a href="debias_fairness_industrial/mitigating_popularity_bias_in_recommendation_with_global_listwise_learning_/">Mitigating Popularity Bias in Recommendation with Global Listwise Learning and Progressive Bi-Weighting</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 IPS 加权多项似然实现列表级全局无偏学习，并以 Bi-Weighting 平滑倾向估计与渐进式去偏，六数据集 Recall@20 平均提升 9.43%。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;重排&quot;,&quot;双边公平&quot;,&quot;后悔理论&quot;,&quot;模糊规划&quot;]"><strong><a href="debias_fairness_industrial/regret_aware_re_ranking_for_guaranteeing_two_sided_fairness_and_accuracy_in/">Regret-aware Re-ranking for Guaranteeing Two-sided Fairness and Accuracy in Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将后悔理论与模糊规划引入双边公平重排，兼顾用户个体公平、平均准确率与提供方曝光。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;精排/推理阶段&quot;,&quot;RAG 外部协同知识&quot;,&quot;对抗扰动净化&quot;,&quot;鲁棒 LLM4Rec&quot;]"><strong><a href="debias_fairness_industrial/retrieval_augmented_purifier_for_robust_llm_empowered_recommendation/">Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用外部协同物品图检索定位并删除或替换提示中的扰动项，推理阶段即插即用，P5 上 H@5 平均回升 2 倍余。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS&quot;,&quot;去偏公平与可信工业架构&quot;,&quot;端到端解释与重排调优&quot;,&quot;Sparse Autoencoder&quot;,&quot;Model Interpretation&quot;,&quot;Targeted Tuning&quot;]"><strong><a href="debias_fairness_industrial/understanding_internal_representations_of_recommendation_models_with_sparse/">Understanding Internal Representations of Recommendation Models with Sparse Autoencoders</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用稀疏自编码器通用探测推荐模型内部表征，抽取单语义概念并支持定向调优，不改动原模型，重建NDCG损失&lt;5%、人评3.89/4。</p>
</dd>
<dt data-paper-tags="[&quot;TOIS 2026&quot;,&quot;去偏公平、可信隐私与工业架构&quot;,&quot;召回/训练&quot;,&quot;Graph Contrastive Learning&quot;,&quot;Targeted Promotion Attack&quot;,&quot;Spectral Smoothing&quot;]"><strong><a href="debias_fairness_industrial/when_graph_contrastive_learning_backfires_spectral_vulnerability_and_defens/">When Graph Contrastive Learning Backfires: Spectral Vulnerability and Defense in Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM Transactions on Information Systems">ACM TOIS</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>揭示GCL谱平滑放大定向推广攻击，提出CLeaR攻击与SIM防御，在三大数据集上显著降H@50且保Recall。</p>
</dd>
</dl>
</section>
</div>
