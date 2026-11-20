---
hide:
  - toc
---

# ✨ 生成式推荐与离散语义 ID

**SIGIR 2026** · **19** 篇论文解读

> **方向说明**：RQ-VAE/VQ 残差量化码本、TIGER 自回归生成式检索、扩散/流匹配生成推荐 (Diffusion/Flow Rec)

📌 **同领域跨会议浏览：** [SIGIR 2023 (1)](../../2023/generative_rec/) · [SIGIR 2024 (2)](../../2024/generative_rec/) · [SIGIR 2025 (4)](../../2025/generative_rec/) · [AAAI (RecSys Track) 2026 (5)](../../../AAAI/2026/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2024 (1)](../../../ACL_EMNLP/2024/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (5)](../../../ACL_EMNLP/2025/generative_rec/) · [ACL / EMNLP (RecSys & CRS) 2026 (2)](../../../ACL_EMNLP/2026/generative_rec/) · [CIKM 2024 (3)](../../../CIKM/2024/generative_rec/) · [CIKM 2025 (6)](../../../CIKM/2025/generative_rec/) · [CIKM 2026 (14)](../../../CIKM/2026/generative_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/generative_rec/) · [ECIR 2026 (1)](../../../ECIR/2026/generative_rec/) · [ICDE 2024 (1)](../../../ICDE/2024/generative_rec/) · [ICDM 2023 (1)](../../../ICDM/2023/generative_rec/) · [ICDM 2026 (2)](../../../ICDM/2026/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2023 (2)](../../../ICLR_NeurIPS_ICML/2023/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (6)](../../../ICLR_NeurIPS_ICML/2025/generative_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2026 (3)](../../../ICLR_NeurIPS_ICML/2026/generative_rec/) · [IJCAI (RecSys Track) 2025 (1)](../../../IJCAI/2025/generative_rec/) · [IJCAI (RecSys Track) 2026 (1)](../../../IJCAI/2026/generative_rec/) · [KDD 2023 (1)](../../../KDD/2023/generative_rec/) · [KDD 2024 (2)](../../../KDD/2024/generative_rec/) · [KDD 2025 (4)](../../../KDD/2025/generative_rec/) · [KDD 2026 (13)](../../../KDD/2026/generative_rec/) · [RecSys 2024 (4)](../../../RecSys/2024/generative_rec/) · [RecSys 2025 (6)](../../../RecSys/2025/generative_rec/) · [RecSys 2026 (12)](../../../RecSys/2026/generative_rec/) · [SIGIR-AP 2023 (1)](../../../SIGIR-AP/2023/generative_rec/) · [SIGIR-AP 2026 (1)](../../../SIGIR-AP/2026/generative_rec/) · [ACM TKDD 2026 (1)](../../../TKDD/2026/generative_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/generative_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/generative_rec/) · [WSDM 2026 (3)](../../../WSDM/2026/generative_rec/) · [WWW (TheWebConf) 2024 (1)](../../../WWW/2024/generative_rec/) · [WWW (TheWebConf) 2025 (4)](../../../WWW/2025/generative_rec/) · [WWW (TheWebConf) 2026 (6)](../../../WWW/2026/generative_rec/)

🔥 **高频主题：** 生成式推荐 ×19 · 语义ID ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">19</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">5</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语义ID" aria-pressed="false">语义ID<span class="paper-tag-filter__count">2</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;级联生成&quot;,&quot;RAG 约束解码&quot;,&quot;教师学生蒸馏&quot;]"><strong><a href="a_cascaded_generative_approach_for_e_commerce_recommendations/">A Cascaded Generative Approach for e-Commerce Recommendations</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出级联生成式货架构建框架：LLM1 生成投放主题、LLM2 经 RAG 约束生成检索关键词，教师-学生蒸馏降本，线上购物车添加 +2.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;双层优化&quot;,&quot;梯度手术&quot;]"><strong><a href="bi_level_optimization_for_generative_recommendation_bridging_tokenization_a/">Bi-Level Optimization for Generative Recommendation: Bridging Tokenization and Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将分词器与推荐器建模为双层优化，用元学习+梯度手术让推荐损失直接监督RQ-VAE码本，Full-Ranking下全面超越TIGER/LETTER/ETEGRec。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID构建&quot;,&quot;非均匀量化&quot;,&quot;视觉语义单元&quot;]"><strong><a href="card_non_uniform_quantization_of_visual_semantic_unit_for_generative_recomm/">CARD: Non-Uniform Quantization of Visual Semantic Unit for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将文本/图像/协同信号渲染为统一视觉语义单元，并用可逆非均匀变换校正偏斜分布，提升码本利用率与生成式推荐精度。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;可微语义ID&quot;,&quot;Gumbel-Softmax 探索&quot;,&quot;码本坍缩&quot;]"><strong><a href="differentiable_semantic_id_for_generative_recommendation/">Differentiable Semantic ID for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="自监督对比学习与交互去噪">SSL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Gumbel 噪声+不确定性衰减实现可微语义ID联合训练，Leave-One-Out 全库排序下 R@10 相对 TIGER 提升约 12%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;离散偏好Token&quot;,&quot;跨模态一致性奖励&quot;]"><strong><a href="discrete_preference_learning_for_personalized_multimodal_generation/">Discrete Preference Learning for Personalized Multimodal Generation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用模态专属LightGCN+RQ-VAE把协同偏好量化成离散token注入LLM与扩散模型，配合跨模态一致性奖励实现个性化图文生成。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;双扩散 Transformer&quot;,&quot;多模态可解释生成&quot;]"><strong><a href="dual_diffusional_generative_fashion_recommendation/">Dual-Diffusional Generative Fashion Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="多模态推荐与内容理解">MM</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>双分支扩散 Transformer 联合生成时尚单品图像与属性级文本，以结构化偏好加权采样建模用户，PFITB/GOR 双任务个性化指标显著领先且训练提速约 7 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;Flow Matching&quot;,&quot;单步生成&quot;,&quot;语义锚先验&quot;]"><strong><a href="fave_flow_based_average_velocity_establishment_for_sequential_recommendatio/">FAVE: Flow-based Average Velocity Establishment for Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用语义锚先验替代高斯噪声、以平均速度场实现单步流匹配生成推荐，全库排序下 N@20 相对提升 9.9%，推理提速 20 倍以上。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;双向线性注意力&quot;,&quot;长序列编码器加速&quot;]"><strong><a href="gated_bidirectional_linear_attention_for_generative_retrieval/">Gated Bidirectional Linear Attention for Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 Conv1D 局部混合、序列级 Key 门控软遗忘与门控 RMSNorm 将核化线性注意力扩展为双向线性时间算子，1:2 混合编码器在 Yandex Music 上追平双向自注意力，32768 长度单层提速 8.2×。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Page-wise NTP&quot;,&quot;GRPO-SR 偏好对齐&quot;]"><strong><a href="genrec_a_preference_oriented_generative_framework_for_large_scale_recommend/">GenRec: A Preference-Oriented Generative Framework for Large-Scale Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以 Page-wise NTP 解一对多歧义、线性 Token Merger 压缩 2× 输入、GRPO-SR 混合奖励对齐偏好，京东线上点击 +9.5%、成交 +8.7%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;GFlowNets&quot;,&quot;曝光偏差&quot;,&quot;奖励对齐微调&quot;]"><strong><a href="gflowgr_fine_tuning_generative_recommendation_frameworks_with_generative_fl/">GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="强化学习与动态探索推荐">RL</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将生成式推荐重构为集合级序列生成，用 GFlowNets 微调使生成概率正比于行为感知奖励，缓解曝光偏差，淘宝线上年收入相对提升 0.4%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;语义ID&quot;,&quot;上下文token组合&quot;,&quot;分解嵌入融合&quot;]"><strong><a href="learning_decomposed_contextual_token_representations_from_pretrained_and_co/">Learning Decomposed Contextual Token Representations from Pretrained and Collaborative Signals for Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>DECOR 用冻结码本+可学协同嵌入分解融合、上下文 token 组合与可学 BOS 查询，缓解静态分词歧义与预训练语义丢失，Full-Ranking 下 NDCG@10 最高提升 14%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;Semantic ID 对齐&quot;,&quot;时序漂移&quot;]"><strong><a href="mitigating_collaborative_semantic_id_staleness_in_generative_retrieval/">Mitigating Collaborative Semantic ID Staleness in Generative Retrieval</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>提出模型无关的 SID 对齐更新，将刷新后的协同语义 ID 双部图匹配映射回旧词表，实现热启动微调，Recall@500 提升且训练算力降 8-9 倍。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;变分多视图集成&quot;,&quot;语义ID与协同ID&quot;]"><strong><a href="mviger_multi_view_variational_integration_of_complementary_knowledge_for_ge/">MVIGER: Multi-View Variational Integration of Complementary Knowledge for Generative Recommender</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将模板-索引组合视为隐变量视图，用可学习先验做变分集成，Full-Ranking 下 H@10 最高提升约 26%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;精排&quot;,&quot;连续隐流形轨迹&quot;,&quot;运动学约束&quot;]"><strong><a href="next_interest_flow_a_generative_pre_training_paradigm_for_recommender_syste/">Next Interest Flow: A Generative Pre-training Paradigm for Recommender Systems by Modeling All-domain Movelines</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>将用户意图建模为隐流形上的连续演化轨迹，用切空间正交与测地平滑约束生成式预训练，并以双向对齐+TSP桥接判别式精排，工业级AUC+0.87pt。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;逆向课程学习&quot;,&quot;稀疏转化建模&quot;]"><strong><a href="rclrec_reverse_curriculum_learning_for_modeling_sparse_conversions_in_gener/">RCLRec: Reverse Curriculum Learning for Modeling Sparse Conversions in Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以逆向课程前缀为稀疏转化目标注入实例级中间监督，配合课程质量感知损失，离线全库排序全面超越 SOTA，线上广告收入 +2.09%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;Request-Only Optimization&quot;,&quot;HSTU&quot;,&quot;特征去重&quot;]"><strong><a href="request_only_optimization_for_recommendation_systems/">Request-Only Optimization for Recommendation Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>以用户请求为训练样本单元，在数据源头去重用户特征，实现训练吞吐 2-6 倍提升并支撑 HSTU 生成式推荐规模化落地。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;召回&quot;,&quot;语义ID&quot;,&quot;RQ-VAE码本坍缩&quot;,&quot;生成式检索&quot;]"><strong><a href="semantic_ids_for_recommender_systems_at_snapchat_use_cases_technical_challe/">Semantic IDs for Recommender Systems at Snapchat: Use Cases, Technical Challenges, and Design Choices</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>Snapchat 工业级 SID 落地：STE 全码本反传与多模态融合缓解码本坍缩，启发式桶内消歧解决 SID-Item 碰撞，线上短视频分享 +4.39%。</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;混合物品分词&quot;,&quot;指令多任务SFT&quot;,&quot;自适应概率融合&quot;]"><strong><a href="sigma_a_semantic_grounded_instruction_driven_generative_multi_task_recommen/">SIGMA: A Semantic-Grounded Instruction-Driven Generative Multi-Task Recommender at AliExpress</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>多视图语义对齐+混合SID/ID分词+指令多任务SFT+三步生成与自适应概率融合，离线HR@10达33.76%，线上GMV+7.84%</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2026&quot;,&quot;生成式推荐&quot;,&quot;通用物品分词&quot;,&quot;树状码本&quot;,&quot;跨域迁移&quot;]"><strong><a href="universal_item_tokenization_for_transferable_generative_recommendation/">Universal Item Tokenization for Transferable Generative Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--category" title="跨域与多场景推荐">Cross</span><span class="paper-tag paper-tag--year" title="发表年份">2026</span></span></dt>
<dd>
<p>用 MLLM+树状码本构建跨域通用物品分词器，结合内容重建与共现协同对齐，实现可迁移生成式推荐，全库排序下全面超越基线。</p>
</dd>
</dl>
