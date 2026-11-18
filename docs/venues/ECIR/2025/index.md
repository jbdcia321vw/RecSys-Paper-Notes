---
hide:
  - toc
---

# 🏛️ ECIR 2025 推荐系统论文解读

11 篇 ECIR 2025 论文解读，覆盖 4 个推荐系统细分方向。每篇均包含一句话总结、核心思想、实验设置与审稿级批判，5 分钟读懂一篇论文。

🔥 **高频主题：** 去偏公平与可信推荐 ×6 · 大语言模型推荐 ×3 · 冷启动 ×2

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
<button type="button" class="paper-tag-filter__chip" data-tag="召回" aria-pressed="false">召回<span class="paper-tag-filter__count">7</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="去偏公平与可信推荐" aria-pressed="false">去偏公平与可信推荐<span class="paper-tag-filter__count">6</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="大语言模型推荐" aria-pressed="false">大语言模型推荐<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重排" aria-pressed="false">重排<span class="paper-tag-filter__count">3</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="冷启动" aria-pressed="false">冷启动<span class="paper-tag-filter__count">2</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="CVaR-DRO" aria-pressed="false">CVaR-DRO<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 元数据增强" aria-pressed="false">LLM 元数据增强<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="LLM 标签抽取" aria-pressed="false">LLM 标签抽取<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Minimax Group Fairness" aria-pressed="false">Minimax Group Fairness<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Next Basket Recommendation" aria-pressed="false">Next Basket Recommendation<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="RAG" aria-pressed="false">RAG<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="SASRec" aria-pressed="false">SASRec<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="Transformer 序列建模" aria-pressed="false">Transformer 序列建模<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="下一篮推荐" aria-pressed="false">下一篮推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="列表解释" aria-pressed="false">列表解释<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="原型矩阵分解" aria-pressed="false">原型矩阵分解<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可复现性研究" aria-pressed="false">可复现性研究<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="可解释性量化" aria-pressed="false">可解释性量化<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="地域公平性" aria-pressed="false">地域公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多利益相关方可解释推荐" aria-pressed="false">多利益相关方可解释推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="多语种句子编码器" aria-pressed="false">多语种句子编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="工业级部署" aria-pressed="false">工业级部署<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="序列与会话推荐" aria-pressed="false">序列与会话推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="异构知识图谱" aria-pressed="false">异构知识图谱<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="描述性货架" aria-pressed="false">描述性货架<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="提供方公平性" aria-pressed="false">提供方公平性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="文化多样性" aria-pressed="false">文化多样性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="新闻推荐" aria-pressed="false">新闻推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="校准多样性" aria-pressed="false">校准多样性<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="流行度去偏" aria-pressed="false">流行度去偏<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="混合整数线性规划" aria-pressed="false">混合整数线性规划<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="生成式推荐" aria-pressed="false">生成式推荐<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="用户画像预处理" aria-pressed="false">用户画像预处理<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="表示解耦" aria-pressed="false">表示解耦<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="语言模型编码器" aria-pressed="false">语言模型编码器<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="重复偏差" aria-pressed="false">重复偏差<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="零样本跨语言迁移" aria-pressed="false">零样本跨语言迁移<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="领域适配" aria-pressed="false">领域适配<span class="paper-tag-filter__count">1</span></button>
<button type="button" class="paper-tag-filter__chip" data-tag="频率感知模块" aria-pressed="false">频率感知模块<span class="paper-tag-filter__count">1</span></button>
  </div>
  <div class="paper-tag-filter__status" data-tag-status></div>
</div>
  </aside>
  <div class="pn-filter-list">
<div class="conf-index">
<section class="paper-section" data-paper-section>
<hr />
<h2 id="sequential_rec">⏱️ 序列与会话推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;序列与会话推荐&quot;,&quot;召回&quot;,&quot;Next Basket Recommendation&quot;,&quot;频率感知模块&quot;,&quot;Transformer 序列建模&quot;]"><strong><a href="sequential_rec/saferec_self_attention_and_frequency_enriched_model_for_next_basket_recomme/">SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 SASRec 上并联频率感知打分模块，用物品出现频次与历史指示向量补足重复购买建模，Leave-One-Basket 全库排序下 Recall@10 最高提升 8%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="llm_rec">🧠 大语言模型推荐 (LLM4Rec) (3)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;重排&quot;,&quot;描述性货架&quot;,&quot;LLM 元数据增强&quot;,&quot;列表解释&quot;]"><strong><a href="llm_rec/contextualizing_spotify_s_audiobook_list_recommendations_with_descriptive_s/">Contextualizing Spotify&#x27;s Audiobook List Recommendations with Descriptive Shelves</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="多任务学习与列表级重排">MTL</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 抽取十类书目描述符并生成多样化描述性货架标题，两轮线上 A/B 测试验证发现性与互动指标提升。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;召回&quot;,&quot;新闻推荐&quot;,&quot;语言模型编码器&quot;,&quot;冷启动&quot;]"><strong><a href="llm_rec/revisiting_language_models_in_neural_news_recommender_systems/">Revisiting Language Models in Neural News Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 MIND 上统一复现 8 种 LM 作为新闻编码器，发现更大模型不必然更优，但显著利好冷启动用户。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;大语言模型推荐&quot;,&quot;生成式推荐&quot;,&quot;LLM 标签抽取&quot;,&quot;RAG&quot;,&quot;工业级部署&quot;]"><strong><a href="llm_rec/text2playlist_generating_personalized_playlists_from_text_on_deezer/">Text2Playlist: Generating Personalized Playlists from Text on Deezer</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--category" title="生成式推荐与离散语义 ID">GenRec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 LLM 抽取显隐式标签 + CF 亲和度重排 + LLM 列表精炼，将宽意图文本查询转为个性化歌单，Deezer 线上 20% 灰度中生成歌单后续收听率达 45% vs 人工 27%。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="cold_start_rec">❄️ 冷启动与长尾推荐 (1)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;冷启动&quot;,&quot;召回&quot;,&quot;多语种句子编码器&quot;,&quot;零样本跨语言迁移&quot;,&quot;领域适配&quot;]"><strong><a href="cold_start_rec/news_without_borders_domain_adaptation_of_multilingual_sentence_embeddings_/">News Without Borders: Domain Adaptation of Multilingual Sentence Embeddings for Cross-lingual News Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--category" title="大语言模型推荐 (LLM4Rec)">LLM4Rec</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>构建 PolyNews 多语新闻语料，以 DAE+MT 领域适配 LaBSE 得 NaSE，冻结嵌入+晚期点击融合在 ZS-XLT 冷启动新闻推荐上达 SOTA。</p>
</dd>
</dl>
</section>
<section class="paper-section" data-paper-section>
<hr />
<h2 id="debias_fairness_industrial">🛡️ 去偏公平、可信隐私与工业架构 (6)</h2>
<dl data-paper-list>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;表示解耦&quot;,&quot;可解释性量化&quot;,&quot;可复现性研究&quot;]"><strong><a href="debias_fairness_industrial/are_representation_disentanglement_and_interpretability_linked_in_recommend/">Are Representation Disentanglement and Interpretability Linked in Recommendation Models? A Critical Review and Reproducibility Study</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>复现5个推荐模型×4数据集，量化解耦度与可解释性，发现解耦与效果无关但与表示可解释性强正相关。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;原型矩阵分解&quot;,&quot;流行度去偏&quot;,&quot;文化多样性&quot;]"><strong><a href="debias_fairness_industrial/embedding_cultural_diversity_in_prototype_based_recommender_systems/">Embedding Cultural Diversity in Prototype-based Recommender Systems</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="协同过滤与双塔召回">CF</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>在 ProtoMF 原型矩阵分解上引入 K-近邻原型过滤与原型正交分布正则，以物品国别为文化代理缓解流行度偏差，全库排序下长尾平均排名降 27%、HR@10 反升 2%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;用户画像预处理&quot;,&quot;提供方公平性&quot;,&quot;校准多样性&quot;]"><strong><a href="debias_fairness_industrial/how_to_diversify_any_personalized_recommender/">How to Diversify any Personalized Recommender?</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="冷启动与长尾推荐">Cold</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>提出用户画像级预处理：用逻辑回归识别用户代表类别、userKNN 生成候选，再按 λ% 增删交互，使任意推荐器在保持精度的同时提升多样性与少数类曝光公平。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;召回&quot;,&quot;CVaR-DRO&quot;,&quot;Minimax Group Fairness&quot;,&quot;SASRec&quot;]"><strong><a href="debias_fairness_industrial/improving_minimax_group_fairness_in_sequential_recommendation/">Improving Minimax Group Fairness in Sequential Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>将 CVaR-DRO 首次引入 SASRec 序列推荐，无需组标注即可提升小众/短序列用户的最差组 NDCG@20，Retailrocket 整体提升约 6%。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;异构知识图谱&quot;,&quot;多利益相关方可解释推荐&quot;,&quot;地域公平性&quot;]"><strong><a href="debias_fairness_industrial/okra_an_explainable_heterogeneous_multi_stakeholder_job_recommender_system/">OKRA: an Explainable, Heterogeneous, Multi-Stakeholder Job Recommender System</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="图神经网络与知识图谱推荐">Graph</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>异构知识图谱+多头GATv2构建求职者/企业双侧嵌入，谐波均值融合打分，nDCG@10较六基线最高提升约18%，并揭示城乡地域偏差。</p>
</dd>
<dt data-paper-tags="[&quot;ECIR 2025&quot;,&quot;去偏公平与可信推荐&quot;,&quot;重排&quot;,&quot;重复偏差&quot;,&quot;混合整数线性规划&quot;,&quot;下一篮推荐&quot;]"><strong><a href="debias_fairness_industrial/repeat_bias_aware_optimization_of_beyond_accuracy_metrics_for_next_basket_r/">Repeat-bias-aware Optimization of Beyond-accuracy Metrics for Next Basket Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：European Conference on Information Retrieval">ECIR</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--category" title="序列与会话推荐">Seq</span><span class="paper-tag paper-tag--year" title="发表年份">2025</span></span></dt>
<dd>
<p>用 MILP 重排同时优化多样性/物品公平与重复偏差，在三个零售数据集上以可接受 Recall 损失显著改善 DS、logDP 与 RepBias。</p>
</dd>
</dl>
</section>
</div>
  </div>
</div>
