---
hide:
  - toc
---

# 📍 时空轨迹与 POI 本地生活推荐

**SIGIR 2023** · **2** 篇论文解读

> **方向说明**：Next-POI 兴趣点推荐、时空序列轨迹建模、LBS 本地生活服务与移动出行推荐

📌 **同领域跨会议浏览：** [SIGIR 2021 (1)](../../2021/poi_spatiotemporal_rec/) · [SIGIR 2022 (2)](../../2022/poi_spatiotemporal_rec/) · [SIGIR 2024 (3)](../../2024/poi_spatiotemporal_rec/) · [SIGIR 2025 (2)](../../2025/poi_spatiotemporal_rec/) · [SIGIR 2026 (1)](../../2026/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2021 (1)](../../../AAAI/2021/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2024 (2)](../../../AAAI/2024/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2025 (1)](../../../AAAI/2025/poi_spatiotemporal_rec/) · [AAAI (RecSys Track) 2026 (1)](../../../AAAI/2026/poi_spatiotemporal_rec/) · [ACL / EMNLP (RecSys & CRS) 2025 (1)](../../../ACL_EMNLP/2025/poi_spatiotemporal_rec/) · [CIKM 2021 (2)](../../../CIKM/2021/poi_spatiotemporal_rec/) · [CIKM 2023 (2)](../../../CIKM/2023/poi_spatiotemporal_rec/) · [CIKM 2025 (1)](../../../CIKM/2025/poi_spatiotemporal_rec/) · [CIKM 2026 (1)](../../../CIKM/2026/poi_spatiotemporal_rec/) · [ECIR 2024 (1)](../../../ECIR/2024/poi_spatiotemporal_rec/) · [ICDE 2026 (2)](../../../ICDE/2026/poi_spatiotemporal_rec/) · [ICDM 2022 (1)](../../../ICDM/2022/poi_spatiotemporal_rec/) · [ICLR / NeurIPS / ICML (RecSys) 2025 (1)](../../../ICLR_NeurIPS_ICML/2025/poi_spatiotemporal_rec/) · [IJCAI (RecSys Track) 2021 (1)](../../../IJCAI/2021/poi_spatiotemporal_rec/) · [KDD 2024 (1)](../../../KDD/2024/poi_spatiotemporal_rec/) · [KDD 2025 (1)](../../../KDD/2025/poi_spatiotemporal_rec/) · [KDD 2026 (3)](../../../KDD/2026/poi_spatiotemporal_rec/) · [RecSys 2025 (1)](../../../RecSys/2025/poi_spatiotemporal_rec/) · [RecSys 2026 (2)](../../../RecSys/2026/poi_spatiotemporal_rec/) · [IEEE TKDE 2023 (1)](../../../TKDE/2023/poi_spatiotemporal_rec/) · [IEEE TKDE 2024 (2)](../../../TKDE/2024/poi_spatiotemporal_rec/) · [IEEE TKDE 2025 (1)](../../../TKDE/2025/poi_spatiotemporal_rec/) · [ACM TOIS 2023 (1)](../../../TOIS/2023/poi_spatiotemporal_rec/) · [ACM TOIS 2024 (2)](../../../TOIS/2024/poi_spatiotemporal_rec/) · [ACM TORS 2026 (1)](../../../TORS/2026/poi_spatiotemporal_rec/) · [WSDM 2021 (1)](../../../WSDM/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2021 (2)](../../../WWW/2021/poi_spatiotemporal_rec/) · [WWW (TheWebConf) 2026 (1)](../../../WWW/2026/poi_spatiotemporal_rec/)

<dl data-paper-list>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;时空轨迹与 POI 本地生活推荐&quot;,&quot;精排&quot;,&quot;时空特征选择&quot;,&quot;动态特征门控&quot;]"><strong><a href="exploring_the_spatiotemporal_features_of_online_food_recommendation_service/">Exploring the Spatiotemporal Features of Online Food Recommendation Service</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="CTR/CVR 预估与特征交叉">CTR</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>系统剖析外卖场景时空特征价值，指出 self-attention 不适用的根因，提出时间差特征与动态上下文自适应模块 DCAM，离线 AUC 与线上 CTR 双赢</p>
</dd>
<dt data-paper-tags="[&quot;SIGIR 2023&quot;,&quot;时空轨迹与POI本地生活推荐&quot;,&quot;端侧推理/召回&quot;,&quot;去中心化协同学习&quot;,&quot;知识蒸馏互信息最大化&quot;]"><strong><a href="model_agnostic_decentralized_collaborative_learning_for_on_device_poi_recom/">Model-Agnostic Decentralized Collaborative Learning for On-Device POI Recommendation</a></strong><span class="paper-tags"><span class="paper-tag paper-tag--venue" title="发表出处：ACM SIGIR Conference on Research and Development in Information Retrieval">SIGIR</span><span class="paper-tag paper-tag--category" title="时空轨迹与 POI 本地生活推荐">POI</span><span class="paper-tag paper-tag--category" title="去偏公平、可信隐私与工业架构">Debias</span><span class="paper-tag paper-tag--year" title="发表年份">2023</span></span></dt>
<dd>
<p>MAC 允许端侧 POI 推荐模型结构异构，用参考数据软决策+互信息蒸馏做去中心化协同，采样邻居降噪，1/6 模型体积下 HR@10 超 DCLR 3.8%</p>
</dd>
</dl>
