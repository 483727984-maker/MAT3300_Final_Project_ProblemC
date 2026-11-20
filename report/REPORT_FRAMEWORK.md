# MAT3300 C 题报告行文框架（对照 main.tex 骨架的逐节填充指南）

> 现状：`C/q1_analysis.py` 已产出完整的 Q1 结果（5 图 + 3 个结果文件 + 全艺术家指标表）；
> `report/main.tex` 已搭好 9 节骨架，全部 TODO。本文档回答"每节写什么、放哪张图、引哪个数"。

---

## 0. 现有产出盘点（写作素材清单）

| 产出 | 内容 | 报告用途 |
|---|---|---|
| `q1_outputs/q1_results.json` | 全网拓扑统计、三种指标秩相关、灵敏度对比 | §4 数据 + §8 验证 |
| `q1_outputs/artist_network_indicators.csv` | 5,603 位艺术家的 out/in-degree、Katz、reverse-PageRank 及排名 | 全文数字唯一来源 |
| `q1_outputs/cohort_results.json` | 9 个 follower-decade cohort 的规模与 top-5 | §5 动态视角 |
| `q1_outputs/jazz_results.json` | Jazz 子网统计 + within-Jazz top-15 | §7 案例研究 |
| `figures/fig0–fig4` | 工作流图 / 全网排名+散点 / 灵敏度双面板 / cohort 热力图 / Jazz 子网 | 见下方分配 |

**可直接引用的关键数字**
- 42,770 条唯一有向边、5,603 位艺术家；最大弱连通分量 5,599（99.9%）
- 733 条时间倒序边（1.7%）保留在主图；同期边 8,477 条
- Katz α=0.10（谱半径 3.614，收敛上界 0.2767，α 距上界余量充分）
- 秩相关：Katz vs out-degree ρ=0.953；Katz vs reverse-PageRank ρ=0.970
- 灵敏度：删 733 条倒序边后 ρ=0.986、top-20 完全一致（20/20）
- Jazz 子网：384 人、1,769 边、密度 0.0120、Jazz→Pop/Rock 出流 409 条；top-3：Charlie Parker / Lester Young / Miles Davis
- cohort 主线：1960s 起 Beatles 连续 4 个 decade 居 top；1990s 出现 Nirvana；2010s cohort 仅 811 边（样本提醒）

---

## 1. 逐节填充指南（含页数预算，合计 ≤25 页）

### 摘要页（1 页，最后写）
四句话结构：问题（量化音乐影响力+流派演化）→ 方法（有向影响网络 + 三种互补指标 + cohort/Jazz 双切片；Model II/III 一句话预告）→ 关键结果（列 3–4 个硬数字：top 影响者、ρ 值、灵敏度 20/20）→ 验证与局限（报告关系≠因果；decade 粒度）。

### §1 Introduction and Problem Restatement（~1.5 页）
背景 2 段 + 7 项任务逐条重述（编号 T1–T7，后文所有结果都挂到这些编号上）+ 路线图 1 段（说明本文结构如何映射 T1–T7）。不要在此放任何数字。

### §2 Assumptions and Justifications（~1 页）
与代码 docstring 对齐，至少 5 条：
- A1 边是"报告的关系"非强度，权重取 1（数据只记录有无）
- A2 active_start 是艺术家近似属性，不是影响发生时间（不做因果时序断言）
- A3 733 条倒序边保留于主图（专家记录误差或同辈互影响），仅作敏感性检验
- A4 重复有向对视为同一关系，不解释为强度
- A5 网络为静态聚合快照；cohort 是"谁的报告"，不是"何时影响"

### §3 Notation（0.5 页）
表：G=(V,E)、A、d^out、K_i（Katz）、PR^rev、α、cohort C_d 等。只收正文实际用到的符号。

### §4 Data Preprocessing and Exploration（~2 页）
写 `load_data()/artist_metadata()` 做的事：ID 去空格、乱码修复（mojibake）、`R&B;` 分号清洗、decade 数值化、去重对。
放**图 0（fig0 工作流图）**+ 一个小拓扑统计表（42,770 / 5,603 / 弱连通 3 / 最大弱分量 5,599 / 零出度 1,829 / 零入度 557）。
口径注意：零出度 = "数据集中无记录追随者"，不等于无影响（这句话本身写进正文）。

### §5 Model I: Directed Influence Network（~5 页，Q1 主战场）
1. **网络定义**（0.5 页）：有向图、边方向 i→j 表示 i 影响 j、权重 1、节点属性挂 genre/decade
2. **三种指标**（1.5 页，各给定义式+一句解释）：
   - out-degree：直接被点名数
   - **outgoing Katz**（主指标）：K=(I−αA)^{-1}1，展开级数解释"多步影响的几何衰减"；报告 α=0.10、收敛界 0.2767、谱半径
   - reverse-PageRank（对照）：显式说明"翻转边使分数从追随者流回影响者"
   - 明确写：三者语义不同、**不做无依据的加权合成**，分列报告 + 秩相关比较（ρ=0.953/0.970，及"高相关但 top 有差异"的具体人名例证——从两个 top-15 列表里找差异节点）
3. **全网结果**（1 页）：**图 1（fig1）**——左面板 top-15 Katz 榜（着色=流派），右面板 out-degree vs Katz log-log 散点（标注 Beatles/Dylan/Chuck Berry/Muddy Waters 讲"多步影响补足直接度数"的故事）
4. **动态视角：cohort**（1.5 页）：定义 follower-start-decade cohort（复述 A5 的口径声明）→ **图 3（fig3 热力图）**→ 叙事：Beatles 的 4-decade 统治、1990s Nirvana 的新面孔、1950s Hank Williams/Elvis 的乡土转移、2010s 样本量警告
   - *此节回应 T1 的"subnetwork + parameters"，cohort 是结构性的动态证据*

### §6 Model II: Measures of Music Similarity（~4 页，待做）
特征清洗（log1p 偏斜特征、loudness/tempo 脏值、key 圆距离、popularity 年内 z-score）→ 白化/PCA → 同流派 vs 跨流派分布对比 + Mann-Whitney + 效应量 → （核心）同流派同年代基线校正 s*(u,v)。此节同时为 Model III 提供 s* 输入。

### §7 Model III: Genre Structure and Influence Validation（~3.5 页，T3+T4）
- T3 流派定义/相关：流派×流派相似度矩阵（Model II 产物聚合）× 跨流派边密度矩阵（Model I 产物聚合），两套独立证据对比；"流派是文化边界还是声音边界"的结论落点
- T4 影响验证：s*(follower, influencer) vs 同流派同年代基线的显著性检验 + 逐特征固定效应回归（控制 decade + genre），标准化系数 = 传染性排行

### §8 Model IV: Temporal Dynamics（~4.5 页，T5+T6+T7）
- T5 革命检测（**主通道单独用 data_by_year 时间数据**）：流派×年份特征曲线 → 成分校正 → 断点检测（PELT 或 Bollinger/MA 斜率），输出"革命时段 + 哪些特征跃变"；再回到网络找对应年代的跨界高中心性艺术家 = 革命者。两条通道各自独立成立、互相印证
- T6 Jazz 案例研究（复用 fig4 + jazz_results.json）：按 decade 切片算 within-Jazz Katz 榜更替（Parker/Young/Davis → 现代），每个更替点对应 T5 检出的断点；口径声明照抄 Q1_paper_structure §5
- T7 时代切片：以 ~1995 分界，比较跨流派边比例、同 decade 边比例、cohort 边数规模（cohort_results.json 现成）；特征侧用 data_by_year 看互联网前后 duration/loudness/energy 趋势变化

### §9 Validation and Sensitivity Analysis（~1.5 页）
集中放 Q1 已算好的三组 + 未来 T4/T5 的同类检验：
1. **图 2（fig2）**：α 稳定性（ρ≥0.975 全谱、top-20 在 α=0.10 峰值 20/20、极端 α 8–9/20 已作语义解释）+ 时间过滤（ρ=0.986、top-20 一致 20/20）
2. 三指标秩相关表 + top-k 重叠率
3. 局限：报告关系≠因果、网络无权、decade 粒度、数据集存活偏差

### §10 Strengths, Weaknesses and Conclusion（~1.5 页）
按 T1–T7 逐条给一句结论；strengths（可解释指标体系、灵敏度闭环、多切片互证）/ weaknesses（无因果识别、标签粒度粗）。

### ICM Society Memo（1 页，硬性）
(i) 方法价值：网络度量可解释+可复现；(ii) 更富数据如何改变工作：边来源标注（自述 vs 专家）、歌词/音频模态、事件级时间戳；(iii) 进一步研究建议。直接复用参考论文 D21018555 的 "Further data" 思路。

### References + Contribution Statement
补引用：Page & Brin (PageRank)、Katz (1953)、Kempe et al. (2003，若 Model III 用级联)、Mariani et al. (rescaled centrality，若做时代校正)、两篇 COMAP 优秀论文。贡献声明按学号升序（已排好）。

---

## 2. 图表分配总表

| 图 | 落位节 | 作用 | caption 必写口径 |
|---|---|---|---|
| fig0 工作流 | §4 | 全文地图 | "decade 为描述性属性" |
| fig1 全网排名+散点 | §5 | 主结果 | 节点着色=main_genre；α=0.10 |
| fig2 灵敏度双面板 | §8 | 稳健性 | 左：α 扫描；右：删倒序边对比 |
| fig3 cohort 热力图 | §5 | 动态视角 | cohort=follower 出道年代分组，非影响发生时间 |
| fig4 Jazz 子网 | §7(Req6) | 案例研究 | 选定 ego 子图的选取规则 |

全网拓扑统计一律用表格（避免 hairball 大图）；每张图 caption 写明节点大小/颜色编码含义。

---

## 3. 与 Q1_paper_structure.md 的两处衔接决定

1. **cohort 热力图放 §5（Model I 内）而不是单独成节**——它是 T1 "dynamic view" 的直接证据；§7 的 Req6 Jazz 案例复用 fig4，不重复画图。
2. **Jazz 案例从 Q1 范围挪到 §7**——同一份 `jazz_results.json` 服务 T1（子网络展示）与 T6（流派案例），写作上在 §5 结尾一句 "see §7 for the Jazz case study" 即可，避免内容重复。

---

## 4. 写作顺序建议

1. §4 → §5 先写（素材 100% 就绪，纯汇编）
2. §2/§3 契约性内容随后（半小时）
3. §6 Model II 实验跑完 → §7 融合层 → §8 把 Q2–Q5 的灵敏度补进去
4. §9 结论、Memo、References
5. **摘要页最后写**（要引用全文最终数字）
6. 全程纪律：正文数字只允许从 `q1_outputs/*.json` 和 `artist_network_indicators.csv` 抄，禁止手改；图表先落盘再引用
