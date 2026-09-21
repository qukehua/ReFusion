# 审稿人三：论文与实现核对

核对日期：2026-09-21。依据用户提供的 `comment.pdf` 第 4 页 Reviewer 3
及 `ReFusion__A_Relation_Aware_Conditional_Latent_Diffusion_Model_for_Diverse_Human_Motion_Prediction.pdf`
第 3–6、8、11 页。审稿意见作为核查材料，不作为提交、公开或修改实验数字的授权。

**范围与结论**：本次修正项目实现，补充公式和实验协议的精确定义。提供的 PDF 没有被改写；
其中几个公式本身缺少维度映射，因此不能声称当前 PDF 与代码已经逐字完全一致。
下面给出必须同步进论文的补充。没有重新训练，也没有生成替代论文表格的实验数值。
按作者要求，`data_loader/` 保留在本地且由 Git 忽略，录用前不公开。

## 1. 按审稿意见核查

| 意见 | 当前核对结果与处理 | 尚需完成的事项 |
|---|---|---|
| R3-1：表格重复、缺少原始结果 | 新评估保存独立时间戳目录、K 个预测、逐窗口指标、邻居集合、数据窗口编号、配置和源码/权重哈希；`utils.rescore_evaluation` 可离线重算 | 现有表格的来源、重复原因无法从模型代码确定；必须重新运行或找回原始实验记录，不能凭代码修改填数 |
| R3-2：3DPW 没有真正使用第二个人 | 现有目标/条件切分已经是 Person 1 预测、两人的观测作条件；新增/保留梯度、未来泄漏和实际 PKL 读取测试 | 需要真实数据上的重新训练与评估；合成测试不能证明表 IV 的性能 |
| R3-2：CMU 合成第二个人 | 历史加载器确有镜像、旋转、平移合成人物。本地加载器和预处理已取消该路径，改为显式同步双人记录 | 作者需提供真实配对数据及来源/划分；单人 TXT 不再能运行双人评估。元数据字段不能证明录制来源的真实性 |
| R3-2：CoMaD 实际仅 HR | 保持论文 CoMaD 实验的 HR 配置，不把 HR-only 宣称为 HH 实验；本地修复根坐标与缺失实体填零 | 若要报告 CoMaD HH，需另行配置、真实数据和独立结果；当前 HR 配置不支持该结论 |
| R3-3：对完整历史+未来扩散 | 本次开始时已修正：训练只对未来 DCT_P 加噪，条件为观测 DCT_T；采样没有历史 inpainting | 旧模型和旧表格不是这版实现的实验结果 |
| R3-3：缺少频率邻接 | 本次开始时已有 `frequency_adj` 与 `spatial_adj`；新增显式 Kronecker 算子数值等价检查 | 无代码缺口；需用新训练结果验证效果 |
| R3-3：第二阶段漏掉 G_intra | 已有 `c_k + G_intra + G_inter`，测试验证每个块和输出头收到的条件 | 维持式 (19)，无需再次改写架构 |
| R3-3：Adam/AdamW 不符 | 已用 AdamW；当前验证实际优化器及解耦权重衰减 | 论文需补充未报告的 weight_decay、betas、eps |
| R3-3：表 VII 的端点不可构造 | 已支持 0/9 至 9/0；同一个模型类，新增默认 9 层、2/7 分配 | 必须说明 0/9 仍注入 intra token、stage II 同时用两类全局条件，见下文 |
| R3-3：式 (3) 投影和 T/P 对齐缺失 | 当前代码有 xyz→d、Linear(T,P)；本文件明确参数维度 | 必须把第 2 节补充到论文原稿，不能只保留含糊的 ψ |
| R3-4：不同关节数 | 现有实现是空间注意力保留所有伙伴关节，频率 K/V 对伙伴关节均值池化并广播。本次提取成有明确定义的 `frequency_partner_context` | 式 (5) 需增加池化/广播算子；不能将其描述成关节解剖对应，也不能宣称支持物体外观/场景语义 |
| R3-5：指标协议不明确 | 明确 K 可配置、flattened xyz L2、last-observation 邻居、lower median；单样本 APD=0，拒绝空多模态集合或非有限值 | 未实现 MotionMap 的多历史/骨架缩放协议；物理合理性、碰撞/接触等需要另做实验，APD 不等于真实感 |
| R3-6：公开复现不完整 | 核心测试可在缺少私有加载器时运行；实际 train/eval 明确说明加载器尚未公开 | 按作者决定仍不公开加载器；缺少完整公开复现的意见仍成立，未实现/发布论文所述所有对比方法 |

## 2. 应同步到论文的公式与维度

记 B 为 batch，T 为观测长度，P 为预测长度，J_h 为**模型实际预测的**人体关节点数，
J_r 为伙伴关节点数，d 为 hidden dimension。去掉的主人体根关节在显示时补零；
CoMaD 保留 9 个输出标记，首个标记作为零锚点。

### DCT 与扩散目标：式 (1)、(8)、(22)–(24)

```text
X_h: [B,T,J_h,3]       X_r: [B,T,J_r,3]       Y_h: [B,P,J_h,3]
Xbar_h = D_T X_h       Xbar_r = D_T X_r        Ybar_0 = D_P Y_h
Ybar_k = sqrt(alpha_bar[k]) Ybar_0 + sqrt(1-alpha_bar[k]) epsilon
epsilon_hat: [B,P,J_h,3]
Yhat_h = IDCT_P(Ybar_0_hat)
```

`utils/motion_latent.py` 与 `models/diffusion.py` 实现上述过程。模型 API 将关节坐标展平为
`[B,P,3*J_h]`。所有论文配置 `n_pre=P`；`n_pre<P` 是显式额外消融，只保留前 M 个频率系数，
解码仍用 `[P,M]` 的 IDCT 子矩阵。不是对 T+P 拼接序列做 DCT。
训练时间索引为 1…1000，`alpha_bar[0]=1`；DDIM 默认 100 个子采样步，最后一步回到 0，eta=0。

### ST-GCN：式 (2)

第一层 `C_0=3`，两层 GCB，输出维度均为 d。每层有独立的
`A_s:[J_h,J_h]`、`A_f:[T,T]`、`W:[C_in,d]`，均可训练。
LayerNorm 沿通道轴，随后无偏置通道线性映射、关节混合、频率混合、ReLU、dropout。
按实现的 frequency-major 展平顺序，整体邻接是 `A_f ⊗ A_s`；论文若使用 joint-major 顺序写
`A_s ⊗ A_f`，需同步说明排列。两个操作等价于相应的张量重排，不能混淆展平顺序。
频率和空间邻接初始均为单位阵，训练时不施加稀疏或非负约束。

### 坐标嵌入：替换式 (3)

```text
Xtilde_h = Linear_h(Xbar_h) + E_f + E_j_h
Xtilde_r = Linear_r(Xbar_r) + E_f + E_j_r
Linear_h/r: R^3 -> R^d
E_f: [1,T,1,d]                 (两分支共享固定 sin/cos 数值)
E_j_h: [1,1,J_h,d]             E_j_r: [1,1,J_r,d]
```

交互分支的 human query 来自投影后的**观测坐标**，没有将 GCN 输出偷偷用作 query。
intra 分支则先完成 GCB，再加频率/关节位置嵌入及轻量通道投影。

### 不等关节数的频率注意力：补充式 (5)

当前 ST-Attn 深度 L_2=1。空间注意力按每个观测频率独立计算：

```text
Q: [B*T,J_h,d]    K,V: [B*T,J_r,d]
S_spa: [B,T,J_h,d]
R[b,f,h,:] = (1/J_r) sum_{r=1..J_r} Xtilde_r[b,f,r,:],  h=1..J_h
Q_f = transpose(S_spa): [B*J_h,T,d]
K_f,V_f = transpose(R): [B*J_h,T,d]
S_frq = transpose_back(MHA(Q_f,K_f,V_f)): [B,T,J_h,d]
```

Q/K/V 输入先做代码中的 LayerNorm，注意力含输出线性投影及 dropout。
上述均值池化只用于频率 K/V；空间阶段没有平均伙伴关节。
这给出了当前实现的完整算子，**不声称原式 (5) 已写明这些步骤**，也不新增未经实验验证的注意力机制。
无伙伴时整个 inter 条件严格为零，避免 projection bias 产生虚假的交互信号。

### 条件长度映射：补充 ψ、式 (7)、(9)、(11)、(17)、(18)

两个分支各自使用独立的频率线性映射 `W_f_intra, W_f_inter:[P,T]`，偏置 `[P]`，
对每个关节、每个通道共享：

```text
F: [B,T,J_h,d]
Ftilde[b,p,j,c] = sum_f W_f[p,f] * psi_channel(F)[b,f,j,c] + b_f[p]
Ftilde: [B,P,J_h,d]
G = phi(AttentionPool(flatten(Ftilde, axes=(P,J_h)))): [B,d]
```

`psi_channel` 为 LayerNorm→Linear(d,d)→SiLU→Linear(d,d)。AttentionPool 的权重由
`Linear(LayerNorm(token),1)` 加 softmax 产生。截断消融在映射到完整 P 后选前 M 个 token，
其全局池化也使用这 M 个 token。这样式 (11)、(18) 的逐元素加法才有定义。

### 层数分配与第二阶段：式 (12)、(19)、表 VII

默认先加 intra token，以 `c1=c_k+G_intra` 运行 2 个块；在边界加 inter token，
以 `c2=c_k+G_intra+G_inter` 运行 7 个块；输出头使用最后阶段的全局条件。
每块按 spatial attention→frequency attention→MLP 顺序执行，含 AdaLN 和残差 gate。

- **0/9**：先后在 block 0 前加入两类 token，9 块均接收 c2；不是“移除 intra 模块”。
- **9/0**：只加入 intra token，9 块均接收 c1；不执行 inter 条件路径。
- 层数消融与“移除 ST-GCN/ST-Attn”的组件消融不是同一个实验。

## 3. 本次额外发现并修复的问题

- 所有六份配置时间反转概率从 0.2 改为论文 IV-C 的 0.3；旋转概率为 0.5。
- HARPER 模型按论文要求接收 21 人体 + 21 Spot，条件去掉主人体根后是 41 个关节。
  作者已确认其项目将原始 23 点数组的 21/22 视为相机相关点，不参与计算。
  两份 YAML 已固定 `harper_spot_joint_indices` 为 0–20；加载器、多模态加载器和预处理
  在归一化、增强及 DCT 之前统一删除 21/22。输入必须为 `[F,23,3]`，防止默默套用另一套序号。
  选点定义位于公共 `utils/harper_layout.py`，显示连边也按选点同步裁剪。
  [官方 links.py](https://raw.githubusercontent.com/intelligolabs/HARPER/main/tools/links.py)
  仍将 21/22 标为 wrist/hand；文档区分官方名称与作者确认的相机相关点命名。
  全部人体/Spot 节点及官方材料的数量矛盾见 [HARPER 节点核查](harper_joint_definition_audit.md)。
  官方可视化代码明确注明 Y-up，本地 HARPER 旋转增强已相应改成绕 Y 轴。
- CHICO 的训练增强以前落入基类镜像增强，与论文的旋转/反转不符；本地已统一。
  CHICO 现在使用 S00/S04 验证集选模型，S02/S03/S18/S19 留作最终测试。
- 本地采样器接纳长度恰好 T+P 的序列和最后有效窗口，包含不足 batch_size 的尾批；
  50,000 样本不会再向下取整成 49,984。训练/验证 loss 按实际样本数加权。
- CoMaD 原来保留首个标记的全局坐标，尽管 `drop_root_joint=False`；本地修复为零锚点。
  不存在的伙伴保持零填充，避免减根操作把其变成虚假实体。
- CoMaD 显示坐标变换原来在送入模型前及预测后分别发生，破坏训练/预测一致性；
  现在仅在显示前统一变换一次，不改变模型输入。
- EMA 开启时，验证和保存使用同一套 EMA 参数，避免用普通模型 loss 选择另一套权重。
- 检查点含版本与模型配置；原始无版本 state_dict 会被拒绝，层数分配不同但参数形状相同时也会拒绝。
  新版仍只支持权重初始化式 `--resume`，不声称恢复了 optimizer/RNG 的精确训练状态。

## 4. 实际配置、协议和实验留档

| 配置 | T | P / n_pre | 输出关节 J_h | 条件关节 J_h+J_r |
|---|---:|---:|---:|---:|
| harper3d_30hz | 25 | 100 | 20 | 41 |
| harper3d_120hz | 100 | 400 | 20 | 41 |
| chico | 10 | 25 | 14 | 23 |
| comad (HR) | 15 | 15 | 9 | 11 |
| 3dpw | 25 | 100 | 23 | 47 |
| cmu_mocap (recorded pairs required) | 25 | 100 | 38 | 77 |

3DPW/CMU 的 T、P 是项目设置，不能当作论文 IV-C 已明确给出的参数。
模型为 9 层、2/7、8 heads、d=512、FFN=1024、dropout=0.2。
AdamW: lr=3e-4、weight_decay=0.01、betas=(0.9,0.999)、eps=1e-8；
100 epochs，每 20 epochs lr×0.8。weight_decay、heads、FFN 等为公开记录的实现选择，需补充到论文。
论文配置没有 velocity 输入/辅助 loss，也不随机丢弃条件。默认使用 EMA(0.995)，前 2000 次更新复制参数。

当前 ADE/FDE 在每帧 `[3*J_h]` 展平向量上算 L2；不是 MPJPE。
APD 是整段 `[P*3*J_h]` 的成对 L2 均值。MM 指标的集合来自测试窗口的最后一个观测姿态，
以 `--multimodal_threshold` 严格小于阈值筛邻居，包括自身；无骨架缩放、无多历史帧匹配。
先对 K 个预测取 min/lower-median/max，再在候选真值和窗口间平均。
K 默认 50，可用 `--eval_samples` 修改；APD 与 best/worst 指标必须连同 K 一起报告。
历史 YAML 中的 multimodal_path/data_candi_path 不决定当前指标集合，当前评估会从实际 test windows 重建。

每次评估输出 `evaluation_<UTC时间>/`：

- `metadata.json`：K、协议、配置、窗口 ID、邻居编号、运行 manifest 副本；失败时标为 incomplete。
- `predictions_*.npz`：分块保存观察、K 个未来预测、真值、每个窗口的多模态真值和 offsets。
- `per_sample.csv` 和 `summary.csv`：逐窗口和最终均值，方法名统一为 ReFusion。
- `stats_latest.csv` / `stats.csv` 只保留最新快照，历史以独立目录为准，不再拼接无运行标识的列。

```bash
python -m pytest tests -q
python main.py --cfg chico --mode train --data_path /path/to/CHICO --exp_name chico_aligned_v2
python main.py --cfg chico --mode eval --data_path /path/to/CHICO --eval_samples 50 --ckpt results/chico_aligned_v2/models/best_ema.pt --exp_name chico_aligned_v2
python -m utils.rescore_evaluation inference/chico_aligned_v2/results/evaluation_<timestamp>
```

train/eval/pred 需要作者私有加载器、实际数据和新训练的检查点；前后两条核心测试/离线重算命令不依赖它们。
表 VII 可在上述训练命令增加 `--num_layers 9 --stage1_num_layers 0` … `9`，
为各次运行使用不同 exp_name。多种子使用 `--seed` 和独立 exp_name；本次没有执行这些昂贵实验。

## 5. 验证范围与仍不能宣称完成的事项

验证包括小张量前后向、实际一轮优化、DDIM 噪声 oracle、条件未来泄漏、全部层数端点、
Kronecker 等价、不同 T/P/J、配置与论文参数、离线预测重评分、私有加载器的合成输入检查。
本地环境为 Windows、Python 3.12.14、PyTorch 2.14.0+cpu；没有 CUDA、实际数据或可归属新版的训练权重。
在本地完整源码上通过 86 项测试；移除整个私有 data_loader 的独立副本通过 77 项测试、
跳过 1 项私有 CoMaD 显示映射测试；`main.py --help` 可正常运行。
测试不证明 A100 上的内存/吞吐、生成质量、重复实验方差、真实交互泛化或表格性能。

提交论文前仍须：用实际 HARPER 文件验证作者确认的映射及数据处理；将上面的公式补充写入原稿；根据真实录制与新结果重新组织表 IV；
检查现有表格的原始记录；补充指标解释、重复运行及匹配输入的 baseline；
将物体/场景语义、导航和碰撞规划等超出 3D 点轨迹与 root-relative 输出的表述收窄。
恢复加载器公开前，不能声称公开仓库足以完整复现实验。
