# 遥感语义分割与表征诊断工作总结

本文档对本阶段围绕 DINOv3 / OLMoEarth 骨干网络在三类遥感数据集上开展的语义分割实验、特征塌缩（collapse）可视化诊断，以及推理框架精简等工作进行系统性总结。全文遵循学术发表风格，所列数值均来源于各实验的原始训练日志与评测结果。

---

## 第一部分　数据集

本研究涉及 PASTIS、Potsdam 与 SVDT 三类遥感语义分割数据集，覆盖卫星时序、高分辨率航空影像与农业专题三种场景。各数据集的分辨率、谱段、时相与模态属性汇总如下。

| 数据集 | 分辨率（GSD） | 谱段/波段 | 时相 | 模态 | 类别数 |
|---|---|---|---|---|---|
| **PASTIS** | 10 m（Sentinel-2） | 10 波段（B02、B03、B04、B05、B06、B07、B08、B08A、B11、B12） | 多时相；时序实验采用 12 个月中值合成（`n_frames=12`），非时序实验将全时相均值塌缩为单帧 | 单模态（光学） | 19（背景 + 18 种作物） |
| **Potsdam** | 5 cm（ISPRS 航空） | 3（RGB） | 单时相 | 单模态光学（HR）；融合实验额外引入 OLMoEarth embedding 作为第二模态上下文 | 5（不透水面、建筑、低矮植被、树、车） |
| **SVDT** | 高分辨率光学（文件名携带 `2024_08m` 标记，约 8 m 级） | 3（RGB） | 单时相 | 单模态（光学） | 2（背景 / 农田，二值） |

**PASTIS** 基于 Sentinel-2 的 10 m 数据集，包含 10 个光谱波段与可变数量的观测时相（`mmseg_dino/custom_datasets/pastis.py:19`）。其非时相数据管线将全部时相通过 `s2.mean(axis=0)` 塌缩为单帧（`pastis.py:40`），而时相实验使用预先计算的 12 个月中值合成体，张量形状为 `(12, 10, 128, 128)`（`pastis_temporal.py:27,34`）。该数据集仅读取光学 S2 数据，为单模态数据集，类别数为 19（背景 0 + 18 种作物，空值映射至 `ignore_index=255`）。

**Potsdam** 为 ISPRS 高分辨率航空遥感基准（5 cm GSD），以 3 通道 RGB 方式加载（`potsdam_unet/custom_datasets/customPotsdam.py:20`），无时间维度，单时相单模态。其类别数为 5，原始标签 0/6 映射至忽略值（`customPotsdam.py:32-37`）。在融合实验中，Potsdam 的光学影像作为高分辨率（HR）分支，并额外加载 768 维的 OLMoEarth embedding 作为低分辨率多模态上下文（`fusion_potsdam/custom_datasets/customPotsdam.py:124-141`）。

**SVDT** 为农业专题高分辨率光学数据集，3 通道 RGB、单时相、单模态（`mmseg_dino_agri/custom_datasets/customAgri.py:72-73,92`）。该数据集为二值分割任务（背景/农田），`mmseg_unet` 与 `mmseg_dino_agri` 中的加载器代码逐字节相同。在融合配置中，SVDT 充当与 OLMoEarth 低分辨率上下文融合的 HR 光学分支。

---

## 第二部分　各数据集上的测试

### 2.1 PASTIS 数据集

**综述。** 本部分工作集中于 `/mnt/ht2-nas2/00-model/00-fb/mmseg_dino/` 目录，配置文件位于 `configs/`，骨干封装位于 `custom_models/`，数据加载位于 `custom_datasets/`。在 DINOv3 ViT-Large 与 ViT-Small（patch=16）骨干上，针对**输入范式与时序聚合方式**两个维度展开对照实验：进行多波段单时相（配置 `dinov3l_m2f_pastis_v2`）、多波段多时相即时序中值/均值池化（配置 `dinov3l_m2f_pastis_temporal_v1`、`dinov3l_m2f_pastis_temporal_v3`、`dinov3s_m2f_pastis_temporal_v2_vits`、`dinov3s_m2f_pastis_zjf`），以及多波段多时相叠加注意力池化（配置 `dinov3l_m2f_pastis_temporal_v4`）等不同方式的测试。其中时序均值配置 `temporal_v1` 采用 1×1 卷积将 10 波段降至 3 通道，`temporal_v3` 及后续配置改用对 patch embedding 与 SPM stem 进行通道膨胀（3→10）的策略；注意力配置 `temporal_v4` 在 scales 2/3/4 上使用带月份位置编码的 `TemporalAttnPool` 多头注意力进行时序池化，scale 1 因显存约束退化为均值池化。所有实验的任务头均为 **Mask2Former**（`num_queries=100`，ViT-L 为 `in_channels=[1024]×4`、ViT-S 为 `[384]×4`，类别数为 19）。

**全参冻结训练说明。** 尽管各时序配置中显式设置了 `freeze_backbone=False`（个别配置进一步通过 `paramwise_cfg` 对 `backbone.adapter.backbone` 设定 `lr_mult`），但该参数实际为无效参数。DINOv3 adapter 上游代码在 `dinov3/eval/segmentation/models/backbone/dinov3_adapter.py:423` 以 `with torch.no_grad():` 上下文包裹了骨干的前向计算（`get_intermediate_layers`），并在构造期 `:326` 调用 `requires_grad_(False)`。骨干封装虽在随后将 `requires_grad` 置回 `True`（如 `dinov3_temporal_backbone_v3.py:181-182`），但 `no_grad()` 前向上下文使梯度图无法建立，反向时骨干参数梯度恒为 `None`，优化器不会更新。因此 ViT 骨干在所有实验中实际处于冻结状态，仅有 adapter（SPM 与 InteractionBlocks）及 Mask2Former 头参与训练。这本质上是上游 ViT-Adapter / DINOv3 "冻结特征提取器" 设计的固有行为。

**mIoU 结果。** 下表记录各已训练实验的最佳 mIoU（来源：`work_dirs/<config>/<timestamp>/vis_data/scalars.json`）。未实际训练的单时相 RGB 配置（`dinov3b_m2f_pastis_raster`）与多波段 1×1 卷积配置（`dinov3l_m2f_pastis_v1`）已从表中剔除。

| 配置（输入范式 / 时序聚合） | 骨干 | 任务头 | Best mIoU (%) | @ iter |
|---|---|---|---|---|
| `dinov3l_m2f_pastis_v2`（多波段单时相，stem 膨胀） | ViT-L | Mask2Former (100) | 39.67 | 20000 |
| `dinov3l_m2f_pastis_temporal_v1`（多波段多时相，mean，1×1 卷积） | ViT-L | Mask2Former (100) | 50.81 | 18000 |
| `dinov3l_m2f_pastis_temporal_v3`（多波段多时相，mean，stem 膨胀） | ViT-L | Mask2Former (100) | 54.97 | 20000 |
| `dinov3s_m2f_pastis_temporal_v2_vits`（多波段多时相，mean） | ViT-S | Mask2Former (100) | 53.56 | 19000 |
| `dinov3s_m2f_pastis_zjf`（多波段多时相，mean） | ViT-S | Mask2Former (100) | 54.26 | 20000 |
| `dinov3l_m2f_pastis_temporal_v4`（多波段多时相 + attention） | ViT-L | Mask2Former (100) | **58.31** | 16000 |

实验表明：时序信息对 PASTIS 至关重要，引入 12 个月中值合成相较单时相带来约 +11 mIoU 的提升（`v2` 单时相 39.67 → `temporal_v3` 时序均值 54.97）；通道膨胀策略优于 1×1 卷积降维（`temporal_v3` 54.97 > `temporal_v1` 50.81，提升约 +4 mIoU）；注意力时序池化取得全局最优（`temporal_v4` 58.31，较均值池化再提升约 +3.3 mIoU）。

### 2.2 SVDT 数据集

**综述。** 本部分涉及 `/mnt/qh2-nas3/00-model/00-fb/mmseg_unet/`（UNet 基线，配置 `configs/unet_customAgri.py`）与 `/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_agri/`（DINOv3，配置 `configs/dinov3l_m2f_agri.py`；`dinov3l_m2f_agri_v1.py` 仅将 `num_queries` 改为继承基线默认值 100，未实际训练）。在 SVDT 数据集上对比了两种任务头：**UNet + FCN**（5 级下采样骨干，主头接最高分辨率层 `in_index=4`，并设 `in_index=3` 的辅助 FCN 头用于深度监督）与 **DINOv3 ViT-Large + Mask2Former**（`num_queries=50`，2 类，`class_weight=[1.0, 2.0, 0.1]`）。优化设置上，UNet 采用 AdamW（lr=1e-3，40k 迭代）；DINOv3 采用 AdamW（lr=5e-5，80k 迭代，3k 线性 warmup + PolyLR）。

**全参冻结训练说明。** `dinov3l_m2f_agri.py:40-41` 显式设置 `freeze_backbone=False` 与 `finetune_vit=True`，但二者同样为无效参数。骨干封装 `mmseg_dino_agri/custom_models/dinov3_backbone_fb.py` 通过 `sys.path` 注入上游 adapter，而该 adapter 在 `dinov3_adapter.py:423` 同样以 `no_grad()` 包裹 ViT 前向；`finetune_vit` 属性在 adapter 代码中从未被读取（全代码树零匹配）。因此 ViT-L 全程保持 sat493m 预训练权重不变，仅 adapter 与 Mask2Former 头参与训练。这也解释了 DINOv3 配置在 iter 2000 即达到 82.66 mIoU 的快速收敛现象。

**mIoU 结果。**

| 配置 | 骨干 | 任务头 | Best mIoU (%) | @ iter |
|---|---|---|---|---|
| `unet_customAgri.py` | UNet（5 stage，base_ch=64） | FCN + 辅助 FCN | 85.91 | 38000 |
| `dinov3l_m2f_agri.py` | DINOv3 ViT-L | Mask2Former (50) | **88.30** | 60000 |

DINOv3 + Mask2Former 较 UNet + FCN 提升约 +2.39 mIoU，验证了冻结 DINOv3 骨干配合 Mask2Former 查询式解码器在农业二值分割任务上的有效性。

### 2.3 Potsdam 数据集

**综述。** 本部分涉及 `/mnt/qh2-nas3/00-model/00-fb/fusion_potsdam/`、`/mnt/qh2-nas3/00-model/00-fb/potsdam_lp/` 与 `/mnt/qh2-nas3/00-model/00-fb/potsdam_unet/` 三个目录，分别对应端到端微调、OLMoEarth 冻结 embedding 线性探针（Linear-Probe, LP）与 UNet 基线。实验使用了**三类权重**进行推理：

1. **DINOv3 官方预训练权重**。包括 ImageNet LVD-1689M 预训练（`/mnt/ht2_nas2/EO_test/weights/Dinov3_pretrained/DINOv3 ViT LVD-1689M/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth`）与遥感 SAT-493M 预训练（`dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`，DINOv3 ViT-L/16 官方遥感预训练）。

2. **自有权重**。浙江遥感数据 SSL 预训练权重，位于 `/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/stage2+stage3-zhejiang/`，涉及多个 epoch 的检查点：`5999.pth`、`9999.pth`、`23999.pth`、`31999.pth`；另有无对比损失版本目录 `stage2+stage3-zhejiang_no_cl/`。

3. **OLMoEarth 权重**。作为线性探针输入的 embedding，涉及三个版本：**v1**（原生，`/mnt/qh2-nas3/00-model/00-limx/datasets/postdam_bgr/postdam_embeddings_v1_*`）、**v1 + hypernet**（`/mnt/qh2-nas3/00-model/00-limx/datasets/postdam_bgr_hyper/postdam_embeddings_v1_hypernet_*`）与 **v1.2**（v1.2 推理输出，`/mnt/qh2-nas3/00-model/guantp/dior_inference/limx/olmoearth_inference/outputs/postdam_v1_2_0`）。

任务头方面：端到端与融合实验采用 **Mask2Former**（`num_queries=50`，5 类，见 `fusion_potsdam/configs/dinov3_m2f_9999.py`）；UNet 基线采用 FCN；线性探针采用 **`OlmoEarthPatchLinearHead`**（patch_size=4，将 128×128 token 展开为 512×512 像素 logits，骨干为恒等直通、不训练）。所有实验均遵循冻结骨干的训练范式：fusion_potsdam 配置显式设定 `freeze_backbone=True`、`finetune_vit=False`，并通过 `paramwise_cfg` 对 `backbone.backbone` 设定 `lr_mult=0.0`；线性探针更仅训练线性头。

**表 1 — DINOv3 官方权重与自有权重的 mIoU 与 mF1。** 下表为端到端/融合实验汇总（来源：`fusion_potsdam/RESULTS.md`），按 mIoU 降序排列。`HR` 表示纯 DINOv3 骨干，`fusion` 表示叠加 OLMoEarth 上下文的融合骨干。

| # | 实验 | 骨干 | 预训练 | Context | 对比损失 | mIoU | mFscore |
|---|---|---|---|---|---|---|---|
| 1 | DINOv3 LVD-1689M (ori) | HR | ImageNet | — | — | **89.00** | 94.07 |
| 2 | nocont HR 9999 | HR | zhejiang | — | No | 88.49 | 93.77 |
| 3 | HR 9999 | HR | zhejiang | — | Yes | 88.43 | 93.73 |
| 4 | gram HR 4999 | HR | gram | — | No (gram) | 88.39 | 93.75 |
| 5 | nocont HR 23999 | HR | zhejiang | — | No | 88.12 | 93.57 |
| 6 | HR 23999 | HR | zhejiang | — | Yes | 88.11 | 93.56 |
| 7 | HR 31999 | HR | zhejiang | — | Yes | 87.98 | 93.47 |
| 8 | fusion 9999 | fusion | zhejiang | mask_token | Yes | 87.85 | 93.41 |
| 9 | nocont fusion 9999 | fusion | zhejiang | mask_token | No | 87.85 | 93.41 |
| 10 | nocont fusion 23999 | fusion | zhejiang | mask_token | No | 87.66 | 93.30 |
| 11 | fusion 31999 (mask) | fusion | zhejiang | mask_token | Yes | 87.64 | 93.29 |
| 12 | fusion 31999 (olmov1) | fusion | zhejiang | olmoearth RGB | Yes | 87.64 | 93.29 |
| 13 | fusion 23999 | fusion | zhejiang | mask_token | Yes | 87.54 | 93.24 |
| 14 | DINOv3 SAT-493M (sat) | HR | SAT RS | — | — | 86.85 | 92.85 |

作为参照，UNet + FCN 基线在 Potsdam 上的最佳 mIoU 为 84.85（来源：`potsdam_unet/work_dirs/unet_potsdam/.../scalars.json`），显著低于上述 DINOv3 实验。结果显示：官方 ImageNet LVD-1689M 预训练最优（89.00）；浙江遥感 SSL 权重（88.43–88.49）显著优于官方遥感 SAT-493M（86.85），差距约 1.6 mIoU；纯 HR 骨干始终优于融合骨干（差距约 0.5–0.9 mIoU）；较早的检查点（9999.pth）在多数实验中优于后期检查点。

**表 2 — OLMoEarth 三种权重在不同学习率与 tile 配置下的 50 epoch 结果（best mIoU %）。** 在线性探针设定下，需特别指出：由于线性头对冻结特征构成凸优化问题，存在全局最优解，因此**设置不同学习率并无必要性**——学习率（1e-1 至 1e-4）仅影响收敛速度，而非最终可达精度上界。下表给出三套特征在各 tile 配置下的最优值（括号标注对应最优学习率），其中 size 4 / 256 / 512 对应 `tile4_patch4` / `tile256_patch4_50_overlap` / `tile512_patch4`。

| 权重 \ size | 4 | 256（50% overlap） | 512 |
|---|---|---|---|
| **v1（原生）** | 48.15 (1e-2) | **69.02** (1e-1) | 60.37 (1e-1) |
| **v1 + hypernet** | 49.52 (1e-1) | 68.79 (1e-1) | 60.26 (1e-1) |
| **v1.2** | 47.46 (1e-2) | 68.70 (1e-1) | **64.98** (1e-1) |

可见 size 256（50% overlap）抽取的特征对线性探针最为有利，三套权重均达到约 69 mIoU；v1.2 在 size 512 上显著优于 v1/v1+hypernet（+4.6 mIoU），说明 v1.2 推理在 tile512 配置下产出了更高质量的 embedding；v1 与 v1+hypernet 两套特征整体持平。

**表 3 — 两个权重训练 600 epoch 的结果对比（tile256 / lr=1e-3）。**

| 权重 | 600 epoch Best mIoU (%) |
|---|---|
| **v1**（`step600_lr1e3`） | **69.49** |
| **v1.2**（`v1_2_step600_lr1e3`） | 68.69 |

延长训练至 600 epoch 后，v1 与 v1.2 基本持平（v1 略优 +0.8 mIoU）。这表明 v1.2 在 50 epoch / size 512 下的短期优势在充分训练后被抹平，与线性探针存在全局最优、长训练趋于收敛上界的结论一致。

---

## 第三部分　可视化方式

本部分工作集中于 `/mnt/qh2-nas3/00-model/00-fb/visualise/` 与 `/mnt/qh2-nas3/00-model/00-fb/fusion_dino/` 两个目录，旨在通过四类互补的可视化手段诊断骨干网络表征的"特征塌缩（collapse）"问题——即表征退化为各 token 或全局向量近乎相同、丧失判别力的失效模式。

**1. PCA（主成分分析）。** PCA 适用于对结果特征进行降维可视化，从而揭示塌缩现象。其做法是对骨干最后一层的逐像素/逐 patch token 特征（维度 D=1024）进行主成分分析，取前 3 个主成分映射为 RGB 图像。当特征发生塌缩时，所有 token 趋于一致，方差几乎全部集中于单一主成分，PCA 可视化图呈现为均匀的单色平面、丧失空间结构。实现上，`visualise/task1_layer_cosine.py:65-72` 的 `pca()` 采用 `np.linalg.svd`（mean-center 后投影到前 3 奇异方向），而 `fusion_dino/visualise2.ipynb` 与 `mmseg_dino_agri/visualise2.ipynb` 采用 `sklearn.decomposition.PCA(n_components=3)`，二者在数学上等价。

**2. Cosine Similarity（余弦相似度）。** 余弦相似度适用于查看不同层的语义相似度与特征融合情况，同时可观察塌缩。其做法是对每像素 token 沿通道维 L2 归一化后与某一参考 token 做点积，得到 H×W 的相似度热力图（`visualise/task1_layer_cosine.py:61-63`；`fusion_dino/visualise_cosine.py:60-62`，并设 `vmin=0.9 / vmax=1.0` 以放大塌缩区域）。通过跨层对比可观察语义层次的演化，通过融合前后对比可评估融合质量；当特征塌缩时，全图相似度饱和趋近 1.0，丧失空间纹理。

**3. Heatmap（全局向量自相似矩阵）。** 该方法通过对比所有图像输出的 class/global token，揭示不同图像全局向量的塌缩问题。具体地，对 N 张图像各提取一个全局向量（ViT 取 CLS token；融合骨干取 patch 均值；OLMoEarth 取模态空间池化均值），构成特征矩阵 F∈R^{N×D}，对各行 L2 归一化后计算 Gram 矩阵 F·Fᵀ 并将对角线置 0（`visualise/task2_global_heatmap.py:254-259`，`task2_cl_vs_nocl.py:109-110`）。在健康模型中，不同图像的全局向量分散，矩阵呈现多彩结构；当发生全局向量塌缩时，所有图像映射至近乎相同的向量，非对角元素整体饱和趋近 +1，热力图呈现为一片亮红。该方法是验证对比损失（contrastive loss）抗塌缩效果的核心手段（`task2_cl_vs_nocl.py` 即用于对比有无对比损失的两行模型）。

**4. 全局向量矩阵的 SVD（奇异值谱）。** 该方法基于奇异值归一化后的谱衰减斜率诊断塌缩。对中心化后的特征矩阵 Fc 进行 SVD 分解，并按最大奇异值归一化得到 σ/σ_max 谱（`visualise/task3_svd_spectrum.py:281-286`，`task3_cl_vs_nocl.py:204-209`），以对数纵轴绘制。塌缩表征的有效秩近似为 1，即存在一个主导奇异值而其余奇异值趋近于 0，谱线表现为陡降；而健康模型的奇异值谱缓慢衰减，存在多个有效奇异值，表征丰富、高秩。因此谱衰减的陡峭程度即为塌缩程度的定量判据。

---

## 第四部分　其余工作

**1. `mmseg_dino_infer`：基于纯 PyTorch 的 Mask2Former 推理。** 该项目位于 `/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_infer/`，旨在使用 Mask2Former 仅基于 PyTorch 完成推理，摆脱对 mmdet / mmcv 及其编译扩展的依赖。其推理流程在 `@torch.no_grad()` 下进行（`infer.py:45-96`）：以纯 PyTorch 算子（`F.interpolate`、`einsum`、`argmax` 与手写混淆矩阵 IoU）组装骨干与头部并完成评测。关键组件 `convert_checkpoint.py:19-109` 将 openmmlab / mmdet 训练所得权重的键名（`pixel_decoder.*`、`transformer_decoder.*` 等）映射至 Meta 官方 `Mask2FormerHead` 的键名（`head.pixel_decoder.*`、`head.predictor.*`），随后 `load_state_dict(strict=True)` 加载。项目通过 `configs.py` 注册网络：`pastis`（temporal，12×10 波段，19 类）与 `agri`（single，RGB，2 类，`num_queries=50`）。值得注意的是，MSDA（Multi-Scale Deformable Attention）在 `dinov3/.../utils/ms_deform_attn.py:18-66` 提供了纯 PyTorch 的前向回退（基于 `grid_sample` 实现），因此推理前向无需编译 CUDA 扩展，仅在训练反向传播时才需要编译。

**2. `olmoearth_inference_v2_1`：OLMoEarth 推理步骤的简化。** 该项目位于 `/mnt/ht2_nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1/`，是 OLMoEarth 多模态对地观测编码器的推理专用精简版本（`config.py` 标注 "Minimal Config for inference-only mode"，`nn/flexi_vit.py` 标注 "inference only version"）。相较完整预训练仓库，其简化体现在三个方面：其一，仅构建 Encoder，丢弃 decoder 与 target-encoder，加载权重时剥离 `decoder.` 与 `target_encoder.` 前缀（`dataload/model.py:70-88`，`load_state_dict(strict=False)`）；其二，默认启用 `fast_pass=True`（`olmoearth_inference_h5.py:45`），推理期不物理移除 masked token、不生成 attention mask（`nn/flexi_vit.py:813-822,888-907`），并跳过对比投影头，从而显著降低推理开销；其三，仅保留 H5 输入入口（`olmoearth_inference_h5.py`），去除了 v2 版本中的 JP2 变体与打包配置文件。骨干为 FlexiViT（depth=12，embed=768，heads=12，patch_size=4，`FlexiPatchEmbed` 可重采样至任意 patch 尺寸）。输入 H5 中每模态张量形状为 `[H, W, T, C]`，时间维 pad/截断至 12；输出为每模态 token `[B, H', W', T, S, 768]`，经池化得到 `[B, H', W', 768]`。相较于 v1（仅 JP2）与 v2（最全、含 JP2 与 H5），v2_1 是面向单 H5 推理、配合 fast_pass 与 Encoder-only 的精简版本。

---

*注：本文档为阶段性总结。各项目目录的逐文件细致说明（代码运行方式、文件含义）将另行整理于 `summerise_per_project/` 目录下，不在本文档范围内。*
