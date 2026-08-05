# 遥感基础模型语义分割与表征分析项目总结报告

# 第一部分 数据集与实验对象

本项目围绕遥感基础模型在不同遥感场景下的迁移能力开展实验研究，涉及三类具有代表性的数据集：

* PASTIS
* Potsdam
* SVDT

三类数据集分别覆盖多时相卫星遥感、高空间分辨率航空遥感以及农业专题遥感任务，在空间分辨率、光谱维度、时间信息和应用场景方面具有明显差异。相关实验主要基于以下工程环境展开：
```text
/mnt/ht2_nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1/
/mnt/ht2-nas2/00-model/00-fb/mmseg_dino/
/mnt/qh2-nas3/00-model/00-fb/fusion_potsdam/
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_agri/
/mnt/qh2-nas3/00-model/00-fb/mmseg_unet/
/mnt/qh2-nas3/00-model/00-fb/potsdam_lp/
/mnt/qh2-nas3/00-model/00-fb/potsdam_unet/
/mnt/qh2-nas3/00-model/00-fb/visualise/
/mnt/qh2-nas3/00-model/00-fb/fusion_dino/
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_infer/
```
---

# 1.1 PASTIS 数据集

PASTIS 是基于 Sentinel-2 卫星遥感影像构建的多时相农业语义分割数据集，主要用于研究基础视觉模型对于时间序列遥感数据的建模能力。其主要属性如下：

| 属性    | 描述         |
| ----- | ---------- |
| 数据来源  | Sentinel-2 |
| 空间分辨率 | 10 m       |
| 光谱信息  | 10 个光谱波段   |
| 时间维度  | 多时相观测      |
| 数据模态  | 单模态光学遥感    |
| 任务类型  | 农作物语义分割    |

PASTIS 的主要特点是同时包含空间、光谱和时间三个维度的信息，因此能够用于评价模型对于多维遥感数据的综合建模能力。本项目中针对 PASTIS 数据主要研究：

1. 多光谱信息对于基础模型迁移能力的影响
2. 多时相信息对于农业类别识别的贡献
3. 不同时间融合策略对于语义分割性能的影响

---

# 1.2 Potsdam 数据集

Potsdam 是 ISPRS 发布的高空间分辨率航空遥感语义分割数据集，主要用于城市区域地物分类任务。其主要属性如下：

| 属性    | 描述       |
| ----- | -------- |
| 数据来源  | 航空遥感影像   |
| 空间分辨率 | 5 cm     |
| 光谱信息  | RGB      |
| 时间维度  | 单时相      |
| 数据模态  | 单模态光学遥感  |
| 任务类型  | 城市地物语义分割 |

Potsdam 包含五类典型城市地物：
* Impervious Surface
* Building
* Low Vegetation
* Tree
* Car

本项目中，Potsdam 数据集主要用于评价不同视觉基础模型在高分辨率遥感场景中的迁移能力，包括：

* DINOv3 不同预训练权重
* 自有遥感 SSL 权重
* OLMoEarth embedding
* UNet 卷积基线

涉及实验目录包括：

```text
/mnt/qh2-nas3/00-model/00-fb/fusion_potsdam/
/mnt/qh2-nas3/00-model/00-fb/potsdam_lp/
/mnt/qh2-nas3/00-model/00-fb/potsdam_unet/
```

---

# 1.3 SVDT 数据集

SVDT 用于农业遥感场景下的专题语义分割任务，主要用于验证基础视觉模型在农业区域识别中的迁移能力。其主要属性如下：

| 属性    | 描述      |
| ----- | ------- |
| 数据来源  | 农业遥感影像  |
| 空间分辨率 | 约 8 m   |
| 光谱信息  | RGB     |
| 时间维度  | 单时相     |
| 数据模态  | 单模态光学遥感 |
| 任务类型  | 农田二分类分割 |

SVDT 实验主要比较：

* 传统 CNN 方法
* 基于 DINOv3 的基础视觉模型方法

涉及工程目录：

```text
/mnt/qh2-nas3/00-model/00-fb/mmseg_unet/
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_agri/
```

---

# 1.4 数据集整体对比

| 数据集     | 分辨率   | 光谱信息           | 时间维度 | 模态 | 主要任务   |
| ------- | ----- | -------------- | ---- | -- | ------ |
| PASTIS  | 10 m  | Sentinel-2 十波段 | 多时相  | 光学 | 农作物分类  |
| Potsdam | 5 cm  | RGB            | 单时相  | 光学 | 城市地物分割 |
| SVDT    | 约 8 m | RGB            | 单时相  | 光学 | 农业区域分割 |

三类数据集覆盖了从低分辨率、多时间维度卫星遥感，到厘米级高分辨率航空遥感的不同应用场景，可用于系统评估基础视觉模型在不同遥感任务中的适应能力。

---
# 第二部分 不同数据集上的模型测试

本部分针对 PASTIS、SVDT 和 Potsdam 三类遥感数据集开展语义分割实验，重点评估不同视觉基础模型、任务头以及训练策略在不同遥感场景中的适应能力。实验整体采用迁移学习与监督学习两类策略：

1. 基于预训练视觉 backbone 的迁移学习：
   * 冻结 DINOv3 等基础模型 backbone
   * 训练语义分割任务头

2. 从头训练卷积模型：
   * 以 UNet 为代表

需要说明的是，部分 DINOv3 配置中虽然存在：

```text
unfreeze_backbone
finetune_vit
```

等参数设置，但由于实际 adapter 结构内部仍保持 backbone 参数冻结，因此这些参数不会改变训练过程中的梯度更新状态。因此，本项目中的 DINOv3 实验均属于：

> Frozen backbone + Trainable task head

的迁移学习模式。

---

# 2.1 PASTIS 数据集实验

## 2.1.1 实验设置

PASTIS 实验主要基于：

```text id="q4j7n2"
/mnt/ht2-nas2/00-model/00-fb/mmseg_dino/
```

该部分主要研究 DINOv3 在 Sentinel-2 多光谱、多时相遥感数据上的迁移能力。实验围绕输入数据组织方式展开，包括：

1. 单时相输入；
2. 多波段输入；
3. 多波段多时相输入；
4. 多波段多时相融合输入。

其中，多时相实验主要探索如何利用 Sentine2 时间序列信息提升农业类别识别能力。

---

## 2.1.2 实验方案

PASTIS 实验主要包含以下几类配置：

| 实验类型                     | 输入形式     | 主要目的       |
| ------------------------ | -------- | ---------- |
| Single Temporal          | 单时相输入    | 评价基础空间表征能力 |
| Multi Spectral           | 多波段输入    | 分析光谱信息贡献   |
| Multi Temporal Median    | 多时相统计融合  | 分析时间聚合效果   |
| Multi Temporal Attention | 多时相注意力融合 | 建模时间动态变化   |

其中：

* 多波段实验用于验证 Sentinel-2 光谱维度对于类别区分的重要性；
* 多时相中值融合用于构建简单时间聚合策略；
* Attention 融合用于学习不同时间点之间的重要性关系。

---

## 2.1.3 模型结构与任务头

PASTIS 实验采用 DINOv3 作为视觉特征提取 backbone，并结合不同语义分割任务头完成预测。涉及任务头包括：

* Mask2Former
* FCN Head
* 其他轻量化 segmentation decoder

整体结构为：DINOv3 Backbone + Segmentation Head，其中：

* backbone 提供预训练视觉表示
* segmentation head 完成像素级分类

训练过程中：

* backbone 参数保持冻结；
* 仅更新任务头和相关适配模块。

---

## 2.1.4 实验结果

PASTIS 数据集实验结果如下表所示。实验主要比较不同输入组织方式、时序聚合策略以及不同规模 DINOv3 backbone 对多时相农业遥感语义分割任务的影响。

| 配置（输入范式 / 时序聚合）                                       | 骨干    | 任务头               | Best mIoU (%) | @ iter |
| ----------------------------------------------------- | ----- | ----------------- | ------------- | ------ |
| `dinov3l_m2f_pastis_v2`（多波段单时相，stem 膨胀）               | ViT-L | Mask2Former (100) | 39.67         | 20000  |
| `dinov3l_m2f_pastis_temporal_v1`（多波段多时相，mean，1×1 卷积）  | ViT-L | Mask2Former (100) | 50.81         | 18000  |
| `dinov3l_m2f_pastis_temporal_v3`（多波段多时相，mean，stem 膨胀） | ViT-L | Mask2Former (100) | 54.97         | 20000  |
| `dinov3s_m2f_pastis_temporal_v2_vits`（多波段多时相，mean）    | ViT-S | Mask2Former (100) | 53.56         | 19000  |
| `dinov3s_m2f_pastis_zjf`（多波段多时相，mean）                 | ViT-S | Mask2Former (100) | 54.26         | 20000  |
| `dinov3l_m2f_pastis_temporal_v4`（多波段多时相 + attention）  | ViT-L | Mask2Former (100) | **58.31**     | 16000  |

---

实验结果表明，输入时间维度的引入能够显著提升模型在 PASTIS 多时相农业遥感任务上的分割性能。

首先，单时相多波段配置：`dinov3l_m2f_pastis_v2`，仅获得：`39.67% mIoU`，说明仅依赖单幅 Sentinel-2 多光谱影像时，模型虽然能够利用光谱信息完成基础类别识别，但对于具有明显季节变化特征的农业类别仍存在较大限制。

相比之下，引入多时相信息后，模型性能明显提升。其中：`dinov3l_m2f_pastis_temporal_v1`，采用多波段多时相 mean 聚合方式，mIoU 提升至：`50.81%`，相比单时相方案提升：`+11.14 mIoU`，说明时间序列信息能够提供额外的农业生长周期特征，有效增强类别区分能力。

进一步比较不同时间融合方式可以发现，stem 结构对于多时相输入具有重要影响。采用：`dinov3l_m2f_pastis_temporal_v3`，通过 stem 膨胀适配多时相输入后，模型达到：`54.97% mIoU`。相比仅使用 1×1 卷积映射的 temporal v1 提升：`+4.16\ mIoU`,说明更加充分的输入维度适配方式能够帮助 backbone 更有效地利用扩展后的时序光谱信息。

不同规模 backbone 的比较结果显示，ViT-S 与 ViT-L 在多时相任务中的性能差距较小。其中：`ViT-S temporal v2`=
`53.56% mIoU`; `ViT-S temporal zjf` = `54.26% \ mIoU`, 均接近 ViT-L mean 融合结果：`54.97% \ mIoU`,说明在引入有效时间建模后，较小规模 backbone 也能够获得具有竞争力的遥感表征能力。

最终，采用：`dinov3l_m2f_pastis_temporal_v4`的多波段、多时相 Attention 融合方案取得最佳性能：`58.31%` mIoU, 相比最佳 mean 聚合方案：`54.97%`, 提升：`+3.34\ mIoU`。该结果表明，相比简单时间平均策略，基于 Attention 的动态时序融合能够自适应学习不同时间节点对于类别识别的重要性，因此更适合处理农业遥感中具有周期变化特征的地物分类任务。

综合来看，PASTIS 实验验证了：

1. 多时相信息是提升农业遥感语义分割性能的关键因素；
2. 输入适配方式会显著影响基础模型对于多光谱时间序列的利用能力；
3. Attention 时序融合相比固定统计聚合方式能够进一步提升模型表达能力。

---

## 2.1.5 实验分析

PASTIS 实验主要验证基础视觉模型对于多维遥感信息的利用能力。实验设计表明，仅使用单幅影像时，模型主要依赖空间和光谱信息进行类别判断；引入时间维度后，模型能够进一步利用农业目标随季节变化产生的动态特征。相比简单时间统计方式，基于注意力机制的时间融合能够更加灵活地学习不同时间节点的重要程度，因此更适合处理具有明显季节变化的农业遥感任务。该实验说明：

* 光谱信息能够增强类别区分能力
* 多时相信息能够提供额外语义约束
* 时间建模是 Sentinel-2 遥感任务中提升基础模型性能的重要方向

---

# 2.2 SVDT 数据集实验

## 2.2.1 实验设置

SVDT 实验主要涉及两个工程目录：

```text id="9p5x2m"
/mnt/qh2-nas3/00-model/00-fb/mmseg_unet/
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_agri/
```

其中：
* `mmseg_unet`

用于构建传统 CNN 分割基线；
* `mmseg_dino_agri`

用于测试 DINOv3 在农业遥感任务中的迁移能力。该部分主要比较：

1. UNet
2. DINOv3 + Mask2Former

---

## 2.2.2 UNet 基线实验

UNet 实验采用传统编码器-解码器结构，用于建立农业遥感语义分割基线。其特点：

* 无预训练 backbone
* 全参数训练
* 完全依赖 SVDT 数据学习特征

该实验用于衡量基础视觉模型相比传统卷积网络的性能提升。

---

## 2.2.3 DINOv3 实验

DINOv3 实验采用：

* DINOv3 ViT backbone
* Mask2Former segmentation head

训练方式：

* 冻结 DINOv3 backbone
* 优化 segmentation decoder

该实验用于评价大规模自监督视觉模型在农业遥感任务中的迁移能力。

---

## 2.2.4 实验结果

| 方法                   | Backbone     | 任务头         | mIoU (%)  |
| -------------------- | ------------ | ----------- | --------- |
| UNet                 | CNN Encoder  | FCN Head    | 85.91     |
| DINOv3 + Mask2Former | DINOv3 ViT-L | Mask2Former | **88.30** |

---

## 2.2.5 实验分析

SVDT 实验结果表明，基于 DINOv3 的方法相比传统 UNet 获得更高的分割性能。其中：

* UNet：mIoU=85.91
* DINOv3 + Mask2Former：mIoU=88.30

提升约：+2.39 mIoU

该结果说明，即使不更新视觉 backbone，仅利用大规模自监督模型学习得到的视觉表征，也能够在农业遥感任务中获得优于传统 CNN 方法的性能。相比从头训练的卷积模型，DINOv3 能够提供更加稳定的空间语义表示，使模型更容易适应遥感场景中的复杂区域结构。

---

# 2.3 Potsdam 数据集实验

## 2.3.1 实验设置

Potsdam 实验主要涉及以下三个工程目录：

```text id="m3l6x9"
/mnt/qh2-nas3/00-model/00-fb/fusion_potsdam/
/mnt/qh2-nas3/00-model/00-fb/potsdam_lp/
/mnt/qh2-nas3/00-model/00-fb/potsdam_unet/
```

分别对应：

1. 基于 DINOv3 的语义分割与特征融合实验
2. 基于 OLMoEarth embedding 的 Linear Probe 实验
3. 基于 UNet 的传统卷积网络基线实验

Potsdam 实验主要研究不同类型视觉表征在高空间分辨率遥感语义分割任务中的迁移能力，包括：

* 大规模自然图像预训练模型
* 遥感领域自监督预训练模型
* 遥感基础模型 embedding
* 从头训练卷积模型

---

# 2.3.2 Potsdam UNet 基线实验

## 实验设置

Potsdam UNet 基线位于：

```text id="h7n4bc"
/mnt/qh2-nas3/00-model/00-fb/potsdam_unet/
```

对应配置：

```text id="v9n4wq"
configs/unet_potsdam.py
```

该实验作为 Potsdam 数据集上的传统卷积网络基准，用于比较预训练视觉 backbone 与从头训练 CNN 模型之间的性能差异。模型采用 mmseg 内置 UNet 结构，任务头采用 FCN。与 DINOv3 系列实验不同，UNet 不包含预训练视觉 backbone，因此：

* 所有参数均参与训练
* 不存在 backbone 冻结
* 模型完全依赖 Potsdam 数据进行特征学习

---

## 实验结果

| 配置                | Backbone                 | 任务头                 | Best mIoU (%) | @ iter            |
| ----------------- | ------------------------ | ------------------- | ------------- | ----------------- |
| `unet_potsdam.py` | UNet（5 stage，base_ch=64） | FCN + Auxiliary FCN | 84.85         | 38000（aAcc 91.22） |

该结果作为 Potsdam 数据集上的卷积网络基线，用于后续 DINOv3 与融合实验比较。

---

# 2.3.3 DINOv3 与自有遥感 SSL 权重实验

## 实验设置

DINOv3 实验主要基于：

```text id="v4b9pm"
/mnt/qh2-nas3/00-model/00-fb/fusion_potsdam/
```

实验中使用三类视觉权重：

---

## （1）DINOv3 官方权重

包括：

* ImageNet LVD-1689M 预训练权重
* SAT-493M 遥感预训练权重

用于比较：

* 通用视觉预训练
* 遥感领域预训练

对于高分辨率遥感任务迁移能力的影响。

---

## （2）自有遥感 SSL 权重

路径：

```text id="4xk8pm"
/mnt/qh2-nas3/00-model/00-limx/Dinov3/ckpt/
```

包含多个训练阶段 checkpoint，用于分析不同训练阶段视觉表征对于下游任务的影响。

---

## （3）OLMoEarth 权重

用于后续 Linear Probe 实验。包括：

* OLMoEarth v1
* OLMoEarth v1 + HyperNet
* OLMoEarth v1.2

---

## 模型结构与训练方式

DINOv3 Potsdam 实验采用：DINOv3 Backbone + Mask2Former Head，训练策略：

* 冻结 DINOv3 backbone
* 优化 Mask2Former decoder

虽然部分配置文件包含：

```text id="m7f1sq"
unfreeze_backbone
```

参数，但由于 adapter 内部仍保持 backbone 冻结，因此实际训练过程中仅更新任务头及适配模块。

---

# 2.3.4 DINOv3 与自有权重实验结果

| 实验配置                            | 权重类型   | Backbone     | 任务头                 | mIoU (%)  | mF1 (%)   |
| ------------------------------- | ------ | ------------ | ------------------- | --------- | --------- |
| DINOv3 LVD-1689M                | 官方预训练  | DINOv3 ViT-L | Mask2Former         | **89.00** | **94.07** |
| Zhejiang SSL 9999               | 自有 SSL | DINOv3 ViT-L | Mask2Former         | 88.49     | 93.77     |
| Zhejiang SSL 9999 (contrastive) | 自有 SSL | DINOv3 ViT-L | Mask2Former         | 88.43     | 93.73     |
| Gram SSL 4999                   | 自有 SSL | DINOv3 ViT-L | Mask2Former         | 88.39     | 93.75     |
| Zhejiang SSL 23999              | 自有 SSL | DINOv3 ViT-L | Mask2Former         | 88.12     | 93.57     |
| Zhejiang SSL 31999              | 自有 SSL | DINOv3 ViT-L | Mask2Former         | 87.98     | 93.47     |
| DINOv3 SAT-493M                 | 遥感预训练  | DINOv3 ViT-L | Mask2Former         | 86.85     | 92.85     |
| UNet baseline                   | 从头训练   | UNet         | FCN + Auxiliary FCN | 84.85     | -         |

---

# 2.3.5 OLMoEarth Linear Probe 实验

## 实验设置

OLMoEarth Linear Probe 实验基于：

```text id="8d0qz6"
/mnt/qh2-nas3/00-model/00-fb/potsdam_lp/
```

该实验固定 OLMoEarth embedding，仅训练线性分类头，用于评价基础模型特征本身的语义表达能力。

实验比较：

* OLMoEarth v1
* OLMoEarth v1 + HyperNet
* OLMoEarth v1.2

由于 Linear Probe 仅优化线性分类层，不涉及复杂特征学习过程，因此不同学习率主要影响收敛速度，而不会显著改变最终性能上限。

---

## 实验结果（50 epoch）

| 权重                      | Tile 设置                | Learning Rate | Best mIoU (%) |
| ----------------------- | ---------------------- | ------------- | ------------- |
| OLMoEarth v1            | Tile 4                 | 1e-2          | 48.15         |
| OLMoEarth v1            | Tile 256 (50% overlap) | 1e-1          | **69.02**     |
| OLMoEarth v1            | Tile 512               | 1e-1          | 60.37         |
| OLMoEarth v1 + HyperNet | Tile 4                 | 1e-1          | 49.52         |
| OLMoEarth v1 + HyperNet | Tile 256               | 1e-1          | 68.79         |
| OLMoEarth v1 + HyperNet | Tile 512               | 1e-1          | 60.26         |
| OLMoEarth v1.2          | Tile 4                 | 1e-2          | 47.46         |
| OLMoEarth v1.2          | Tile 256               | 1e-1          | 68.70         |
| OLMoEarth v1.2          | Tile 512               | 1e-1          | 64.98         |

---

# 2.3.6 OLMoEarth 长周期训练实验

为了验证 Linear Probe 是否充分收敛，进一步进行了 600 epoch 长周期训练。

实验结果：

| 权重             | Epoch | Best mIoU (%) |
| -------------- | ----- | ------------- |
| OLMoEarth v1 (Tile256)  | 600   | **69.49**     |
| OLMoEarth v1.2 (Tile256) | 600   | 68.69         |

---

# 2.3.7 Potsdam 实验分析

Potsdam 实验结果表明，预训练视觉模型能够显著提升高分辨率遥感语义分割性能。首先，从传统卷积模型来看：
* UNet baseline：mIoU=84.85
* 冻结 DINOv3 backbone：mIoU=89.00

提升：+4.15 mIoU，说明大规模预训练视觉模型能够提供更加有效的空间语义表示。

同时，自有遥感 SSL 权重能够达到接近官方 DINOv3 权重的性能，说明针对遥感数据开展自监督预训练能够获得具有竞争力的视觉表征。另一方面，SAT-493M 权重性能低于 LVD-1689M，说明预训练数据规模、训练策略以及任务匹配程度对于迁移性能具有重要影响。

OLMoEarth Linear Probe 实验显示，OLMoEarth v1 (Tile 256) 可实现最优结果，即空间位置编码插值带来的信息缺失要大于全图信息带来的提升，其中相对位置编码的影响要小于绝对位置编码。

---

# 第三部分 表征可视化与特征诊断方法

为了分析基础视觉模型迁移至遥感任务后的特征表达质量，本项目进一步开展了特征空间分析工作。

相关实验主要涉及：

```text id="v6z1hc"
/mnt/qh2-nas3/00-model/00-fb/visualise/
/mnt/qh2-nas3/00-model/00-fb/fusion_dino/
```

该部分不直接评价语义分割性能，而是从特征空间角度分析：

* 不同模型层级的语义表达变化
* 特征融合过程中的信息保持情况
* 模型输出是否出现 feature collapse（特征塌缩）

本项目采用四类表征分析方法：

1. PCA 降维分析
2. Cosine Similarity 分析
3. Global Feature Heatmap 分析
4. SVD Spectrum 分析

四种方法分别从：

* 低维空间分布
* 特征方向一致性
* 样本级语义区分能力
* 特征空间有效维度

四个角度评价模型表征质量。

---

# 3.1 PCA 降维分析

PCA 用于将高维视觉特征映射至低维空间，从而观察不同输入样本或不同网络层输出特征的空间分布。

该方法主要用于：

* 分析模型不同层级的特征变化
* 比较融合前后的特征分布
* 判断特征空间是否发生退化

当模型具有较好的表征能力时，不同输入区域或不同样本应在低维空间中保持一定分布差异。若模型发生特征塌缩，则不同输入产生的高维特征趋于一致，使 PCA 投影后的空间分布更加集中，表现为特征方差降低。因此，PCA 可以作为观察特征空间退化的直观分析工具。

---

# 3.2 Cosine Similarity 分析

Cosine Similarity 用于衡量不同视觉 token 或不同层级特征之间的方向一致性。

该方法主要用于：

* 分析不同网络层之间的语义演化
* 判断融合后的特征是否保留互补信息
* 检测局部 token 是否趋同

在正常情况下：

* 不同区域 token 应保持一定差异
* 不同层级特征应逐渐形成更加抽象的语义表示

如果大量 token 之间具有接近 1 的余弦相似度，则说明特征空间可能过度集中，存在局部表征塌缩风险。因此，Cosine Similarity 能够用于分析模型内部不同阶段的语义变化以及融合模块的信息有效性。

---

# 3.3 Global Feature Heatmap 分析

Global Feature Heatmap 用于分析不同输入样本经过模型编码后，其全局表示之间的关系。

该方法主要关注：

* 图像级语义表示差异
* 不同样本之间的特征分布
* 模型是否保持全局区分能力

通过计算不同图像 global feature 之间的相似关系，可以观察模型输出空间结构。当模型具有良好的全局表示能力时：

* 不同样本之间存在明显相似度差异
* heatmap 呈现具有结构性的分布

若发生 global feature collapse：

* 不同输入产生近似一致的表示
* 样本间相似度普遍升高
* heatmap 呈现均匀化趋势

因此，该方法主要用于评价模型是否仍保持图像级语义区分能力。

---

# 3.4 SVD Spectrum 分析

SVD Spectrum 从奇异值分布角度分析视觉特征空间结构。该方法通过观察特征矩阵奇异值变化，评价：

* 特征空间有效维度
* 信息分布情况
* 表征退化程度

当模型输出具有丰富表达能力时：

* 多个奇异值保持较高比例
* 特征空间具有较高有效秩

当模型发生特征塌缩时：

* 第一奇异值占据主要能量
* 后续奇异值快速下降
* 有效特征维度降低

相比 PCA 和相似度分析，SVD Spectrum 能够从整体特征矩阵角度提供更加定量化的表征评价。

---

# 3.5 表征诊断方法总结

| 方法                     | 分析对象          | 主要用途          |
| ---------------------- | ------------- | ------------- |
| PCA                    | 高维视觉特征        | 观察特征空间分布及退化趋势 |
| Cosine Similarity      | Token 或层级特征关系 | 分析语义一致性和融合效果  |
| Global Feature Heatmap | 图像级表示         | 判断全局特征是否塌缩    |
| SVD Spectrum           | 特征矩阵          | 分析有效维度和空间退化   |

上述四类方法共同构成基础模型表征分析体系，从局部 token、样本级表示以及整体特征空间三个层面评价模型输出质量。

---

# 第四部分 推理流程优化

除模型训练和性能分析外，本项目进一步针对基础模型推理流程进行了工程优化。主要涉及两个方向：

1. Mask2Former 纯 PyTorch 推理流程
2. OLMoEarth 推理流程简化

相关工程目录：

```text id="9c3h8p"
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_infer/
/mnt/ht2_nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1/
```
---

# 4.1 Mask2Former 纯 PyTorch 推理流程

## 4.1.1 工程概述

工程路径：

```text id="k8m0ps"
/mnt/qh2-nas3/00-model/00-fb/mmseg_dino_infer/
```

该工程主要实现 Mask2Former 在纯 PyTorch 环境下的独立推理流程。传统 MMSegmentation 推理流程依赖完整 OpenMMLab 生态，包括：
* mmcv
* mmengine
* mmdetection 相关组件

为了降低部署复杂度，本项目重新组织推理流程，使模型能够脱离训练环境完成：
* backbone 特征提取
* segmentation head 推理
* mask 生成
* 性能评价

---

## 4.1.2 推理流程特点

该推理框架主要具有以下特点：

### （1）降低框架依赖

将原本依赖完整训练框架的推理流程转化为：

> PyTorch + CUDA

独立运行模式。

---

### （2）支持基础模型迁移

统一支持：
* DINOv3 backbone
* Mask2Former segmentation head

使训练阶段得到的模型权重能够直接用于独立推理。

---

### （3）适用于遥感语义分割任务

该流程支持：

* PASTIS 多时相遥感任务
* 农业遥感分割任务
* 其他基于 DINOv3 的语义分割实验

---

# 4.2 OLMoEarth 推理流程简化

## 4.2.1 工程概述

工程路径：

```text id="w2d7mz"
/mnt/ht2_nas2/00-model/00-fb/olmo_test/olmoearth_inference_v2_1/
```

该工程主要用于简化 OLMoEarth 原始推理流程，使其适用于大规模遥感 embedding 提取任务。原始 OLMoEarth 流程包含较多训练阶段组件，而实际推理阶段主要需求为：

> 获取稳定、高质量的视觉表征。

因此，本项目针对推理阶段进行了结构简化。

---

# 4.2.2 推理流程优化

主要优化包括：

#### （1）保留视觉 Encoder 

推理版本主要保留：

* Vision Encoder；
* Feature extraction。

删除推理阶段不需要的训练模块。

#### （2）减少训练相关计算

针对 inference 场景：

* 删除 mask 相关处理；
* 删除训练阶段辅助模块；
* 删除对比学习相关计算。

使模型更加适用于 embedding 提取任务。

#### （3）统一输出形式

简化后的流程能够输出标准化视觉 embedding，用于：

* Linear Probe；
* 下游语义分割；
* 特征空间分析。

---

# 4.3 推理优化总结

本阶段完成了两个基础模型推理流程优化工作。其中：

`mmseg_dino_infer`

实现了 Mask2Former 从完整训练框架到独立 PyTorch 推理环境的迁移，降低了部署复杂度。

`olmoearth_inference_v2_1`

实现了 OLMoEarth 推理流程轻量化，仅保留视觉表征提取能力，提高 embedding 生成效率。上述优化工作为后续基础模型的大规模遥感应用和表征分析提供了更加稳定、简洁的工程基础。

---

