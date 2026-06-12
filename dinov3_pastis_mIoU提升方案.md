# DINOv3 + M2F 在 PASTIS-R 上从 ~20 到 60 mIoU 的改进方案

> 配套文档：[dinov3+mmseg分割下游任务.md](./dinov3+mmseg分割下游任务.md)（现状基线）
> 撰写日期：2026-06-11

---

## 1. 现状诊断：为什么卡在 ~20

训练日志（`work_dirs/dinov3b_m2f_pastis_raster`）显示 val mIoU 在 iter 8k 达到峰值 **19.53**，之后在 17.7–19.5 之间震荡甚至缓慢下降，aAcc 稳定在 ~56%。**模型已经收敛，瓶颈不在训练时长，而在输入信息量和 backbone 适配度。**

### 1.1 根本问题：输入端丢掉了 ~97% 的判别信息

PASTIS 的 18 类作物在**单张 RGB 合成图上几乎不可分**。区分小麦/大麦/黑小麦、草地/苜蓿靠的是：

1. **物候曲线（时序）**：不同作物的播种、返青、抽穗、收割时间不同。U-TAE 之所以能到 63，本质是它看到了整条 43 时相的生长曲线。
2. **光谱（10 波段）**：NIR（B08）、红边（B05-B07）、SWIR（B11/B12）对植被状态的区分度远高于可见光 RGB。

当前流水线的信息瓶颈：

```
原始输入：43 时相 × 10 波段 = 430 个观测通道/像素
   ↓ 时相均值（丢物候曲线，且把云像素平均进去造成污染）
10 波段合成图
   ↓ 只取 RGB（丢 NIR/红边/SWIR）
3 通道单帧  ← 仅剩 ~0.7% 的原始观测，且 RGB 是判别力最弱的 3 个波段
```

文献参照：U-TAE 论文的**单时相**消融中，最优单日期模型 mIoU 也只有 ~30 出头。当前 19.5 = 单帧 RGB 上限（~30）再打折扣（frozen backbone + 域差距），完全符合预期。**不解决时序和波段问题，任何 backbone/头的改动都到不了 60。**

### 1.2 次要问题清单

| # | 问题 | 影响估计 |
|---|---|---|
| A | **时相均值** 丢物候 + 云污染 | 最大，-20~25 mIoU |
| B | **只用 RGB 3 波段**，丢 NIR/红边/SWIR | -5~8 |
| C | **backbone 冻结** 且为 web 图像预训练（LVD-1689M），Sentinel-2 反射率域差距大 | -4~6 |
| D | 未用 **SAT-493M 卫星预训练权重**（本地已有 `dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`） | -3~6 |
| E | **评估协议与 U-TAE 不对齐**：U-TAE 的 63.1 是 19 类（背景 0 参与 mIoU，仅 ignore void 19）；当前把背景也 ignore，只算 18 个难类，数值天然偏低 | 对比口径差 ~3-5 |
| F | ViT-S 容量小 | -2~4 |
| G | 增强只有 RandomFlip；40k iter ≈ 110 epoch，头部过拟合（日志中 mIoU 后期下降） | -1~2 |

---

## 2. 改进路线图（按优先级）

```
P0-1  多时相输入：月度中值合成 T=12，逐帧过 backbone，特征级时序聚合   ★核心
P0-2  全 10 波段输入（先 1×1 投影，后 patch-embed 通道扩展）
P1-1  换 dinov3_vitl16_sat493m 卫星预训练权重（本地已有）
P1-2  解冻 backbone 全参微调（backbone lr ×0.1）
P2-1  评估协议对齐 U-TAE：背景作为第 0 类参与训练和 mIoU
P2-2  时序聚合从 mean 升级为轻量时序注意力（LTAE-lite）
P2-3  增强（rot90）、缩短 schedule、加 warmup
```

### 阶段目标

| 阶段 | 配置 | 预期 mIoU |
|---|---|---|
| 基线（当前） | ViT-S frozen + RGB 时相均值 | 19.5 ✅实测 |
| **Phase 1** | + 10 波段 + 解冻微调 + 协议对齐 + 增强 | **32–38** |
| **Phase 2** | + 月度中值 T=12 + 特征 mean 池化 | **45–52** |
| **Phase 3** | + sat493m ViT-L + 时序注意力池化 | **55–63** |

每阶段只改一组变量，能定位每项改动的真实贡献。

---

## 3. Phase 1：单帧最强基线（约 1–2 天）

目的：把"单帧合成图"路线推到上限，为时序版提供干净的对照组。

### 3.1 全 10 波段输入（1×1 投影方案）

最小改动：在 backbone 前加一个可学习的 `1×1 Conv(10→3)`，预训练 stem 不动。

**`LoadPASTISRaster` 改动**：去掉"取 RGB"步骤，输出 `(H, W, 10)`；归一化直接在 loader 内完成（`NORM_S2_patch.json` 的 10 波段 mean/std），`data_preprocessor` 改为 `mean=[0]*10, std=[1]*10`（或干脆不设 size 归一化只做 pad）。

**wrapper 改动**（`dinov3_backbone.py`）：

```python
class DINOv3BackboneMmseg(BaseModule):
    def __init__(self, ..., in_bands=3):
        ...
        if in_bands != 3:
            self.input_proj = nn.Conv2d(in_bands, 3, kernel_size=1)
            # 初始化：B04/B03/B02 → R/G/B 直通，其余波段权重置 0，训练中自行学习
            nn.init.zeros_(self.input_proj.weight)
            for rgb_idx, band_idx in enumerate([2, 1, 0]):   # R←B04, G←B03, B←B02
                self.input_proj.weight.data[rgb_idx, band_idx, 0, 0] = 1.0
            nn.init.zeros_(self.input_proj.bias)
        else:
            self.input_proj = nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)
        out = self.adapter(x)
        return (out["1"], out["2"], out["3"], out["4"])
```

> 这个初始化保证第 0 步与 RGB 基线完全等价，之后梯度自动把 NIR/红边信息混进来，训练稳定。
> 进阶方案（Phase 3 可选）：把 ViT `patch_embed.proj` 和 Adapter SpatialPriorModule 的首层 conv 从 3 通道扩展到 10 通道（RGB 槽位拷贝原权重、其余通道用均值×缩放初始化），表达力更强，但要同时改两处 stem，建议等微调流程跑通后再做。

### 3.2 解冻 backbone

config 中 `freeze_backbone=False`（`backbone lr_mult=0.1` 已配好）。配合：

```python
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1, decay_mult=1.0),
        'input_proj': dict(lr_mult=1.0),
        ...}))
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),  # warmup 必须加
    dict(type='PolyLR', eta_min=0, power=0.9, begin=1500, end=20000, by_epoch=False)]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=20000, val_interval=1000)
```

> 40k → 20k：1455 张图 bs=4 下 20k iter ≈ 55 epoch 已足够；当前日志显示 8k 之后纯属过拟合。

### 3.3 评估协议对齐 U-TAE（重要，影响所有对比结论）

U-TAE 官方代码用 20 类混淆矩阵、`ignore_index=19`（void），**背景 0 参与 mIoU**，即对 19 类取平均。背景占像素大头且容易分，纳入后 mIoU 数值上浮明显。当前"背景也 ignore、只算 18 个难类"的口径比 U-TAE 严格，直接对比会系统性吃亏。

改法（`LoadPASTISRaster` 标签映射）：

```python
# 旧：0→255, 1-18→0-17, ≥19→255   （18 类）
# 新：0-18→0-18（背景=第 0 类）, 19→255  （19 类）
```

config：`num_classes=19`，`class_weight=[1.0]*19 + [0.1]`。

> 两种口径各有意义：对外报告/和 U-TAE 比用 19 类口径；分析作物辨识能力时可同时输出 18 类作物的 per-class IoU（IoUMetric 本来就逐类打印，手动剔除背景行重新平均即可，无需重训）。

### 3.4 增强

遥感图无方向先验，加 90° 旋转：

```python
train_pipeline = [
    dict(type='LoadPASTISRaster', img_size=img_size, bands='all'),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomRotate90', prob=0.75),   # 若 mmseg 版本无此 transform，用 RandomRotate degree=(90,90) 组合或自写 np.rot90
    dict(type='PackSegInputs'),
]
```

不要加光度类增强（ColorJitter/PhotoMetricDistortion）——反射率的物理量纲就是判别信息。

**Phase 1 验收：val mIoU ≥ 32（19 类口径）。** 达不到 30 先排查 10 波段归一化和 input_proj 初始化。

---

## 4. Phase 2：多时相输入 + 特征级时序聚合（核心，约 3–5 天）

### 4.1 时序表示：月度中值合成（推荐起步方案）

把 ~43 个不规则时相整理成 **12 个月度中值帧**，每帧 `(10, 128, 128)`：

- **天然去云**：中值对云污染帧鲁棒，不需要显式云检测；
- **定长 + 固定语义**：第 k 帧恒等于第 k 月，时序位置编码可以直接用可学习的 per-month embedding，不需要把采集日期穿透 mmseg 流水线（这是工程上最大的简化）;
- 物候曲线的月度采样对作物分类已足够（作物物候以旬/月为尺度变化）。

**`LoadPASTISRasterTemporal`**（新增，`pastis.py`）：

```python
@TRANSFORMS.register_module()
class LoadPASTISRasterTemporal(BaseTransform):
    """输出 (H, W, 12*10) 的月度中值合成堆叠图。"""

    def transform(self, results):
        s2 = np.load(results['s2_path']).astype(np.float32)      # (T, 10, 128, 128)
        dates = results['dates_s2']                              # 来自 metadata.geojson 的 dates-S2
        months = np.array([int(str(d)[4:6]) for d in dates])     # YYYYMMDD → 月
        frames = []
        for m in range(1, 13):
            sel = s2[months == m]
            if len(sel) == 0:                # 个别月份缺数据：用相邻月份兜底
                sel = s2[np.abs(months - m).argsort()[:3]]
            frames.append(np.median(sel, axis=0))                # (10, H, W)
        img = np.stack(frames)                                   # (12, 10, H, W)
        img = (img - NORM_MEAN[None, :, None, None]) / NORM_STD[None, :, None, None]
        img = img.reshape(12 * 10, *img.shape[2:]).transpose(1, 2, 0)  # (H, W, 120)
        # resize 到 img_size（双线性），标签 NEAREST，与现有逻辑一致
        ...
```

`PASTISRasterDataset` 需要把 `feature.properties["dates-S2"]` 一并放进每条样本的 `results`。

`data_preprocessor`：`mean=[0]*120, std=[1]*120`（归一化已在 loader 完成），`size=(img_size, img_size)`。

> 月度中值可以离线预计算缓存成 `S2M_{ID}.npy`（12×10×128×128 float16 ≈ 3.9MB/瓦片，全集 ~9.5GB），训练时 IO 和 CPU 都省一个量级，强烈建议先跑一次缓存脚本。

### 4.2 backbone：逐帧编码 + 时序池化

```python
@MODELS.register_module()
class DINOv3TemporalBackbone(DINOv3BackboneMmseg):
    """输入 (B, T*C, H, W)，逐帧过 ViT+Adapter，4 个尺度上分别做时序聚合。"""

    def __init__(self, *args, n_frames=12, in_bands=10, temporal_agg='mean', **kw):
        super().__init__(*args, in_bands=in_bands, **kw)
        self.n_frames, self.in_bands = n_frames, in_bands
        if temporal_agg == 'attn':
            self.agg = nn.ModuleList([TemporalAttnPool(self.embed_dim, n_frames)
                                      for _ in range(4)])
        else:
            self.agg = None    # mean

    def forward(self, x):
        B, TC, H, W = x.shape
        T = self.n_frames
        x = x.view(B * T, self.in_bands, H, W)
        x = self.input_proj(x)
        out = self.adapter(x)                      # 每尺度 (B*T, D, h, w)
        feats = []
        for i, key in enumerate(["1", "2", "3", "4"]):
            f = out[key]
            f = f.view(B, T, *f.shape[1:])         # (B, T, D, h, w)
            f = self.agg[i](f) if self.agg else f.mean(dim=1)
            feats.append(f)
        return tuple(feats)
```

Phase 2 先用 `temporal_agg='mean'`：实现 10 分钟，效果已能体现时序合成（中值月帧本身就保留了物候，mean 池化保留"各月特征的平均响应"）。

### 4.3 显存与速度预算（RTX 5090 Laptop, 24GB）

backbone 前向次数 ×12 是主要开销，对策按顺序使用：

| 措施 | 说明 |
|---|---|
| `img_size=256`（而非 512） | 128 原生 → 256 已是 2× 上采样；token 数 16×16/帧，12 帧 ≈ 3072 token，相当于单帧 512 的 3 倍。PASTIS 地块大，stride-4 的 M2F mask 分辨率 64×64 足够 |
| AMP | `optim_wrapper=dict(type='AmpOptimWrapper', loss_scale='dynamic', ...)` |
| 梯度检查点 | ViT 上开 grad checkpointing（dinov3 vision_transformer 支持逐 block ckpt；不行就手动 `torch.utils.checkpoint` 包 blocks） |
| bs=2 + `accumulative_counts=2` | 等效 bs4 |
| 训练时随机抽 T'=8 个月、推理用全 12 个月 | 进一步降一档显存，且相当于时序 dropout 增强 |

参考量：ViT-S、256 输入、T=12、bs2 + AMP 约 8–10GB，完全可行；ViT-L 同配置约 18–22GB，需要 grad-ckpt + bs1×4。

> 注意：附录 A 的纯 PyTorch MSDA fallback（`grid_sample`）在 T=12 下 Adapter 内的 MSDA 调用也 ×12，速度损失会被放大。建议此时重新验证 mmcv 的 `MultiScaleDeformableAttention` CUDA 实现（wrapper 已有 `_MmcvMSDeformAttn` 替换路径）在 sm_120 上是否可用——mmcv 编译时若带了 12.0 arch 就能省回一大块时间。

**Phase 2 验收：val mIoU ≥ 45。** 此时和 Phase 1 的差值就是"时序信息"的定量贡献，值得单独记录。

---

## 5. Phase 3：sat493m ViT-L + 时序注意力（冲 60）

### 5.1 换卫星预训练 ViT-L（本地已有，零下载成本）

```python
DINO_CKPT = '/home/zifei/.cache/modelscope/hub/models/facebook/dinov3pth/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
backbone=dict(arch='vit_large', checkpoint=DINO_CKPT,
              interaction_indexes=[5, 11, 17, 23], freeze_backbone=False)
decode_head=dict(in_channels=[1024, 1024, 1024, 1024], ...)
```

SAT-493M 是 DINOv3 在 4.93 亿张卫星影像上的预训练版本，对 Sentinel-2 纹理/光谱分布的先验远好于 web 图像版，是本方案里**性价比最高的一次换权重**。先在 Phase 1 的单帧配置上快速 AB 一次（frozen 线性探针 8k iter 即可对比出差距），确认增益后再上时序版。

> 注意核对 sat493m 权重的 `n_storage_tokens` 等结构超参与 wrapper 里 OmegaConf 配置是否一致（加载时 strict 报 missing/unexpected keys 就是这里），ViT-L 的 `layerscale`、`untie_*` 配置也要和官方发布配置对齐。

### 5.2 时序注意力池化（LTAE-lite）

mean 池化对"关键月份"不敏感（比如冬小麦 4–5 月的抽穗期权重应远高于 1 月）。换成 per-pixel 时序注意力：

```python
class TemporalAttnPool(nn.Module):
    """对 (B, T, D, h, w) 在 T 维做注意力加权，输出 (B, D, h, w)。"""

    def __init__(self, dim, n_frames, n_heads=4):
        super().__init__()
        self.month_embed = nn.Parameter(torch.zeros(n_frames, dim))  # 月份位置编码
        self.query = nn.Parameter(torch.zeros(1, 1, dim))            # 可学习聚合 query
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.month_embed, std=0.02)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, x):
        B, T, D, h, w = x.shape
        x = x.permute(0, 3, 4, 1, 2).reshape(B * h * w, T, D)
        x = self.norm(x + self.month_embed)
        q = self.query.expand(B * h * w, -1, -1)
        out, _ = self.attn(q, x, x)                  # (B*h*w, 1, D)
        return out.reshape(B, h, w, D).permute(0, 3, 1, 2)
```

参数量极小（每尺度一个 MHA），先只在 stride-8/16/32 三个尺度用、stride-4 尺度保持 mean（h×w=64×64 时 B·h·w 较大，attention 显存敏感），观察增益后再决定是否全开。

### 5.3 最终训练配置要点

- ViT-L + T=12 + 256 输入 + AMP + grad-ckpt，bs1 × accum 4；
- lr：head 1e-4 / backbone 1e-5（lr_mult=0.1），warmup 1500 iter，Poly 到 30k；
- `loss_mask` 保持 M2F 默认的 CE+Dice 组合（M2F 头本身带 DiceLoss，无需额外加）；
- 最终对外数字：**Fold 5 test set**（不是 val 的 Fold 4），与 U-TAE 论文同折。

---

## 6. 实验记录表（建议照此推进）

| Exp | 改动（相对上一行） | backbone | 输入 | mIoU(19类) | 备注 |
|---|---|---|---|---|---|
| E0 | 基线 | ViT-S frozen | RGB 均值合成 512 | 19.5（18类口径） | 已完成 |
| E1 | 协议对齐 + 10 波段 + 解冻 + 增强 + 20k | ViT-S ft | 10b 均值合成 512 | 预期 32–38 | Phase 1 |
| E2 | E1 权重口径下快速 AB：sat493m vs lvd | ViT-L frozen | 同 E1 | 看 Δ | 8k iter 即可 |
| E3 | 月度中值 T=12 + mean 池化 | ViT-S ft | 10b×12帧 256 | 预期 45–52 | Phase 2 |
| E4 | 换 sat493m ViT-L | ViT-L ft | 同 E3 | 预期 52–58 | Phase 3 |
| E5 | + 时序注意力池化 | ViT-L ft | 同 E3 | 预期 55–63 | Phase 3 |
| E6 | （可选）patch-embed 10 通道扩展 / 加 S1 SAR | ViT-L ft | +S1A/S1D | +1~3 | 冲刺项 |

到 E5 若仍差 2–3 个点，可继续上的冲刺项：
- **S1 SAR 融合**（PASTIS-R 的本意）：S1A/S1D 月度中值各 3 通道，`input_proj` 改 16→3 或独立分支后特征相加；U-TAE 论文中 S1+S2 融合 +1~2 点；
- 多尺度测试（TTA：flip + 多 scale）通常 +0.5~1；
- 5 折交叉验证取平均（论文口径）。

---

## 7. 风险与排查清单

| 风险 | 信号 | 对策 |
|---|---|---|
| 10 波段归一化错误 | E1 反而低于 E0 | 检查 NORM_S2_patch.json 波段顺序与 loader 通道顺序一致（S2 原始顺序 B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12） |
| sat493m 权重加载不匹配 | console 大量 missing keys | 对照官方 vitl16 sat 配置改 OmegaConf（storage tokens / layerscale） |
| 解冻后训崩 | loss 发散或 mIoU 归零 | warmup 拉长到 3000；backbone lr_mult 降到 0.05；确认 clip_grad 生效 |
| T=12 训练太慢 | 单 iter > 5s | 预缓存月度合成 npy；验证 mmcv MSDA CUDA 核；训练时随机抽 8 帧 |
| 月度合成某些月全缺 | loader 报错或全零帧 | 兜底逻辑（取时间最近 3 帧中值）+ 离线统计缺月分布 |
| M2F 对 19 类背景主导 | 背景 IoU 高但作物类塌 | class_weight 背景项降到 0.5；或保持 18 类口径训练、仅评估时换口径对比 |

---

## 8. 实测结果（Phase 1+2 合并实验，2026-06-11）

把第 1 项（10 波段 stem inflation）、第 2 项（U-TAE 19 类协议）、第 3 项（rot90）、第 4 项（月度中值 T=12 + 逐帧编码 + mean 池化）一次性全部改好，做了一次 ViT-S 实验：

**最终：Fold-5 test mIoU = 53.64（19 类协议） / 52.56（18 类作物协议，背景排除）**，对比原基线 19.5 → **提升 +34 个点，达到方案预期区间的上沿，已超过 OlmoEarth 线性探针（50.6）和接近 U-Net 3D（58）。**

验证集 mIoU 收敛轨迹（每 1000 iter）：

```
iter   1k    2k    3k    4k    5k    6k    7k    8k    9k   10k
mIoU  12.6  30.6  33.9  36.9  40.6  41.2  47.4  47.8  48.2  50.2
iter  11k   12k   13k   14k   15k   16k   17k   18k   19k   20k
mIoU  50.8  51.1  50.3  52.2  52.6  52.4  53.0  53.6  53.4  53.9
```

Fold-5 test 逐类 IoU（识别难度一目了然）：

| 易分（>70） | IoU | 中等（40–70） | IoU | 难分（<40） | IoU |
|---|---|---|---|---|---|
| sorghum | 84.6 | meadow | 62.2 | triticale | 39.9 |
| corn | 82.2 | grain_maze | 54.7 | beans | 36.8 |
| winter_peas | 81.2 | beet | 51.6 | maize | 32.8 |
| soft_wheat | 80.4 | dead_plant | 44.3 | winter_spelt | 30.9 |
| background | 73.0 | barley | 40.8 | peas | 27.9 |
| sunflower | 71.9 | | | potato | 27.5 |
| hard_wheat | 70.3 | | | rapeseed | 26.3 |

> 难分类普遍是样本稀少的小类（potato/rapeseed/peas）。下一步空间：(1) Phase 3 换 sat493m ViT-L（预期到 ~58）；(2) 对小类加 class-balanced 采样或调高 loss 权重；(3) 时序 mean → 时序注意力池化。

### 关键工程问题与修复（复现必读）

| 问题 | 现象 | 修复 |
|---|---|---|
| **fp16 下 M2F mask BCE 溢出** | `loss_mask=nan`（loss_cls/dice 正常） | AMP 用 `dtype='bfloat16'` 而非默认 fp16 |
| **mmcv MSDA 无 bf16 kernel** | `ms_deform_attn_forward_cuda not implemented for BFloat16` | patch `MultiScaleDeformableAttnFunction.apply`，bf16 输入时转 fp32 跑再转回（见 `jzf/dinov3_temporal_backbone.py` 顶部） |
| **DataLoader worker 被 OOM kill** | iter 500 后 `worker killed by signal: Killed`，主机 RAM 涨到 13GB/worker | dataset `__getitem__` 用 `dict(self.data_list[idx])` 浅拷贝，避免 31.5MB/样本的 pipeline 输出被写回 data_list 并被 worker 永久缓存 |

### 复现文件（全部在 `segmentation/mmsegmentation/jzf/`）

| 文件 | 作用 |
|---|---|
| `pastis_temporal.py` | `LoadPASTISRasterTemporal`（月度中值 12 帧 ×10 波段 + 19 类协议）、`PASTISRandomRotate90`、`PASTISRasterTemporalDataset` |
| `dinov3_temporal_backbone.py` | `DINOv3TemporalBackbone`（stem 3→10 通道 inflation + 逐帧编码 + 时序 mean 池化）、bf16 MSDA patch |
| `cache_monthly_median.py` | 离线预计算 `DATA_S2_M12/S2M_*.npy`（2433 瓦片，已跑完） |
| `configs/dinov3s_m2f_pastis_temporal.py` | 完整 mmseg config（bf16 AMP、warmup+poly、20k iter） |
| `train.sh` | 启动脚本（设 `PYTHONPATH` 让 `custom_imports` 找到 `jzf.*`） |
| `smoke_test.py` | 冒烟测试（数据形状 + 1 train/val step + 显存） |
| `work_dirs/dinov3s_m2f_pastis_temporal/` | 训练日志、checkpoint、Fold-5 test 结果 |

**启动命令**：`cd segmentation/mmsegmentation && bash jzf/train.sh`
**资源占用**：ViT-S / 256 输入 / T=12 / bs4 / bf16，显存 5.9GB，单 iter 0.37s，20k iter 约 2.5h（RTX 5090 Laptop）。

---

## 9. 一句话总结

**20 → 60 的钥匙不在头和训练技巧，而在输入端：把"1 帧 RGB"还原成"12 帧 × 10 波段"的物候立方体（贡献 ~25 点），再用卫星预训练 ViT-L 微调（~10 点）、对齐 U-TAE 评估口径（~4 点）。** Mask2Former 头保持不动即可。
