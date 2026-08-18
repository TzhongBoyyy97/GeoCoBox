# 论文方法到代码的映射

## 1. 输入与监督

论文将 3D 图像记为 `X in R^(D x H x W)`，框标注记为六个角点坐标，并把框内填 1 得到 `B`。代码在 `geocobox/data.py` 中使用
`[z0, y0, x0, z1, y1, x1]` 半开坐标生成 `box_mask`。像素级 `mask` 不传入任何训练损失。

论文对每个已标注肿瘤裁剪 `96 x 96 x 96` patch，并明确不考虑漏诊肿瘤。`TumorPatchDataset` 相应地把每个框展开成一个样本，以框中心裁剪/补齐固定大小 patch。

## 2. 主干与 AD-CAM

| 论文 | 代码 |
|---|---|
| 3D U-Net backbone | `UNet3DBackbone` |
| `z = FC_soft(f(x))`，式 (2) | `soft_layer(features)` |
| `f^m = z elementwise-multiply f(x)`，式 (3) | `soft_logits * features`（广播到通道） |
| `z^m = FC_seed(f^m)`，式 (4) | `seed_layer(...)` |
| Soft Margin，式 (5) | `soft_margin_mask_loss` |
| Binary Dice，式 (7) | `binary_dice_loss` |
| `L_AD-CAM`，式 (6) | `adcam_loss` |

论文称两个层为 FC，但任务输出是体素图，图 3 也显示保留空间结构。本复现将其解释为共享于每个体素的 FC，即 `1x1x1 Conv3d`。论文式 (5) 写 `B in {0,1}`，而 SoftMargin 的带符号形式需要 `{-1,+1}`；代码显式执行 `2B-1`。Dice 使用 `1-Dice`，与论文最小化 `-Dice` 只差常数。

论文没有给出 U-Net 通道数、归一化层和激活。本复现使用 base channel 16、InstanceNorm 和 LeakyReLU；它们都在配置或模型构造参数中可改。

## 3. 对比头预训练

`ContrastiveHead` 对应论文的两层 point-wise convolution，输出逐体素 L2 归一化 embedding。

对每个框：

1. 用框中心作为几何中心；
2. 在中心欧氏距离 `<= tau`、位于框内且与中心 HU 差 `<= delta` 的体素中采正样本；
3. 在框外且与中心 HU 差 `> delta` 的体素中采负样本；
4. 使用 `local_supervised_contrastive_loss` 计算式 (8)。

论文式 (8) 的分母只写负样本集合 `N`，不同于标准 InfoNCE。默认 `denominator: negatives` 忠实保留该形式；可设为 `all` 使用正、负样本共同分母。

预训练优化 backbone 与 contrastive head，不优化 AD-CAM 两层。完成后冻结对比头参数；主训练时仍允许梯度穿过冻结的映射回到 backbone。

## 4. GCL 中心—边界精修

已明确部分：

- AD-CAM 阈值化粗掩码产生边界位置；
- 在边界距离 `tau` 内选体素；代码以迭代 3D 膨胀实现 Chebyshev 边界带；
- 框中心为 anchor；
- 计算中心 embedding 与边界 embedding 的 cosine similarity（式 9）；
- 前 40 轮只训练 AD-CAM，后 40 轮加入 `L_GCL`。

论文未明确部分：

- 式 (9) 的 `f` 是向量 embedding，但式 (10) 将 `f` 写成可做 `p(1-p)` 的标量概率；
- 式 (11) 只显示 `1(s_ce > 0)`，但图 3 同时包含正边界和负边界；
- 没有定义如何将边界相似度与粗预测合成最终预测。

本复现采用以下最小、可微且与图 3 一致的补全：

1. `cos(center, edge) / temperature` 作为“与中心同类”的 logit；
2. 粗掩码边界带内，粗前景是同类目标 1、粗背景是异类目标 0；
3. 用 binary cross entropy 训练中心—边界同类概率；
4. 仅在边界带内以该同类概率替换粗概率，其他体素保持粗预测；
5. 利用框的确定信息，把框外概率钳制为 0。

这段实现位于 `GeoCoBox.forward` 和 `geometric_coembedding_loss`。它不是论文未公开细节的事实声明，而是为得到确定、可复查代码所做的工程化解释。

## 5. 训练和评测

- PyTorch 2.1+；SGD；基础学习率 0.01。
- 对比头：50 epochs，batch size 2。
- 分割：80 epochs，batch size 32；epoch 1-40 为 AD-CAM，41-80 为 AD-CAM + GCL。
- 默认 `tau=8`、`delta=20 HU`，来自论文 LIDC-IDRI 消融最优值。
- 概率阈值 0.5；指标为 Dice 和 HD95。

论文只说明四个数据集按 4:1 划分，没有给出病例列表/随机种子；本复现工具默认种子 2026，并按病例划分。HD95 应使用重采样后的真实 spacing；当前 patch `.npy` 默认 spacing 为 `(1,1,1)`，调用 `hd95` 时可以显式传入。

## 6. 不能由论文单独恢复的内容

- 四个数据集的精确病例清单与 train/test IDs；
- LIDC-IDRI 的完整 LUNA16 纳入规则和结节合并规则；
- 各数据集重采样 spacing、CT window、数据增强；
- U-Net 通道数、归一化、学习率计划、权重衰减；
- 多框/多肿瘤在单个 patch 中的冲突处理；
- GCL 的式 (10)-(11) 实际张量定义与最终解码方式；
- 两张 GPU 上 batch size 32 的并行策略。

因此，报告实验结果时应把上述选择与随机种子一并记录，不能只凭论文 PDF 将数值差异归因于模型。
