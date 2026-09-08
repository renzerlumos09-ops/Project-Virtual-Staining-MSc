# MVP-2 设计说明：IHC 域自监督预训练编码器

## 目标（对应 RQ2）

验证假说：**领域预训练编码器能提供含染色语义的特征，从而激活 Proj 残差精修模块**（在 ImageNet VGG 特征下 Proj 三组损失恒零 = 阴性对照已存在）。

## 总体策略：同构替换，接口零改动

不重新设计架构。做一个与 MVP-1 中 FrozenVGG **完全同构**的 VGG16 features 编码器，
在 trainB（IHC，~52k 张）上自监督预训练，权重灌进 FrozenVGG 即完成替换。
三层特征 tap 不变：relu2_2（128ch@1/2）、relu3_3（256ch@1/4）、relu4_3（512ch@1/8）。

## Stage 1：MIM 预训练（本阶段交付 `mae_pretrain.py`）

- 方法：SimMIM 式掩码图像建模——随机遮住 50% 的 32×32 块（置中灰），
  编码器 + 轻量解码器重建被遮像素，L1 损失只算遮挡区域；
- 初始化：默认 ImageNet 初始化（收敛快），`--init scratch` 可纯从零训（对照用）；
- 数据：trainB 全部 52,303 张，归一化 [-1,1]；
- 资源预估：batch 8 + AMP，8GB 显存可跑；约 6500 iter/epoch，预计 10~20 min/epoch；
- 产出：`mae_encoder_latest.pth`（VGG16 features state_dict 格式）。

### Stage 1 验收标准

1. 损失曲线持续下降并趋稳；
2. 每 epoch 的重建对比图（vis/epochXXX.png）里，被遮区域的重建应呈现
   合理的 IHC 结构（核形态、腺体轮廓），而非涂抹色块——肉眼判断即可；
3. 训练 30 epoch 后若重建仍模糊，优先考虑：加 epoch、降 mask_ratio 到 0.4。

## Stage 2：替换 FrozenVGG 并重跑快速实验（预训练完成后进行）

1. 在 `mvp1_modules.py` 的 FrozenVGG 增加 `weights_path` 参数：
   若提供则加载 `mae_encoder_latest.pth` 覆盖 ImageNet 权重，随后照常冻结；
2. 重跑 MVP-1 同配置快速实验（3 epoch, 10000 张, batch 2），实验名 `mvp2_fastcut`；
3. 判定 RQ2：
   - **核心指标**：Proj_distill / Proj_pyramid / Proj_sparse 是否不再恒零；
   - 次要指标：G_NCE 收敛速度、fake_B 细节锐度（对比 mvp1_tuned 同 epoch 图）；
4. 结果无论正负都有价值：
   - Proj 激活 → 假说成立，进入 MVP-3；
   - Proj 仍恒零 → 说明问题不在特征语义而在损失设计，回到方案文档修订风险 2。

## Stage 3（暂缓，属完整方案）：跨染色特征回归

AF 侧下层特征回归预测 IHC 编码器特征（预测编码的跨域实例），
等 Stage 2 判定后再决定是否启用。AF 与 IHC 无配对，该损失只能作用在
分布/风格层面，设计时需重新审视——先不做。

## 风险记录

| 风险 | 应对 |
|---|---|
| 52k 张图对 MIM 预训练偏小 | ImageNet 初始化起步；观察重建质量；必要时降 mask_ratio |
| 8GB 显存 | batch 8 + AMP；爆显存先降 batch 到 4，不要降输入分辨率 |
| 重建任务太简单（同纹理拷贝） | 32px 块 + 50% 比例已是 SimMIM 验证过的配置；重建图肉眼验收 |
| 训完替换后 G_NCE 变差 | 说明新特征空间与 PatchNCE 温度不匹配，调 nce_temp (0.07→0.1/0.2) |
