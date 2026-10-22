# 训练好的模型与来源

本仓库包含运行当前本地融合识别所需的模型文件，无需用户重新训练。安装融合依赖并启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Fusion
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-fusion.ps1
```

## 本项目训练的模型

- `backend/models/piece-kind-cnn-v1.onnx`：棋种分类 CNN；训练信息及哈希见同名 JSON。
- `backend/models/board-color-v1.npz`、`board-color-v1.xml`：棋盘与棋子颜色参数。

这些文件是训练后的推理参数，不包含原始训练视频或图片集。

## DeepXiangQi 检测模型

来源项目：[DuyLeTran/DeepXiangQi](https://github.com/DuyLeTran/DeepXiangQi)。原作者 / 维护者账号：[DuyLeTran](https://github.com/DuyLeTran)。感谢原作者提供象棋检测模型及研究工作。

来源版本：`a9265ae50d0ccaba1dae09ef83b2f4c4cc0dac19`。

| 本仓库文件 | 用途 | SHA256 |
| --- | --- | --- |
| `backend/models/fusion/detect-ultra.pt` | 主棋子检测器 | `798795334ee7538eb6a0ec37d0cdca15fff4b230a5fdf6b9b402d7f3c154fb30` |
| `backend/models/fusion/detect-aux.pt` | 辅助棋子检测器 | `a5db1b87fb12fa99ec9eac4fe751593eb9a08aa508a61d64463128ba72063886` |

这两份权重保持本地已验证版本的文件内容，未由本项目重新训练。程序按 `backend/models/fusion/weights.json` 校验 SHA256，任意同名模型不一定兼容。

截至 2026-10-19，上游 [License 说明](https://github.com/DuyLeTran/DeepXiangQi#license)原文为：

> This project is for educational and research purposes. Please contact the author before using it in commercial products.

本项目以免费作品展示与学习研究为发布目的，未取得额外的模型再分发授权。这里保留来源和上游说明，不声称署名可以替代许可或扩大上游授权。

## 整盘布局辅助模型

- 来源项目：[Chinese_Chess_Recognition](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition)。原作者 / 维护者账号：[yolo12138](https://huggingface.co/yolo12138)。感谢原作者提供布局识别模型。
- 来源版本：`19a60be042982793322d36d51e8aecada98f0b76`。
- 上游文件：`onnx/layout_recognition/nano_v3-0319.onnx`。
- 本仓库文件：`backend/models/board-layout-v1.onnx`。
- SHA256：`da66ba9809f15127f8ae729b1755e42ee61c100c4f9979ce0ef13602ac471298`。
- 上游该版本的 [README 元数据](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition/blob/19a60be042982793322d36d51e8aecada98f0b76/README.md)标注 `license: mit`；仓库树中未见单独的 LICENSE 文件。

文件内容未修改，仅以本项目文件名保存；来源、哈希和输入格式记录在同名 JSON 中。此模型用于辅助核对，不独立添加棋子或静默覆盖棋种。

## 推理组件

融合模式使用 [Ultralytics](https://github.com/ultralytics/ultralytics)，其 [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 为 AGPL-3.0；另有企业授权选项。该组件的许可与上述第三方模型的来源说明分别保留，来源署名不代表已经取得企业许可。
