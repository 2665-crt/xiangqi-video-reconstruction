# 模型配置与来源

公开仓库没有打包以下第三方模型权重，也不会自动下载它们：

| 文件 | 来源与用途 |
| --- | --- |
| `backend/models/fusion/detect-ultra.pt` | DeepXiangQi 主检测器 |
| `backend/models/fusion/detect-aux.pt` | DeepXiangQi 辅助检测器 |
| `backend/models/board-layout-v1.onnx` | 可选的整盘布局交叉核对模型 |

融合模型来源：<https://github.com/DuyLeTran/DeepXiangQi>。该项目说明需自行获取权重。上游版本记录为 `a9265ae50d0ccaba1dae09ef83b2f4c4cc0dac19`；这里仅记录来源，不代表已经取得第三方再分发授权。请向权利人获取允许使用的权重，按文件名放置，并与 `backend/models/fusion/weights.json` 的 SHA256 对照。任意同名模型不一定兼容。

布局模型来源和哈希记录在 `backend/models/board-layout-v1.json`，上游为 <https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition>。该模型缺失时不会加载此辅助核对器。

本仓库包含本项目训练生成的棋种分类模型及颜色参数（`piece-kind-cnn-v1.*`、`board-color-v1.*`），不包含训练视频或图片集。完整融合自动识别仍需要上面两个检测权重。

Ultralytics 的授权说明：<https://www.ultralytics.com/license>。使用、修改、提供服务或再分发时请分别核对推理组件与模型权重的条款；不要仅因可下载就认为可以任意分发。
