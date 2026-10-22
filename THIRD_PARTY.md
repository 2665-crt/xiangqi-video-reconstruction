# 第三方组件与模型署名

本项目以免费作品展示、学习与研究为发布目的。感谢以下项目的原作者与维护者。第三方组件和权重遵循原有许可说明，本仓库的署名不扩大其授权范围。

| 组件或模型 | 原作者 / 维护者账号 | 来源及说明 |
| --- | --- | --- |
| 两份 YOLO 棋子检测权重 | [DuyLeTran](https://github.com/DuyLeTran) | [DeepXiangQi](https://github.com/DuyLeTran/DeepXiangQi)，来源版本 `a9265ae50d0ccaba1dae09ef83b2f4c4cc0dac19`；上游说明为教育研究用途，商业产品需先联系作者；未取得额外再分发授权。 |
| 整盘布局 ONNX 权重 | [yolo12138](https://huggingface.co/yolo12138) | [Chinese_Chess_Recognition](https://huggingface.co/spaces/yolo12138/Chinese_Chess_Recognition)，来源版本 `19a60be042982793322d36d51e8aecada98f0b76`；该版本 README 元数据标注 `license: mit`。 |
| YOLO 推理组件 | [Ultralytics](https://github.com/ultralytics) 及贡献者 | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)，[AGPL-3.0 许可](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)；另有企业授权选项。 |
| SVG 界面图标 | Lucide Contributors | [Lucide](https://lucide.dev/)，保留依赖包中的许可。 |

两份 YOLO 权重和整盘布局模型已随仓库提供，文件内容保持本地已验证版本。逐文件用途、SHA256、来源路径与上游说明见 [模型说明](docs/models.md)。模型并非全部由本项目原创或训练。

React、Vite、FastAPI、OpenCV、PyAV、ONNX Runtime、PyTorch 等运行依赖通过包管理器安装，各自版权和许可证保留在依赖包中。整个项目尚未指定统一开源许可证，不据此声称已满足所有第三方分发条件。

本项目训练生成的 CNN 与颜色参数随仓库提供；原始训练图片、个人视频、私人任务和真实视频回归素材不在发布范围内。山水背景为本项目创作流程生成的界面资源。
