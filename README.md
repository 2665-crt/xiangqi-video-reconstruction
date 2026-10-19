# 象棋视频棋局复刻 · Xiangqi Video Reconstruction

从实拍象棋对局视频中定位棋盘、跟踪棋子移动，重建可回放的棋谱。支持人工校正、区域逐帧重检与虚拟棋盘录像导出。

本仓库发布当前正式版的源码快照。它是本地运行的 Windows Web 应用，不是微信小程序，也不是托管的在线识别服务；当前没有发布免安装 EXE。

## 功能

- 上传 MP4 视频，选择分析时间段，自动或手动校准棋盘四角。
- 视频对照与棋盘演绎回放，每步可定位原视频动作开始前一秒。
- 校正、插入、追加走棋；支持不计入走棋的审核记录，规则问题只提示。
- 选择两个步骤边界，对中间区间逐帧核对，自动补入已确认漏步、更新错误走法和动作时间。
- 保留遮挡、悔棋与无法确定的变化及截图证据，避免靠规则猜造棋谱。
- 导出虚拟棋盘 MP4 录像，支持每手 1 / 2 / 3 秒。
- 山水风格工作台，可折叠区域重检详情，适配减少动态效果设置。

## 运行环境

- Windows 10 / 11，Python 3.12，Node.js 22 或更高版本。
- Chrome：仅录像导出需要，渲染器使用 Playwright 的 `chrome` 通道。
- 固定使用本机 `127.0.0.1:5278`（前端）和 `127.0.0.1:8278`（后端）。端口被占用时启动脚本会停止，不会关闭其他程序。

## 安装与启动

```powershell
git clone https://github.com/2665-crt/xiangqi-video-reconstruction.git
cd xiangqi-video-reconstruction
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -Fusion
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-fusion.ps1
```

打开 <http://127.0.0.1:5278/>。上述入口启用本地融合识别，所需训练好的模型已随仓库提供，无需重新训练或填写云端 API Key。安装依赖需要联网；识别在本机执行。

停止本项目服务：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/stop.ps1
```

### 开启本地融合识别与深度重检

仓库包含本项目训练的 CNN、颜色模型，以及第三方 YOLO 检测和整盘布局模型。使用上面的 `setup.ps1 -Fusion` 和 `start-fusion.ps1` 即可加载，无需另外下载权重。各模型的来源、作者账号、版本、SHA256 和上游许可说明见 [模型配置与来源](docs/models.md) 与 [第三方署名](THIRD_PARTY.md)。部分无法确定的内容仍会保留人工审核。

### 可选的模型接口配置

需要模型接口时，将 `.env.example` 复制为 `.env`，自行填写接口地址、模型名和密钥，再使用 `scripts/start.ps1` 普通入口启动。普通入口不启用本地融合识别。模型接口可能将棋盘截图发送给你配置的服务商，并产生费用。密钥只保存在本地，默认不会提交。

## 基本操作

1. 上传本人有权使用的视频，选择起止时间，核对棋盘四角。
2. 确认起始棋局，再开始识别；也可直接手动设置并编辑走棋。
3. 点击步骤定位视频，按实际画面校正、插入或追加；删除走棋会保留必要审核信息。
4. 区域重检：点击“区域范围内重新检测”，直接选择两个步骤，核对前一步落定至后一步起动之间的区间。确认结果自动写回，未解释变化留在待核对列表。
5. 检查结果后导出录像。数据、原视频和证据保存在本机 `data/`，录像保存在项目的 `录屏存放/`。

## 技术实现与复用

本项目实现视频事件跟踪、走棋状态与时间整理、人工编辑、选区重检与写回、任务持久化、复盘交互及录像导出。技术栈为 React / TypeScript / Vite、FastAPI / Pydantic / SQLite、OpenCV / PyAV / ONNX Runtime。

识别还复用了第三方视觉模型及 Ultralytics 推理组件；不是所有模型都由本项目训练。来源、署名及上游许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 验证与限制

- 手动编辑、选区漏步补入、错序修正和重复走法保留具有自动化测试。
- 一段实际选区 501.133—746.835 秒的本地验收扫描了 4669 帧，确认 10 手，补入 7 手、更新 3 手；另有 10 处疑点保留。此结果不是任意视频的准确率保证。
- 重遮挡、摆棋、悔棋、光照及模糊仍可能需要人工核对。程序不会保证每次都完整自动还原。
- 本次公开快照不包含个人视频、任务数据库、训练图片集或真实录像回归素材。
- 训练好的模型、依赖安装流程与 Windows 脚本随源码提供；新电脑仍需具备上述运行环境。当前尚未提供免安装 EXE。

## 开发检查

```powershell
.venv/Scripts/python.exe -m pytest
cd frontend
npm test -- --run
npm run build
```

依赖私人录像的测试会明确跳过；缺少专用 Node 渲染回归夹具的测试也会跳过，其他 Python 导出测试继续执行。公开快照的测试范围见 [测试说明](docs/testing.md)。

## 仓库结构

```text
backend/app/       视频、识别、走棋、持久化和导出 API
backend/models/    训练好的本地模型及第三方模型来源清单
backend/tests/     单元和合成视频回归测试
frontend/src/      复盘工作台、人工编辑和录像渲染页面
scripts/           Windows 安装、启动、停止和录像渲染入口
docs/              模型与测试说明
```

## 许可证状态

本项目以免费作品展示及学习研究为发布目的，尚未指定整个项目的统一开源许可证。第三方组件和模型保留原作者署名与各自许可说明；本仓库的来源标注不构成额外授权，也不扩大上游允许的使用范围。详情见 [THIRD_PARTY.md](THIRD_PARTY.md)。
