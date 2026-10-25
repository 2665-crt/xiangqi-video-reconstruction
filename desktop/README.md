# Windows 便携桌面版

应用名：**XiangQi Vedio Reconstruction**。

推荐下载 `XiangQi-Vedio-Reconstruction-0.3.1-win-x64-portable.exe`，单个文件即可放到可写目录或桌面双击运行。启动时显示准备画面并自动展开内置运行环境，无需手动解压。无需安装 Python、Node.js 或浏览器，无需配置云端 API；使用内置本地模型和 CPU 推理。

旧版 ZIP 或源码构建生成的完整程序目录仍须保留全部文件，桌面应创建指向内部 EXE 的快捷方式。内部 EXE 不能脱离 `icudtl.dat`、`resources` 等同级依赖单独使用。

首次启动会加载模型。上传 MP4 后，按现有界面完成棋盘定位、初始局面核对、分析和人工校正。模型可能留下待核对区间，不能保证每段视频完全自动还原。任务保存在应用目录的 `data`，导出视频保存在 `录屏存放`。移动软件时请移动整个文件夹，保留这两个目录即可保留个人数据。关闭应用会停止自己的后台服务。

单文件模式的“应用目录”指下载 EXE 所在目录，不是临时解压目录。每次启动分别使用独立临时运行环境；关闭后临时文件可清理，任务和录屏继续保留在 EXE 相邻目录。显式配置 `XIANGQI_DATA_ROOT`、`XIANGQI_RECORDING_ROOT` 时优先使用配置路径。

此构建未作代码签名。在 Windows x64 本机完成验收，未在其他实体电脑上验证。第三方组件、权重来源和许可见 `licenses`；对应应用源码随 `licenses/application-source.zip` 提供。

## 多窗口与 CPU 状态

重复双击 EXE 可以打开独立窗口，标题显示窗口编号。每个窗口拥有自己的后台、任务、浏览器配置和日志，停止分析或关闭一个窗口不会关闭其他窗口。当前先验证双窗口；更高并发需要逐步验证，实际适合的并发数取决于 CPU、内存和散热，不保证多开按比例提速。

起始局面有占位或棋种疑点时，该窗口立即进入待核对状态，显示候选棋局并允许编辑，不再先扫描完整个视频。核对并保存初始棋局后，点击“自动解析并复刻所选区间”继续；已有扫描结果保留。其他窗口可继续独立解析，候选棋局不会自动当成已确认局面。

窗口 1 继续使用原来的 `data` 和 `录屏存放`，不迁移历史任务。窗口 2 及之后的数据位于 `data/instances/<编号>`，录像位于 `录屏存放/instances/<编号>`。再次启动会使用最小的空闲编号，因此可以继续访问该编号之前保存的任务。若要移动或备份软件，仍应保留完整的这两个目录。

右上角 CPU 状态每秒采样：总量以整机 100% 计，显示逻辑处理器数、整机已用、本窗口占用和当前空闲。本窗口包含桌面主进程及所属界面、后台、分析和导出子进程，不包含其他实例。首次采样与数据不可用会明确标注。当前空闲只是采样余量，并非本窗口独享的 CPU 配额，也不是硬性限额。源码网页版只能统计其后台进程树，显示“本服务占用”。

CPU 统计依赖 BSD-3-Clause 许可的 [psutil](https://github.com/giampaolo/psutil)；窗口编号占用使用 Node 原生命名管道，Windows 在所属进程退出后自动释放。后台另持有数据目录的独占文件锁，防止异常关闭期间重复打开同一任务库。

CPU 数值为采样估计：完全在两次采样之间启动并退出的短进程可能漏计，不能用它替代精确的累计 CPU 时间审计。

## 从源码构建

构建机需要 Windows x64、Python 3.12、Node.js 和 npm，安装依赖时需要联网。最终使用者无需这些开发环境。

```powershell
powershell -ExecutionPolicy Bypass -File .\desktop\build.ps1 -OutputRoot 'D:\XiangQi-build'
```

`OutputRoot` 应选择不存在成品文件夹的目录；脚本不会替换已有成品。输出包括完整程序目录和 portable 单文件 EXE，工具依赖位于 `build`。完整目录中的 `resources` 不能删除；单文件 EXE 可独立分发。

单文件构建固定使用 MIT 许可的 electron-builder 26.18.0 portable 目标。该版本实际实现中 `unpackDirName: true` 使用每次启动独立的 `$PLUGINSDIR`；网站部分文档描述与该版本不一致，请保留仓库配置并执行双开关闭隔离验收。

实现基于 [Electron](https://www.electronjs.org/docs/latest/tutorial/tutorial-packaging) 与 [PyInstaller](https://pyinstaller.org/en/stable/operating-mode.html)。Electron 内置窗口显示现有 React 页面，后端使用随机本机端口，录像使用独立的离屏渲染窗口和原有编码流程。
