# Windows 便携桌面版

应用名：**XiangQi Vedio Reconstruction**。

将整个 ZIP 解压到可写目录，双击同名 EXE。保留 EXE 旁边的全部文件，不能单独复制 EXE。无需安装 Python、Node.js 或浏览器，无需配置云端 API；使用随包提供的本地模型和 CPU 推理。

首次启动会加载模型。上传 MP4 后，按现有界面完成棋盘定位、初始局面核对、分析和人工校正。模型可能留下待核对区间，不能保证每段视频完全自动还原。任务保存在应用目录的 `data`，导出视频保存在 `录屏存放`。移动软件时请移动整个文件夹，保留这两个目录即可保留个人数据。关闭应用会停止自己的后台服务。

此构建未作代码签名。在 Windows x64 本机完成验收，未在其他实体电脑上验证。第三方组件、权重来源和许可见 `licenses`；对应应用源码随 `licenses/application-source.zip` 提供。

## 从源码构建

构建机需要 Windows x64、Python 3.12、Node.js 和 npm，安装依赖时需要联网。最终使用者无需这些开发环境。

```powershell
powershell -ExecutionPolicy Bypass -File .\desktop\build.ps1 -OutputRoot 'D:\XiangQi-build'
```

`OutputRoot` 应选择不存在成品文件夹的目录；脚本不会替换已有成品。临时依赖位于该目录的 `build`。确认成品后可以删除 `build`，不要删除应用文件夹内的 `resources`。

实现基于 [Electron](https://www.electronjs.org/docs/latest/tutorial/tutorial-packaging) 与 [PyInstaller](https://pyinstaller.org/en/stable/operating-mode.html)。Electron 内置窗口显示现有 React 页面，后端使用随机本机端口，录像使用独立的离屏渲染窗口和原有编码流程。
