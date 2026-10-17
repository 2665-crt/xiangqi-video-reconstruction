# Board Recording Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 用户核对后导出所选区间右侧棋盘的紧凑 MP4，自动保存到指定目录。

**Architecture:** 后端校验并快照任务，异步启动 Node 渲染进程。专用 React 页复用 Board，按帧暂停现有动画，PNG 流由 PyAV 编码为 H.264。任务状态单独持久化，前端查询进度。

**Tech Stack:** FastAPI / React / 已安装 Playwright + Chrome / PyAV。

**Spec:** `docs/superpowers/specs/2026-10-12-board-recording-design.md`

## Global Constraints

- 只录棋盘，1080×1192 / 60 fps / MP4-H.264，无原视频等待与音轨。
- 固定保存到 `F:\codex产出\虚拟棋盘对战录制\录屏存放`，不覆盖已有文件。
- 沿用 5278/8278，保留现有识别和任务数据；不调用云端识别。
- 目录没有 Git，不创建仓库或 worktree，不执行提交；当前用户已明确授权执行完整方案，不重复请求阶段审批。

## Review Focus

1. 中断分析虽经校正仍可能缺失末尾走棋，必须检查 processed_video_s 覆盖选区。
2. 审核通过的无走棋记录不能造成缺失局面或额外录像片段。
3. 页面刷新、换任务和重复点击不能错绑录像或产生并发重录。
4. 录制画面须按确定时间暂停动画，并且文件最后一帧已落稳。
5. 编码失败或服务退出不能将半成品标记为成功或覆盖旧录像。

### Task 1: 完整性校验、快照与异步导出状态

Files: 新增 `backend/app/recording.py`、`backend/tests/test_recording.py`；修改 schemas.py/main.py。

Interfaces: `build_recording_playlist(task, interval_s)` 返回 initial_position/records/interval_s；`RecordingManager.start(task, revision, interval_s)` 返回 job；get/latest 返回持久化 job。

- [x] 写失败测试：疑点的具体步骤、缺失初始局面、分析未完成、版本冲突、审核记录不计走棋、中文命名安全和重复请求。
- [x] 运行 `.venv/Scripts/python.exe -m pytest backend/tests/test_recording.py -q` 观察失败。
- [x] 实现校验、唯一文件命名、快照、状态和 POST/GET API；输出固定目录，失败不给完成文件。
- [x] 验证该测试与完整后端测试集。

### Task 2: 复用棋盘渲染并实际编码

Files: 新增 `frontend/src/RecordingRenderer.tsx`、`scripts/render-recording.cjs`；修改 main.tsx/recording.py。

Interfaces: 渲染页 `/recording-render` 的 `loadRecording(payload)`、`renderRecordingStep(index, elapsedMs)`；stdout JSON 行 `{bytes,repeat}` 后紧接 PNG bytes，Python 逐包解码并编码重复帧。

- [x] 先写实际生成 MP4 的失败测试/验证脚本，要求总帧数 `(变化数+1)*interval_s*60`，尺寸 1080×1192，末帧完整。
- [x] 实现暂停并定位浏览器动画、静态画面重复编码，流输出与 PyAV 编码，完成后才发布文件。
- [x] 解码验证初始/移动中/最终局面，检验 1/2/3 秒节奏。

### Task 3: 导出按钮、进度与真实 UI 验证

Files: 新增 frontend/src/components/RecordingPanel.tsx；修改 types.ts/api.ts/App.tsx/styles.css、README.md；验证脚本放 output/playwright/recording。

- [x] 写 API 与导出 UI 的失败验证，包括待复核拒绝、正常生成、进度完成及刷新恢复。
- [x] 接入导出按钮，未保存设置/校正时禁用，状态轮询，显示绝对路径与录像链接。
- [x] 运行后端完整测试、前端完整测试和构建，真实浏览器点击并生成实际 MP4。
- [x] 运行原棋盘动画验证，邀请只读最终审查并修复实质问题。
