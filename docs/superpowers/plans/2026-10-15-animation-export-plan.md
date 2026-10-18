# 走棋动画与当前步骤录像导出 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 隐藏的疑点记录不影响正常相邻走棋动画；有实际步骤即可按当前已保存版本导出录像。

**Architecture:** 主界面向 Board 传递可见步号，保持真实跨步跳转和多子局面替换的处理。前后端同时取消待校正、未分析完、重检及编辑状态对当前版本导出的限制；录像清单使用保存局面，无局面但已有走法时在导出副本中执行该走法，纯疑点记录不计步。保留任务版本校验与已有编码管线。

**Tech Stack:** React / TypeScript / Vitest / FastAPI / pytest / Playwright / PyAV。

**Spec:** 本对话已确认的两个问题及用户 2026-10-15 的执行授权。

## Global Constraints

- 修改融合实验 `experiments/fusion-recognition`；保留当前未提交修改，不改正式版本。
- 不调用识别模型，不修改原用户棋谱。
- 导出保持 MP4、1080×1192、60 fps 与现有编码画质。
- 有未保存编辑时导出已保存步骤，不自动保存编辑。
- 不提交、不推送代码；当前工作目录已有用户修改，原地进行有限改动。

## Review Focus

- 相邻可见步骤间有一条或多条隐藏记录，以及反向吃子，仍播放单子动画。
- 真正跨多个可见步骤、拖动视频、多子变化仍为局面替换。
- 待校正走法无 position_after 时仍能从已有局面生成录像，不改变原记录状态。
- 分析、等待恢复、重检或手动编辑不阻止导出已保存步骤。
- 导出清单不可因随后校正而改变；没有实际步骤不生成空棋谱录像。

### Task 1: 相邻可见走棋动画

**Files:** `frontend/src/App.tsx`、`frontend/src/board-motion.test.ts`。

- [x] 添加隐藏记录间隔与反向吃子回归测试；验证原始编号触发 replace，显示编号触发 move。
- [x] App 的 Board.step 改为 displayStep(task, step)，保持原始编号用于查找记录。
- [x] 实际浏览器验证用户列出的全部 15 步及正常相邻、跨步对照。

### Task 2: 导出当前已有步骤

**Files:** `frontend/src/components/RecordingPanel.tsx`、`frontend/src/App.tsx`、`frontend/src/recording.test.ts`、`backend/app/recording.py`、`backend/tests/test_recording.py`。

- [x] 用新需求替换旧阻断测试，新增未校正、疑点、未完成分析、仅走法无局面、空步骤与导出快照测试；先运行确认失败。
- [x] 清单生成不以校正状态和扫描完成度为前提，跳过非走棋疑点；仅在清单副本补全已有走法的局面。
- [x] 前端按钮取消校正/重检/分析前置阻断，并提示按已保存步骤导出；后端保留版本与技术参数检查。
- [x] 运行前后端回归及生产构建；通过正常页面导出含疑点的真实任务，解码核对尺寸、帧数、帧率和末局。

### Verification ledger

- 初查：当前服务未监听 5288/8288；实验 Git 有既存修改，不覆盖这些修改。

- 完成：实际页面 20 个动画用例通过，含用户指定全部 15 步。
- 完成：前端 57 通过、后端 416 通过/2 跳过/1 依赖弃用警告，生产构建通过。
- 完成：真实任务 76 手由未保存编辑状态直接导出，77 秒/4620 帧，全部帧解码通过，任务与模型请求数未改变。
- Ruling: 保留只有人工摆局校正的既有导出能力，将空清单判断移至构造之后 — 全量回归证明是已有能力 — 没有步与摆局的纯疑点仍不生成录像。
- Ruling: 导出改为从 get_task 读取保存快照 — 审查与失败 API 测试证明 task_detail 会清空规则模式的待校正保存局面 — GET 详情原行为保持不变。
- Ruling: 5288 由腾讯视频占用，临时使用 5289 并配置录制渲染 URL — 不停止其他应用 — 原固定端口启动脚本未改，进程记录与启动脚本保存在验收目录。
- 最终验收：output/animation-export-validation/动画与录像导出整改验收.md。
