# 象棋视频复盘第一版 Implementation Plan

> **For agentic workers:** 执行时使用 superpowers:executing-plans 按任务推进；如用户选择委派方式，再使用 superpowers:subagent-driven-development。准备阶段不启动子代理。

**Goal:** 将用户录屏提取成可核对的棋谱，并在网页右侧逐步复现。

**Architecture:** 本地 FastAPI 服务接收视频，以 PyAV 解码、OpenCV 分析画面，通过视觉模型识别棋子，结合规则和时序验证。React 网页显示视频、棋盘和校正界面，SQLite 与任务文件保存结果。

**Tech Stack:** Python 3.12、FastAPI、PyAV、OpenCV、React、TypeScript、Vite、SVG、SQLite、pytest、Vitest、Playwright。

**Spec:** ../specs/2026-10-10-xiangqi-video-design.md

## Global Constraints

- 第一步定义为一方一次走棋，初始局面为第 0 步；红方在下。
- 只实现 PC 网页上传、识别、校正、回放和保存，不增加录制导出或第三方控制。
- 使用工作目录内的 .venv、frontend/node_modules 和 data，不修改全局 Python 环境。
- 上传上限暂定 2 GiB；首批验收 MP4/H.264，实际验证规模另行记录。
- 服务默认 127.0.0.1:8278、前端 127.0.0.1:5278，严格检查占用、不自动切换，不自动暴露外网。
- 密钥只存在后端环境，不写入前端、日志、示例数据或文档。
- 缺少模型或视频时明确报告未验证，不用固定棋谱冒充视频识别结果。
- 棋盘坐标 file=0..8、rank=0..9；模型、自报置信度和单独规则合法性都不能替代证据。
- 测试先运行并观察预期失败，再实现对应逻辑；各任务完成后跑相关测试，最终运行整个套件。

## Review Focus

1. 变帧率与快速连续落子不能因固定帧率假设而时间错位或漏步，归任务 3、5。
2. 从中盘开始及黑方先走不能强制标准开局，归任务 1、4。
3. 手部遮挡产生多种合理走法时不能继续确认后续步骤，归任务 5、6。
4. 修改早期一步不能只更新当前画面而留下不一致的后续棋局，归任务 6。
5. 刷新或服务重启不能丢失校正、显示虚假运行状态，归任务 2、7。

## 文件职责

| 计划文件 | 职责 |
|---|---|
| backend/app/schemas.py | Spec 中数据类型及请求体 |
| backend/app/chess.py | 棋子规则、落子、局面回放 |
| backend/app/storage.py | SQLite、任务文件与修订号 |
| backend/app/video.py | 元数据、时间戳解码、证据图 |
| backend/app/calibration.py | 四角映射和棋盘定位 |
| backend/app/vision.py | 视觉请求、结构化响应验证 |
| backend/app/analysis.py | 局面序列、事件检测与消歧 |
| backend/app/main.py | Spec 中 HTTP 接口、静态媒体、任务调度 |
| frontend/src/types.ts、api.ts | 数据契约及接口调用 |
| frontend/src/App.tsx | 页面布局与任务状态 |
| frontend/src/components/Board.tsx | SVG 棋盘与棋子动画 |
| frontend/src/components/VideoPanel.tsx | 播放、范围和四角编辑 |
| frontend/src/components/MoveList.tsx、ReviewPanel.tsx | 棋谱列表、疑点确认 |
| frontend/src/replay.ts | 选步、步间隔、对照模式状态 |
| backend/tests、frontend/src/*.test.*、frontend/e2e | 规则、接口、视频和实际页面验证 |
| scripts/start.ps1、.env.example、README.md | 项目启动、配置占位与操作说明 |

## Task 1：棋局数据与正确回放基础

**Files:** backend/app/schemas.py、backend/app/chess.py；backend/tests/test_chess.py、test_schemas.py；backend/requirements.txt、pytest.ini。

**Interfaces:** 产生 Position、Move、MoveRecord 等 Spec 类型；`legal_moves(position: Position) -> list[Move]`；`apply_move(position: Position, move: Move) -> Position`；`replay(initial: Position, moves: list[Move]) -> list[Position]`，返回结果含第 0 步。

- [ ] 创建项目 .venv；安装并记录本任务依赖，避免全局安装。
- [ ] 写失败测试：`test_replay_capture_and_restore` 断言吃子后目标棋子移除、原始 Position 未改变、回放第 0 步完整；`test_black_to_move_midgame` 断言保留中盘与黑方轮次；`test_illegal_horse_cannon_and_facing_kings` 分别拒绝蹩马腿、错误炮架和将帅照面。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_chess.py backend/tests/test_schemas.py -q`；确认因缺少功能失败。
- [ ] 实现上述类型、走棋规则及纯回放，不加入对弈引擎。
- [ ] 运行同一命令，要求全部通过；记录具体通过数和覆盖局面。

## Task 2：视频任务与持久化 API

**Files:** backend/app/storage.py、backend/app/main.py；backend/tests/test_tasks.py。

**Interfaces:** 消费 Task、VideoMetadata；产生 `TaskStore.create(filename: str, metadata: VideoMetadata) -> Task`、`get(task_id: str) -> Task`、`list_tasks() -> list[Task]`、`save(task: Task, expected_revision: int) -> Task`。实现 health、tasks 列表/详情、上传和 setup 接口。

- [ ] 写失败测试：`test_task_survives_reopen` 在新建存储实例后读到相同校正与修订号；`test_stale_revision_conflict` 拒绝旧 revision；`test_interrupted_task_on_restart` 将残留 analyzing 转为 interrupted；`test_health_hides_configuration` 验证响应中无密钥和接口详情。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_tasks.py -q`，确认预期失败。
- [ ] 实现项目内 SQLite 与服务器生成任务 ID；上传流式保存，失败任务保留可理解错误，实际解码检查接任务 3。
- [ ] 运行任务测试并实际访问 `/api/health`；记录返回及存储重开结果。

## Task 3：真实视频解码、校准及证据

**Files:** backend/app/video.py、backend/app/calibration.py、main.py；backend/tests/test_video.py、test_calibration.py。

**Interfaces:** `probe_video(path: Path) -> VideoMetadata`；`iter_frames(path: Path, start_s: float, end_s: float) -> Iterator[tuple[float, ndarray]]`；`rectify_board(frame: ndarray, calibration: Calibration) -> ndarray`；`detect_calibration(frames: list[ndarray]) -> Calibration | None`。提供 video Range 与 evidence 接口。

- [ ] 安装 PyAV 与 opencv-python-headless；依赖安装失败时记录具体阻塞，不改变算法假装可运行。
- [ ] 写失败测试：`test_variable_timestamps_preserved` 保留合成短视频真实 PTS；`test_range_keeps_absolute_time` 检查区间时间不是从 0 重置；`test_rotation_maps_intersections` 断言方向与四角正确；`test_video_range_request` 返回正确 206 与字节范围；`test_invalid_media_rejected` 损坏 MP4 返回明确错误。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_video.py backend/tests/test_calibration.py -q`，确认预期失败。
- [ ] 实现元数据、顺序解码、缩略图、透视映射及媒体接口，上传接口接入真实解码验证。
- [ ] 运行测试；用可解码的视频实际上传并拖动网页视频进度。合成视频只验证解码，不算真人识别验收。

## Task 4：视觉识别及初始局面

**Files:** backend/app/vision.py、analysis.py、schemas.py；backend/tests/test_vision.py。

**Interfaces:** `VisionClient.recognize_position(images: list[ndarray]) -> list[Position]`，返回候选局面；配置来自三个 VISION_* 环境变量。初始下一方若无证据确定，进入 needs_review 等待用户设置。

- [ ] 写失败测试：`test_missing_config_blocks_analysis` 断言模型未配置时自动分析不启动；`test_vision_response_validation` 拒绝重复格子、越界及未知棋子；`test_midgame_not_replaced_by_opening` 保留中盘；`test_unknown_turn_requires_review` 不默认红方先走。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_vision.py -q`，确认预期失败。
- [ ] 实现 HTTP 图像请求和响应验证；测试可模拟外部响应，但不将模拟结果暴露为真实识别。优先发送棋盘裁剪图；记录可用的用量元数据。
- [ ] 运行测试。配置存在时实测一组真实棋盘图；配置缺失时报告该实测未执行，不索取或打印密钥。

## Task 5：走棋提取、多帧消歧与分析进度

**Files:** backend/app/analysis.py、main.py；backend/tests/test_analysis.py。

**Interfaces:** 消费解码帧、VisionClient、legal_moves 和 apply_move；`extract_moves(initial: Position, observations: list[tuple[float, list[Position]]]) -> list[MoveRecord]`；`analyze_task(task_id: str, store: TaskStore, vision: VisionClient) -> None`。提供 analyze 接口及实际视频分析位置。

- [ ] 写失败测试：`test_hand_motion_not_a_move` 同一落稳局面无重复步；`test_fast_two_moves_not_collapsed` 两次变化保留两步；`test_blur_preserves_piece_identity` 邻近证据充分时保持棋子类型；`test_ambiguous_occlusion_stops_prefix` 多种解释产生 needs_review，后续为 tentative；`test_analysis_progress_monotonic` 进度来自处理时间戳。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_analysis.py -q`，确认预期失败。
- [ ] 实现顺序事件检测、局面候选、证据保存、身份跟踪、规则过滤及唯一解释接受。后台单任务运行，异常持久化而非无限转圈。
- [ ] 运行测试，再用真实录屏逐步与参考棋谱核对。没有真实输入时只能报告模块测试通过，M2 尚未验收。

## Task 6：人工校正与后续一致性

**Files:** backend/app/analysis.py、main.py、storage.py；backend/tests/test_corrections.py。

**Interfaces:** `correct_move(task: Task, index: int, move: Move) -> Task`；`revalidate_suffix(task: Task, start_index: int) -> Task`。实现 moves 校正接口，setup 更新初始局面时调用同一后续校验逻辑。

- [ ] 写失败测试：`test_correct_early_move_revalidates_suffix` 修改早期走法后重新检查后续身份、证据和局面；`test_legal_but_unsupported_suffix_stays_tentative` 合法但无图像支持的后续不能自动确认；`test_corrected_move_persists` 新存储实例读到 corrected 状态；`test_initial_position_edit_invalidates_prefix` 初始变化触发重验。
- [ ] `.venv/Scripts/python.exe -m pytest backend/tests/test_corrections.py -q`，确认预期失败。
- [ ] 实现 revision 检查、局面重建、证据重验和持久化。用户点确认才写入。
- [ ] 运行测试，实际修改任务中一步并重新打开检查结果。

## Task 7：完整网页与回放

**Files:** frontend/package.json、index.html、vite.config.ts、tsconfig.json、src/types.ts、api.ts、App.tsx、replay.ts、样式及 Spec 对应组件；src/replay.test.ts、e2e/replay.spec.ts；scripts/start.ps1、.env.example。

**Interfaces:** TypeScript 类型严格对应 Spec。`positionAt(task: Task, step: number): Position | null`；`stepAtVideoTime(task: Task, timeS: number): number`。所有 API 调用走 api.ts，开发代理 /api 到后端。

- [ ] 建立 React/TypeScript/Vite 项目与 Vitest。写失败测试：`test_position_zero_and_capture`、`test_playback_stops_before_review`、`test_video_time_selects_move`，分别验证第 0 步与吃子、疑点前暂停、证据时间映射。
- [ ] 在 frontend 运行 `npm.cmd test -- --run`，确认预期失败。
- [ ] 实现左视频右棋盘、上传状态、四角叠层、中盘设置、步骤列表、校正面板、1/2/3 秒播放、联动和任务重开；未配置模型时显示说明。
- [ ] 运行前端测试及 `npm.cmd run build`，要求通过。
- [ ] 使用浏览器实际检查：上传后视频可播、棋盘范围能编辑、步骤点击联动、吃子与后退正确、跳步取消动画、刷新后校正仍存在、疑点阻止自动播放。
- [ ] 启动脚本定位项目目录并启动两个本地服务；配置模板只包含空值和名称。页面尚未识别时不出现假棋谱。

## Task 8：真实验收与交付

**Files:** README.md、docs/验收记录.md；测试素材仅在用户提供后纳入工作目录，原始视频不提交到外部。

- [ ] 运行 `.venv/Scripts/python.exe -m pytest backend/tests -q`、frontend 的 `npm.cmd test -- --run`、`npm.cmd run build` 和实际 UI 验证，记录通过和失败项目。
- [ ] 使用 Spec 验收集逐步核对，记录自动正确步数/参考总步数、漏步、重复、待确认、人工修正及完整匹配情况；记录各视频时间与模型用量。
- [ ] 清晰视频达不到全部正确时定位错误事件与证据，先写失败测试再修复，不以最终棋局相同代替逐步正确。
- [ ] 写完整 Windows 启动、后端模型配置、操作 SOP、限制与已验证输入范围。
- [ ] 只有真实验收通过才报告 M4 完成；样本或配置缺失时交付可运行功能并明确列出尚未验证项目。

## 自审与进度汇报

已检查：上传、初始棋局、自动识别、模糊遮挡、校正、保存、媒体定位及回放均有任务负责；Spec 字段和接口名在各任务一致；五项 Review Focus 有对应测试。

任务完成时在对话报告：完成时间、功能、实际验证、剩余问题、下一任务。当前只有 M0 文档准备，所有实现任务均未勾选。

实施方式建议：当前会话由主助手逐项执行。正式实现前需要用户审阅本计划与设计文档；本文件不是功能完成证明。
