# 网络与模型连接故障自动恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 将连接建立失败、模型端或网关中途断开接入持久保存和后台自动恢复，联网后无需点击重连即可完成分析。

**Architecture:** 复用现有 SQLite、任务目录、单分析进程和 FastAPI 生命周期。请求前保存证据及可恢复检查点，临时故障进入等待队列，调度器领取到期任务并恢复；运行代次防止旧工作覆盖新任务。

**Tech Stack:** Python 3.12、FastAPI、httpx 0.28.1、SQLite、OpenCV、React 19、TypeScript、pytest、Vitest。

**Spec:** `docs/superpowers/specs/2026-10-12-automatic-network-recovery-design.md`。用户于 2026-10-12 要求将网络中断和模型连接中断两类处理实际用于程序；设计方向已确认，用户已确认按计划直接执行；实施证据见验收报告及阶段记录。

## Global Constraints

- 固定前端 5278、后端 8278，不占用其他项目端口，不安装额外消息队列，不修改模型或代理配置。
- 断网保存、后台自动恢复；主动停止取消自动运行，关闭网页不取消。
- 初始识别、实际局面与合法棋谱分析使用统一错误分类，各自保存完整恢复状态。
- 连接失败重试间隔 5、15、30、60、120 秒，随后上限 300 秒，约 20% 随机浮动；服务端 Retry-After 是下限。
- 同一操作的结果不确定推理最多额外重试两次，不无限重复发送可能已计费的请求。
- 无法恢复的配置或存储错误进入管理员处理状态；不能将部分 JSON、模型列表成功或模拟测试视为有效识别。
- 开发使用隔离任务和独立数据库；不直接覆盖已有 24 条记录的生产任务，不输出密钥。
- 项目当前没有 Git 仓库，不初始化 Git，不执行分支或提交命令；用阶段记录和测试结果记录完成情况。

## Review Focus

- 请求已得到有效响应，但棋谱提交前退出：恢复复用响应，不重复推理或追加记录。
- 在同一视频帧的多个事件之间中断：恢复保持事件阶段，既不漏步也不重复。
- 等待服务时人工停止或改变任务版本：晚到结果不得写回，也不得后台重新启动。
- 云端已接收请求但流式连接中断：保留部分用量和请求阶段，按结果不确定预算处理。
- 网页连接本地服务失败后恢复：重新同步真实状态，不要求刷新页面。

## 文件职责

`vision.py` 负责结构化调用错误和诊断；`schemas.py` 定义恢复状态；`storage.py` 负责原子保存、运行代次和领取事务；新增 `recovery.py` 负责调度策略及后台协调；新增 `checkpoint.py` 负责版本化状态和图片持久化；`actual.py`、`analysis.py`、`turns.py` 保存及恢复检测运行状态；`main.py` 接入生命周期和操作；前端接入等待状态和辅助恢复入口。

## Task 1：结构化模型调用错误及分阶段诊断

**Files:** 修改 `backend/app/vision.py`；测试 `backend/tests/test_vision.py`。

**Interfaces:** 产生 `VisionRequestError(ValueError)`，字段 `code: str`、`phase: str`、`retryable: bool`、`outcome_uncertain: bool`、`retry_after_s: float | None`。`last_response` 保存安全诊断：已收到 HTTP 响应、首块时间、已接收字符数、终止状态及请求编号；不保存 Authorization 或原始异常中的凭证。

- [x] 先写 `test_remote_disconnect_preserves_phase_partial_usage_and_safe_diagnostic`：接收部分 SSE 后抛 `RemoteProtocolError`，断言 phase 为 response、outcome_uncertain 为真、部分用量保留、凭证不在诊断中。
- [x] 写 `test_connect_timeout_is_retryable_without_uncertain_inference`：断言连接阶段可重试、请求只发一次、结果不确定为假。
- [x] 写 `test_http_status_selects_recovery_and_retry_after`：401/403 不重试，429/503 可重试且解析 Retry-After；StreamError 和缺少正常结束的 SSE 不接受为成功。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest backend/tests/test_vision.py -q`，确认新增断言因缺少分类失败。
- [x] 实现上述接口，兼容现有 ValueError 调用方；模型端接收/协议中断不能提示客户检查网络。
- [x] 重跑同一命令，确认全部通过。

## Task 2：任务恢复状态、原子检查点和写入保护

**Files:** 修改 `backend/app/schemas.py`、`backend/app/storage.py`；新增 `backend/app/checkpoint.py`、`backend/tests/test_recovery_storage.py`。

**Interfaces:** `Task.recovery: RecoveryState | None`；RecoveryState 包含 auto_run、operation、generation、checkpoint、pending_request、attempts、uncertain_attempts、next_attempt_at、last_error、lease_token。`CheckpointStore(root)` 产生 `save(task_id, generation, state) -> str`、`load(task_id, generation, name) -> dict`。`TaskStore.claim_recovery(task_id, generation, now) -> Task | None` 事务领取到期任务；`save_analysis` 检查当前运行代次和租约，拒绝旧结果。

- [x] 写 `test_checkpoint_survives_store_recreation`，保存图片、检测数组和棋局后重建存储，断言内容一致。
- [x] 写 `test_only_one_scheduler_claims_due_operation`，两个连接领取同一任务，断言仅一个领取成功。
- [x] 写 `test_old_generation_cannot_overwrite_stopped_or_changed_task`，断言主动停止后晚到保存被拒绝、原棋谱完整。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest backend/tests/test_recovery_storage.py -q`，确认缺少行为导致失败。
- [x] 实现状态、事务和原子文件写入；使用带版本 JSON 和无损图片，数据库只引用完整文件；保存有效响应并使用稳定请求标识去重。
- [x] 重跑该测试，并运行 `backend/tests/test_tasks.py backend/tests/test_corrections.py backend/tests/test_review_isolation.py`，确认旧任务兼容和校正保护。

## Task 3：可恢复的事件检测与棋谱提交

**Files:** 修改 `backend/app/actual.py`、`backend/app/analysis.py`、`backend/app/turns.py`；新增 `backend/tests/test_recovery_pipeline.py`。

**Interfaces:** 实施裁决：使用 `ReplayStore(store, current)` 保留已提交前缀，`RecoveryVision(vision, store, task)` 保存请求和有效响应；`run_saved_operation(task_id, store, vision=None)` 完整重放原始输入以重建生成器和回合状态，不重写现有检测器/TurnReplay。成功响应复用，提交前对齐旧记录，检测不一致时保护原数据。额外本地解码成本已列入验收报告。

- [x] 写 `test_network_wait_preserves_pending_turn_and_completed_records`：云端失败后原棋谱和未完成回合保留，不添加“分析中断”人工疑点。
- [x] 写 `test_resumed_actual_replay_matches_uninterrupted_result`，覆盖吃子、取子未落下、跨回合和短暂复位，断言最终棋局、记录、时间和疑点与不中断基线一致。
- [x] 写 `test_resume_between_events_from_one_frame_has_no_missing_or_duplicate_moves`、`test_saved_response_is_reused_after_commit_interruption`、`test_legal_analysis_and_initial_operation_restore`。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest backend/tests/test_recovery_pipeline.py -q`，确认缺少恢复流程失败。
- [x] 局部提取可持久化检测状态；请求前保存证据和检查点；网络等待释放进程，不调用回合中断逻辑。周期检查点覆盖意外退出；响应落盘后方可提交棋谱。
- [x] 保留最新人工校正；更改设置须先取消运行，不把扫描时间当成完成检查点。
- [x] 重跑新增测试及 `test_actual.py test_tracking.py test_turn_rules.py test_pipeline.py`，确认一致性。

## Task 4：后台自动调度与恢复接口

**Files:** 新增 `backend/app/recovery.py`、`backend/tests/test_recovery_api.py`；修改 `backend/app/main.py`、`backend/app/storage.py`。

**Interfaces:** `RecoveryCoordinator(store, workers, analysis_lock)` 提供 `start()`、`close()`、`tick(now=None)`、`request_resume(task_id, revision)`、`cancel(task_id)`；`run_recovery_worker(task_id, data_root, generation, lease_token)` 分派已保存操作。新增 `POST /api/tasks/{task_id}/resume`，提交 revision，返回 Task，重复请求不启动额外进程。

- [x] 写 `test_waiting_task_runs_automatically_when_due`，无需 resume API 即自动领取恢复；任务占用情况变化后可继续调度。
- [x] 写 `test_restart_recovers_authorized_tasks_but_not_manual_stop`、`test_repeated_resume_creates_only_one_worker`、`test_service_disconnect_respects_uncertain_request_budget`。
- [x] 写 `test_starting_new_analysis_invalidates_old_pending_operation`、`test_initial_recognition_network_failure_is_scheduled_in_background`。
- [x] 运行 `.\.venv\Scripts\python.exe -m pytest backend/tests/test_recovery_api.py -q`，确认缺少调度行为失败。
- [x] 实现每两秒调度、退避及 Retry-After，启动时回收失效租约并恢复授权任务；停止时撤销自动运行和租约。单独识别提取可由后台运行的操作，保持正常成功响应兼容。
- [x] 永久错误进入 action_required；管理员修正后 resume 重新检查。旧失败任务没有检查点时拒绝冒充直接续跑，保留生产棋谱并提供独立恢复入口。
- [x] 重跑新增测试及 `backend/tests/test_stop.py backend/tests/test_initial_recognition.py`。

## Task 5：网页自动同步与辅助恢复按钮

**Files:** 修改 `frontend/src/types.ts`、`frontend/src/api.ts`、`frontend/src/App.tsx`、`frontend/src/styles.css`、`frontend/src/components/RecordingPanel.tsx`、`backend/app/recording.py`；新增 `frontend/src/recovery.test.ts`，修改 `frontend/src/api.test.ts`。

**Interfaces:** 任务类型增加 waiting_service、action_required 和安全恢复摘要；`api.resume(task)` 提交版本，不调用 analyze。界面轮询 analyzing 和 waiting_service；本地服务短暂不可达后自动重新同步。

- [x] 写等待状态轮询、本地服务恢复同步、resume 请求携带 revision、重复点击只提交一次、停止取消、等待期间禁止完整录像导出的测试。
- [x] 运行 `npm.cmd test -- --run`，确认新增功能测试失败。
- [x] 实现提示“视觉服务暂时不可用，任务和进度已保存，将自动恢复”，显示已保存记录数量、等待时间和下次尝试；按钮仅提供“立即尝试恢复”“停止任务”，管理员错误单独显示补救信息。
- [x] 界面恢复时保留用户回放位置；等待期间不可不经取消就修改前序棋局和设置。
- [x] 运行 `npm.cmd test -- --run` 和 `npm.cmd run build`，确认测试及构建通过。

## Task 6：完整验收与交付

**Files:** 新增 `output/automatic-recovery-validation/` 下的独立验收脚本与结果，更新 `README.md`、`docs/迭代进度.md`。

- [x] 完整运行 `.\.venv\Scripts\python.exe -m pytest -q` 和前端测试、构建，记录所有失败及修复。
- [x] 在独立数据目录和独立测试端口启动真实 HTTP 视觉服务替身，分别模拟连接不可用、SSE 中途断开、503、响应保存后退出；恢复真实后端进程，验证正常 API 路径自动完成，不仅测试辅助函数。
- [x] 正常浏览器界面验证等待提示、后台自动恢复、主动停止、刷新恢复和按钮去重；保存实际状态及最终棋谱对照。
- [x] 做一次有范围及请求次数限制的真实模型识别验收，区分模拟和云端证据；如果需要单独批准模型请求费用，则在其余验证和具体请求准备完成后再提出，不先动生产任务。
- [x] 依据执行技能进行一次独立代码审查，重要问题以失败测试复现后修复，跑完整相关回归；不把尚未通过的检查写成完成。
- [x] 更新使用说明和阶段记录，报告最终自动恢复结果、测试数量及剩余阻碍。

## 计划自审

六项任务覆盖错误分类、落盘边界、检测器/回合状态、进程调度、网页恢复和实际验收。无外部队列、供应商或新端口依赖。旧生产任务缺少检查点的限制单列，不能把它等同于新任务的无损断点恢复。

执行方式建议：在当前对话由我直接执行，保持接口修改和测试连续；仅最终审查按执行技能使用独立审查。用户已确认并完成实施；未逐项序列化生成器的技术裁决见上述Task 3及执行记录。
