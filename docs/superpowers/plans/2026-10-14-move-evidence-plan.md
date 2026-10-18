# 逐手证据与整盘一致性 Implementation Plan

> **For agentic workers:** 使用 superpowers:executing-plans 在当前会话执行，使用 TDD；用户已明确要求开始整改，持续执行直到完成验证。

**Goal:** 保证每手核对可追溯，发现整盘差异后回查漏步，并完成选区末手的延伸追踪。

**Architecture:** 在实际局面事件与回合合并之间加入证据核对、区间补查与末手过滤；保留现有识别和校正流程，新增可选字段兼容历史任务。

**Tech Stack:** Python/OpenCV/FastAPI/Pydantic、React/TypeScript/Vitest。

**Spec:** ../specs/2026-10-14-move-evidence-design.md

## Global Constraints
- 仅修改 experiments/fusion-recognition，保留当前未提交改动。
- 不调用云端，不覆盖原任务；验收使用独立数据库与输出。
- 不根据时间间隔或合法走法编造中间步骤。
- 未知格保留未知，起终点必须有直接证据。

## Review Focus
- 遮挡下额外棋子变化、错误初始局面及多手聚合不能通过校验。
- 吃子离盘不能触发错误的选区外边界。
- 末手在选区外连续调整、下一手不进入结果。
- 回查失败和 EOF 未完成操作保留审核信息。
- 校正及恢复不能丢失证据时间或沿用已失效验证状态。

### Task 1: 核对与漏步补查
- [x] 新增 test_move_evidence.py：前后冲突拒绝、未知格不当空格、多手观察回查且不按规则直接发布。
- [x] 运行并确认失败，再实现 schemas.py / board_evidence.py / actual.py / turns.py。
- [x] 运行实际检测、回合与恢复回归。

### Task 2: 末手延伸与审核
- [x] 测试选区内拿起、选区外落定和再调整、排除下一手、EOF 未落定、选区外校正。
- [x] 实现事件选区过滤和完整视频读取；记录证据时间，更新已有 ReviewPanel 与类型。
- [x] 前后端测试及构建。

### Task 3: 真实视频与正常入口验证
- [x] 原任务只读核对与原视频短片重跑，检查相邻局面及跨选区落定。
- [x] 全量后端、前端测试与构建；代码审查。
- [x] 核对实验服务身份后加载修复，验证正常 API/UI。
- [x] 写验收报告、标记完成项并汇报限制。

## Progress
- 基线：实验版本相关36项测试通过，1项既有 Starlette 弃用警告。
- 已确认非 Git 根目录；当前实验目录处于 experiment/fusion-recognition，直接在已授权实验目录整改。
- 最终：22项本轮回归；后端354 passed、2 skipped，前端37 passed，构建通过。真实7～40秒15手ready、0云端；7～14.5秒末手15.455秒落定，3手ready。只读独立审查发现的5个边界问题均修复。
- Ruling: 当前分支及用户既有未提交改动保留，不提交、合并或推送；授权范围为在当前实验程序完成整改与验收。
