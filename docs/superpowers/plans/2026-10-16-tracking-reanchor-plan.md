# 历史疑点隔离与画面重新同步 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 历史无法还原时保留人工审核区间，从独立识别的清晰画面恢复跟踪，后续逐步自动确认。

**Architecture:** 在现有实际视频流水线增加受限的当前画面恢复器；连续独立类型/占位证据确认后发出重新同步事件。审核记录保存单独的跟踪锚点，不把缺失历史或同步动作记成自动走棋。TurnReplay 结束旧段并开启新段，持久化和前端均能使用锚点。

**Tech Stack:** Python / Pydantic / OpenCV / 本地融合检测器；TypeScript / Vitest。

**Spec:** 本对话上一条已获用户授权的五步方案；实现约束补充如下。

## Global Constraints

- 在现有 experiments/fusion-recognition 分支原地完成；保留既有未提交修改，不创建丢失这些改动的新 checkout。
- 疑点不自动升为确认；同步前后分别保存局面，已确认历史不重写。
- 不改变活动训练权重；不放宽普通走棋确认；保留 15 秒/同一参考边界一次的局部回查上限。
- 同步由当前画面的独立类型识别和跨帧一致证据决定，不由超时决定。
- 正常逐帧路径不增加逐帧重模型调用；恢复调用限频且失败后退避。

## Review Focus

- 一帧误识别或手部遮挡不能建立锚点。
- 身份种类错配、同类内部编号不确定，不能无限污染后续自动确认。
- 初始轮到方错误不能使同步后的全部步骤变为人工审核。
- 保存、重载、重新校验和视频同步必须保留独立锚点，不伪造走棋。
- 旧片 44 步/15 步保持完整走法、身份、时间、悔棋回归；当前片扫描完整，回查保持有界。

### Task 1: 恢复器与证据门槛

Files: 新建 backend/app/tracking_recovery.py；新建 backend/tests/test_tracking_reanchor.py。

- [x] 测试错误身份、多子漏步、单帧噪声、未知格、限频/退避。
- [x] 运行失败测试。
- [x] 实现 VisualRecovery.request(time_s) 与 observe(time_s, image, reference, cells)，类型读取接口为 detector.readings(image)。至少两次相隔 1 秒的独立同类型证据；90 格覆盖允许相邻兼容帧互补，不把未知当空。
- [x] 测试通过。

### Task 2: 隔离历史并恢复自动确认

Files: backend/app/{actual,analysis,schemas,turns,board_evidence}.py；backend/tests/test_tracking_reanchor.py。

- [x] 测试审核区间封口后连续两步自动确认、轮到方重建、已确认历史保留、存储重载/校验、普通回查策略不变。
- [x] 运行失败测试。
- [x] VideoEvent 增加 tracking_anchor；MoveRecord 增加 tracking_anchor 和 tracking_anchor_s；TurnReplay.reanchor 封口并清空旧段临时状态，下一步独立推断行动方；review_base/revalidate_actual 识别锚点。
- [x] 主事件生成器支持独立恢复回调；消费者接受锚点后更新跟踪参考与证据，清理旧来源/历史缓存，保持正常确认逻辑。
- [x] 相关测试通过。

### Task 3: 前端与实际回归

Files: frontend/src/{types,replay}.ts；frontend/src/tracking-reanchor.test.ts；output/tracking-reanchor-20261016/。

- [x] 测试锚点不计步、视频时间对应新局面、后续自动确认独立可定位、历史缺口仍提示审核。
- [x] 实现并通过前端回归和构建。
- [x] 后端完整测试、旧片实际回归与当前片实际恢复验证，保存精确结果和限制；独立代码审查。

## Execution ledger

用户已明确要求落实上述方案，直接在当前会话执行。当前实验分支含大量已验证未提交改动，保留当前目录；不自动提交已有工作。

完成：2026-10-17T09:35:32.569112+08:00。后端515通过/2跳过；前端61通过；构建与独立审查通过。最终旧片44/15步强门禁通过；当前录屏故障注入78.870秒同步后18步自动确认；长视频47～2100秒完整扫描，交付保留89自动+1人工校正，18疑点待审核。
