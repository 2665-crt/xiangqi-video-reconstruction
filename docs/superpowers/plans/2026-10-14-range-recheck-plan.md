# 区域重检与插入走棋 Implementation Plan

> 使用 executing-plans 在当前会话按任务执行。用户已经确认设计并明确要求开始整改，直接实施。

**Goal:** 在融合识别实验版实现可点选边界的逐帧重检及中间手动插入。
**Architecture:** 独立 recheck 模块维护后台运行与候选序列，原棋谱在应用前保留。复用逐帧解码、局部占位检测、回合追踪和校正传播。
**Tech Stack:** FastAPI/Pydantic/SQLite/OpenCV/PyAV、React/TypeScript/Vitest/Playwright。
**Spec:** ../specs/2026-10-14-range-recheck-design.md

## Global Constraints
- 仅修改实验版；保留当前未提交工作；不修改原任务或凭证；端口保持 5288/8288。
- 使用本地模型，无付费模型调用；完整过程缺少证据时不能显示已补齐。
- 任务进度留在指定程序区域；对话显示整改阶段进度。

## Review Focus
- 倒序及同一步选择、动作时间缺失或边界重叠：排序明确，拒绝不可信范围。
- 中间插入后旧索引与人工编辑：后缀与关联索引保留、重算。
- 部分恢复或遮挡：候选保留但不冒充完整。
- 后台重检与编辑冲突、停止/重启：版本保护，不覆盖原谱。
- 末端同局面但中间存在回退：保留过程疑点，不能仅凭端点一致确认完整。

### Task 1: 棋谱插入与候选替换
- [x] 编写插入、边界、重编号、人工校正、区间外保留的失败测试。
- [x] 实现 schemas.py、turns.py 的插入及 recheck.py 的范围/替换契约。
- [x] 定向运行及回归。

### Task 2: 逐帧重检后台与 API
- [x] 编写真实编码视频恢复两手、部分恢复、端点冲突、路由版本保护测试。
- [x] 实现 recheck.py、main.py，后台进度持久化、取消、候选应用、历史留存。
- [x] 运行后端完整回归。

### Task 3: 页面交互与进度布局
- [x] 编写倒序点选、插入请求及进度位置的行为测试。
- [x] 修改 App.tsx、MoveList.tsx、ReviewPanel.tsx、types.ts、api.ts、styles.css；新增 RecheckPanel.tsx。
- [x] 前端测试、构建、真实浏览器测试。

### Task 4: 本地运行验收
- [x] 在任务副本上走真实 API/UI，检查刷新、回放、录制播放列表及指定布局对齐。
- [x] 复核修改，记录可验证结果与仍需用户视频核对的限制。

不创建新工作树：当前运行目录含既有未提交修复且需原位交付；备份本次涉及的文件到 output/range-recheck-validation/source-before。不自动提交或推送。
