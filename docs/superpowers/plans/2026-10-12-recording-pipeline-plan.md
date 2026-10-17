# 录像生成与编码衔接优化 Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. The user has authorized inline execution and a reduction of at least 10% from the current exporter.

**Goal:** 在第一轮快速 PNG 导出的基础上，生成与编码衔接优化使同棋谱独立实测耗时再减少至少 10%。

**Architecture:** 两个独立页面并行截图相邻画面，按原帧序写入管道；读取与 PNG 解码各有一个线程，通过容量为 2 的两个 FIFO 向原 H.264 编码循环供帧。编码器继续独占时间戳、重复帧、封装和进度更新；退出时停止线程并终止本次拥有的渲染进程树。

**Tech Stack:** Python threading/queue、OpenCV、PyAV、既有 Node/Chrome 渲染器。

**Spec:** 当前对话的执行授权；视觉质量标准见 `docs/第一轮导出提速整改与验收.md`。

## Global Constraints

- 1080×1192、60 fps、H.264、CRF 18、preset fast、yuv420p；保持完整帧数、顺序、动画和初末局面。
- 沿用已确认的无可见画质损失标准及第一轮质量门槛，不减少动画采样帧。
- 仅优化生成并行度、衔接与必要生命周期处理；前端及截图参数保持不变。
- 有界缓冲，不保存整段原始画面；异常不可发布半成品；服务停止后不遗留本次渲染进程。
- 使用第一轮代码快照做本次对照，独立运行计时；主要性能计时不加入逐帧哈希计算。
- 至少三个基准/优化运行的中位数减少 ≥10%；同时核对相对原 54.234 秒的结果。
- 复用 5278/8278 服务；重启前验证进程身份、没有分析及导出任务。测试数据使用独立目录。

## Review Focus

1. 队列满时主线程失败，读取线程和渲染进程都必须结束。
2. 等待首帧时停止服务，原有取消与半成品清理行为保持。
3. 渲染或 PNG 解码异常必须传到导出任务。
4. 帧数组在异步交接后保持正确生命周期、顺序与重复次数。
5. EOF 后须排空已读取帧并完整刷新编码器。

## Task 1: 有界读取/解码与编码衔接

Files: `backend/app/recording.py`、`scripts/render-recording.cjs`、`backend/tests/test_recording_pipeline.py`、`output/playwright/recording-pipeline/test-renderer.cjs`。

Interfaces: `encode_recording(manifest_path, output_path, progress, render_url, on_process=None)` 签名与帧流 `{bytes,repeat}+PNG` 不变。

- [x] 写真实子进程/真实编码的预读、顺序与重复、解码异常、满队列退出测试。
- [x] 运行预读测试，确认旧循环因编码期间没有预读而失败。
- [x] 保存第一轮编码文件快照及计时脚本，完成本次基准计时。
- [x] 实现双页面有序截图、两个容量 2 的 FIFO、读取和解码线程、错误传播与可取消退出；共用本次渲染进程终止逻辑。
- [x] 运行上述测试及原有关闭服务回归。

## Task 2: 实测、质量与正常页面验收

Files: `output/playwright/recording-pipeline/*`、`docs/第二轮生成与编码衔接优化验收.md`。

- [x] 同一 24 手/25 秒/1500 帧棋谱，在无并行测试负载下运行三组第一轮快照与第二轮代码计时。
- [x] 对比分辨率、fps、codec、manifest、重复次数、总帧数与帧时间；运行已确认的全帧及棋子局部质量门控。
- [x] 跑完整后端、前端、构建及实际页面导出/刷新恢复验收。
- [x] 对线程退出、进程归属与队列错误路径做一次独立只读审查。
- [x] 修正实际问题，确认没有运行任务后加载最终后端，清理本次临时服务，保存 UTF-8 验收报告并读回。

Ruling: 工作目录不是 Git 仓库，使用代码快照与源文件 SHA-256 记录对照，直接在已授权工作目录执行，不创建 worktree 或提交。

