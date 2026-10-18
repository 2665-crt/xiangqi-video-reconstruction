# 融合实验提升正式版执行计划

> **For agentic workers:** Use superpowers:executing-plans to implement this confirmed migration in the current session.

**Goal:** 将当前融合实验完整提升到根目录本地 main，保留正式端口、数据和回退点。

**Architecture:** 先提交原正式版，再复制实验工作区源码、测试和模型；只适配目录、启动环境和端口。

**Tech Stack:** Git、PowerShell、Python/FastAPI、React/Vite、现有 CNN 与 DeepXiangQi。

**Spec:** 本对话中用户确认的三步方案；`docs/正式版提升记录.md`。

## Global Constraints

不覆盖 `.env`、原任务数据库或视频；不修改实验副本；保留 `5278/8278`；不推送远程。

## Review Focus

- 进程身份检查匹配新的融合 Python 路径，且不停止其他项目。
- 原 78 个正式任务可由新 schema 读取，实验任务保持独立。
- 所有融合权重和新增颜色模型随代码提升，哈希与来源一致。
- 前端代理、后端重启、录像渲染 URL 继续使用正式端口。
- 新增脚本不沿用实验目录的上两级路径假设。

## Tasks

- [x] 保存正式版提交、回退标签、备份分支和数据库只读备份。
- [x] 复制实验源码、模型、测试、依赖及文档；适配正式脚本与文档。
- [x] 运行后端、前端测试和生产构建，检查失败的具体原因。
- [x] 验证正式启动、历史任务读取、页面交互和录像链路。
- [x] 核对源文件、实验副本与数据保留情况；记录验证结果并准备 main 提交内容。

## 验证与审查记录

后端540通过、前端61通过、build通过；正式启动、78个历史任务读取、零云端调用的本地推理、120帧实际录像和后端重启均通过。独立审查无阻塞。历史8290专项脚本标注为历史工具，未扩大为额外改造。最终提交完成后以 git log 和清洁工作区确认。
