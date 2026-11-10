# VLA 面试练习室前端

中文 React 18 / Next.js 14 静态应用。正式运行由项目根目录的 FastAPI 服务同源提供 `web/out/`；所有浏览器请求为 `/api/*`，不会读取或展示模型密钥。

```bash
npm ci
npm test
npm run typecheck
npm run build
```

`npm run build` 将 Monaco 的完整 AMD 编辑器与 worker 资源复制到 `public/monaco/vs/`，再导出到 `out/`。运行时不依赖 Monaco CDN。不要使用 `next start` 运行静态导出应用。

本地开发可用 `npm run dev`，请求代理至 `http://127.0.0.1:8765`；后端地址可通过 `VLA_API_BASE` 调整。正式构建不写入代理规则。

界面包括真实进度概览、章节/题型/掌握状态筛选、单多选与填空、错题收藏、原文公式与可放大图解、本地代码草稿/测试、导入预览、AI 草稿审核、题目完整 JSON 编辑及版本保存、统计与备份恢复。未连接后端、未配置 AI、隔离运行器未就绪、测试失败和超时均显示实际状态。

前端测试保护旧题目版本不会被算作掌握、错题按最近作答更新、多重筛选、选项打乱时稳定 ID 和 API 验证错误展示。浏览器端综合验收由根项目进行。
