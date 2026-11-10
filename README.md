# VLA 面试练习室

面向 VLA / 具身智能算法岗位的本机面试练习工具。内置 84 条原始问答、168 道选择与填空题、20 道 Python/PyTorch 代码题，支持图解、论文原图、错题复习、收藏及学习进度备份。

## 启动

需要 Python 3.11 或更新版本、Node.js 和 npm。代码判题还需要安装并启动 Docker。首次安装需要网络。

```bash
./scripts/build-runner.sh  # 首次运行，构建代码测试镜像
./start.sh
```

在同一台电脑上访问 <http://127.0.0.1:8765>。Docker 未就绪时仍可使用普通题库与代码编辑器。运行记录保存在 `.local/`，可在「学习统计」导出备份。

可导入 Markdown、TXT、PDF、DOCX 或结构化 JSON；普通文档作为来源材料入库。配置 Ollama 或兼容模型接口后可生成新题，人工审核后入库。详见 [使用说明](README_使用说明.md)。

## 项目来源

本项目基于 [docs-to-quiz](https://github.com/iqbal-rahmatullah/docs-to-quiz) 扩展，复用了文档解析、文本切分、BM25 检索和模型访问能力，并新增了练习界面、个人题库、学习记录和隔离代码判题。上游项目以 MIT 许可证发布，其原始许可见 [LICENSE](LICENSE)；复用说明及第三方资料见 [THIRD_PARTY.md](THIRD_PARTY.md)。
