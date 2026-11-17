# 代码与资料来源说明

本项目是在课程示例代码基础上开展的个人学习与改造项目，不将课程基础功能表述为全部原创。

## 待补充的来源

原课程名称、作者、原仓库地址以及原代码可公开范围仍待项目持有人确认。当前没有足够信息为整份衍生代码指定 MIT、Apache-2.0 等许可证；本说明不授予未确认的第三方代码或资料的再分发权。

## 公开候选版本的数据范围

`scripts/prepare_release.py` 默认不复制以下本地材料：

- 原课程 FAQ CSV、PDF、Markdown、图片与分类训练数据；
- 原始问答、参考答案、评分提示与课程资料摘录等评测明细；
- 模型权重、私有配置、日志、会话数据库及 Docker 数据目录。

公开候选目录使用 `examples/` 下本轮自编的软件演示资料。原本地资料保持原样。排除原资料不等于已经解决衍生源码的许可问题；上游来源与适用许可的记录仍待补充。

## 依赖与模型

第三方 Python 包和模型分别适用其上游许可证。模型权重不随源码分发；完整模式需要自行准备 BGE-M3、BGE Reranker、BERT 基础模型及已训练的查询分类器。Qwen 通过用户自行配置的 API 调用。

## 检索到的来源线索（尚未确认直接来源）

- [Happy-Chen-CH/Educational_RAG_System](https://github.com/Happy-Chen-CH/Educational_RAG_System)：包含同名 FAQ 文件及相近的 BERT、BGE-M3、MySQL/Milvus 模块；尚未确认是本地代码的直接上游。
- [spongebobgit/Hybrid-Retrieval-QA-System](https://github.com/spongebobgit/Hybrid-Retrieval-QA-System)：包含同名 FAQ 文件与相近目录结构；同名文件或另一个仓库的许可证不能单独证明本项目来源。

这些链接仅供项目持有人核对，不据此补造作者、授权关系或整库许可证。当前 GitHub 仓库已公开，课程原始资料与模型权重不随源码分发。
