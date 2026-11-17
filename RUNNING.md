# 本地运行

源码版本为 1.2.1。应用入口是根目录 `app.py`，启动后通过 `/health` 核对实际版本与模式，通过 `/ready` 检查依赖。

## 轻量演示

新环境推荐 Python 3.12；使用独立虚拟环境：

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-demo.txt
$env:EDURAG_MODE = "demo"
.\venv\Scripts\python.exe app.py
```

已有环境可用 `.\run_demo.ps1 -NoBrowser` 后台启动。默认端口为 8088，端口被占用时脚本会显示新的访问地址。

源码候选使用 `examples/faq_sample.csv` 的 8 条自编题答；本地课程 CSV 存在时优先使用。显式指定示例：

```powershell
$env:EDURAG_FAQ_CSV = "examples/faq_sample.csv"
```

演示模式不调用模型，历史存入 `data/runtime/demo_sessions.db`，每个会话保留最近五轮。

## 完整环境

`requirements.txt` 记录本机完整模型环境；不是已验证的全新安装方案。先自行准备完整依赖、Docker 及模型文件。轻量演示和离线开发依赖分别记录在 `requirements-demo.txt` 和 `requirements-dev.txt`。

`config.example.ini` 复制为 `config.ini` 后填写：
- MySQL 与 Redis 的连接信息，需与 Docker Compose 环境一致。
- Milvus 连接信息及目标集合。
- 通义千问兼容接口地址、模型名和自己的 API Key。

不要上传 `config.ini`。新机器的模型目录需要：
- `rag_qa/models/bge-m3`
- `rag_qa/models/bge-reranker-large`
- `rag_qa/models/bert-base-chinese`
- `rag_qa/models/bert_query_classifier`（已训练查询分类器）

分类器权重不包含在源码候选中；原始训练数据来源仍待核实，不提供未经确认的再分发。原脚本的训练入口保留，但在自行准备数据和完成训练前，不能宣称新环境完整模式开箱即用。

Docker Compose 定义 MySQL、Redis、Milvus、etcd 和 MinIO。首次没有索引时，使用独立集合导入自编样例：

```powershell
.\venv\Scripts\python.exe scripts/build_index.py --data-dir examples/knowledge --collection edurag_examples_v1
```

先确保 Milvus 等服务已启动；入库后设置 `config.ini` 中 `milvus.collection_name = edurag_examples_v1`。本脚本不会自动清理已删除文件的旧块，不要将原资料与示例结果混为同一评测版本。

启动已有索引的完整模式：

```powershell
.\run_full.ps1 -Port 8090 -NoBrowser -SkipIndex
```

脚本会尝试启动 Docker Desktop 和依赖容器。若本机已有完整的 `edurag-*` 容器组，核对服务标签和项目一致性后复用原容器，不替换数据挂载。挂载仍可能指向原项目目录，请保留该目录的数据。容器组不完整或归属不一致时停止并提示检查。

也可直接运行应用（依赖须已就绪）：

```powershell
$env:EDURAG_MODE = "full"
$env:HOST = "127.0.0.1"
$env:PORT = "8090"
.\venv\Scripts\python.exe app.py
```

不要把 `auto` 回退到演示模式当作完整 RAG 启动成功。网页成功加载也不代表模型接口可用。

## 接口与验收

- `POST /api/query`：默认标准问答直接返回，生成路径提示前端连接 WebSocket；设置 `stream: false` 等待完整答案。
- `WS /api/stream`：接收 JSON，返回 `start/token/end/error`；途中失败报告错误。
- `GET /api/history/{session_id}`：读取最近五轮。
- `POST /api/retrieve`：完整模式下对比检索方式，不执行回答生成。
- `GET /api/sources`：当前可选范围，`ai`、`java` 与 `Python学科`、`JAVA学科` 是不同资料范围。

```powershell
.\venv\Scripts\python.exe scripts/check_demo.py
.\venv\Scripts\python.exe scripts/check_routing.py --base-url http://127.0.0.1:8090
```

第二条会调用真实模型。历史课程题组相关的 `check_full.py`、`check_round2.py` 和生成评测脚本需要原本地资料，不适合作为自编样例的开箱验收。

日志为 `server*.log` 与 `logs/app.log`。不删除 `.runtime` 中的数据库卷。当前缺少用户登录与会话归属鉴权，默认仅监听本机。
