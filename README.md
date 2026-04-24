# 扫地机器人智能客服 Agent 系统 (Cleaner-Customer-Service-Agent)

这是一个基于 **LangChain** 和 **ReAct** 架构开发的扫地机器人智能客服系统。它集成了 **RAG (检索增强生成)**、**混合检索**、**重排序** 以及 **外部 API 感知** 等先进技术，旨在提供工业级的智能问答与个性化服务体验。

## 🌟 核心特性

- **智能 Agent 决策**：采用 ReAct 架构，支持自主思考、工具调用与多轮对话。
- **工业级 RAG 流水线**：
  - **混合检索**：结合语义向量检索与 BM25 关键词检索，通过 RRF 融合提升召回精度。
  - **重排序 (Reranking)**：集成 DashScope GTE-Rerank 模型进行二次精排。
  - **引用溯源**：回答自动标注来源 [n]，并列出参考文档及页码，抑制大模型幻觉。
- **实时环境感知**：集成高德地图 API，根据用户实时位置与天气/湿度提供维护建议。
- **自动化评估**：引入 Ragas 框架，从忠实度、相关性等维度量化评估系统表现。
- **现代化 UI**：基于 Streamlit 构建，支持流式输出与交互式对话。

## 🛠️ 技术栈

- **LLM**: DashScope (Qwen-Max)
- **Framework**: LangChain, LangGraph
- **Vector DB**: ChromaDB
- **Evaluation**: Ragas
- **Frontend**: Streamlit
- **API**: Amap (高德地图)

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. 安装依赖
建议使用虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量
在项目根目录创建 `.env` 文件，并填入您的 API Key：
```env
DASHSCOPE_API_KEY=您的通义千问API_KEY
AMAP_API_KEY=您的高德地图API_KEY
```

### 4. 准备数据
将您的知识库文档（PDF/TXT）放入 `data/` 目录。

### 5. 启动应用
```bash
streamlit run app.py
```

## 📊 评估系统
运行自动化评估脚本：
```bash
python evaluation/ragas_eval.py
```

## 📄 开源协议
MIT License
