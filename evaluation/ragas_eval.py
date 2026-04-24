from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from model.factory import chat_model, embed_model
from rag.rag_service import RagSummarizeService
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 1. 准备评估器
# Ragas 需要一个 LLM 和一个 Embedding 模型来作为评估者
# 这里我们直接复用项目中已经配置好的 DashScope 模型
evaluator_llm = LangchainLLMWrapper(chat_model)
evaluator_embeddings = LangchainEmbeddingsWrapper(embed_model)

# 2. 准备测试数据 (黄金数据集)
test_questions = [
    {
        "question": "小户型适合哪种扫地机器人？",
        "ground_truth": "小户型适合基础激光导航机型，如米家 1C、石头 T4。"
    },
    {
        "question": "有宠物的家庭在选购扫地机器人时应该注意什么？",
        "ground_truth": "有宠物的家庭需强化吸力和防缠绕功能。"
    },
    {
        "question": "扫地机器人的吸力建议选择多少 Pa 以上？",
        "ground_truth": "家用建议选择≥3000Pa吸力机型，地毯场景需≥4000Pa。"
    }
]

def run_evaluation():
    rag_service = RagSummarizeService()
    
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("正在生成 RAG 回答以进行评估...")
    for item in test_questions:
        question = item["question"]
        print(f"处理问题: {question}")
        
        # 获取 RAG 的回答
        answer = rag_service.rag_summarize(question)
        
        # 获取检索到的上下文
        docs = rag_service.retriever_docs(question)
        # Ragas 期望 contexts 是一个字符串列表
        contexts = [doc.page_content for doc in docs]
        
        data["question"].append(question)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])
    
    # 转换为 datasets.Dataset 格式
    dataset = Dataset.from_dict(data)
    
    # 3. 执行评估
    print("\n开始 Ragas 评估 (使用 DashScope 作为评委)...")
    # 这步会调用 LLM 对回答进行打分，耗时取决于问题数量和模型响应速度
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,       # 忠实度：回答是否源自上下文
            answer_relevancy,   # 回答相关性：回答是否解决了问题
            context_precision,  # 检索精度：检索到的文档是否真的相关
            context_recall,     # 检索召回率：正确答案是否在检索到的文档中
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    
    # 4. 输出结果
    print("\n" + "="*50)
    print("RAG 系统自动化评估报告")
    print("="*50)
    
    df = result.to_pandas()
    # 打印详细得分
    print("\n详细得分表:")
    print(df)
    
    print("\n总体平均分:")
    print(result)
    
    print("\n" + "="*50)
    print("面试提示：")
    print("1. Faithfulness 高说明系统不胡编乱造。")
    print("2. Answer Relevancy 高说明回答切题。")
    print("3. Context Precision/Recall 高说明 RAG 检索部分（混合检索+重排序）非常给力。")
    print("="*50)

if __name__ == "__main__":
    run_evaluation()
