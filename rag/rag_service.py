
"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model, rerank_model
from utils.config_handler import chroma_config



def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagSummarizeService(object):
    def __init__(self):
        self.vector_store = VectorStoreService()
        
        # 1. 向量检索器 (Semantic Search)
        self.vector_retriever = self.vector_store.get_retriever()
        
        # 2. 关键字检索器 (Lexical Search)
        all_docs = self.vector_store.get_all_documents()
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = chroma_config["k"]  # 同样召回 config 中指定的候选
        
        # 3. 混合检索器 (Hybrid Search)
        # 使用 RRF (Reciprocal Rank Fusion) 算法融合两个检索器的结果
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.5, 0.5]
        )
        
        # 4. 重排序检索器 (Rerank)
        # 在混合检索的基础上进行精排
        self.retriever = ContextualCompressionRetriever(
            base_compressor=rerank_model,
            base_retriever=self.ensemble_retriever
        )
        
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def rag_summarize(self, query: str) -> dict:

        context_docs = self.retriever_docs(query)

        context = ""
        sources = []
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】: 参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
            sources.append({
                "id": counter,
                "source": doc.metadata.get("source", "未知来源"),
                "page": doc.metadata.get("page", "N/A")
            })

        answer = self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
        
        return {
            "answer": answer,
            "sources": sources
        }


if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪些扫地机器人"))
