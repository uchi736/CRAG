import os
import re
from typing import List, TypedDict, Dict, Any

# ✨ .envファイルから環境変数を読み込む
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --------------------------------------------------------
# 実行前の準備:
# 1. pip install python-dotenv langchain-google-genai langgraph langchain-community chromadb
# 2. このスクリプトと同じ階層に .env ファイルを作成し、
#    APIキーなどを記述してください。
# --------------------------------------------------------

# グラフの状態定義
class GraphState(TypedDict):
    question: str
    documents: List[str]
    recursion_depth: int
    max_recursions: int
    processed_targets: List[str]
    new_targets: List[str]
    answer: str

# LLMとベクトルストアの初期化
# ライブラリは環境変数から自動的にAPIキーを読み込みます
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# (以下、コードの他の部分は変更ありません)

# サンプルドキュメント
sample_documents = [
    Document(page_content="RAGシステムの基本的な仕組みについては、「RAG入門ガイド」を参照してください。", metadata={"source": "rag_overview.md"}),
    Document(page_content="RAG入門ガイド：RAGとは検索拡張生成の略で、LLMの回答精度を向上させる手法です。詳細はhttps://example.com/rag-guideをご覧ください。", metadata={"source": "RAG入門ガイド"}),
    Document(page_content="https://example.com/rag-guideの内容：RAGは大規模言語モデルに外部知識を統合する強力な手法です。ベクトルデータベースと組み合わせることで高精度な回答が可能になります。", metadata={"source": "https://example.com/vector-db-benchmark"}),
    Document(page_content="ベクトルデータベースの選定については「ベクトルDB比較表」を参照してください。", metadata={"source": "vector_db_intro.md"}),
    Document(page_content="ベクトルDB比較表：Chroma、Pinecone、Weaviateなどの主要なベクトルDBの比較。性能詳細はhttps://example.com/vector-db-benchmarkを参照。", metadata={"source": "ベクトルDB比較表"}),
]

# ベクトルストアの作成
vectorstore = Chroma.from_documents(documents=sample_documents, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Recursive Triggerロジック
def extract_new_targets(texts: List[str], processed_targets: List[str]) -> List[str]:
    url_pattern = r'https?://[^\s/$.?#].[^\s]*'
    doc_pattern = r'「(.*?)」を参照'
    new_targets = []
    for text in texts:
        found_urls = re.findall(url_pattern, text)
        for url in found_urls:
            if url not in processed_targets:
                new_targets.append(url)
        found_docs = re.findall(doc_pattern, text)
        for doc in found_docs:
            if doc not in processed_targets:
                new_targets.append(doc)
    return list(set(new_targets))

# ノード関数の定義
def retrieve_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n[Retrieve] 再帰深度: {state['recursion_depth']}")
    if state['recursion_depth'] == 0:
        query = state['question']
    else:
        query = state['new_targets'][0] if state.get('new_targets') else state['question']
    print(f"検索クエリ: {query}")
    docs = retriever.get_relevant_documents(query)
    retrieved_texts = [doc.page_content for doc in docs]
    print(f"取得したドキュメント数: {len(retrieved_texts)}")
    updated_documents = state.get('documents', []) + retrieved_texts
    return {"documents": updated_documents, "recursion_depth": state['recursion_depth'] + 1}

def generate_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n[Generate] ドキュメント数: {len(state['documents'])}")
    prompt = ChatPromptTemplate.from_template("以下の情報を基に質問に回答してください。\n\n質問: {question}\n\n利用可能な情報:\n{documents}\n\n回答:")
    documents_text = "\n\n".join(state['documents'][:5])
    chain = prompt | llm
    response = chain.invoke({"question": state['question'], "documents": documents_text})
    all_texts = state['documents'] + [response.content]
    new_targets = extract_new_targets(all_texts, state.get('processed_targets', []))
    print(f"抽出された新しいターゲット: {new_targets}")
    updated_processed = state.get('processed_targets', []) + new_targets
    return {"new_targets": new_targets, "processed_targets": updated_processed}

def final_generate_node(state: GraphState) -> Dict[str, Any]:
    print(f"\n[Final Generate] 総ドキュメント数: {len(state['documents'])}")
    prompt = ChatPromptTemplate.from_template("以下の全ての情報を統合して、質問に対する包括的な回答を生成してください。\n\n質問: {question}\n\n収集した全情報:\n{all_documents}\n\n最終回答:")
    all_documents_text = "\n\n".join(state['documents'])
    chain = prompt | llm
    response = chain.invoke({"question": state['question'], "all_documents": all_documents_text})
    return {"answer": response.content}

def should_continue(state: GraphState) -> str:
    print(f"\n[Condition Check] 再帰深度: {state['recursion_depth']}/{state['max_recursions']}")
    if not state.get('new_targets'):
        print("-> 新しいターゲットなし。終了。")
        return "end"
    if state['recursion_depth'] >= state['max_recursions']:
        print("-> 最大再帰深度に到達。終了。")
        return "end"
    all_processed = all(target in state.get('processed_targets', []) for target in state['new_targets'])
    if all_processed:
        print("-> 全てのターゲットが処理済み。終了。")
        return "end"
    print("-> 再帰を継続。")
    return "continue"

# グラフの構築
def build_graph():
    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("retrieve", retrieve_node)
    graph_builder.add_node("generate", generate_node)
    graph_builder.add_node("final_generate", final_generate_node)
    graph_builder.set_entry_point("retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_conditional_edges("generate", should_continue, {"continue": "retrieve", "end": "final_generate"})
    graph_builder.add_edge("final_generate", END)
    return graph_builder.compile()

# メイン実行関数
def run_recursive_rag(question: str, max_recursions: int = 2):
    print(f"=== 再帰的RAG開始 ===")
    print(f"質問: {question}")
    print(f"最大再帰深度: {max_recursions}")
    graph = build_graph()
    initial_state = {
        "question": question, "documents": [], "recursion_depth": 0,
        "max_recursions": max_recursions, "processed_targets": [],
        "new_targets": [], "answer": ""
    }
    
    trace_config = {
        "configurable": {
            "run_name": f"Recursive RAG - {question[:20]}..."
        }
    }
    
    result = graph.invoke(initial_state, config=trace_config)
    
    print(f"\n=== 最終回答 ===")
    print(result["answer"])
    return result

# 使用例
if __name__ == "__main__":
    question = "RAGシステムの仕組みとベクトルデータベースの選定について教えてください。"
    result = run_recursive_rag(question, max_recursions=3)