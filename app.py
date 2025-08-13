import streamlit as st
import os
import re
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# .envファイルから環境変数を読み込む
load_dotenv()

# ページ設定を最初に行う（ワイド表示、テーマ設定）
st.set_page_config(
    page_title="Azure OpenAI Chat",
    page_icon="🤖",
    layout="wide",  # ワイド表示で読みやすく
    initial_sidebar_state="expanded"
)

# カスタムCSS - 表示を大幅に改善
st.markdown("""
<style>
/* 全体のフォント設定 */
.stApp {
    font-family: 'Noto Sans JP', 'Helvetica Neue', Arial, sans-serif;
}

/* チャットメッセージのスタイル改善 */
.stChatMessage {
    padding: 1.5rem;
    margin-bottom: 1rem;
    border-radius: 10px;
}

/* コードブロックのスタイル改善 */
.stCodeBlock {
    margin: 1rem 0;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* コード内のフォント */
.stCodeBlock pre {
    font-family: 'Monaco', 'Consolas', 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.6 !important;
    padding: 1rem !important;
}

/* マークダウンのスタイル改善 */
.stMarkdown {
    line-height: 1.8;
}

/* 見出しのスタイル */
.stMarkdown h1 {
    font-size: 2rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e0e0e0;
}

.stMarkdown h2 {
    font-size: 1.5rem;
    margin-top: 1.5rem;
    margin-bottom: 0.8rem;
    color: #333;
}

.stMarkdown h3 {
    font-size: 1.2rem;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
    color: #555;
}

/* リストのスタイル改善 */
.stMarkdown ul, .stMarkdown ol {
    margin-left: 2rem;
    margin-bottom: 1rem;
}

.stMarkdown li {
    margin-bottom: 0.5rem;
    line-height: 1.8;
}

/* インラインコードのスタイル */
.stMarkdown code:not(.hljs) {
    background-color: #f0f0f0;
    padding: 0.2rem 0.4rem;
    border-radius: 3px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 0.9em;
    color: #d14;
}

/* 段落の間隔 */
.stMarkdown p {
    margin-bottom: 1rem;
    line-height: 1.8;
}

/* 引用のスタイル */
.stMarkdown blockquote {
    border-left: 4px solid #ddd;
    padding-left: 1rem;
    margin-left: 0;
    color: #666;
    font-style: italic;
}

/* テーブルのスタイル */
.stMarkdown table {
    border-collapse: collapse;
    width: 100%;
    margin-bottom: 1rem;
}

.stMarkdown th, .stMarkdown td {
    border: 1px solid #ddd;
    padding: 0.5rem;
    text-align: left;
}

.stMarkdown th {
    background-color: #f5f5f5;
    font-weight: bold;
}

/* スピナーのスタイル */
.stSpinner > div {
    color: #1f77b4;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Azure OpenAI Chat")
st.caption("by LangChain & Streamlit - 改善版")

# サイドバーに情報を表示
with st.sidebar:
    st.header("📋 設定情報")
    
    # フォーマット設定
    st.subheader("表示設定")
    code_theme = st.selectbox(
        "コードテーマ",
        ["monokai", "solarized-dark", "solarized-light", "default"],
        index=0
    )
    
    show_raw = st.checkbox("元のメッセージも表示", value=False)
    
    st.divider()
    
    # 使い方のヒント
    st.subheader("💡 使い方のヒント")
    st.markdown("""
    - コードブロックは自動的に検出されます
    - Markdownフォーマットに対応
    - 太字は見出しに自動変換
    """)

# --- 改善されたコード表示関数 ---
def display_message_enhanced(message_content):
    """
    大幅に改善されたメッセージ表示関数
    """
    # オリジナルを表示するオプション
    if show_raw:
        with st.expander("元のメッセージを表示"):
            st.text(message_content)
    
    # コンテンツの前処理
    content = preprocess_content(message_content)
    
    # セクションごとに分割して処理
    sections = split_into_sections(content)
    
    for section in sections:
        if section['type'] == 'code':
            display_code_section(section['content'], section.get('language', 'python'))
        elif section['type'] == 'heading':
            st.markdown(f"## {section['content']}")
        elif section['type'] == 'list':
            display_list_section(section['content'])
        else:
            display_text_section(section['content'])

def preprocess_content(content):
    """
    コンテンツの前処理 - 改行とフォーマットの修正
    """
    # 基本的な改行の修復
    content = re.sub(r'([.!?。！？])\s*(?=[A-Z])', r'\1\n\n', content)
    
    # コードパターンの前で改行
    content = re.sub(r'(\w+\s*=\s*[^=\n]+)(?=[a-zA-Z_])', r'\1\n', content)
    content = re.sub(r'(print\([^)]+\))(?=[a-zA-Z_])', r'\1\n', content)
    content = re.sub(r'(\])(?=\s*print)', r'\1\n', content)
    
    # 数字付きリストの改行
    content = re.sub(r'(\d+\.\s+[^.]+\.)(?=\s*\d+\.)', r'\1\n', content)
    
    # **太字**を見出しに変換（改善版）
    content = re.sub(r'\*\*([^*]+)\*\*', r'\n\n## \1\n\n', content)
    
    return content

def split_into_sections(content):
    """
    コンテンツをセクションに分割
    """
    sections = []
    
    # コードブロックを先に抽出
    code_block_pattern = r'```(\w*)\n?(.*?)```'
    
    parts = re.split(code_block_pattern, content, flags=re.DOTALL)
    
    i = 0
    while i < len(parts):
        if i % 3 == 0:  # テキスト部分
            if parts[i].strip():
                # テキスト内のさらなる分析
                text_sections = analyze_text_content(parts[i])
                sections.extend(text_sections)
        elif i % 3 == 2:  # コード部分
            language = parts[i-1] if parts[i-1] else 'python'
            sections.append({
                'type': 'code',
                'content': parts[i].strip(),
                'language': language
            })
        i += 1
    
    return sections

def analyze_text_content(text):
    """
    テキストコンテンツを分析してセクションに分割
    """
    sections = []
    lines = text.split('\n')
    current_section = []
    current_type = 'text'
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if current_section:
                sections.append({
                    'type': current_type,
                    'content': '\n'.join(current_section)
                })
                current_section = []
                current_type = 'text'
            continue
        
        # 見出しの検出
        if line.startswith('##'):
            if current_section:
                sections.append({
                    'type': current_type,
                    'content': '\n'.join(current_section)
                })
                current_section = []
            sections.append({
                'type': 'heading',
                'content': line.replace('##', '').strip()
            })
            current_type = 'text'
        
        # リストの検出
        elif re.match(r'^(\d+\.|-|\*)\s+', line):
            if current_type != 'list':
                if current_section:
                    sections.append({
                        'type': current_type,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                current_type = 'list'
            current_section.append(line)
        
        # インラインコードを含む行
        elif detect_inline_code(line):
            if current_section:
                sections.append({
                    'type': current_type,
                    'content': '\n'.join(current_section)
                })
                current_section = []
            sections.append({
                'type': 'code',
                'content': line,
                'language': 'python'
            })
            current_type = 'text'
        
        else:
            if current_type != 'text':
                if current_section:
                    sections.append({
                        'type': current_type,
                        'content': '\n'.join(current_section)
                    })
                    current_section = []
                current_type = 'text'
            current_section.append(line)
    
    # 最後のセクション
    if current_section:
        sections.append({
            'type': current_type,
            'content': '\n'.join(current_section)
        })
    
    return sections

def detect_inline_code(line):
    """
    インラインコードの検出
    """
    code_patterns = [
        r'^\s*(print\s*\([^)]+\))',
        r'^\s*(\w+\s*=\s*[^=]+)$',
        r'^\s*(for\s+\w+\s+in\s+[^:]+:)',
        r'^\s*(if\s+[^:]+:)',
        r'^\s*(def\s+\w+\s*\([^)]*\):)',
        r'^\s*(import\s+\w+)',
        r'^\s*(from\s+\w+\s+import\s+\w+)'
    ]
    
    return any(re.match(pattern, line) for pattern in code_patterns)

def display_code_section(code, language='python'):
    """
    コードセクションの表示
    """
    # コードのインデント修正
    lines = code.split('\n')
    formatted_lines = []
    indent_level = 0
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # インデントレベルの調整
        if stripped.endswith(':'):
            formatted_lines.append('    ' * indent_level + stripped)
            indent_level += 1
        elif stripped in ['else:', 'elif', 'except:', 'finally:']:
            indent_level = max(0, indent_level - 1)
            formatted_lines.append('    ' * indent_level + stripped)
            indent_level += 1
        elif stripped == 'pass' or stripped.startswith('return'):
            formatted_lines.append('    ' * indent_level + stripped)
            indent_level = max(0, indent_level - 1)
        else:
            formatted_lines.append('    ' * indent_level + stripped)
    
    formatted_code = '\n'.join(formatted_lines)
    st.code(formatted_code, language=language)

def display_list_section(content):
    """
    リストセクションの表示
    """
    st.markdown(content)

def display_text_section(content):
    """
    テキストセクションの表示
    """
    # 段落ごとに分割
    paragraphs = content.split('\n\n')
    for para in paragraphs:
        if para.strip():
            st.markdown(para.strip())

# --- 環境変数の読み込みとチェック ---
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-endpoint.openai.azure.com/")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name")

if not all([azure_endpoint, api_key, api_version, deployment_name]):
    st.error("必要な環境変数が設定されていません。")
    st.stop()

# --- Azure OpenAIモデルの初期化 ---
try:
    llm = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        azure_deployment=deployment_name,
        api_version=api_version,
        max_completion_tokens=2000,
    )
except Exception as e:
    st.error(f"モデルの初期化に失敗しました: {e}")
    st.stop()

# --- Streamlit UI の設定 ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        AIMessage(content="こんにちは！どのようなご用件でしょうか？")
    ]

# メッセージ履歴の表示
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            display_message_enhanced(message.content)

# ユーザー入力
if prompt := st.chat_input("メッセージを入力してください。"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                response = llm.invoke(st.session_state.messages)
                display_message_enhanced(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")