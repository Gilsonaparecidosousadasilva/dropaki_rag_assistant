import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Import do Chroma
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# 1. Configuração de caminhos base
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
DB_DIR = BASE_DIR / "chroma_db"

# Load do arquivo .env
load_dotenv(dotenv_path=ENV_PATH)

# Configuração da página Streamlit
st.set_page_config(
    page_title="DropAki | Assistente Virtual",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Paleta: laranja (energia/ação, comum em e-commerce) + azul-marinho (confiança) */
    :root {
        --dropaki-primary: #9D4EDD;
        --dropaki-primary-dark: #7B2FBF;
        --dropaki-dark: #121212;
        --dropaki-light: #F5F7FA;
        --dropaki-gray: #9CA3AF;
    }

    /* Fundo geral */
    .stApp {
        background: linear-gradient(180deg, #121212 0%, #1E1E24 100%);
    }

    /* Texto padrão em fundo escuro */
    .stApp, .stApp p, .stApp label, .stApp span, .stApp h1,
    .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp div[data-testid="stMarkdownContainer"] {
        color: #FFFFFF;
    }

    /* Esconde o menu/rodapé padrão do Streamlit */
    #MainMenu, footer {visibility: hidden;}

    /* Cabeçalho customizado */
    .dropaki-header {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 18px 22px;
        background: var(--dropaki-dark);
        border-radius: 14px;
        margin-bottom: 22px;
        box-shadow: 0 4px 14px rgba(26, 35, 50, 0.25);
    }
    .dropaki-header .icon {
        font-size: 34px;
        background: var(--dropaki-primary);
        border-radius: 10px;
        padding: 8px 12px;
    }
    .dropaki-header .title {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
    }
    .dropaki-header .subtitle {
        color: #B8C2D0;
        font-size: 13.5px;
        margin: 2px 0 0 0;
    }

    /* Bolhas de chat */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 4px 6px;
        margin-bottom: 6px;
        border: 1px solid rgba(0,0,0,0.05);
    }

    /* Mensagens do usuário */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: #2A2A35;
        border-color: rgba(157,78,221,0.4);
    }

    /* Mensagens do assistente */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #23232C;
        border-color: rgba(255,255,255,0.08);
    }

    /* Campo de input do chat */
    div[data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
    }
    div[data-testid="stChatInput"] {
        border: 1.5px solid var(--dropaki-primary) !important;
        border-radius: 14px !important;
    }

    /* Botões */
    .stButton > button {
        background-color: var(--dropaki-primary);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: var(--dropaki-primary-dark);
        color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--dropaki-dark);
    }
    section[data-testid="stSidebar"] * {
        color: #E7ECF3 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Chips de sugestão de pergunta */
    .dropaki-chip {
        display: inline-block;
        background: #2A2A35;
        color: var(--dropaki-primary);
        border: 1px solid var(--dropaki-primary);
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px 6px 4px 0;
        font-size: 13px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 2. Carregamento seguro da API Key (evita o StreamlitSecretNotFoundError)
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("❌ A chave 'GOOGLE_API_KEY' não foi encontrada! Verifique se o arquivo .env existe na raiz do projeto.")
    st.stop()

# 3. Inicialização do Banco Vetorial e da Chain RAG
@st.cache_resource
def carregar_rag():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    # Carrega o banco vetorial existente
    vectorstore = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=api_key,
        temperature=0.3
    )

    # Prompt do sistema
    system_prompt = (
        "Você é um assistente especialista encarregado de responder perguntas sobre os documentos fornecidos.\n"
        "Use os seguintes trechos de contexto recuperados para responder à pergunta.\n"
        "Se você não souber a resposta, diga claramente que não encontrou essa informação nos documentos.\n"
        "Mantenha a resposta clara, objetiva e baseada apenas no contexto fornecido.\n\n"
        "Contexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 📦 DropAki")
    st.caption("Assistente virtual de suporte")
    st.markdown("---")
    st.markdown("**Posso ajudar com:**")
    st.markdown("- 🚚 Envios e rastreio\n- 🔄 Trocas e reembolso\n- 💳 Pagamentos\n- 📋 Políticas da loja")
    st.markdown("---")
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# CABEÇALHO
# =========================================================
st.markdown("""
<div class="dropaki-header">
    <div class="icon">🤖</div>
    <div>
        <p class="title">Assistente Virtual DropAki</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Garante que o banco vetorial exista
if not DB_DIR.exists():
    st.warning("⚠️ O banco de dados vetorial ainda não foi gerado!")
    st.info("Execute primeiro o arquivo `python ingest.py` no terminal para processar seus arquivos antes de fazer perguntas.")
    st.stop()

try:
    rag_chain = carregar_rag()
except Exception as e:
    st.error(f"❌ Erro ao inicializar o modelo ou carregar o banco vetorial: {e}")
    st.stop()

# Histórico da conversa na sessão
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sugestões de pergunta (aparecem só se ainda não houver conversa)
if not st.session_state.messages:
    st.markdown("**Perguntas rápidas:**")
    col1, col2, col3 = st.columns(3)
    sugestao_clicada = None
    with col1:
        if st.button("📦 Como rastreio meu pedido?", use_container_width=True):
            sugestao_clicada = "Como rastreio meu pedido?"
    with col2:
        if st.button("🔄 Como funciona a troca?", use_container_width=True):
            sugestao_clicada = "Como funciona o processo de troca?"
    with col3:
        if st.button("💳 Quais formas de pagamento?", use_container_width=True):
            sugestao_clicada = "Quais formas de pagamento vocês aceitam?"
else:
    sugestao_clicada = None

# Exibe mensagens antigas no chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input do usuário (via campo de texto ou clique em sugestão)
user_input = st.chat_input("Digite sua pergunta sobre seu pedido, troca ou pagamento...")
if sugestao_clicada:
    user_input = sugestao_clicada

if user_input:
    # Mostra a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Gera a resposta do Agente
    with st.chat_message("assistant"):
        with st.spinner("Buscando informações nos documentos..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response.get("answer", "Não foi possível gerar uma resposta.")
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Erro ao processar a resposta: {e}")

    if sugestao_clicada:
        st.rerun()