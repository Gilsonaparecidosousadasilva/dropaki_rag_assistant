import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    BSHTMLLoader,
    JSONLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
DB_DIR = BASE_DIR / "chroma_db"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

LOADERS_POR_EXTENSAO = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".md": UnstructuredMarkdownLoader,
    ".markdown": UnstructuredMarkdownLoader,
    ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
    ".json": lambda p: JSONLoader(p, jq_schema=".", text_content=False),
}


def carregar_arquivo_unico(file_path: str):
    """Identifica a extensão do arquivo e aplica o loader correto."""
    ext = Path(file_path).suffix.lower()
    loader_factory = LOADERS_POR_EXTENSAO.get(ext)

    if loader_factory is None:
        return []

    try:
        loader = loader_factory(file_path)
        print(f"📖 Carregando: {file_path}")
        docs = loader.load()
        if not docs:
            print(f"⚠️  Nenhum conteúdo extraído de: {file_path}")
        return docs
    except Exception as e:
        print(f"❌ Erro ao carregar {file_path}: {e}")
        return []


def executar_ingestao():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ ERRO: Chave 'GOOGLE_API_KEY' não encontrada no arquivo .env!")
        print(f"👉 Verifique o arquivo em: {ENV_PATH}")
        sys.exit(1)

    documentos = []
    arquivos_processados = 0
    for root, _, files in os.walk(str(DOCS_DIR)):
        for file in files:
            file_path = os.path.join(root, file)
            docs = carregar_arquivo_unico(file_path)
            if docs:
                arquivos_processados += 1
            documentos.extend(docs)

    if not documentos:
        print(f"\n⚠️ Nenhum arquivo válido foi encontrado dentro de '{DOCS_DIR}'.")
        print("👉 Confirme se os arquivos suportados estão dentro da pasta 'docs'.")
        return

    print(f"\n✅ Arquivos processados com sucesso: {arquivos_processados}")
    print(f"✅ Total de documentos/páginas lidos: {len(documentos)}")

    print("✂️ Fragmentando textos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"🧩 Total de chunks gerados: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ ERRO: O divisor de texto não gerou nenhum chunk.")
        print("👉 Verifique se os arquivos possuem texto selecionável (ex: PDF não escaneado).")
        return

    print("🧠 Gerando Embeddings e salvando no ChromaDB...")
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )

        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(DB_DIR),
        )
    except Exception as e:
        print(f"❌ ERRO durante geração de embeddings ou gravação no ChromaDB: {e}")
        print("👉 Verifique se a GOOGLE_API_KEY é válida e se você não excedeu a cota da API.")
        sys.exit(1)

    print(f"\n🎉 Ingestão concluída com sucesso! Banco vetorial gerado em: {DB_DIR}")


if __name__ == "__main__":
    executar_ingestao()