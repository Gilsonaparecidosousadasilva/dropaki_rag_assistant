# 🤖 DropAki – Assistente Virtual com RAG

Assistente virtual de suporte para a DropAki, capaz de responder perguntas de clientes sobre **envios e rastreio, trocas e reembolso, pagamentos e políticas da loja**, com respostas baseadas exclusivamente nos documentos internos da empresa.

O projeto usa a técnica de **RAG (Retrieval-Augmented Generation)**: em vez de o modelo de linguagem "inventar" respostas, ele busca os trechos mais relevantes em uma base de documentos e gera a resposta com base apenas nesse contexto — reduzindo alucinações e mantendo o suporte alinhado às políticas reais da loja.

---

**1. Ingestão (`ingest.py`)**
- Varre a pasta `docs/` e carrega qualquer arquivo suportado (PDF, DOCX, PPTX, Markdown, CSV, XLSX, HTML, JSON) usando o loader apropriado para cada extensão.
- Divide os documentos em pedaços menores (*chunks*) com `RecursiveCharacterTextSplitter` (1000 caracteres, 200 de sobreposição), para que o retriever encontre trechos relevantes com precisão.
- Gera embeddings de cada chunk com o modelo `gemini-embedding-001` do Google e salva tudo em um banco vetorial **ChromaDB**, persistido em disco na pasta `chroma_db/`.

**2. Consulta / Chat (`app.py`)**
- Interface web em **Streamlit**, com visual customizado (dark mode) e chips de perguntas rápidas.
- Ao receber uma pergunta, o **retriever** busca no ChromaDB os 4 chunks mais similares semanticamente à pergunta.
- Esses chunks são inseridos como contexto em um prompt e enviados ao modelo **Gemini** (`gemini-flash-latest`), que gera a resposta final.
- O histórico da conversa é mantido na sessão do Streamlit (`st.session_state`).

---

## 🛠️ Tecnologias e ferramentas

| Categoria | Ferramenta |
|---|---|
| Linguagem | Python |
| Interface web | [Streamlit](https://streamlit.io/) |
| Orquestração RAG | [LangChain](https://www.langchain.com/) (`langchain_classic`, `langchain_core`, `langchain_text_splitters`, `langchain_community`) |
| LLM e Embeddings | Google Gemini (`gemini-flash-latest` + `gemini-embedding-001`) via `langchain_google_genai` |
| Banco vetorial | [ChromaDB](https://www.trychroma.com/) (`langchain_chroma`) |
| Carregamento de documentos | `PyPDFLoader`, `Docx2txtLoader`, `UnstructuredPowerPointLoader`, `UnstructuredMarkdownLoader`, `CSVLoader`, `UnstructuredExcelLoader`, `BSHTMLLoader`, `JSONLoader` |
| Configuração | `python-dotenv` (variáveis de ambiente via `.env`) |

---

## ▶️ Como executar o projeto

### 1. Pré-requisitos
- Python 3.10+
- Uma chave de API do Google Gemini ([Google AI Studio](https://aistudio.google.com/))

### 2. Instalar as dependências

\`\`\`bash
pip install streamlit python-dotenv langchain langchain-classic langchain-core \
    langchain-community langchain-text-splitters langchain-google-genai langchain-chroma \
    pypdf docx2txt unstructured python-pptx openpyxl beautifulsoup4
\`\`\`

> Dependendo dos tipos de arquivo que você for indexar (PDF, DOCX, PPTX, XLSX, HTML, etc.), pode ser necessário instalar bibliotecas extras exigidas pelos respectivos loaders do `unstructured`.

### 3. Configurar a chave de API

Crie um arquivo `.env` na raiz do projeto:

\`\`\`env
GOOGLE_API_KEY=sua_chave_aqui
\`\`\`

### 4. Adicionar os documentos da base de conhecimento

Coloque os arquivos que o assistente deve consultar (políticas, FAQs, manuais etc.) dentro da pasta `docs/` na raiz do projeto. Formatos suportados: `.pdf`, `.docx`, `.pptx`, `.ppt`, `.md`, `.markdown`, `.csv`, `.xlsx`, `.xls`, `.html`, `.htm`, `.json`.

### 5. Gerar o banco vetorial

\`\`\`bash
python ingest.py
\`\`\`

Esse comando processa os documentos e cria a pasta `chroma_db/` com os embeddings.

### 6. Rodar a aplicação

\`\`\`bash
streamlit run app.py
\`\`\`

A interface abrirá automaticamente no navegador (geralmente em `http://localhost:8501`).

> ⚠️ Sempre que adicionar ou alterar documentos em `docs/`, é necessário rodar `python ingest.py` novamente para atualizar o banco vetorial.

---

## 💬 Exemplos de perguntas que o agente consegue responder

- "Como faço para rastrear meu pedido?"
- "Qual o prazo para solicitar uma troca ou reembolso?"
- "Quais formas de pagamento são aceitas?"
- "O produto que recebi veio com defeito, o que eu faço?"
- "Vocês entregam para todo o Brasil?"
- "Posso cancelar um pedido depois de confirmado o pagamento?"

---

## 📋 Exemplos de respostas geradas pelo agente

**Pergunta:** *Como faço para rastrear meu pedido?*
**Resposta:** *Após a confirmação do envio, você recebe um código de rastreio por e-mail. Com esse código, é possível acompanhar o status da entrega diretamente na seção "Meus Pedidos" do site ou no site dos Correios/transportadora responsável.*

**Pergunta:** *Qual o prazo para solicitar troca?*
**Resposta:** *Você tem até 7 dias corridos após o recebimento do produto para solicitar troca ou devolução, desde que o item esteja em sua embalagem original e sem sinais de uso.*

**Pergunta:** *Vocês entregam em Fernando de Noronha?*
**Resposta:** *Não encontrei essa informação nos documentos disponíveis. Recomendo entrar em contato com o suporte para confirmar a disponibilidade de entrega nessa região.*

> As respostas exatas dependem do conteúdo real inserido na pasta `docs/`, já que o agente responde **apenas** com base nos documentos indexados — os exemplos acima são ilustrativos.