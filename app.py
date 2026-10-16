import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from google import genai
import numpy as np
import re


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# TITLE / LANDING PAGE
# ============================================================

st.title("📚 AI Research Assistant using RAG")

st.markdown(
    "Upload a research paper and ask questions about its content."
)

st.write("### Project Details")

st.write(
    "**Project Topic:** AI Research Assistant using RAG"
)

st.write(
    "**Full Name:** Aysha hanna k"
)

st.write(
    "**Registered Email ID:** ayshahanna900@gmail.com"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📚 Project Menu")

st.sidebar.write(
    "Retrieval-Augmented Generation Research Assistant"
)

st.sidebar.divider()

st.sidebar.header("🔑 AI Configuration")

api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password"
)

client = None

if api_key:
    client = genai.Client(
        api_key=api_key
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# SESSION STATE
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0


# ============================================================
# CLEAR CHAT
# ============================================================

if st.sidebar.button("🗑️ Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload Research Paper")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    st.session_state.document_name = uploaded_file.name

    reader = PdfReader(
        uploaded_file
    )

    st.session_state.page_count = len(
        reader.pages
    )

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"


    st.success(
        "PDF uploaded successfully! ✅"
    )


    # --------------------------------------------------------
    # DOCUMENT INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📄 Pages",
            st.session_state.page_count
        )

    with col2:

        st.metric(
            "📝 Characters",
            len(text)
        )

    with col3:

        st.metric(
            "✂️ Chunks",
            0
        )


    st.write(
        f"**Document:** {st.session_state.document_name}"
    )


    # --------------------------------------------------------
    # TEXT CHUNKING
    # --------------------------------------------------------

    st.header("✂️ Text Processing")

    chunk_size = 1200
    overlap = 200

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap


    st.write(
        f"**Text divided into {len(chunks)} chunks.**"
    )


    # --------------------------------------------------------
    # BUILD KNOWLEDGE BASE
    # --------------------------------------------------------

    if st.button(
        "🧠 Build Knowledge Base",
        type="primary"
    ):

        with st.spinner(
            "Creating document embeddings..."
        ):

            embeddings = embedding_model.encode(
                chunks,
                normalize_embeddings=True
            )


        st.session_state.chunks = chunks

        st.session_state.embeddings = embeddings

        st.success(
            "Knowledge base created successfully! ✅"
        )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    chunks,
    embeddings,
    top_k=8
):

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )

    semantic_scores = cosine_similarity(
        question_embedding,
        embeddings
    )[0]


    # --------------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------------

    question_words = set(
        re.findall(
            r"\b[a-zA-Z]{3,}\b",
            question.lower()
        )
    )

    keyword_scores = []

    for chunk in chunks:

        chunk_words = re.findall(
            r"\b[a-zA-Z]{3,}\b",
            chunk.lower()
        )

        word_counts = Counter(
            chunk_words
        )

        score = 0

        for word in question_words:

            if word in word_counts:

                score += 1

        keyword_scores.append(score)


    keyword_scores = np.array(
        keyword_scores,
        dtype=float
    )


    if keyword_scores.max() > 0:

        keyword_scores = (
            keyword_scores /
            keyword_scores.max()
        )


    # --------------------------------------------------------
    # COMBINE SCORES
    # --------------------------------------------------------

    final_scores = (
        0.7 * semantic_scores
        +
        0.3 * keyword_scores
    )


    # --------------------------------------------------------
    # GET TOP RESULTS
    # --------------------------------------------------------

    top_indices = np.argsort(
        final_scores
    )[::-1][:top_k]


    results = []

    for index in top_indices:

        results.append(
            {
                "index": index,
                "chunk": chunks[index],
                "score": final_scores[index]
            }
        )


    return results


# ============================================================
# RESEARCH ASSISTANT
# ============================================================

if st.session_state.embeddings is not None:

    st.divider()

    st.header("💬 Research Assistant")

    st.write(
        "Ask questions about your uploaded research paper."
    )


    # --------------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------------

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask a question about your research paper..."
    )


    if question:

        # ----------------------------------------------------
        # DISPLAY USER QUESTION
        # ----------------------------------------------------

        with st.chat_message("user"):

            st.markdown(question)


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        # ----------------------------------------------------
        # CHECK API KEY
        # ----------------------------------------------------

        if client is None:

            answer = (
                "⚠️ Please enter your Gemini API key "
                "in the sidebar."
            )

            sources = []


        else:

            # ------------------------------------------------
            # RETRIEVE RELEVANT CHUNKS
            # ------------------------------------------------

            results = hybrid_search(
                question,
                st.session_state.chunks,
                st.session_state.embeddings,
                top_k=8
            )


            # ------------------------------------------------
            # BUILD CONTEXT
            # ------------------------------------------------

            context_parts = []

            for i, result in enumerate(results):

                context_parts.append(
                    f"""
SOURCE {i + 1}

{result["chunk"]}
"""
                )


            context = "\n".join(
                context_parts
            )


            # ------------------------------------------------
            # RAG PROMPT
            # ------------------------------------------------

            prompt = f"""
You are an AI Research Assistant.

Answer the user's question using the provided
sections from the uploaded research paper.

IMPORTANT RULES:

1. Read all provided sources carefully.
2. Use information from the research paper.
3. Combine information from multiple sources
   when necessary.
4. Do not invent facts.
5. Do not use outside knowledge.
6. If the answer cannot be found in the
   provided sources, clearly say so.

RESEARCH PAPER SOURCES:

{context}

USER QUESTION:

{question}

Give a clear, accurate and concise answer.
"""


            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            try:

                with st.spinner(
                    "🔎 Searching the paper and generating answer..."
                ):

                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt
                    )


                answer = response.text

                sources = results


            except Exception as e:

                answer = (
                    "⚠️ An error occurred:\n\n"
                    f"{e}"
                )

                sources = []


        # ----------------------------------------------------
        # DISPLAY ANSWER
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)


        # ----------------------------------------------------
        # SAVE ANSWER
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


        # ----------------------------------------------------
        # DISPLAY SOURCES
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "🔎 View Retrieved Sources"
            ):

                for i, result in enumerate(
                    sources
                ):

                    st.write(
                        f"### Source {i + 1}"
                    )

                    st.write(
                        f"**Retrieval Score:** "
                        f"{result['score']:.3f}"
                    )

                    st.write(
                        result["chunk"]
                    )

                    st.divider()


# ============================================================
# INSTRUCTIONS WHEN NO KNOWLEDGE BASE EXISTS
# ============================================================

else:

    st.info(
        "👆 Upload a PDF and click "
        "'Build Knowledge Base' to start."
    )


    st.markdown(
        """
### 🔄 How the RAG system works

**1. 📄 Upload PDF**  
Upload your research paper.

**2. 📖 Text Extraction**  
Text is extracted from the PDF.

**3. ✂️ Text Chunking**  
The document is divided into smaller sections.

**4. 🧠 Embeddings**  
The sections are converted into numerical vectors.

**5. 🔎 Hybrid Retrieval**  
Semantic and keyword search find relevant sections.

**6. 🤖 AI Generation**  
Gemini generates an answer using the retrieved
research-paper content.

**7. 📚 Sources**  
The retrieved sections are shown to the user.
"""
    )