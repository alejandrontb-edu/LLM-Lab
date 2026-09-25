import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq

# Configuración inicial de la página
st.set_page_config(
    page_title="Plataforma NLP con Groq",
    page_icon="🤖",
    layout="wide"
)

# --- BARRA LATERAL: API KEY Y CONFIGURACIÓN ---
st.sidebar.header("🔑 Configuración")
api_key_input = st.sidebar.text_input(
    "Ingresa tu API Key de Groq:",
    type="password",
    help="Consigue tu API key en el portal de Groq."
)

if api_key_input:
    st.session_state["groq_api_key"] = api_key_input
    st.sidebar.success("¡API Key guardada en sesión!")
else:
    if "groq_api_key" not in st.session_state:
        st.sidebar.warning("Por favor ingresa tu API Key para usar la sección de generación.")

st.sidebar.markdown("---")
st.sidebar.info("Navega por las secciones principales desde la interfaz central.")

# --- TÍTULO PRINCIPAL ---
st.title("🚀 Plataforma Interactiva de NLP y Modelos Groq")
st.write("Explora tokenización, vectores de texto, similitudes semánticas y generación con modelos de lenguaje de alta velocidad.")

# --- PESTAÑAS DE LA APLICACIÓN ---
tab1, tab2, tab3, tab4 = st.tabs([
    "a) Tokenización y Colores",
    "b) Bag of Words",
    "c) Similitud de Coseno",
    "d) Generador Groq"
])

# --- SECCIÓN A: TOKENIZACIÓN Y COLORES ---
with tab1:
    st.header("a) Esquemas de Tokenización y Visualización")
    st.write("Divide el texto de entrada bajo diferentes criterios, asigna IDs y resalta visualmente cada token.")
    
    text_input_tok = st.text_area(
        "Texto de prueba para tokenizar:",
        "El procesamiento de lenguaje natural (NLP) con Groq es increíblemente rápido y eficiente.",
        key="tok_text"
    )
    
    scheme = st.selectbox(
        "Selecciona el esquema de tokenización:",
        ["Por Palabras", "Por Caracteres", "Por Oraciones"]
    )
    
    if text_input_tok:
        if scheme == "Por Palabras":
            tokens = text_input_tok.split()
        elif scheme == "Por Caracteres":
            tokens = list(text_input_tok)
        else:
            tokens = [s.strip() for s in text_input_tok.split('.') if s.strip()]
            
        # Generar representación en colores
        colors = ["#FFB3BA", "#BAFFC1", "#BAE1FF", "#FFFFBA", "#FFDFBA", "#E8BAFF", "#BFFCC6"]
        colored_html = "<div style='line-height: 2.5; font-size: 16px;'>"
        for i, token in enumerate(tokens):
            color = colors[i % len(colors)]
            colored_html += f"<span style='background-color: {color}; color: #000; padding: 6px 10px; margin: 3px; border-radius: 6px; display: inline-block; font-weight: 500;'>{token} <sub style='font-size:10px;'>id:{i}</sub></span>"
        colored_html += "</div>"
        
        st.markdown("### Tokens Visuales")
        st.markdown(colored_html, unsafe_allow_html=True)
        
        # Tabla de Tokens e IDs
        st.markdown("### Tabla de Datos de Tokens")
        df_tokens = pd.DataFrame([{"Token ID": i, "Token": t} for i, t in enumerate(tokens)])
        st.dataframe(df_tokens, use_container_width=True)

# --- SECCIÓN B: BAG OF WORDS ---
with tab2:
    st.header("b) Bag of Words (Bolsa de Palabras)")
    st.write("Convierte un corpus de frases en una matriz numérica de conteo de frecuencias.")
    
    bow_corpus_input = st.text_area(
        "Ingresa frases (una por línea):",
        "El gato come pescado.\nEl perro corre en el parque.\nEl gato y el perro son amigos.",
        key="bow_text"
    )
    
    docs = [line.strip() for line in bow_corpus_input.split('\n') if line.strip()]
    
    if docs:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        
        st.subheader("Vocabulario extraído:")
        st.write(list(vocab))
        
        st.subheader("Matriz BoW (Frecuencias):")
        df_bow = pd.DataFrame(X.toarray(), columns=vocab, index=[f"Frase {i+1}" for i in range(len(docs))])
        st.dataframe(df_bow, use_container_width=True)

# --- SECCIÓN C: SIMILITUD DE COSENO ---
with tab3:
    st.header("c) Similitud de Coseno entre Frases")
    st.write("Calcula la cercanía semántica y léxica entre dos enunciados utilizando vectores de conteo.")
    
    col1, col2 = st.columns(2)
    with col1:
        phrase_1 = st.text_input("Frase 1:", "La inteligencia artificial transforma la industria tecnológica.")
    with col2:
        phrase_2 = st.text_input("Frase 2:", "El machine learning y la IA están cambiando el sector tecnológico.")
        
    if st.button("Calcular Similitud de Coseno"):
        if phrase_1 and phrase_2:
            vectorizer_cos = CountVectorizer()
            vectors = vectorizer_cos.fit_transform([phrase_1, phrase_2])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            st.metric(label="Coeficiente de Similitud de Coseno", value=f"{similarity:.4f}")
            if similarity > 0.7:
                st.success("Alta similitud léxica/conceptual entre las frases.")
            elif similarity > 0.3:
                st.info("Similitud moderada.")
            else:
                st.warning("Baja similitud entre las frases.")
        else:
            st.error("Por favor completa ambas frases.")

# --- SECCIÓN D: GENERACIÓN CON GROQ ---
with tab4:
    st.header("d) Generación de Respuestas (Catálogo de Modelos Groq)")
    st.write("Interactúa con los modelos avanzados de Groq configurando hiperparámetros personalizados (Nota: No se utiliza GPT).")
    
    # Catálogo de modelos Groq actualizados
    groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    
    selected_model = st.selectbox("Selecciona un Modelo del Catálogo Groq:", groq_models)
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        temperature = st.slider("Temperatura (Creatividad)", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    with col_p2:
        max_tokens = st.slider("Max Tokens (Longitud de salida)", min_value=10, max_value=2048, value=512, step=10)
        
    prompt_text = st.text_area(
        "Ingresa tu Prompt:",
        "Explica en tres viñetas las ventajas de usar modelos open source optimizados en hardware especializado."
    )
    
    if st.button("Generar Respuesta con Groq"):
        current_api_key = st.session_state.get("groq_api_key", "")
        if not current_api_key:
            st.error("⚠️ Falta la API Key de Groq. Por favor ingrésala en la barra lateral izquierda.")
        else:
            try:
                client = Groq(api_key=current_api_key)
                with st.spinner(f"Generando respuesta con {selected_model}..."):
                    chat_completion = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": "user", "content": prompt_text}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    response_content = chat_completion.choices[0].message.content
                    
                st.subheader("Respuesta del Modelo:")
                st.markdown(response_content)
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la solicitud con Groq: {e}")
