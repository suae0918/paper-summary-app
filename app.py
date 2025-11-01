import streamlit as st
import openai
import fitz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

st.title("🧠 논문 자동 요약 & 주제 분석기")

# ✅ Streamlit Cloud Secrets에서 불러오기
openai.api_key = st.secrets["OPENAI_API_KEY"]

uploaded_file = st.file_uploader("📄 논문 PDF 업로드", type=["pdf"])

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    st.subheader("📘 PDF 텍스트 미리보기")
    st.write(text[:2000] + "...")

    vectorizer = CountVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform([text])
    lda = LatentDirichletAllocation(n_components=3, random_state=42)
    lda.fit(X)
    topics = lda.components_

    st.subheader("🧩 LDA 주제 비율")
    for idx, topic in enumerate(topics):
        st.write(f"Topic {idx+1}: {round(topic.sum() / topics.sum() * 100, 2)}%")

    st.subheader("🪶 GPT 요약 결과")
    prompt = f"다음 논문 내용을 간결하게 요약해줘:\n\n{text[:6000]}"
    with st.spinner("요약 중..."):
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
    st.write(response.choices[0].message.content)
