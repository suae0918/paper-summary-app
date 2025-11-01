import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI

# ✅ 새로운 OpenAI 클라이언트 방식
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🧠 논문 자동 요약 & 주제 분석기")

uploaded_file = st.file_uploader("PDF 논문 업로드", type=["pdf"])

if uploaded_file:
    pdf = PdfReader(uploaded_file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()

    st.subheader("📄 논문 내용 추출")
    st.write(text[:1000] + "...")

    if st.button("요약 및 분석 실행"):
        with st.spinner("요약 및 LDA 주제 분석 중..."):

            # ✅ 요약 프롬프트
            prompt = f"""
            다음 논문 텍스트를 요약하고 주요 주제(분야)를 분석해줘.
            형식:
            - 요약:
            - 주요 주제 비율:
            - 핵심 문장:
            
            논문 내용:
            {text[:6000]}
            """

            # ✅ 새로운 호출 방식
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )

            summary = response.choices[0].message.content
            st.subheader("🪶 GPT 요약 결과")
            st.write(summary)

