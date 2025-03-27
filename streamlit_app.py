import streamlit as st
import pandas as pd
import faiss
import json
import ollama
import os
from sentence_transformers import SentenceTransformer

@st.cache_resource
def load_encoder():
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource
def build_index_from_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)

    if "name" not in df.columns or "description" not in df.columns:
        st.error("❌ CSV phải có 2 cột: name, description")
        return None, None

    encoder = load_encoder()
    embeddings = encoder.encode(df['description'].tolist(), convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return df, index

# --- Bước 1: Trích sản phẩm từ câu hỏi bằng Mistral ---
def extract_products_mistral(user_input):
    prompt = (
        "Bạn là trợ lý kỹ thuật nói tiếng Việt. Phân tích đoạn sau và trích xuất danh sách sản phẩm dưới dạng JSON. "
        "Mỗi sản phẩm gồm: tên sản phẩm, số lượng (nếu có), kích thước (nếu có).\n"
        f"Đoạn văn: {user_input}\n"
        "Trả kết quả JSON đơn thuần, không giải thích."
    )

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý kỹ thuật chuyên phân tích sản phẩm."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response['message']['content']
    try:
        json_start = content.find("[")
        json_end = content.rfind("]") + 1
        return json.loads(content[json_start:json_end])
    except Exception as e:
        st.warning("Không phân tích được JSON từ mô hình.")
        st.text(content)
        return []

# --- Bước 2: Tìm kiếm sản phẩm ---
def search_by_product_names(product_list, df, index, encoder, top_k=2):
    results = []
    for p in product_list:
        name = p.get("tên sản phẩm", "")
        if not name:
            continue
        q_vector = encoder.encode([name])
        D, I = index.search(q_vector, top_k)
        found = df.iloc[I[0]].to_dict(orient="records")
        results.extend(found)
    return results

# --- Bước 3: Gọi lại Mistral để trả lời ---
def chatbot_reply_with_context(user_input, product_info_list):
    product_text = "\n".join([f"- {p['name']}: {p['description']}" for p in product_info_list])
    final_prompt = (
        f"Người dùng hỏi: \"{user_input}\"\n\n"
        f"Dưới đây là các sản phẩm liên quan tìm được:\n{product_text}\n\n"
        "Hãy trả lời người dùng bằng tiếng Việt, rõ ràng, gợi ý sản phẩm phù hợp."
    )

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "Bạn là trợ lý kỹ thuật chuyên tư vấn sản phẩm, nói tiếng Việt."},
            {"role": "user", "content": final_prompt}
        ]
    )

    return response['message']['content']


# --- Giao diện Streamlit ---
st.title("🛠️ Chatbot Tư vấn Sản phẩm (Upload CSV + Mistral + FAISS)")

uploaded_file = st.file_uploader("📂 Tải lên file sản phẩm (.csv)", type=["csv"])
user_input = st.text_area("💬 Nhập câu hỏi về sản phẩm:", height=100)

if uploaded_file:
    df, index = build_index_from_csv(uploaded_file)
    encoder = load_encoder()

    if st.button("🚀 Gửi câu hỏi") and user_input:
        with st.spinner("🤖 Đang phân tích..."):
            products = extract_products_mistral(user_input)
            matches = search_by_product_names(products, df, index, encoder)
            answer = chatbot_reply_with_context(user_input, matches)

        st.subheader("📋 Danh sách sản phẩm phân tích:")
        st.json(products)

        st.subheader("🔍 Kết quả tìm kiếm:")
        for item in matches:
            st.markdown(f"- **{item['name']}**: {item['description']}")

        st.subheader("🧠 Trợ lý phản hồi:")
        st.markdown(answer)
else:
    st.info("⬆️ Vui lòng tải lên file `product.csv` trước khi hỏi.")
