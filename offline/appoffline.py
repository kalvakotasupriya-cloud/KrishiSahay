def run_offline():

    import streamlit as st
    import faiss
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image

    st.title("🌾 KrishiSahay - Offline Agriculture Assistant")

    st.markdown("### 🌱 Ask your agriculture questions")

    language = st.selectbox(
        "Select Answer Language",
        ["English", "Telugu"]
    )

    @st.cache_resource
    def load_embed_model():
        return SentenceTransformer(
            "all-MiniLM-L6-v2",
            local_files_only=True
        )

    @st.cache_resource
    def load_index():
        return faiss.read_index("offline/kcc_index.faiss")

    @st.cache_data
    def load_data():
        return pd.read_csv("offline/clean_kcc.csv")

    model = load_embed_model()
    index = load_index()
    df = load_data()

    query = st.text_input(
        "Type your question in English:"
    )

    if st.button("🔍 Get Answer"):

        if query.strip() == "":
            st.warning("Please enter a question.")
        else:

            query_embedding = model.encode([query])
            D, I = index.search(
                np.array(query_embedding),
                k=1
            )

            result = df.iloc[I[0][0]]
            full_answer = str(result["answer"])

            st.subheader("Answer")
            st.write(full_answer)

    st.markdown("---")
    st.subheader("🐛 Pest Detection")

    uploaded_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image)

        class_names = torch.load("offline/class_names.pth")

        model = models.mobilenet_v2(weights=None)

        model.classifier[1] = torch.nn.Linear(
            model.last_channel,
            len(class_names)
        )

        model.load_state_dict(
            torch.load(
                "offline/pest_model.pth",
                map_location="cpu"
            )
        )

        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        img = transform(image).unsqueeze(0)

        with torch.no_grad():

            outputs = model(img)

            _, pred = torch.max(outputs,1)

        label = class_names[pred.item()]

        st.success(label)