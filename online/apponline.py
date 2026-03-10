def run_online():

    import streamlit as st
    import random
    from datetime import datetime

    from online.utils.translations import LANGUAGES, get_ui_string
    from online.utils.query_engine import answer_query
    from online.utils.schemes import get_recommended_schemes
    from online.utils.weather import get_weather

    st.title("🌾 Kisan Call Centre Assistant")

    st.write("Online AI assistant loaded successfully.")

    # ---------------- CHATBOT ----------------

    st.subheader("Ask your agriculture question")

    query = st.text_input("Enter your question")

    if st.button("Send"):

        if query.strip() == "":
            st.warning("Please enter a question")
        else:
            result = answer_query(query)
            st.write(result["answer"])

    # ---------------- WEATHER ----------------

    st.markdown("---")
    st.subheader("🌤️ Weather")

    weather = get_weather("Hyderabad")

    st.write(weather)

    # ---------------- GOVERNMENT SCHEMES ----------------

    st.markdown("---")
    st.subheader("📋 Government Schemes")

    schemes = get_recommended_schemes()

    for scheme in schemes[:3]:
        st.write("•", scheme["name"], "-", scheme["benefit"])

    # ---------------- DAILY TIP ----------------

    st.markdown("---")
    tips = [
        "Water crops early morning",
        "Monitor pests regularly",
        "Use soil testing before fertilizer",
    ]

    st.info(random.choice(tips))