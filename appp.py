
import streamlit as st
import pandas as pd
import joblib

# =========================
# Configuration de la page
# =========================
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# =========================
# Style CSS
# =========================
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        background-color: #f5f7fa;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
    .risk-high {
        color: #c0392b;
        font-size: 24px;
        font-weight: bold;
    }
    .risk-low {
        color: #27ae60;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Chargement des fichiers
# =========================
@st.cache_resource
def load_files():
    saved = joblib.load("best_churn_model_with_scalerr.pkl")

    model = saved["model"]
    scaler = saved["scaler"]
    feature_columns = joblib.load("feature_columns.pkl")

    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_files()
except FileNotFoundError:
    st.error("Erreur : Vérifiez que best_churn_model_with_scalerr.pkl et feature_columns.pkl sont dans le même dossier que appp.py.")
    st.stop()
# =========================
# Titre
# =========================
st.markdown("<div class='main-title'>🏦 Bank Customer Churn Prediction App</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Cette application prédit si un client risque de quitter la banque à partir de ses informations.</div>",
    unsafe_allow_html=True
)

# =========================
# Sidebar
# =========================
st.sidebar.header("📌 Informations du client")
client_name = st.sidebar.text_input("Nom du client", "sohaib bakkali")

# =========================
# Inputs
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    credit_score = st.number_input(
        "Score de crédit",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Pays",
        ["France", "Germany", "Spain"]
    )

    gender_label = st.selectbox(
        "Genre",
        ["Homme", "Femme"]
    )

with col2:
    age = st.number_input(
        "Âge",
        min_value=18,
        max_value=100,
        value=40
    )

    tenure = st.number_input(
        "Ancienneté",
        min_value=0,
        max_value=10,
        value=3
    )

    balance = st.number_input(
        "Solde bancaire",
        min_value=0.0,
        max_value=300000.0,
        value=60000.0
    )

with col3:
    num_of_products = st.selectbox(
        "Nombre de produits bancaires",
        [1, 2, 3, 4]
    )

    has_cr_card_label = st.selectbox(
        "Possède une carte de crédit",
        ["Oui", "Non"]
    )

    is_active_member_label = st.selectbox(
        "Client actif",
        ["Oui", "Non"]
    )

estimated_salary = st.number_input(
    "Salaire estimé",
    min_value=0.0,
    max_value=250000.0,
    value=50000.0
)

# =========================
# Transformation des inputs
# =========================
gender = "Male" if gender_label == "Homme" else "Female"
has_cr_card = 1 if has_cr_card_label == "Oui" else 0
is_active_member = 1 if is_active_member_label == "Oui" else 0

input_data = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_of_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active_member,
    "EstimatedSalary": estimated_salary
}])

# =========================
# Bouton de prédiction
# =========================
st.markdown("---")

if st.button("🔍 Prédire le risque de churn", use_container_width=True):

    data_encoded = pd.get_dummies(input_data, drop_first=True)
    data_encoded = data_encoded.reindex(columns=feature_columns, fill_value=0)

    data_scaled = scaler.transform(data_encoded)

    prediction = model.predict(data_scaled)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(data_scaled)[0][1]
    else:
        probability = None

    st.subheader("📊 Résultat de la prédiction")

    col_res1, col_res2 = st.columns(2)

    with col_res1:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)
        st.write(f"**Nom du client :** {client_name}")
        st.write(f"**Pays :** {geography}")
        st.write(f"**Genre :** {gender_label}")
        st.write(f"**Âge :** {age}")
        st.write(f"**Score de crédit :** {credit_score}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res2:
        st.markdown("<div class='result-box'>", unsafe_allow_html=True)

        if prediction == 1:
            st.markdown("<p class='risk-high'>⚠️ Client susceptible de quitter la banque</p>", unsafe_allow_html=True)
            st.warning("Ce client présente un risque élevé de churn.")
        else:
            st.markdown("<p class='risk-low'>✅ Client susceptible de rester</p>", unsafe_allow_html=True)
            st.success("Ce client présente un faible risque de churn.")

        st.write(f"**Classe prédite :** {prediction}")

        if probability is not None:
            st.write(f"**Probabilité de churn :** {probability:.2%}")
            st.progress(float(probability))

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🧾 Données utilisées pour la prédiction")
    st.dataframe(input_data, use_container_width=True)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption("Projet Machine Learning - Bank Customer Churn Prediction")
