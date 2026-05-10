import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from matplotlib.patches import Patch

# Chargement du model
with open("heart_disease_model.pkl", "rb") as f:
    classifier = pickle.load(f)
#Importation du dataset
df = pd.read_csv('heartdisease.csv')
#Titre de la page
st.set_page_config(page_title='Heart Alert', page_icon='❤️', layout="wide")

# Personalisation du style et du visuel
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    [data-testid="stSidebar"] * { color: white !important; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 16px;
        border-left: 4px solid #a32d2d;
        margin-bottom: 8px;
    }
    .result-danger {
        background: #fff0f0;
        border: 1px solid #a32d2d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #a32d2d;
    }
    .result-success {
        background: #f0fff4;
        border: 1px solid #639922;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #3b6d11;
    }
</style>
""", unsafe_allow_html=True)

# Séparation de l'écran
with st.sidebar:
    st.markdown("## ❤️ Heart Alert")
    st.caption("Classification ML — Random Forest")
    st.divider()
    rubrique = st.radio("Navigation", [
        "📊 Données",
        "🔍 Prédiction",
        "🌲 Modèle"
    ])
    st.divider()
    st.caption(f"Dataset : {len(df)} patients · {df.shape[1]-1} variables")



# PAGE 1 — DONNÉES
if rubrique == "📊 Données":
    st.title("Exploration des données")
    st.markdown("Aperçu statistique du dataset Heart Disease")
    st.divider()

    # Métriques
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total patients", len(df))
    col2.metric("Cas à risque", int(df['target'].sum()))
    col3.metric("Cas sains", int((df['target'] == 0).sum()))
    col4.metric("Variables", df.shape[1] - 1)

    st.divider()
    c1, c2 = st.columns(2)

    # Graphe 1 : Distribution âge
    with c1:
        st.subheader("Distribution de l'âge par classe")
        fig, ax = plt.subplots(figsize=(5, 3))
        bins = list(range(20, 85, 5))
        ax.hist(df[df['target'] == 0]['age'], bins=bins,
                color='#639922', alpha=0.7, label='Sain')
        ax.hist(df[df['target'] == 1]['age'], bins=bins,
                color='#a32d2d', alpha=0.7, label='Malade')
        ax.set_xlabel("Âge")
        ax.set_ylabel("Nombre de patients")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graphe 2 : Camembert
    with c2:
        st.subheader("Répartition des classes")
        fig, ax = plt.subplots(figsize=(5, 3))
        vals = df['target'].value_counts()
        ax.pie(vals, labels=["Malade", "Sain"],
               colors=['#a32d2d', '#639922'],
               autopct='%1.1f%%', startangle=90,
               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graphe 3 : Scatter cholestérol vs FC
    st.subheader("Cholestérol vs Fréquence cardiaque maximale")
    fig, ax = plt.subplots(figsize=(9, 3.5))
    colors = df['target'].map({0: '#639922', 1: '#a32d2d'})
    ax.scatter(df['cholesterol'], df['max heart rate'],
               c=colors, alpha=0.5, s=20)
    ax.set_xlabel("Cholestérol (mg/dl)")
    ax.set_ylabel("FC maximale (bpm)")
    ax.legend(handles=[
        Patch(color='#a32d2d', label='Malade'),
        Patch(color='#639922', label='Sain')
    ])
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()



# PAGE 2 — PRÉDICTION
elif rubrique == "🔍 Prédiction":
    st.title("Prédiction du risque cardiaque")
    st.markdown("Renseignez les informations médicales du patient.")
    st.divider()

    # Ligne 1
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Âge", min_value=1, max_value=120, value=30)
    with col2:
        sex = st.selectbox("Sexe", options=[1, 0],
                           format_func=lambda x: "Homme" if x == 1 else "Femme")
    with col3:
        chest_pain_type = st.selectbox(
            "Type de douleur thoracique", options=[1, 2, 3, 4],
            format_func=lambda x: {
                1: "Angine typique", 2: "Angine atypique",
                3: "Douleur non angineuse", 4: "Asymptomatique"}[x])

    # Ligne 2
    col4, col5, col6 = st.columns(3)
    with col4:
        resting_bp_s = st.number_input("Pression artérielle (mmHg)",
                                        min_value=50, max_value=250, value=120)
    with col5:
        cholesterol = st.number_input("Cholestérol (mg/dl)",
                                       min_value=20, max_value=1000, value=200)
    with col6:
        fasting_blood_sugar = st.selectbox(
            "Glycémie à jeun > 120 mg/dl", options=[1, 0],
            format_func=lambda x: "Oui" if x == 1 else "Non")

    # Ligne 3
    col7, col8, col9 = st.columns(3)
    with col7:
        resting_ecg = st.selectbox(
            "ECG au repos", options=[0, 1, 2],
            format_func=lambda x: {
                0: "Normal", 1: "Anomalie ST-T",
                2: "Hypertrophie ventriculaire"}[x])
    with col8:
        max_heart_rate = st.number_input("FC maximale (bpm)",
                                          min_value=50, max_value=400, value=150)
    with col9:
        exercise_angina = st.selectbox(
            "Angine à l'effort", options=[1, 0],
            format_func=lambda x: "Oui" if x == 1 else "Non")

    # Ligne 4
    col10, col11 = st.columns(2)
    with col10:
        oldpeak = st.number_input("Oldpeak (Dépression ST)",
                                   min_value=0.0, max_value=10.0,
                                   value=1.0, step=0.1)
    with col11:
        st_slope = st.selectbox(
            "Pente du segment ST", options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "Non renseigné", 1: "Montante",
                2: "Plate", 3: "Descendante"}[x])

    st.divider()

    # ── BOUTON PRÉDICTION ──
    if st.button("🔍 Analyser", use_container_width=True):

        input_data = pd.DataFrame({
            'age'                : [age],
            'sex'                : [sex],
            'chest pain type'    : [chest_pain_type],
            'resting bp s'       : [resting_bp_s],
            'cholesterol'        : [cholesterol],
            'fasting blood sugar': [fasting_blood_sugar],
            'resting ecg'        : [resting_ecg],
            'max heart rate'     : [max_heart_rate],
            'exercise angina'    : [exercise_angina],
            'oldpeak'            : [oldpeak],
            'ST slope'           : [st_slope]
        })

        prediction = classifier.predict(input_data)[0]

        # Résultat
        if prediction == 1:
            st.markdown('<div class="result-danger">⚠️ Risque de maladie cardiaque détecté</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-success">✅ Aucun risque détecté</div>',
                        unsafe_allow_html=True)

        # Graphique feature importance — visible dans les 2 cas
        st.divider()
        st.subheader("Facteurs les plus déterminants")
        feat_names = np.array([
            'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
            'fasting blood sugar', 'resting ecg', 'max heart rate',
            'exercise angina', 'oldpeak', 'ST slope'
        ])
        importances = classifier.feature_importances_
        idx = np.argsort(importances)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(feat_names[idx], importances[idx],
                       color='#a32d2d', edgecolor='white')
        ax.set_xlabel("Score d'importance")
        ax.set_title("Importance des variables — Random Forest")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()



# PAGE 3 — MODÈLE
elif rubrique == "🌲 Modèle":
    st.title("Caractéristiques du modèle")
    st.markdown("Performance et paramètres du Random Forest entraîné.")
    st.divider()

    # ── Paramètres ──
    st.subheader("⚙️ Paramètres du modèle")
    params = classifier.get_params()
    keys = ['n_estimators', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'max_features', 'random_state']
    p1, p2, p3 = st.columns(3)
    p1.metric("Nombre d'arbres", params.get('n_estimators', 'N/A'))
    p2.metric("Profondeur max", str(params.get('max_depth', 'None')))
    p3.metric("Min samples split", params.get('min_samples_split', 'N/A'))
    st.divider()

    # ── Préparation X_test / y_test ──
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)[:, 1]

    # ── Graphiques ──
    c1, c2 = st.columns(2)

    # Matrice de confusion
    with c1:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                    xticklabels=['Sain', 'Malade'],
                    yticklabels=['Sain', 'Malade'],
                    linewidths=0.5, ax=ax)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title("Matrice de confusion — Test")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Courbe ROC
    with c2:
        st.subheader("Courbe ROC")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='#a32d2d', lw=2,
                label=f'AUC = {roc_auc:.3f}')
        ax.fill_between(fpr, tpr, alpha=0.08, color='#a32d2d')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel("Taux de faux positifs")
        ax.set_ylabel("Taux de vrais positifs")
        ax.set_title("Courbe ROC — Test")
        ax.legend(loc='lower right')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Feature importance
    st.subheader("Importance des variables")
    feat_names = np.array([
        'age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
        'fasting blood sugar', 'resting ecg', 'max heart rate',
        'exercise angina', 'oldpeak', 'ST slope'
    ])
    importances = classifier.feature_importances_
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(feat_names[idx], importances[idx],
            color='#a32d2d', edgecolor='white')
    ax.set_xlabel("Score d'importance")
    ax.set_title("Contribution de chaque variable au modèle")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()