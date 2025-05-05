
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
from skimage.feature import local_binary_pattern, greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler
from skimage import io, color
import os

# Load pre-trained models
skin_type_model = joblib.load("models/skin_type_model.pkl")
acne_model = joblib.load("models/acne_model.pkl")
wrinkle_model = joblib.load("models/wrinkle_model.pkl")
scaler = joblib.load("models/scaler.pkl")
pca = joblib.load("models/pca.pkl")

def extract_features(image_path):
    img = Image.open(image_path).resize((128, 128)).convert("RGB")
    img_np = np.array(img)

    # Color histogram
    color_hist = []
    for i in range(3):
        hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        color_hist.extend(hist)

    # LBP
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    # Haralick features
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    glcm = greycomatrix(gray, [1], [0], symmetric=True, normed=True)
    haralick = [greycoprops(glcm, prop).flatten()[0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]

    return np.array(color_hist + lbp_hist.tolist() + haralick)

def predict_skin_type(image_path):
    features = extract_features(image_path)
    features_scaled = scaler.transform([features])
    features_pca = pca.transform(features_scaled)
    prediction = skin_type_model.predict(features_pca)[0]
    return prediction

def predict_image(image_path):
    img = Image.open(image_path).resize((128, 128)).convert("RGB")
    img_np = np.array(img)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-6)

    prediction = acne_model.predict([lbp_hist])[0]
    prob = acne_model.predict_proba([lbp_hist])[0]
    confidence = prob[list(acne_model.classes_).index(prediction)]
    return prediction, confidence

def predict_wrinkles(image_path):
    img = Image.open(image_path).resize((128, 128)).convert("L")
    img_np = np.array(img)
    glcm = greycomatrix(img_np, [1], [0], symmetric=True, normed=True)
    haralick = [greycoprops(glcm, prop).flatten()[0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
    prediction = wrinkle_model.predict([haralick])[0]
    return prediction

def generate_recommendation(skin_type, acne_level, wrink_lvl, age, profession, work_hours, free_time, using_products):
    recs = []

    # Skin type based
    if skin_type == "dry":
        recs.append("Use hydrating cleansers and moisturizers.")
    elif skin_type == "oily":
        recs.append("Use oil-free, non-comedogenic products.")
    else:
        recs.append("Use a balanced skincare routine.")

    # Acne level
    if acne_level[0] == "acne" and acne_level[1] > 50:
        recs.append("Use products with salicylic acid or benzoyl peroxide.")
    elif acne_level[0] == "acne":
        recs.append("Maintain proper skin hygiene and avoid oily food.")

    # Wrinkle level
    if wrink_lvl == "high":
        recs.append("Use retinol-based creams and apply sunscreen daily.")
    elif wrink_lvl == "medium":
        recs.append("Hydrate well and consider anti-aging creams.")

    # Lifestyle
    if work_hours > 8:
        recs.append("Ensure skin hydration during long work hours.")
    if free_time < 1:
        recs.append("Try a quick 5-min daily skincare routine.")
    if using_products == "no":
        recs.append("Start with basic products: cleanser, moisturizer, sunscreen.")

    recs.append("Stay hydrated and get enough sleep!")
    return recs

# Streamlit UI
st.set_page_config(page_title="Skincare Analyzer", page_icon="💆‍♀️")
st.title("💆‍♀️ Skincare Analysis & Personalized Recommendations")

uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())
    st.image("temp.jpg", caption="Uploaded Image", use_column_width=True)

    st.subheader("📋 Additional Info")
    age = st.number_input("Age", 10, 100, 25)
    profession = st.text_input("Profession")
    work_hours = st.slider("Work hours/day", 0, 16, 8)
    free_time = st.slider("Free time/day (hours)", 0.0, 10.0, 1.0)
    using_products = st.radio("Are you using skincare products?", ("yes", "no"))

    if st.button("🔍 Analyze"):
        skin_type = predict_skin_type("temp.jpg")
        acne_level = predict_image("temp.jpg")
        wrinkle_level = predict_wrinkles("temp.jpg")

        st.markdown(f"**Skin Type**: {skin_type}")
        st.markdown(f"**Acne Detection**: {acne_level[0]} ({acne_level[1]*100:.2f}%)")
        st.markdown(f"**Wrinkle Level**: {wrinkle_level}")

        recs = generate_recommendation(
            skin_type, acne_level, wrinkle_level,
            age, profession, work_hours, free_time, using_products
        )

        st.subheader("🎯 Recommendations")
        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec}")
