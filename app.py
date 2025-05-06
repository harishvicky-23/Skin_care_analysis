import streamlit as st
import numpy as np
import cv2
from joblib import load
import joblib
import mahotas
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import os
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

st.set_page_config(page_title="SkinCare Analyzer")


# Load models
skin_model = joblib.load("models/skin_type_svm_model.pkl")
skin_pca = joblib.load("models/skin_type_pca.pkl")
skin_scaler = joblib.load("models/skin_type_scaler.pkl")

acne_model = joblib.load("models/svm_acne_vgg_model.pkl")

wrinkle_model = load("models/wrinkle_model.joblib")
wrinkle_pca = load("models/wrinkle_pca.joblib")
wrinkle_le = load("models/wrinkle_label_encoder.joblib")

skin_classes = ['dry', 'normal', 'oily']

# --- Feature extraction functions ---
def extract_skin_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color_feat = np.concatenate([
        cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0,256,0,256,0,256]).flatten(),
        cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0,180,0,256,0,256]).flatten()
    ])

    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), range=(0, 10))
    lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)

    haralick_feat = mahotas.features.haralick(gray).mean(axis=0)
    return np.concatenate([color_feat, lbp_hist, haralick_feat])

"""def extract_acne_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    rgb_hist = np.concatenate([cv2.calcHist([img], [i], None, [32], [0, 256]).flatten() for i in range(3)])
    hsv_hist = np.concatenate([cv2.calcHist([hsv], [i], None, [32], [0, 256]).flatten() for i in range(3)])

    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))

    return np.concatenate([rgb_hist, hsv_hist, lbp_hist])"""

def extract_acne_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    # Load the VGG16 model (without top layers)
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)
    
    # Extract features
    features = feature_extractor.predict(img)
    return features.flatten()

def extract_wrinkle_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick[:6]

# --- Prediction functions ---
def predict_skin_type(img):
    skin_classes = ['dry', 'normal', 'oily']
    features = extract_skin_features(img)
    features_scaled = skin_scaler.transform([features])
    features_pca = skin_pca.transform(features_scaled)
    prediction = skin_model.predict(features_pca)[0]
    return skin_classes[prediction]

def predict_acne(img):
    features = extract_acne_features(img)
    pred = acne_model.predict([features])[0]
    prob = acne_model.predict_proba([features])[0][pred]
    label = "acne" if pred else "no acne"
    if label == "acne" and prob > 70:
        return (label, prob * 100)
    elif label == "acne":
        return (label, prob * 100)
    else:
        return (label, prob * 100) 

def predict_wrinkles(img):
    features = extract_wrinkle_features(img)
    features_pca = wrinkle_pca.transform([features])
    prediction = wrinkle_model.predict(features_pca)[0]
    prob = wrinkle_model.predict_proba(features_pca)[0]
    label = wrinkle_le.inverse_transform([prediction])[0].lower()
    confidence = prob[prediction]
    # Apply confidence condition
    if label == 'wrinkled' and confidence >= 0.99:
        final_label = 'wrinkled'
    elif label == 'wrinkled':
        final_label = 'not wrinkled'  # downgrade if not confident
    else:
        final_label = label  # keep 'not wrinkled' as is
    return final_label

# --- Recommendation Logic ---
def generate_recommendation(skin_type, acne_lvl, wrink_lvl, age, profession, work_hours, free_time, using_products):
    acne_label, acne_prob = acne_lvl
    recommendations = []

    if skin_type == 'dry':
        recommendations += ["Use a rich, hydrating moisturizer twice daily.",
                            "Drink plenty of water to maintain skin hydration.",
                            "Avoid alcohol-based products that strip natural oils."]
    elif skin_type == 'oily':
        recommendations += ["Use an oil-free, foaming cleanser.",
                            "Try water-based moisturizers.",
                            "Avoid heavy makeup that clogs pores."]
    elif skin_type == 'normal':
        recommendations += ["Moisturize daily.",
                            "Use SPF daily even indoors."]
    
    if acne_label == 'acne' and acne_prob > 70:
        recommendations += ["Avoid oily/spicy food.",
                            "Use salicylic acid-based cleansers.",
                            "Clean pillowcases and phone screen regularly.",
                            "Do not pop pimples."]
    elif acne_label == 'acne':
        recommendations.append("Mild acne signs detected — use gentle, non-comedogenic products.")
    else:
        recommendations.append("No major acne detected. Maintain your current routine.")

    if wrink_lvl == "wrinkled":
        recommendations += ["Use retinol-based products at night.",
                            "Eat antioxidant-rich foods.",
                            "Get enough sleep and reduce stress."]
    else:
        recommendations.append("Skin looks smooth! Maintain hydration and SPF.")

    if age < 20:
        recommendations.append("Stick to gentle cleansers and light moisturizers.")
    elif 20 <= age <= 35:
        recommendations.append("Use SPF and light exfoliation weekly.")
    elif 36 <= age <= 50:
        recommendations.append("Add collagen-boosting serums and night creams.")
    else:
        recommendations.append("Use anti-aging products and consult a dermatologist yearly.")

    if "construction" in profession or "outdoor" in profession:
        recommendations += ["Use strong SPF (50+) and reapply every 2-3 hours.",
                            "Cleanse thoroughly after work."]
    elif "student" in profession or "jobless" in profession:
        recommendations.append("Great time to build a skincare routine. Stay hydrated.")
    elif "office" in profession or "indoor" in profession:
        recommendations += ["Use humidifiers indoors.",
                            "Take screen breaks to reduce eye strain."]

    if work_hours >= 10:
        recommendations.append("Long work hours — don’t skip your nightly skincare.")
    if work_hours >= 4:
        recommendations.append("Apply broad-spectrum sunscreen daily.")

    if free_time < 1:
        recommendations.append("Even 5 mins twice a day helps. Cleanse + moisturize.")
    else:
        recommendations.append("Use free time for masks or gentle exfoliation weekly.")

    if using_products == "no":
        recommendations.append("Start with a simple routine: Cleanser, Moisturizer, SPF.")
    else:
        recommendations.append("Check product ingredients for harsh chemicals.")

    return recommendations

st.title("🧴 AI-Based SkinCare Recommendation System")

# Image upload
uploaded_file = st.file_uploader("📤 Upload a face image", type=["jpg", "jpeg", "png"])

image_path = None  # Initialize image path

if uploaded_file is not None:
    # Save and display image
    image = Image.open(uploaded_file).convert("RGB")
    image_path = "temp.jpg"
    image.save(image_path)
    
    st.image(image_path, caption="Uploaded Image", use_container_width=True)

# Step 2: User inputs
st.subheader("🧍 Step 2: Enter Lifestyle Details")
age = st.number_input("Enter your age", min_value=10, max_value=100, value=25)
profession = st.text_input("Enter your profession", value="Student")
work_hours = st.number_input("Average work hours per day", min_value=0, max_value=24, value=6)
free_time = st.number_input("Free time per day (hours)", min_value=0.0, max_value=24.0, value=2.0)
using_products = st.radio("Are you currently using skincare products?", options=["yes", "no"])

# Step 3: Button to trigger prediction
if st.button("💡 Generate Recommendations"):
    if image_path and os.path.exists(image_path):
        st.subheader("🔍 Step 1: Analyzing Skin Details")

        # Call your models here (make sure they are defined above this code)
        skin_type = predict_skin_type(image_path)
        acne_label, acne_prob = predict_acne(image_path)
        wrink_lvl = predict_wrinkles(image_path)

        # Display predictions
        st.write(f"**Skin Type:** {skin_type}")
        st.write(f"**Acne Level:** {acne_label} ({acne_prob:.2f}%)")
        st.write(f"**Wrinkle Level:** {wrink_lvl}")

        # Step 4: Recommendations
        st.subheader("📋 Step 4: Personalized Recommendations")
        recs = generate_recommendation(
            skin_type.lower(),
            (acne_label.lower(), acne_prob),
            wrink_lvl.lower(),
            age, profession, work_hours, free_time,
            using_products.lower()
        )

        for i, rec in enumerate(recs, 1):
            st.markdown(f"{i}. {rec}")
    else:
        st.warning("Please upload a valid image before generating recommendations.")
