import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.h5")
    return model

model = load_model()

# Judul aplikasi
st.title("🦴 Fracture Detection Model")
st.write("Deteksi citra X-Ray tulang: **Normal** vs **Fractured**")

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar X-Ray untuk dideteksi(JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing sesuai input model
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    class_names = ["Normal", "Fractured"]

    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")
    st.write(f"**{result}** dengan tingkat keyakinan {confidence:.2f}%")
