import streamlit as st
from PIL import Image
import os
import requests
import io
from io import BytesIO
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
import time
import cv2
from tensorflow.keras.models import load_model
import google.generativeai as genai
import pytesseract
import tensorflow as tf
import torch
from transformers import pipeline
import gdown
import sys
import threading






SERVICES = {
    "Chest X-ray": {
        "description": "AI-powered chest X-ray analysis",
        "page": "chest_xray_page",
        "image": "https://images.stockcake.com/public/1/1/3/113c0212-bbec-4994-b370-7965a4157df9_large/doctor-reviews-x-ray-stockcake.jpg"
    },
    "Brain Tumor Diagnosis": {
        "description": "Brain MRI tumor detection",
        "page": "brain_tumor_page",
        "image": "https://www.intercoastalmedical.com/wp-content/uploads/sites/494/2020/04/iStock-1199813214.jpg"
    },
    "Liver Disease Diagnosis": {
        "description": "Liver fibrosis imaging and analysis",
        "page": "liver_page",
        "image": "https://static.biospace.com/dims4/default/1c559b7/2147483647/strip/true/crop/622x350+2+0/resize/1000x563!/format/webp/quality/90/?url=https%3A%2F%2Fk1-prod-biospace.s3.us-east-2.amazonaws.com%2Fbrightspot%2Flegacy%2FBioSpace-Assets%2FE601C051-24B9-4C00-B352-954397EAEF32.jpg"
    },
    "Analysis": {
        "description": "Automated test interpretation",
        "page": "analysis_page",
        "image": "https://img.freepik.com/premium-photo/healthcare-team-bustling-hospital-with-doctor-standing-out-holding-folder_207634-13088.jpg"
    },
    "Eye Scan": {
        "description": "Retinal image analysis",
        "page": "eye_scan_page",
        "image": "https://www.virginia-lasik.com/wp-content/uploads/2021/01/Retinal-Exam-Nguyen--2048x1360.jpg"
    },
    "Fracture Detection": {
        "description": "Bone fracture identification",
        "page": "fracture_page",
        "image": "https://wp02-media.cdn.ihealthspot.com/wp-content/uploads/sites/309/2022/05/iStock-840336238-1024x576.jpg"
    },
}


SERVICES_PER_PAGE = 3


def load_image_from_url(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return img
    except:
        return None


def main():
   
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 0

   
    total_pages = (len(SERVICES) + SERVICES_PER_PAGE - 1) // SERVICES_PER_PAGE

   
    if 'selected_service' not in st.session_state:
        show_main_page(total_pages)
    else:
        show_service_page(st.session_state.selected_service)


def show_main_page(total_pages):
    
    st.markdown("""
    <style>

      .stApp {
        background-color: #f0f2f6;
    }

    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .title-text {
        font-size: 2.5rem;
        color: #2c3e50;
        font-weight: bold;
    }
    .service-card {
        border: 1px solid #e0e0e0;
        border-radius: 15px;
        padding: 0;
        margin: 15px;
        text-align: center;
        transition: transform 0.3s, box-shadow 0.3s;
        background-color: #f9f9f9;
        height: 380px;
        display: flex;
        flex-direction: column;
        position: relative;
        overflow: hidden;
    }
    .service-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border: 1px solid #4CAF50;
    }
    .service-image-container {
        width: 100%;
        height: 220px;
        overflow: hidden;
    }
    .service-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.5s;
    }
    .service-card:hover .service-image {
        transform: scale(1.05);
    }
    .service-content {
        padding: 15px;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
    }
    .service-title {
        font-weight: bold;
        margin: 5px 0;
        color: #2c3e50;
        font-size: 1.2rem;
    }
    .service-description {
        font-size: 0.95rem;
        color: #7f8c8d;
        margin-bottom: 15px;
        flex-grow: 1;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 8px 15px;
        cursor: pointer;
        transition: background-color 0.3s;
        width: 100%;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .centered-container {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        margin: 30px 0;
    }
    .navigation-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 20px;
        padding: 0 20px;
    }
    .page-indicator {
        color: #7f8c8d;
        font-size: 0.9rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

   
    st.markdown("""
    <div class="title-container">
        <div class="title-text">Our Medical Images Multi Model</div>
        <div style="color: #7f8c8d;">Select a service to get started</div>
    </div>
    """, unsafe_allow_html=True)

    
    start_idx = st.session_state.current_page * SERVICES_PER_PAGE
    end_idx = min(start_idx + SERVICES_PER_PAGE, len(SERVICES))
    current_services = list(SERVICES.items())[start_idx:end_idx]

   
    cols = st.columns(len(current_services))
    for idx, (service_name, service_info) in enumerate(current_services):
        with cols[idx]:
            st.markdown(f"""
            <div class="service-card">
                <div class="service-image-container">
                    <img class="service-image" src="{service_info["image"]}" alt="{service_name}">
                </div>
                <div class="service-content">
                    <div class="service-title">{service_name}</div>
                    <div class="service-description">{service_info["description"]}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Streamlit button that will navigate to the service page
            if st.button(f"Select", key=f"select_{service_name}"):
                st.session_state.selected_service = service_name
                st.rerun()

    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        prev_disabled = st.session_state.current_page == 0
        if st.button("← Previous", disabled=prev_disabled, key="prev_button"):
            st.session_state.current_page -= 1
            st.rerun()

    with col2:
        st.markdown(f"""
        <div class="page-indicator">
            Page {st.session_state.current_page + 1} of {total_pages}
        </div>
        """, unsafe_allow_html=True)

    with col3:
        next_disabled = st.session_state.current_page == total_pages - 1
        if st.button("Next →", disabled=next_disabled, key="next_button"):
            st.session_state.current_page += 1
            st.rerun()


def show_service_page(service_name):
    st.title(f"🔬 {service_name}")
    st.markdown(f"{SERVICES[service_name]['description']}")

    
    st.image(SERVICES[service_name]["image"], use_container_width=True)


    if service_name == "Chest X-ray":
        chest_xray_page()
    elif service_name == "Brain Tumor Diagnosis":
        brain_tumor_page()
    elif service_name == "Liver Disease Diagnosis":
        liver_page()
    elif service_name == "Analysis":
        analysis_page()
    elif service_name == "Eye Scan":
        eye_scan_page()
    elif service_name == "Fracture Detection":
        fracture_page()

    if st.button("← Back to Services"):
        del st.session_state.selected_service
        st.rerun()


def chest_xray_page():
    @st.cache_resource
    def load_model():
        files = {
            "config.json": {
                "file_id": "14M2rmv00uGCT7xbq7nHu7jkUaSsTQ5OG",
                "output": "config.json"
            },
            "model.safetensors": {
                "file_id": "1v90JJcPsad13gtxMluqCRau5HBmonjUH",
                "output": "model.safetensors"
            },
            "preprocessor_config.json": {
                "file_id": "1ycZG5YhATFS67-zODHZhLNY8WE7hphH9",
                "output": "preprocessor_config.json"
            }
        }

        model_dir = "chest_xray_model"
        os.makedirs(model_dir, exist_ok=True)

        try:
            for file_name, file_info in files.items():
                output_path = os.path.join(model_dir, file_name)

                if not os.path.exists(output_path):
                    st.info(f"Downloading {file_name}...")
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_info['file_id']}",
                        output_path,
                        quiet=False
                    )

                if not os.path.exists(output_path):
                    st.error(f"❌ Failed to download: {file_name}")
                    return None

            st.success("✅ Model files loaded successfully.")

            return pipeline(
                "image-classification",
                model=model_dir,
                device="cpu"
            )

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

    
    model = load_model()

    if model is None:
        st.warning("Model couldn't be loaded.")
    else:
        uploaded_file = st.file_uploader("Upload an X-ray image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert("RGB")
                st.image(img, caption="Uploaded Image", use_container_width=True)

                if st.button("🔍 Analyze Image"):
                    with st.spinner("Analyzing..."):
                        predictions = model(img)
                        top_prediction = predictions[0]
                        label = top_prediction['label']
                        score = top_prediction['score'] * 100

                        st.markdown(f"### 🩺 Diagnosis: *{label}*")
                        st.markdown(f"*Confidence:* {score:.2f}%")

            except Exception as e:
                st.error(f"❌ Error analyzing image: {str(e)}")

                

def brain_tumor_page():
    
    @st.cache_resource
    def load_classification_model():
        try:
            file_id = "1nTRy7Kn5nHDlAuXoB3ffFhwiV1I3KTRg"
            output = "brain_classification_model.h5"

            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

            if not os.path.exists(output):
                st.error("Failed to load classification form")
                return None

            return tf.keras.models.load_model(output)
        except Exception as e:
            st.error(f"An error occurred while loading the classification form: {str(e)}")
            return None

   
    @st.cache_resource
    def load_yolo_model():
        try:
            file_id = "1r9KPgGMVdQvzsLqAd0qxDRRwWzwEcGa4"
            output = "brain_detection_model.pt"

            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

            if not os.path.exists(output):
                st.error("YOLOv8 form failed to load")
                return None

            return YOLO(output)
        except Exception as e:
            st.error(f"An error occurred while loading the YOLOv8 form: {str(e)}")
            return None

    # Load the models
    classification_model = load_classification_model()
    yolo_model = load_yolo_model()

    # Check if the models were loaded
    if classification_model is None or yolo_model is None:
        st.error("One or both forms could not be loaded. Please try again later.")
        return

    st.write("### Upload Brain MRI")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load the image using PIL
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded MRI", use_container_width=True)

        if st.button("Detect Tumors"):
            with st.spinner('Analyzing...'):
                # Convert the image to OpenCV format
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                # Save the image temporarily for YOLO use
                temp_img_path = "temp_brain_image.jpg"
                cv2.imwrite(temp_img_path, img_bgr)

                # Run the YOLOv8 detection model
                try:
                    predictions = yolo_model.predict(source=temp_img_path, save=False, conf=0.1)
                    tumor_detected = False
                    result_img = None

                    for r in predictions:
                        if r.boxes:  # If anything is detected
                            tumor_detected = True
                        result_img = r.plot(labels=True, boxes=True)  #Image with bounding boxes

                    # Convert the processed image to RGB
                    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                    # Prepare the image for the classification model
                    img_resized = img.resize((224, 224))
                    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                    #Run the classification model
                    brain_classes = ['No_Tumor', 'Tumor']
                    prediction = classification_model.predict(img_array)
                    predicted_class = int(prediction[0][0] > 0.5)  # Binary classification based on threshold 0.5

                    #Display the results
                    st.success("Analysis complete!")
                    st.markdown(f"### Diagnosis: **{brain_classes[predicted_class]}**")
                    st.image(result_img_rgb, caption="YOLOv8 Detection Result", use_container_width=True)

                    if tumor_detected:
                        st.write("*Detection Results:* Potential tumor regions detected!")
                    else:
                        st.write("*Detection Results:* No tumor regions detected.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

                # Delete the temporary image
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path) 

                    
def liver_page():
    # Function to load the classification model
    @st.cache_resource
    def load_classification_model():
        try:
            file_id = "1lOBTOBoDCEtndw5RAOCoRukUtYgMc8xF"
            output = "liver_classification_model.h5"

            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

            if not os.path.exists(output):
                st.error("Failed to load the classification model")
                return None

            return tf.keras.models.load_model(output)
        except Exception as e:
            st.error(f"An error occurred while loading the classification model: {str(e)}")
            return None

    # Function to load the YOLOv8 model
    @st.cache_resource
    def load_yolo_model():
        try:
            file_id = "1JjVXZ9Ng41oQ53poTk1j-K5rRpC2RAMr"
            output = "liver_detection_model.pt"

            if not os.path.exists(output):
                gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

            if not os.path.exists(output):
                st.error("Failed to load the YOLOv8 model")
                return None

            return YOLO(output)
        except Exception as e:
            st.error(f"An error occurred while loading the YOLOv8 model: {str(e)}")
            return None

    
    classification_model = load_classification_model()
    yolo_model = load_yolo_model()

    # Check if the models were successfully loaded
    if classification_model is None or yolo_model is None:
        st.error("Failed to load one or both models. Please try again later.")
        return

    st.write("### Upload Liver Scan")
    view_mode = st.radio("Choose display type:", ["Classification Only", "Detection Only"])
    uploaded_file = st.file_uploader("Upload a radiology image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image using PIL
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Scan"):
            with st.spinner('Examining...'):
                
                img_array = np.array(img)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                #Save the image temporarily for YOLO use
                temp_img_path = "temp_liver_image.jpg"
                cv2.imwrite(temp_img_path, img_bgr)

                try:
                    if view_mode == "Classification Only":
                        # Prepare image for the classification model
                        img_resized = img.resize((128, 128))
                        img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

                        # Run the classification model
                        liver_classes = ['No_Fibrosis', 'Fibrosis']
                        prediction = classification_model.predict(img_array)
                        predicted_class = int(prediction[0][0] > 0.5)  #Binary Classifcion

                        #show results
                        st.success("Analysis complete!")
                        st.markdown(f"### Diagnosis: **{liver_classes[predicted_class]}**")

                    elif view_mode == "Detection Only":
                        
                        predictions = yolo_model.predict(source=temp_img_path, save=False, conf=0.1)
                        fibrosis_detected = False
                        result_img = None

                        for r in predictions:
                            if r.boxes:  # If anything is detected
                                fibrosis_detected = True
                            result_img = r.plot(labels=True, boxes=True)  

                      
                        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                        
                        st.success("Analysis complete!")
                        st.image(result_img_rgb, caption="YOLOv8 Detection Result", use_container_width=True)
                        if fibrosis_detected:
                            st.write("*Detection Results:* Potential fibrosis regions detected!")
                        else:
                            st.write("*Detection Results:* No fibrosis regions detected.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

                
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)

def analysis_page():
    @st.cache_resource
    def load_model():
        genai.configure(api_key="AIzaSyBsA6ixodO7_ODrKGV6kRqiswdN_3n958A")
        return genai.GenerativeModel(model_name="gemini-2.0-flash")

    model = load_model()

    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    st.header("Upload Test Image")
    uploaded_file = st.file_uploader("Upload the medical test image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)

        if st.button("Analyze Image"):
            with st.spinner("Processing analysis..."):

                extracted_text = pytesseract.image_to_string(image, lang='ara+eng')

                # Display the extracted text in a collapsible text box
                with st.expander("Extracted text from the image"):
                    st.text_area("Extracted Text:", value=extracted_text, height=200)

                messages = [
                    {
                        "role": "user",
                        "parts": [f"""You are an intelligent medical assistant.
    Your task is:
    - Read the attached test results.
    - Summarize the results into a clear medical report.
    - Provide general medical advice based on the results.
    - Suggest the appropriate medical specialty if needed.
    Extracted test results:
    {extracted_text}"""]
                    }
                ]

                try:
                    response = model.generate_content(messages)

                    st.success("The image was successfully analyzed!")
                    st.subheader("Analysis results:")
                    st.markdown(f"""
                    <div style="background-color: #f0f8ff; padding: 20px; border-radius: 10px; border: 1px solid #ccc;">
                        {response.text}
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred while analyzing the image: {str(e)}")
                    st.info(
                        "The reason might be exceeding the usage limits of the Google Gemini API. Please try again later.")

    st.sidebar.header("Usage Instructions")
    st.sidebar.markdown("""
    1. Upload a medical test image
    2. Click the "Analyze Image" button
    3. Wait for the results to appear
    """)

    st.sidebar.header("About the Service")
    st.sidebar.info("""
    This service uses:
    - Google Gemini AI for result analysis
    - Tesseract OCR for text extraction from images
    - Streamlit for the user interface.
    """)

def eye_scan_page():
    @st.cache_resource
    def load_model():
        url = "https://drive.google.com/uc?id=1sACluiNwV__kosazzRVN-42vnHijYUBK&export=download"
        output = "cnn_model.h5"

        # Download the model only if it doesn't already exist
        if not os.path.exists(output):
            st.info("Downloading model for the first time. Please wait...")
            gdown.download(url, output, quiet=False)

        # Check if the file exists after attempted download
        if not os.path.exists(output):
            st.error("Failed to download the model file!")
            return None

        try:
            return tf.keras.models.load_model(output)
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")
            return None


    model = load_model()
    st.write("Upload an OCT image to detect eye diseases or use live video detection.")

    # Define the classes for prediction
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    # Define the predict_image function
    def predict_image(img):
        # Check if the image is empty
        if img is None or img.size == 0:
            st.error("The image is empty or not loaded correctly.")
            return None, None

        # Debugging output to check image shape
        st.text(f"Image shape before resizing: {img.shape}")

        # Only resize if the image is not already in the target shape
        if img.shape != (1, 128, 128, 3):
            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.reshape(img, (1, 128, 128, 3))

        prediction = model.predict(img)
        class_idx = np.argmax(prediction)
        predicted_class = classes[class_idx]

        return predicted_class, prediction[0]

    # Image upload option
    uploaded_file = st.file_uploader("Upload an OCT image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_container_width=True)

        if st.button("Classify Uploaded Image", key="classify_uploaded"):
            with st.spinner('Classifying...'):
                if model is not None:
                    predicted_class, _ = predict_image(opencv_image)
                    st.success(f"Prediction: {predicted_class}")
                else:
                    st.error("Model was not loaded properly")

    # Video capture option
    def start_video_capture():
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        captured_image = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                st.error("Failed to capture video or frame is empty")
                break

            # Display the frame
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Capture image when button is pressed
            if st.button("Capture Image for Prediction", key=f"capture_image_{time.time()}"):
                captured_image = frame.copy()
                st.image(captured_image, channels="BGR", caption="Captured Image", use_container_width=True)
                break

            # Add a small delay to allow the frame to update
            time.sleep(0.03)

        cap.release()

        # Predict on the captured image
        if captured_image is not None:
            with st.spinner('Classifying captured image...'):
                if model is not None:
                    # Preprocess the captured image
                    captured_image_resized = cv2.resize(captured_image, (128, 128))
                    captured_image_rgb = cv2.cvtColor(captured_image_resized, cv2.COLOR_BGR2RGB)
                    captured_image_np = np.reshape(captured_image_rgb, (1, 128, 128, 3))

                    predicted_class, _ = predict_image(captured_image_np)
                    st.success(f"Prediction: {predicted_class}")
                else:
                    st.error("Model was not loaded properly")

    if st.button("Start Live Video Detection", key="start_video"):
        start_video_capture()

    st.markdown("---")
    st.subheader("How to Use the Service")
    st.markdown("""
    1. Upload an OCT image using the "Browse files" button
    2. Click the "Classify Image" button to get the result
    3. Use the "Start Live Video Detection" button to capture a clear image for prediction
    """)

    st.subheader("Important Notes")
    st.markdown("""
    - This service is for educational purposes only and does not replace a medical consultation
    - Diagnosis accuracy depends on the quality of the uploaded image
    - Make sure the uploaded image is a real OCT scan
    """)



def fracture_page():
    @st.cache_resource
    def load_model():
            yolo_model_file_id = "1_gKzNIMSSMBH_uHPz4eq9QMrPIXceDzf"
            model_output = "best.pt"

            try:
                gdown.download(f"https://drive.google.com/uc?id={yolo_model_file_id}",
                               model_output, quiet=False)


                if not os.path.exists(model_output):
                    return None

                from ultralytics import YOLO
                model = YOLO(model_output)

                return model

            except Exception as e:
               return None


    model = load_model()


    if model is None:
        return
    st.write("### Upload Bone X-ray")
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
       
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-ray",use_container_width=True)

        if st.button("Check for Fractures"):
            with st.spinner('Analyzing...'):
                
                img_array = np.array(img)
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

               #Save the image temporarily for YOLO processing
                temp_img_path = "temp_image.jpg"
                cv2.imwrite(temp_img_path, img_rgb)


                # Run model on the image
                predictions = model.predict(source=temp_img_path, save=False, conf=0.1)

                # Extract processed image
                fracture_detected = False
                for r in predictions:
                    if r.boxes:  
                        fracture_detected = True
                    im_show = r.plot(labels=True, boxes=True) #Image with fractures drawn

                # Convert the processed image to RGB format for Streamlit
                im_show_rgb = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)

                
                st.image(im_show_rgb, caption="YOLOv8 Prediction", use_container_width=True)


                st.success("Analysis complete!")
                if fracture_detected:
                    st.write("*Results:* Fractures detected!")
                else:
                    st.write("*Results:* No fractures detected")

                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)



if __name__ == "__main__":
    main()