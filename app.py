import os
from get_gt_mask import get_gt_mask
from load_model import load_model
from postprocess_prediction import postprocess_prediction
from predict import predict
from preprocess_image import preprocess_image
import streamlit as st
from PIL import Image
from cfg import model_9_epochs_path, model_30_epochs_path, GT_TEST_DIRECTORY, TEST_DIRECTORY


def st_load_model(model_name):
    if model_name == "U-net trained for 9 epochs":
        model = load_model(model_9_epochs_path)
    else:
        model = load_model(model_30_epochs_path)
    return model


test_images = [os.path.join(TEST_DIRECTORY, img) for img in os.listdir(TEST_DIRECTORY)]
gt_masks = [os.path.join(GT_TEST_DIRECTORY, img) for img in os.listdir(GT_TEST_DIRECTORY)]


st.title("Image Segmentation Prediction App")


model_name = st.selectbox(
    "Choose Model", ["U-net trained for 9 epochs", "U-net trained for 30 epochs"]
)
model = st_load_model(model_name)

# Image selection
st.sidebar.title("Image Selection")
index = st.sidebar.slider("Select an Image", 0, len(test_images) - 1)
selected_image_path = test_images[index]
selected_gt_path = gt_masks[index]

# Load and display selected test image
st.subheader("Selected Test Image")
original_image = Image.open(selected_image_path).convert("L")
st.image(
    original_image,
    caption="Original Image",
    use_column_width=True,
    clamp=True,
    channels="L",
)


st.subheader("Model Prediction and Post-Processing")
prepr_image = preprocess_image(original_image)
# Run prediction and post-processing
predicted_mask = predict(model, prepr_image)
postprocessed_mask = postprocess_prediction(predicted_mask)

# Load ground truth mask
gt_mask = get_gt_mask(selected_image_path)


col1, col2, col3 = st.columns(3)
col1.image(
    predicted_mask,
    caption="Predicted Mask",
    use_column_width=True,
    clamp=True,
    channels="L",
)
col2.image(
    postprocessed_mask,
    caption="Post-Processed Mask",
    use_column_width=True,
    clamp=True,
    channels="L",
)
col3.image(gt_mask, caption="Ground Truth Mask", use_column_width=True, clamp=True)

# Upload image for prediction
st.sidebar.subheader("Upload an Image for Prediction")
uploaded_image_path = st.sidebar.file_uploader("Choose an image...", type=["tif"])

if uploaded_image_path is not None:
    uploaded_image = Image.open(uploaded_image_path)
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    prepr_uploaded_image = preprocess_image(uploaded_image)

    uploaded_predicted_mask = predict(model, prepr_uploaded_image)
    uploaded_postprocessed_mask = postprocess_prediction(uploaded_predicted_mask)

    col1, col2 = st.columns(2)
    col1.image(
        uploaded_predicted_mask,
        caption="Predicted Mask",
        use_column_width=True,
        clamp=True,
        channels="L",
    )
    col2.image(
        uploaded_postprocessed_mask,
        caption="Post-Processed Mask",
        use_column_width=True,
        clamp=True,
        channels="L",
    )
