import streamlit as st
import tensorflow as tf
import numpy as np
from keras.applications.resnet import preprocess_input
from PIL import Image
import pickle

IMG_SIZE = (224, 224)
IMG_ADDRESS = "https://content.presspage.com/uploads/2110/d237c19a-da4a-4b83-a7fe-057d80f50483/1920_breast-tissue-image.jpg?10000"
IMAGE_NAME = "user_image.png"
LABELS = [
    "Adenosis",
    "Fobroadenoma",
    "Lobular Carcinoma",
    "Mucinous Carcinoma",
    "Papillary Carcinoma",
    "Phyllodes Tumor",
    "Tubular Adenona"
]
LABELS.sort()


@st.cache_resource
def get_convext_model():
    
    from keras.layers import GlobalAveragePooling2D
    from keras.models import Model
    from keras.layers import GlobalAveragePooling2D

    # Download the model, valid alpha values [0.25,0.35,0.5,0.75,1]
    base_model = tf.keras.applications.ConvNeXtLarge(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add average pooling to the base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    model_frozen = Model(inputs=base_model.input,outputs=x)

    return model_frozen


@st.cache_resource
def load_sklearn_models(model_path):

    with open(model_path, 'rb') as model_file:
        final_model = pickle.load(model_file)

    return final_model


def featurization(image_path, model):

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = model.predict(img_preprocessed)

    return predictions


# get the featurization model
featurized_model = get_convext_model()
# load biopsy model
biopsy_model = load_sklearn_models("biopsy_best_featurizor_model_no_carcinoma_CORRECT")
# load ultrasound image
ultrasound_model = load_sklearn_models("ultrasound_new_semi_diffused_featurizor_model_XGB")


# web app

# title
st.title("Breast Cancer Classification")
# image
st.image(IMG_ADDRESS, caption = "Breast Cancer Classification")

# input image
st.subheader("Please Upload a Biopsy Scan")

# file uploader
image = st.file_uploader("Please Upload a Biopsy Image", type = ["jpg", "png", "jpeg"], accept_multiple_files = False, help = "Uploade an Image")

if image:
    user_image = Image.open(image)
    # save the image to set the path
    user_image.save(IMAGE_NAME)
    # set the user image
    st.image(user_image, caption = "User Uploaded Image")

    #get the features
    with st.status("Processing......."):
        st.write("Predicting......")
        image_features = featurization(IMAGE_NAME, featurized_model)
        model_predict = biopsy_model.predict(image_features)
        st.subheader("Cancer Type is {}".format(LABELS[model_predict[0]]))
