import streamlit as st
import cv2
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define the tab functions
def tab_3d_plot_visualization():
    st.header('3D visualization ')
    df = pd.read_csv('WomensClothingE-CommerceReviews.csv', header=0,index_col=0)
    df.head()
def tab_image_processing():
    st.header("Image Processing")
    st.write("This tab will allow users to perform various image processing tasks such as filtering, segmentation, object detection, and more using libraries like OpenCV or Pillow.")

    # File uploader for the image
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

    if uploaded_file is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Resize the image
        width = 300
        height = 200
        resized_image = cv2.resize(image, (width, height))

        # Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Cropping the image
        start_row, end_row = 50, 150
        start_col, end_col = 100, 200
        cropped_image = image[start_row:end_row, start_col:end_col]

        # Image rotation
        angle = 45
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        # Display the images
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # Original Image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Resized Image
        axes[0, 1].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title('Resized Image')
        axes[0, 1].axis('off')

        # Grayscale Image
        axes[0, 2].imshow(gray_image, cmap='gray')
        axes[0, 2].set_title('Grayscale Image')
        axes[0, 2].axis('off')

        # Cropped Image
        axes[1, 0].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('Cropped Image')
        axes[1, 0].axis('off')

        # Rotated Image
        axes[1, 1].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Rotated Image')
        axes[1, 1].axis('off')

        # Hide the empty subplot
        axes[1, 2].axis('off')
        plt.tight_layout()
        st.pyplot(fig)

def tab_text_similarity_analysis():
    st.header("Text Similarity Analysis")
    st.write("This tab will enable users to analyze the similarity between two or more text documents using techniques like cosine similarity, Levenshtein distance, or natural language processing models.")

    # Read the CSV file
    df = pd.read_csv('ModifiedName.csv')

    # Function to preprocess text
    def preprocess_text(text):
        if pd.isnull(text):
            return ''

        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [token for token in tokens if token not in stop_words]
        preprocessed_text = ' '.join(filtered_tokens)

        return preprocessed_text

    # Apply text preprocessing to the 'Review Text' column
    df['preprocessed_review'] = df['Review Text'].apply(preprocess_text)

    # Select reviews for different divisions
    general_reviews = df[df['Division Name'] == 'General']['preprocessed_review'].tolist()
    general_petite_reviews = df[df['Division Name'] == 'General Petite']['preprocessed_review'].tolist()
    intimates_reviews = df[df['Division Name'] == 'Intimates']['preprocessed_review'].tolist()

    # Function to compute cosine similarity between two lists of reviews
    def compute_cosine_similarity(reviews1, reviews2):
        all_reviews = reviews1 + reviews2
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_reviews)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        similarity_scores = similarity_matrix[:len(reviews1), len(reviews1):]

        return similarity_scores

    # Compute cosine similarity between different divisions
    general_general_petite_similarity = compute_cosine_similarity(general_reviews, general_petite_reviews)
    general_intimates_similarity = compute_cosine_similarity(general_reviews, intimates_reviews)
    general_petite_intimates_similarity = compute_cosine_similarity(general_petite_reviews, intimates_reviews)

    # Display the similarity scores
    st.write("Cosine Similarity between General and General Petite:")
    st.write(general_general_petite_similarity)
    st.write("\nCosine Similarity between General and Intimates:")
    st.write(general_intimates_similarity)
    st.write("\nCosine Similarity between General Petite and Intimates:")
    st.write(general_petite_intimates_similarity)

# Create the tabs
tabs = ["3D Plot Visualization", "Image Processing", "Text Similarity Analysis"]
selected_tab = st.sidebar.selectbox("Select a tab", tabs)

# Display the selected tab
if selected_tab == "3D Plot Visualization":
    tab_3d_plot_visualization()
elif selected_tab == "Image Processing":
    tab_image_processing()
elif selected_tab == "Text Similarity Analysis":
    tab_text_similarity_analysis()