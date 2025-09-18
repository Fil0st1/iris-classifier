import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="wide")

# Title
st.title("🌸 Iris Flower Species Classifier")
# st.write("An interactive demo using scikit-learn and Streamlit")

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Sidebar sliders for features
st.sidebar.header("Input flower measurements:")
sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.0)
sepal_width  = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 4.0)
petal_width  = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 1.0)

# User input array
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Make prediction
prediction = model.predict(input_data)[0]
pred_species = iris.target_names[prediction]

st.subheader("Prediction")
st.write(f"The model predicts this is a **{pred_species.capitalize()}** 🌱")

# Show species image – reduced size
image_paths = {
    "setosa": "images/setosa.jpg",
    "versicolor": "images/versicolor.jpg",
    "virginica": "images/virginica.jpg"
}
img_path = image_paths.get(pred_species)
if img_path:
    # display correct image
    st.image(img_path, caption=f"Iris {pred_species.capitalize()}", width=300)

# Probabilities chart
proba = model.predict_proba(input_data)[0]
prob_df = pd.DataFrame({"Species": iris.target_names, "Probability": proba})
st.subheader("Prediction probabilities")
st.bar_chart(prob_df.set_index("Species"))

# Show user input + predicted species separately
input_df = pd.DataFrame(input_data, columns=feature_names)
input_df['Predicted species'] = [pred_species]
st.subheader("Your input and predicted species")
st.dataframe(input_df)

# Model accuracy
st.write("**Training accuracy:**", f"{model.score(X, y)*100:.2f}%")

# Dataset preview & scatterplot
st.subheader("Explore the dataset")
iris_df = pd.DataFrame(X, columns=feature_names)
iris_df['species'] = pd.Series(y).map(lambda x: iris.target_names[x])
st.dataframe(iris_df.head())

st.subheader("Sepal length vs width (colored by species)")
fig, ax = plt.subplots()
sns.scatterplot(data=iris_df, x=feature_names[0], y=feature_names[1], hue="species", ax=ax)
st.pyplot(fig)

