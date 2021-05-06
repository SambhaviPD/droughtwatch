from PIL import Image
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import requests

st.set_page_config(
	page_title = "Drought Watch",
	page_icon = "cactus"
	)
	
st.title(":cactus: Weights & Biases Public Benchmark")
st.title("- for Drought Prediction :cow:")
st.subheader("_My course project for FSDL Spring 2021 - Sambhavi Dhanabalan_ ")

st.sidebar.title("Table of Contents")
st.sidebar.write("1. Introduction")
st.sidebar.write("2. Project Goals")
st.sidebar.write("3. Stretch Goals")
st.sidebar.write("4. About the Data set")
st.sidebar.write("5. Planned Approach")
st.sidebar.write("6. Initial Findings")
st.sidebar.write("7. Final Model & Output")
st.sidebar.write("8. Next Steps")

st.sidebar.title("Author")
st.sidebar.write("[Sambhavi Dhanabalan] (https://www.linkedin.com/in/sambhavi-dhanabalan/)")

st.sidebar.title("Git links")

expander = st.beta_expander("1. Introduction", expanded=True)
expander.write(":trophy: Public benchmarks in Weights & Biases encourages collaboration from the community for a variety of problem statements. The one that I worked on as part of FSDL's final course project was the Drought Watch. The intent of this project is to predict drought severity from satellite imagery and ground-level photos. Complete details about the project from Weights & Biases can be found here: [Drought Watch] (https://arxiv.org/pdf/2004.04081.pdf)")

expander = st.beta_expander("2. Project Goals")
expander.write(":up: 77.8 is the top accuracy in the project's leader board. To surpass that is my primary goal. In order to do that, an initial understanding of satellite imagery is required, followed by the right model architecture.")
expander.write(":rain_cloud: Secondary goal would be to isolate images obscured with clouds. Cloud detection algorithms need to be studied and applied to the input data set. If I'm able to isolate successfully, then need to study what impact it makes on improving the accuracy.")

expander = st.beta_expander("3. Stretch Goals")
expander.write(":clipboard: If I am able to enter the leaderboard, submitting a report to W&B is a goal that I'd planned for.")
expander.write(":earth_asia: eo-learn is a package that provides a set of tools to work with Earth Observation(EO) data. Did a high level study and decided to explore more on eo-patch and eo-workflows in order to view the green patches as well as clouds in images. In short, a deeper visualization of images will give more directions.")

expander = st.beta_expander("4. About the Data set")
expander.write(":cactus: Drought watch data set can be downloaded from the [git repo] (https://github.com/wandb/droughtwatch). A total of 86,317 training images and 10,778 validation images is present. Human experts (pastoralists) have labeled these with the number of cows that the geographic location at the center of the image could support (0, 1, 2, or 3+ cows). Each pixel represents a 30 meter square, so the images at full size are 1.95 kilometers across. Pastoralists are asked to rate the quality of the area within 20 meters of where they are standing, which corresponds to an area slightly larger a single pixel.") 

expander.subheader('4a. Let\'s take a look at Training and Validation split')

col1, col2 = expander.beta_columns(2)
train_clicked = col1.button("Click to see Training Data Split")
val_clicked = col2.button("Click to see Validation Data Split")

# Training Data Plot
train_figure = plt.figure(figsize=(15, 6))
plt.title('Forage Quality Analysis of Training Data')
plt.xlabel('How many cows can be fed?')
plt.ylabel('Number of landsat images')
# TODO: To send this data from a API
counts_elements = [51814, 12851, 13609, 8043]
unique_elements = [0, 1, 2, 3]
plt.bar(unique_elements, counts_elements, color='green')

# To display count of each label
for index, data in enumerate(counts_elements):
  plt.text(x=index, y=data+1, s=f'{data}', fontdict=dict(fontsize=15))
 
if train_clicked:
	expander.pyplot(train_figure)
	expander.write('On looking at the data, it is evident that there is a huge class imbalance. On later iterations, this will be handled to see how it affects the accuracy.')
	
# Validation Data Plot
val_figure = plt.figure(figsize=(15, 6))
plt.title('Forage Quality Analysis of Validation Data')
plt.xlabel('How many cows can be fed?')
plt.ylabel('Number of landsat images')
# TODO: To send this data from a API
counts_elements = [6443, 1710, 1673, 952]
unique_elements = [0, 1, 2, 3]
plt.bar(unique_elements, counts_elements, color='blue')

# To display count of each label
for index, data in enumerate(counts_elements):
  plt.text(x=index, y=data+1, s=f'{data}', fontdict=dict(fontsize=15))
 
if val_clicked:
	expander.pyplot(val_figure)
	
expander.subheader('4b. Let\'s take a look at few samples')
expander.write('Label 0 denotes that the images has forage quantity that is insufficient to feed even a single cow. Label 1 denotes that the image has forage quantity that is sufficient to feed just 1 cow. Likewise for Label 2 and 3+.')

label_option = expander.selectbox(
	'Choose a Label to see corresponding images',
	('--Please select--', '0', '1', '2', '3+'))

# To display few sample images
if label_option == '0':
	col1, col2, col3 = expander.beta_columns(3)

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-1.jpg")
	image1 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-2.jpg")
	image2 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/sunflower-1.jpg")
	image3 = Image.open(BytesIO(response.content))

	col1.image(image1, caption='Rose', width=150)
	col2.image(image2, caption='Rose', width=150)
	col3.image(image3, caption='Sunflower', width=150)
	
if label_option == '1':
	col1, col2, col3 = expander.beta_columns(3)

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-1.jpg")
	image1 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-2.jpg")
	image2 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/sunflower-1.jpg")
	image3 = Image.open(BytesIO(response.content))

	col3.image(image1, caption='Rose', width=150)
	col2.image(image2, caption='Rose', width=150)
	col1.image(image3, caption='Sunflower', width=150)
	
if label_option == '2':
	col1, col2, col3 = expander.beta_columns(3)

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-1.jpg")
	image1 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-2.jpg")
	image2 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/sunflower-1.jpg")
	image3 = Image.open(BytesIO(response.content))

	col1.image(image1, caption='Rose', width=150)
	col2.image(image2, caption='Rose', width=150)
	col3.image(image3, caption='Sunflower', width=150)
	
if label_option == '3+':
	col1, col2, col3 = expander.beta_columns(3)

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-1.jpg")
	image1 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/rose-2.jpg")
	image2 = Image.open(BytesIO(response.content))

	response = requests.get("https://storage.googleapis.com/droughtwatch/images/sunflower-1.jpg")
	image3 = Image.open(BytesIO(response.content))

	col1.image(image1, caption='Rose', width=150)
	col2.image(image2, caption='Rose', width=150)
	col3.image(image3, caption='Sunflower', width=150)
		
expander = st.beta_expander("5. Planned Approach")		
expander.write("Step 1: Transfer learning applying feature extraction on multiple models like Squeezenet, Densenet-121, Resnet-152, UNet, FasterRCNN and analyze derived accuracy from each model. Sweeps from Weights & Biases to be used in order to tune hyperparameters. Data augmentation techniques to be applied and see what impact it makes.")
expander.write("Step 2: Understand the working of Data efficient image transformers (DeiT) and apply it to the Drought watch landsat images and analyze the accuracy it results in.")
expander.write("Step 3: Depending on the accuracy, submit a report to W&B.")	
