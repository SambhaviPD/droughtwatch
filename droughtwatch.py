from PIL import Image
from io import BytesIO
from requests_toolbelt.multipart.encoder import MultipartEncoder

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import requests
import urllib.request as url

st.set_page_config(
	page_title = "Drought Watch",
	page_icon = "cactus"
	)
	
@st.cache(suppress_st_warning=True)
def get_sample_images(label):
	if label == "0":
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%200/train_1_label%200.jpg")
		image1 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%200/train_2_label%200.jpg")
		image2 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%200/train_3_label%200.jpg")
		image3 = Image.open(BytesIO(response.content))
		
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%200/train_4_label%200.jpg")
		image4 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%200/train_5_label%200.jpg")
		image5 = Image.open(BytesIO(response.content))
		
	elif label == "1":
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%201/train_1_label%201.jpg")
		image1 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%201/train_2_label%201.jpg")
		image2 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%201/train_3_label%201.jpg")
		image3 = Image.open(BytesIO(response.content))
		
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%201/train_4_label%201.jpg")
		image4 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%201/train_5_label%201.jpg")
		image5 = Image.open(BytesIO(response.content))
		
	elif label == "2":
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%202/train_1_label%202.jpg")
		image1 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%202/train_2_label%202.jpg")
		image2 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%202/train_3_label%202.jpg")
		image3 = Image.open(BytesIO(response.content))
		
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%202/train_4_label%202.jpg")
		image4 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%202/train_5_label%202.jpg")
		image5 = Image.open(BytesIO(response.content))
		
	else:
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%203%2B/train_1_label%203.jpg")
		image1 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%203%2B/train_2_label%203.jpg")
		image2 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%203%2B/train_3_label%203.jpg")
		image3 = Image.open(BytesIO(response.content))
		
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%203%2B/train_4_label%203.jpg")
		image4 = Image.open(BytesIO(response.content))

		response = requests.get("https://storage.googleapis.com/droughtwatch/images/train/label%203%2B/train_4_label%203.jpg")
		image5 = Image.open(BytesIO(response.content))
		
	return image1, image2, image3, image4, image5

@st.cache(suppress_st_warning=True)
def get_charts(model_name):
	if model_name == "Squeezenet":
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/charts/squeezenet-charts.jpg")
		image = Image.open(BytesIO(response.content))
	elif model_name == "Densenet-121":
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/charts/densenet121-charts.jpg")
		image = Image.open(BytesIO(response.content))
	else:
		return('Incorrect argument')
	return image		
	
def get_val_images(image_name):
	if image_name == 'Image 1':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%200/val_1_label%200.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 2':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%200/val_2_label%200.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 3':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%201/val_1_label%201.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 4':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%201/val_2_label%201.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 5':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%202/val_1_label%202.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 6':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%202/val_2_label%202.jpg")
		image = Image.open(BytesIO(response.content))	
	elif image_name == 'Image 7':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%203%2B/val_1_label%203.jpg")
		image = Image.open(BytesIO(response.content))
	elif image_name == 'Image 8':
		response = requests.get("https://storage.googleapis.com/droughtwatch/images/val/label%203%2B/val_2_label%203.jpg")
		image = Image.open(BytesIO(response.content))
	else:
		return None
	return image	
		
st.title(":cactus: Weights & Biases Public Benchmark")
st.title("- for Drought Prediction :cow:")
st.subheader("_My course project for FSDL Spring 2021_ ")

st.sidebar.title("Author")
st.sidebar.write("[Sambhavi Dhanabalan] (https://www.linkedin.com/in/sambhavi-dhanabalan/)")

st.sidebar.title("Tools used")
st.sidebar.write(":one: Pytorch :two: Google Colab Pro :three: Streamlit  :four: FastAPI :five: Heroku")

st.sidebar.title("Git links")
st.sidebar.write("[Streamlit App] (https://github.com/SambhaviPD/droughtwatch/tree/master)")
st.sidebar.write("[Training Notebooks] (https://github.com/SambhaviPD/droughtwatch/tree/main)")

expander = st.beta_expander("1. Introduction")
expander.write(":trophy: Public benchmarks in Weights & Biases encourages collaboration from the community for a variety of problem statements. The one that I worked on as part of FSDL's final course project was the Drought Watch. The intent of this project is to predict drought severity from satellite imagery and ground-level photos. Complete details about the project from Weights & Biases can be found here: [Drought Watch] (https://arxiv.org/pdf/2004.04081.pdf)")

expander = st.beta_expander("2. Project Goals")
expander.write(":up: 77.8 is the top accuracy in the project's leader board. To surpass that is my primary goal. In order to do that, an initial understanding of satellite imagery is required, followed by the right model architecture.")
expander.write(":rain_cloud: Secondary goal would be to isolate images obscured with clouds. Cloud detection algorithms need to be studied and applied to the input data set. If I'm able to isolate successfully, then need to study what impact it makes on improving the accuracy.")

expander = st.beta_expander("3. Stretch Goals")
expander.write(":clipboard: If I am able to enter the leaderboard, submitting a report to W&B is a goal that I'd planned for.")
expander.write(":earth_asia: eo-learn is a package that provides a set of tools to work with Earth Observation(EO) data. Did a high level study and decided to explore more on eo-patch and eo-workflows in order to view the green patches as well as clouds in images. In short, a deeper visualization of images will give more directions.")

expander = st.beta_expander("4. About the Data set")
expander.write(":cactus: Drought watch data set can be downloaded from the [git repo] (https://github.com/wandb/droughtwatch). A total of 86,317 training images and 10,778 validation images is present. Human experts (pastoralists) have labeled these with the number of cows that the geographic location at the center of the image could support (0, 1, 2, or 3+ cows). Each pixel represents a 30 meter square, so the images at full size are 1.95 kilometers across. Pastoralists are asked to rate the quality of the area within 20 meters of where they are standing, which corresponds to an area slightly larger a single pixel.") 

expander.subheader(':female-detective: 4a. Let\'s take a look at Training and Validation split')

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
	
expander.subheader(':cow: 4b. Let\'s take a look at few samples')
expander.write('Label 0 denotes that the images has forage quantity that is insufficient to feed even a single cow. Label 1 denotes that the image has forage quantity that is sufficient to feed just 1 cow. Likewise for Label 2 and 3+.')

label_option = expander.selectbox(
	'Choose a Label to see corresponding images',
	('--Please select--', '0', '1', '2', '3+'))

# To display few sample images
if label_option == '0':
	
	image1, image2, image3, image4, image5 = get_sample_images("0")
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image1, caption='Label 0', width=200)
	col2.image(image2, caption='Label 0', width=200)
	col3.image(image3, caption='Label 0', width=200)
	
	col1, col2, col3 = expander.beta_columns(3)
	
	col1.image(image4, caption='Label 0', width=200)
	col2.image(image5, caption='Label 0', width=200)
	
if label_option == '1':

	image1, image2, image3, image4, image5 = get_sample_images("1")
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image1, caption='Label 1', width=200)
	col2.image(image2, caption='Label 1', width=200)
	col3.image(image3, caption='Label 1', width=200)
	
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image4, caption='Label 1', width=200)
	col2.image(image5, caption='Label 1', width=200)
	
if label_option == '2':

	image1, image2, image3, image4, image5 = get_sample_images("2")
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image1, caption='Label 2', width=200)
	col2.image(image2, caption='Label 2', width=200)
	col3.image(image3, caption='Label 2', width=200)
	
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image4, caption='Label 2', width=200)
	col2.image(image5, caption='Label 2', width=200)
	
if label_option == '3+':
	image1, image2, image3, image4, image5 = get_sample_images("3+")
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image1, caption='Label 3+', width=200)
	col2.image(image2, caption='Label 3+', width=200)
	col3.image(image3, caption='Label 3+', width=200)
	
	col1, col2, col3 = expander.beta_columns(3)

	col1.image(image4, caption='Label 3+', width=200)
	col2.image(image5, caption='Label 3+', width=200)
		
expander = st.beta_expander("5. Planned Approach")	
expander.write(":top: A high level plan below to make it to the leader board")	
expander.write("Step 1: Transfer learning applying feature extraction on multiple models like Squeezenet, Densenet-121, Resnet-152, UNet, FasterRCNN and analyze derived accuracy from each model. Sweeps from Weights & Biases to be used in order to tune hyperparameters. Data augmentation techniques to be applied and see what impact it makes.")
expander.write("Step 2: Understand the working of Data efficient image transformers (DeiT) and apply it to the Drought watch landsat images and analyze the accuracy it results in.")
expander.write("Step 3: Depending on the accuracy, submit a report to W&B.")

expander = st.beta_expander("6. Initial Findings")
expander.write(":pencil2: Each pre-trained model was trained for 50-75 epochs on the landsat dataset and results captured in W&B.")

model_choice = expander.radio("Choose a model to analyze it\'s output",
							("Squeezenet", "Densenet-121", "Resnet-152"))
expander.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

if model_choice == "Squeezenet":
	col1, col2 = expander.beta_columns(2)

	image1 = get_charts("Squeezenet")
	col1.image(image1, caption='Squeezenet - Training & Validation Loss & Accuracy - 50 epochs. Image snapshot taken from corresponding run from my wandb dashboard.', width=800)
	
	expander.write("On analyzing the charts, val accuracy plateaus right from early epochs and it's not on the upward trend always. Val loss has varied spikes without a pattern. Plan is to see what happens when number of epochs is increased as a first step.")
	
	expander.write("Post 50 epochs, the metrics are : Training Accuracy: 0.7457, Training Loss: 0.6315, Val Accuracy: 0.683, Val Loss: 1.395")
	
	expander.write("________________________________________________________________________________________________")

if model_choice == "Densenet-121":
	col1, col2 = expander.beta_columns(2)

	image1 = get_charts("Densenet-121")
	col1.image(image1, caption='Densenet 121 - Training & Validation Loss & Accuracy - 75 epochs. Image snapshot taken from corresponding run from my wandb dashboard.', width=800)
	
	expander.write("On analyzing the charts, val accuracy plateaus right from 40th epoch. Hence increasing the number of epochs with just feature extraction with set config parameters will not help. Plan is to play around with finetuning hyperparameters using sweeps.")
	
	expander.write("Post 75 epochs, the metrics are : Training Accuracy: 0.801, Training Loss: 0.5085, Val Accuracy: 0.7523, Val Loss: 0.752")
	
	expander.write("________________________________________________________________________________________________")
	
expander.write("**Since test images are unavailable, I have uploaded a couple of images from val dataset in my git repo under images folder.**")
expander.write("**Please ensure to use them if you would like to test. Other images may throw an exception at the moment.**")

expander.subheader('Upload a landsat image')
uploaded_file = expander.file_uploader(label="Make sure you use the provided sample image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False, key="None", help="Only .png, .jpg, and .jpeg files are supported")

expander.write(":vertical_traffic_light: Select a model to see it's prediction vs the true value.")

col1, col2 = expander.beta_columns(2)
model_option = col1.selectbox('Select a model',
	('Squeezenet', 'Densenet-121'))
print(model_option)
	
col1, col2 = expander.beta_columns(2)

#classifyapi_url = "https://rocky-basin-83323.herokuapp.com/predict"

if uploaded_file is not None:
	
	m = MultipartEncoder(
		fields={'input_image': ('filename', uploaded_file, 'image/jpeg')}
	
	)

	# crude fix to avoid sending both model name and file to a single post method
	# Will be changed
	if model_option == 'Squeezenet':
		classifyapi_url = "https://rocky-basin-83323.herokuapp.com/squeezenetpredict"
	else:
		classifyapi_url = "https://rocky-basin-83323.herokuapp.com/densenet121predict"
		
	response = requests.post(classifyapi_url, 
					data=m,
					headers={
						'Content-Type': m.content_type
					})
	
	if response is None:
		st.subheader('Classify API is down. Please try later!')
	else:
		output = response.json()
		
		col1, col2 = expander.beta_columns(2)
		col1.subheader('True Label:')
		# Another crude way, since images should be chosen
		# only from mentioned git folder
		col2.subheader(uploaded_file.name[12:13])
		
		col1, col2 = expander.beta_columns(2)
		col1.subheader('Predicted Label:')
		col2.subheader(output['Prediction'])
		
		col1, col2 = expander.beta_columns(2)
		col1.subheader('Prediction score:')
		col2.subheader(output['Score'])
		
		expander.subheader('Prediction scores of all classes')
		col1, col2 = expander.beta_columns(2)
		col1.write(output['labels'])
		col2.write(output['Scores'])

