import streamlit as st
import matplotlib.pyplot as plt

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
	st.pyplot(train_figure)
	st.write('On looking at the data, it is evident that there is a huge class imbalance. On later iterations, this will be handled to see how it affects the accuracy.')
	
# Validation Data Plot
val_figure = plt.figure(figsize=(15, 6))
plt.title('Forage Quality Analysis of Validation Data')
plt.xlabel('How many cows can be fed?')
plt.ylabel('Number of landsat images')
# TODO: To send this data from a API
counts_elements = [51814, 12851, 13609, 8043]
unique_elements = [0, 1, 2, 3]
plt.bar(unique_elements, counts_elements, color='blue')

# To display count of each label
for index, data in enumerate(counts_elements):
  plt.text(x=index, y=data+1, s=f'{data}', fontdict=dict(fontsize=15))
 
if val_clicked:
	st.pyplot(val_figure)
