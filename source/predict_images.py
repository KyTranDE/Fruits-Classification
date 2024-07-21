# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing import image
# import matplotlib.pyplot as plt
# import joblib

# base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
# def predict_image(img_path, model,base_model):
#     img = image.load_img(img_path, target_size=(224, 224))
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = img_array / 255.0

#     features = base_model.predict(img_array)
#     predictions = model.predict(features)
#     # [0] : Apple, [1] : Banana, [2] : Grape, [3] : Mango, [4] : Strawberry
#     data = {0:'Apple', 1:'Banana', 2:'Grape', 3:'Mango', 4:'Strawberry'}
#     title_color = 'red'
#     title_font = {'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': 14}

#     plt.rcParams['toolbar'] = 'None'
#     plt.figure(num='Predict Result',figsize=(10, 6))
#     plt.title(
#         "[0] : Apple, [1] : Banana, [2] : Grape, [3] : Mango, [4] : Strawberry \n" +
#         "Predicted : " + data.get(int(predictions[0])),
#         color=title_color,
#         fontdict=title_font
#     )
#     plt.axis('off')
#     plt.tight_layout()
#     plt.imshow(img)
#     plt.show()

# loaded_model = joblib.load('lgb_model.pkl')

# def extract_path(path):
#     path = path.replace("'\'", '/')
#     return path

# if __name__ == '__main__':
#     # path = input("Enter image path : ")
#     predict_image(extract_path("D:\Fruits Classification\download (1).jpg"), loaded_model,base_model)

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import joblib
import tkinter as tk
from tkinter import filedialog

# Load the pre-trained base model
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Function to predict the image class
def predict_image(img_path, model, base_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    features = base_model.predict(img_array)
    predictions = model.predict(features)
    # [0] : Apple, [1] : Banana, [2] : Grape, [3] : Mango, [4] : Strawberry
    data = {0: 'Apple', 1: 'Banana', 2: 'Grape', 3: 'Mango', 4: 'Strawberry'}
    title_color = 'red'
    title_font = {'family': 'serif', 'color': 'darkred', 'weight': 'bold', 'size': 14}

    plt.rcParams['toolbar'] = 'None'
    plt.figure(num='Predict Result', figsize=(10, 6))
    plt.title(
        "[0] : Apple, [1] : Banana, [2] : Grape, [3] : Mango, [4] : Strawberry \n" +
        "Predicted : " + data.get(int(predictions[0])),
        color=title_color,
        fontdict=title_font
    )
    plt.axis('off')
    plt.tight_layout()
    plt.imshow(img)
    plt.show()

# Load the model
loaded_model = joblib.load('./models/lgb_model.pkl')

# Function to select an image file
def select_image():
    img_path = filedialog.askopenfilename()
    if img_path:
        predict_image(img_path, loaded_model, base_model)

# Function to create the GUI
def create_gui():
    root = tk.Tk()
    root.title('Image Classifier')

    canvas = tk.Canvas(root, height=150, width=400)
    canvas.pack()

    frame = tk.Frame(root)
    frame.place(relwidth=1, relheight=1)

    button = tk.Button(frame, text='Select Image', padx=10, pady=5, command=select_image)
    button.pack()

    root.mainloop()

if __name__ == '__main__':
    create_gui()
