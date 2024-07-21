import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import joblib
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

def predict_image(img_path, model, base_model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    features = base_model.predict(img_array)
    predictions = model.predict(features)
    # [0] : Apple, [1] : Banana, [2] : Grape, [3] : Mango, [4] : Strawberry
    data = {0: 'Apple', 1: 'Banana', 2: 'Grape', 3: 'Mango', 4: 'Strawberry'}
    predicted_label = data.get(int(predictions[0]))
    return predicted_label

loaded_model = joblib.load('../models/lgb_model.pkl')

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Classifier GUI")
        self.geometry("800x400")
        self.resizable(False, False)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2, 3, 4), weight=1)

        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Image Classifier", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.sidebar_button_1 = ctk.CTkButton(self.sidebar_frame, text="Select Image", command=self.browse_file)
        self.sidebar_button_1.grid(row=1, column=0, padx=20, pady=10)
        
        self.sidebar_button_2 = ctk.CTkButton(self.sidebar_frame, text="Classify Image", command=self.classify_image)
        self.sidebar_button_2.grid(row=2, column=0, padx=20, pady=10)

        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"], command=self.change_appearance_mode)
        self.appearance_mode_optionmenu.grid(row=6, column=0, padx=20, pady=(10, 10))

        self.scaling_label = ctk.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0))
        self.scaling_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["80%", "90%", "100%", "110%", "120%"], command=self.change_scaling)
        self.scaling_optionmenu.grid(row=8, column=0, padx=20, pady=(10, 20))

        self.entry_frame = ctk.CTkFrame(self)
        self.entry_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.entry_frame.grid_columnconfigure(1, weight=1)

        self.image_path_label = ctk.CTkLabel(self.entry_frame, text="Image Path:")
        self.image_path_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")
        self.entry_image_path = ctk.CTkEntry(self.entry_frame)
        self.entry_image_path.grid(row=0, column=1, padx=10, pady=10, sticky="w")

        self.image_label = ctk.CTkLabel(self.entry_frame, text="")
        self.image_label.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.result_label = ctk.CTkLabel(self.entry_frame, text="", font=ctk.CTkFont(size=16, weight="bold"))
        self.result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def browse_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.entry_image_path.delete(0, ctk.END)
            self.entry_image_path.insert(0, file_path)
            self.display_image(file_path)

    def display_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((224, 224), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img)
        self.image_label.image = img

    def classify_image(self):
        img_path = self.entry_image_path.get()
        if img_path:
            try:
                predicted_label = predict_image(img_path, loaded_model, base_model)
                self.result_label.configure(text=f"Predicted: {predicted_label}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
        else:
            messagebox.showwarning("Input Error", "Please select an image file.")

    def change_appearance_mode(self, new_mode):
        ctk.set_appearance_mode(new_mode)

    def change_scaling(self, new_scaling):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

if __name__ == "__main__":
    app = App()
    app.mainloop()
