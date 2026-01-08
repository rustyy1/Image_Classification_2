import gradio as gr
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model("F:\Image_Classification_2\code\models\cat_breed_model_final.keras")

class_names = sorted(['Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair', 'Egyptian_Mau', 
                      'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', 'Sphynx']) 

def classify_cat_breed(img):
 
    img = img.resize((224, 224))
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img))
    img_array = np.expand_dims(img_array, axis=0)
    
 
    predictions = model.predict(img_array, verbose=0)[0]
    top5 = np.argsort(predictions)[-5:][::-1]
    return {class_names[i]: float(predictions[i]) for i in top5}


interface = gr.Interface(
    fn=classify_cat_breed,
    inputs=gr.Image(type="pil", label="Upload Cat Image"),
    outputs=gr.Label(num_top_classes=5, label="Top 5 Predictions"),
    examples=[
        "data/train/Abyssinian/abyssinian_1.jpg",
        "data/train/Bengal/bengal_1.jpg",
        "data/train/Siamese/siamese_1.jpg"
    ],
    title="Cat Breed Classifier",
    description="Upload a cat image to detect its breed."
)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=True)