from flask import Flask, request, jsonify
from PIL import Image
import io
from infer import ImageCaptioningInference
from models.model import ImageCaptioningModel

app = Flask(__name__)

model_dir = 'model'
    
# Initialize inference class
model = ImageCaptioningModel()
model.load(model_dir)

inference_model = ImageCaptioningInference(model)

# # Path to the input image
# image_path = 'test_img.jpg'

# # Perform inference and print the generated caption
# caption = inference_model.infer_image(image_path)
# print("Generated Caption:", caption)

@app.route('/')
def home():
    return "Welcome to the Flask API"

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'})
    
    image = request.files['image']
    # print(image)

    # try :
    image = Image.open(io.BytesIO(image.read()))
    # image.show()

    generated_caption = inference_model.infer_image(image)

    return jsonify({'generated_caption': generated_caption})

    
    # except Exception as e:
    #     return jsonify({'error': f'{e}'}), 500
    
if __name__ == '__main__':
    app.run(debug=True)