from flask import Flask, send_file
import torch
from generator import Generator, generate_image
from torchvision.utils import save_image

app = Flask(__name__)

# Load the generator model
generator = Generator(nz=100, ngf=64, nc=3)
generator.load_state_dict(torch.load('../models/generator_model.pth'))

@app.route('/generate-face', methods=['GET'])
def generate_face():
    # Generate a face
    fake_face = generate_image(generator)
    
    # Save the generated image
    save_image(fake_face, 'static/generated_face.png')
    
    # Return the image to the frontend
    return send_file('static/generated_face.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
