import os
import tensorflow as tf
import librosa
from flask import Flask, request, send_file

# Initialize the Flask app
app = Flask(__name__)

# Path to the pre-trained model directory
MODEL_PATH = os.path.join('model', 'model.ckpt-1000000')

# Define the model (assuming it’s based on U-Net)
class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        # Define the layers of the model (this is a simplified version)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')

    def call(self, inputs):
        x = self.conv1(inputs)
        return x

# Create a model instance
model = UNet()

# Load the checkpoint
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(MODEL_PATH).expect_partial()

# Function to separate audio using the model
def separate_audio(file_path):
    audio, sr = librosa.load(file_path, sr=44100, mono=False)
    
    # Ensure the input shape matches the model's expected input
    audio_input = audio[np.newaxis, ...]  # Add batch dimension

    # Run the model to separate the audio
    separated = model(audio_input)  # Model prediction
    
    # Extract the separated components (vocals and accompaniment)
    vocals = separated[0][0]
    accompaniment = separated[0][1]

    # Save the separated audio files
    vocals_path = os.path.join('separated', 'vocals.wav')
    accompaniment_path = os.path.join('separated', 'accompaniment.wav')

    # Save using librosa
    librosa.output.write_wav(vocals_path, vocals, sr)
    librosa.output.write_wav(accompaniment_path, accompaniment, sr)

    return vocals_path, accompaniment_path

# Route to upload file and separate audio
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']  # Get the file from the form
    if file:
        file_path = os.path.join('uploads', file.filename)  # Save the file
        file.save(file_path)

        # Call the separation function
        vocals_path, accompaniment_path = separate_audio(file_path)

        # Send the separated files back to the user
        return send_file(vocals_path), send_file(accompaniment_path)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
