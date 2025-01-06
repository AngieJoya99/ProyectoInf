from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms

# Lista de clases
clases = ['Avión', 'Carro', 'Pájaro', 'Gato', 'Ciervo', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

# Definición de la clase del modelo
class Network(nn.Module):
    def __init__(self):
        out_1 = 32
        out_2 = 64
        out_3 = 128
        p = 0.5
        
        super(Network, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=3, out_channels=out_1, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1_bn = nn.BatchNorm2d(out_1)
        self.drop_conv = nn.Dropout(p=0.2)
        
        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv2_bn = nn.BatchNorm2d(out_2)
        
        self.cnn3 = nn.Conv2d(in_channels=out_2, out_channels=out_3, kernel_size=5, padding=2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv3_bn = nn.BatchNorm2d(out_3)
        
        # Hidden layers
        self.fc1 = nn.Linear(out_3 * 4 * 4, 1000)
        self.drop = nn.Dropout(p=p)
        self.fc1_bn = nn.BatchNorm1d(1000)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, 1000)
        self.fc3_bn = nn.BatchNorm1d(1000)
        
        self.fc4 = nn.Linear(1000, 1000)
        self.fc4_bn = nn.BatchNorm1d(1000)
        
        # Final layer
        self.fc5 = nn.Linear(1000, 10)
        self.fc5_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.conv1_bn(x)
        x = self.maxpool1(x)
        x = self.drop_conv(x)
        
        x = self.cnn2(x)
        x = self.conv2_bn(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        x = self.drop_conv(x)
        
        x = self.cnn3(x)
        x = self.conv3_bn(x)
        x = torch.relu(x)
        x = self.maxpool3(x)
        x = self.drop_conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc2(x)
        x = self.fc2_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc3(x)
        x = self.fc3_bn(x)
        
        x = F.relu(self.drop(x))
        x = self.fc4(x)
        x = self.fc4_bn(x)

        x = F.relu(self.drop(x))
        x = self.fc5(x)
        x = self.fc5_bn(x)
        
        return x

# Configuración del servidor Flask
app = Flask(__name__)

# Carga del modelo
model_path = os.path.join("dataset", "modelo-entrenado.pth")
model = Network()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Transformación de imagen a tensor
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Asegura que la imagen tenga el tamaño adecuado
    transforms.ToTensor(),        # Convierte la imagen a un tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normaliza la imagen
])

@app.route('/predict', methods=['POST'])
def predict():
    # Recibe la imagen desde el dispositivo local en formato multipart/form-data
    file = request.files['image']
    image = Image.open(BytesIO(file.read())).convert('RGB')  # Convierte a RGB

    # Aplica la transformación a la imagen
    input_tensor = transform(image)
    input_tensor = input_tensor.unsqueeze(0)  # Añade la dimensión del batch

    # Realiza la predicción
    outputs = model(input_tensor)
    predicted_class_idx = outputs.argmax(dim=1).item()  # Obtiene el índice de la clase predicha

    # Obtiene el nombre de la clase predicha
    predicted_class = clases[predicted_class_idx]

    # Imprime el nombre de la clase predicha
    print(f'Predicción: {predicted_class}')

    return jsonify(predictions=predicted_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
