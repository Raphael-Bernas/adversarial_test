import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

def image_to_vector(image_path, n_size=64):
    
    img = Image.open(image_path)
    
    img = img.convert('L')
    
    img = img.resize((n_size, n_size))

    img_array = np.array(img).flatten()
    
    print(f"Flattened image shape: {img_array.shape}")
    
    return img_array

# Example usage
vector = image_to_vector('logo/e1.png')

def get_image_vector(N, label, directory='logo', n_size=64):

    if label not in ['e', 'p']:
        raise ValueError("Label must be 'e' or 'p'")
    
    image_filename = f"{label}{N}.png"
    
    image_path = os.path.join(directory, image_filename)
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found")
    
    return image_to_vector(image_path, n_size=n_size)

try:
    vector = get_image_vector(1, 'e')
    print(vector)
except Exception as e:
    print(e)

def plot_resized_image(image_path, n_size=64):

    img = Image.open(image_path)
    
    img = img.convert('L')
    
    img = img.resize((n_size, n_size))
    
    img_array = np.array(img)
    
    # Plot the image
    plt.imshow(img_array, cmap='gray')
    plt.title('Resized'+n_size+'X'+n_size+'Grayscale Image')
    plt.axis('off')
    plt.show()

try:
    vector = get_image_vector(1, 'e')
    print(vector)
except Exception as e:
    print(e)

try:
    plot_resized_image('logo/e1.png')
except Exception as e:
    print(e)



class ImageDataset(Dataset):
    def __init__(self, directory, n_size=64):
        self.directory = directory
        self.n_size = n_size
        self.image_labels = []
        self.images = []

        for label in ['e', 'p']:
            for i in range(1, 8):
                image_path = os.path.join(directory, f"{label}{i}.png")
                image_vector = image_to_vector(image_path, self.n_size)
                self.images.append(image_vector)
                self.image_labels.append(0 if label == 'e' else 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.image_labels[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Neural network model
class FullyConnectedNN(nn.Module):
    def __init__(self):
        super(FullyConnectedNN, self).__init__()
        self.fc1 = nn.Linear(4096, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

# Training the model
def train_model(directory, num_epochs=100, batch_size=2, learning_rate=0.001):
    dataset = ImageDataset(directory)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FullyConnectedNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return model


# Fast Gradient Sign Method 
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return perturbed_image

def test_adversarial_attack(model, dataset, epsilon, plot=False):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    model.eval()

    for inputs, labels in dataloader:
        inputs.requires_grad = True

        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()

        data_grad = inputs.grad.data
        perturbed_data = fgsm_attack(inputs, epsilon, data_grad)

        perturbed_output = model(perturbed_data)

        print("Original Label:", labels.item())
        print("Adversarial Label:", perturbed_output.argmax(dim=1).item())
        if plot:
            plot_adversarial_image(inputs[0], perturbed_data[0], labels.item(), perturbed_output.argmax(dim=1).item())

        break  # test one image
    return labels.item(), perturbed_output.argmax(dim=1).item()

def plot_adversarial_image(original_image, adversarial_image, label, adversarial_label, n_size=64):
    original_image = original_image.detach().numpy().reshape((n_size, n_size))
    adversarial_image = adversarial_image.detach().numpy().reshape((n_size, n_size))

    if label == 0:
        label = 'ENSTA'
    else:
        label = 'Polytechnique'
    if adversarial_label == 0:
        adversarial_label = 'ENSTA'
    else:
        adversarial_label = 'Polytechnique'

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image. Label: ' + label)
    axes[0].axis('off')

    axes[1].imshow(adversarial_image, cmap='gray')
    axes[1].set_title('Adversarial Image. Label: ' + adversarial_label)
    axes[1].axis('off')

    plt.show()

model = train_model('logo')
test_dataset = ImageDataset('logo')
label, adv_label = 0, 0
i = 0
epsilons = np.linspace(2, 100, num=1000)
while label == adv_label:
    epsilon = epsilons[i]
    i += 1
    label, adv_label = test_adversarial_attack(model, test_dataset, epsilon=epsilon, plot=True)
    if label != adv_label:
        print("Got you a misclassified one !")
    print("Epsilon:", epsilon)