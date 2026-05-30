import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

print("Loading MNIST Dataset")

transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    batch_size=4, 
    shuffle=True
)

print(f"Loaded {len(train_dataset)} training images.")

data_iter = iter(train_loader)
images, labels = next(data_iter)

fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for i in range(4):
    img = images[i].squeeze().numpy()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis('off')

print("Displaying window, close it to exit.")
plt.show()