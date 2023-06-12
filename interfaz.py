import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms, models
import numpy as np

from SSTree2 import SSTree
from animals import Encoder

class App:
    def __init__(self, root, tree, model, device):
        self.root   = root
        self.tree   = tree
        self.model  = model
        self.device = device
        self.image_size = (300, 300)

        self.frame = tk.Frame(root)
        self.frame.pack()

        self.left_frame = tk.Frame(self.frame)
        self.left_frame.pack(side=tk.LEFT)

        self.right_frame = tk.Frame(self.frame)
        self.right_frame.pack(side=tk.RIGHT)

        self.input_path = tk.StringVar()

        self.entry = tk.Entry(self.left_frame, textvariable=self.input_path)
        self.entry.pack()

        self.browse_button = tk.Button(self.left_frame, text="Browse", command=self.browse_image)
        self.browse_button.pack()

        self.search_button = tk.Button(self.left_frame, text="Search", command=self.search_similar)
        self.search_button.pack()

        self.image_label = tk.Label(self.left_frame)
        self.image_label.pack()

        self.result_labels = [tk.Label(self.right_frame) for _ in range(4)]
        for i, label in enumerate(self.result_labels):
            label.grid(row=i//2, column=i%2)

    def browse_image(self):
        file_path = filedialog.askopenfilename()
        self.input_path.set(file_path)
        image = Image.open(file_path)
        image = image.resize(self.image_size, Image.Resampling.LANCZOS)

        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def search_similar(self):
        embedding = self.get_image_embedding(self.input_path.get(), self.model)
        result = self.tree.knn(embedding, 4)
        for i, res in enumerate(result):
            image = Image.open(res['path'])
            image = image.resize(self.image_size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.result_labels[i].config(image=photo)
            self.result_labels[i].image = photo
        
    def get_image_embedding(self,image_path, model):
        image = Image.open(image_path).convert("RGB")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image = preprocess(image).unsqueeze(0).to(self.device)
        model.eval()
        with torch.no_grad():
            output = model(image)
        embedding = output.cpu().numpy()[0]
        norm = np.linalg.norm(embedding)
        return embedding/(norm+np.finfo(np.float32).eps)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root = tk.Tk()
    root.title("SS-Tree")
    model = Encoder().to(device)
    for param in model.parameters():
        param.requires_grad = False
    
    tree = SSTree(filename='tree.ss')
    app = App(root, tree, model, device)
    root.mainloop()
