# %%
import os
import clip
# for macos pip install scikit-image
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch

model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()

# images in skimage to use and their textual descriptions
descriptions = {
    "celtics": "the Boston Celtics",
    "bucks": "the Milwaukee Bucks",
    "bulls": "the Chicago Bulls",
    "cavs": "the Cleveland Cavaliers",
    "heat": "the Miami Heat",
    "lakers": "the Los Angeles Lakers",
    "mavs": "the Dallas Mavericks",
    "nets": "the Brooklyn Nets",
}

original_images = []
images = []
texts = []

plt.rcParams.update({
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "black",
    "axes.facecolor": "black",
    "axes.edgecolor": "lightgray",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "lightgray",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black"})

plt.figure(figsize=(16, 5), facecolor='black')

# parent directory of python file
current_directory = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))+"/test-images/"
print(current_directory)

for filename in [filename for filename in os.listdir(current_directory) if filename.endswith(".png") or filename.endswith(".jpg")]:
    name = os.path.splitext(filename)[0]
    print(name)
    if name not in descriptions:
        continue
    fileLoc = current_directory+filename
    image = Image.open(fileLoc).convert("RGB")

    plt.subplot(2, 4, len(images) + 1)
    plt.imshow(image)
    plt.title(f"{filename}\n{descriptions[name]}", color='gray')
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))
    texts.append(descriptions[name])
plt.tight_layout()
plt.show()

# ======================================================================

image_input = torch.tensor(np.stack(images))
text_tokens = clip.tokenize(["This is " + desc for desc in texts])

with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

count = len(descriptions)

plt.figure(figsize=(20, 14), facecolor='black')
plt.imshow(similarity, vmin=0.1, vmax=0.3)
plt.colorbar()
plt.yticks(range(count), texts, fontsize=10, color='white')
plt.xticks([], color='white')
for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6),
               origin="lower")
for x in range(similarity.shape[1]):
    for y in range(similarity.shape[0]):
        plt.text(x, y, f"{similarity[y, x]:.2f}",
                 ha="center", va="center", size=10, color='white')

for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity between text and image features",
          size=20, color='white')
plt.show()
