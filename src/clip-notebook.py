# %%
import clip
import numpy as np
import torch
from pkg_resources import packaging

from collections import OrderedDict

print("Torch version:", torch.__version__)
clip.available_models()

model, preprocess = clip.load("ViT-B/32", device="cpu")
model.eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:",
      f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

print(clip.tokenize("Hello World!"))

# ======================================================================
