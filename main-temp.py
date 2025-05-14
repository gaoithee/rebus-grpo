import unsloth
from unsloth import FastLanguageModel
import torch

from utils import estrai_rebus_e_chiave

from datasets import load_dataset

# Model uploading 
# Passa parametri da terminale?

# Inference
template = """
    <bos><start_of_turn>user
    Risolvi gli indizi tra parentesi per ottenere una prima lettura, e usa la chiave di lettura per ottenere la soluzione del rebus.

    Rebus: {rebus}
    Chiave risolutiva: {key}<end_of_turn>
    <start_of_turn>model
"""


rebus_raw, chiave = estrai_rebus_e_chiave(dataset_ood[10]['conversations'][0]['value'])