"""Cabeza de clasificación de emoción sobre los embeddings de HuBERT.

Una sola cabeza: capa densa oculta con ReLU + dropout que toma el embedding de
768 dimensiones y devuelve los logits de las 7 clases de la taxonomía unificada.
HuBERT está congelado y no forma parte de este módulo: la cabeza se entrena sola
sobre los embeddings ya extraídos.
"""

import torch.nn as nn

from . import config


class CabezaEmocion(nn.Module):
    def __init__(self, dim_entrada=None, dim_oculta=None,
                 num_clases=None, dropout=None):
        super().__init__()
        dim_entrada = dim_entrada or config.EMBEDDING_DIM
        dim_oculta = dim_oculta or config.HIDDEN_DIM
        num_clases = num_clases or config.NUM_CLASSES
        dropout = config.DROPOUT if dropout is None else dropout

        self.red = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim_entrada, dim_oculta),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_oculta, num_clases),
        )

    def forward(self, x):
        return self.red(x)
