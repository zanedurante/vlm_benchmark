from enum import Enum
import numpy as np


'''
Enum for different available similarity metrics
'''

class Similarity(Enum):
    DOT = 1
    COSINE = 2
    EUCLID = 3
    
    '''
    Args:
        a (np.array):               Shape = (A, embed_dim)
        b (np.array):               Shape = (B, embed_dim)
    Returns:
        similarities (np.array):    Shape = (A, B)
    '''
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self is Similarity.DOT:
            return np.sum(a[:, None, :] * b[None, :, :], axis=2)
        
        if self is Similarity.COSINE:
            a_mag = np.sqrt(np.sum(np.square(a), axis=1))
            b_mag = np.sqrt(np.sum(np.square(b), axis=1))
            return np.sum(a[:, None, :] * b[None, :, :], axis=2) / (a_mag * b_mag)
        
        if self is Similarity.EUCLID:
            return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)