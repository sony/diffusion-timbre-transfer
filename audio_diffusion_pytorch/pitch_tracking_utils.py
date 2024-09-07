import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import io
import base64

import sys
basic_pitch_dir = os.path.join(os.path.dirname(__file__), 'basic-pitch-torch')
sys.path.append(basic_pitch_dir)
from basic_pitch_torch.model import BasicPitchTorch


def remove_low_amplitude_frames(tensor: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    mask = np.any(tensor > threshold, axis=1)
    return tensor[mask]

def aggregate_to_semitones(note_array: np.ndarray) -> np.ndarray:
    batch_size, time_axis, num_notes = note_array.shape
    semitone_matrix = np.zeros((batch_size,time_axis, 12))

    for note in range(num_notes):
        semitone = note % 12
        semitone_matrix[:,:, semitone] = np.maximum(semitone_matrix[:,:, semitone], note_array[:,:, note])

    return semitone_matrix

def extract_dominating_melody(semitone_array: np.ndarray, window_size: int = -1) -> np.ndarray:
    dominating_melody_smoothed = np.argmax(semitone_array, axis=-1)
    #print(f'Mean Note: {np.mean(dominating_melody_smoothed)}')
    if window_size > 0:
        dominating_melody_smoothed = medfilt(dominating_melody_smoothed, kernel_size=window_size)
    
    return dominating_melody_smoothed

def dtw_distance(vector1: np.ndarray, vector2: np.ndarray) -> float:
    distance, path = fastdtw(vector1[:, None], vector2[:, None], dist=euclidean)
    normalized_distance = distance / len(path)
    return normalized_distance

def jaccard_distance(set1: np.ndarray, set2: np.ndarray) -> float:
    set1_unique = np.unique(set1)
    set2_unique = np.unique(set2)
    intersection = len(np.intersect1d(set1_unique, set2_unique))
    union = len(np.union1d(set1_unique, set2_unique))
    return 1 - intersection / union

def plot_dominating_melodies(dominating_melody1: np.ndarray, note_pt1: np.ndarray, label1: str,
                             dominating_melody2: np.ndarray, note_pt2: np.ndarray, label2: str):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Create the first figure and subplot
    fig1, ax1 = plt.subplots(figsize=(9, 4))
    im1 = ax1.imshow(note_pt1.T, aspect='auto', origin='lower', cmap='viridis')
    fig1.colorbar(im1, ax=ax1, label='Pitch Class Likelihood')
    ax1.plot(dominating_melody1, 'r--', label='Melody')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Pitch Class')
    ax1.set_yticks(ticks=range(len(notes)))
    ax1.set_yticklabels(labels=notes)
    ax1.set_title(f'{label1}')
    ax1.legend()

    # Save the first subplot to a BytesIO object
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png')
    buf1.seek(0)
    image_base64_1 = base64.b64encode(buf1.read()).decode('utf-8')
    buf1.close()
    plt.close(fig1)

    # Create the second figure and subplot
    fig2, ax2 = plt.subplots(figsize=(9, 4))
    im2 = ax2.imshow(note_pt2.T, aspect='auto', origin='lower', cmap='viridis')
    fig2.colorbar(im2, ax=ax2, label='Pitch Class Likelihood')
    ax2.plot(dominating_melody2, 'r--', label='Melody')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Pitch Class')
    ax2.set_yticks(ticks=range(len(notes)))
    ax2.set_yticklabels(labels=notes)
    ax2.set_title(f'{label2}')
    ax2.legend()

    # Save the second subplot to a BytesIO object
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png')
    buf2.seek(0)
    image_base64_2 = base64.b64encode(buf2.read()).decode('utf-8')
    buf2.close()
    plt.close(fig2)

    # Return two base64 strings to be embedded in HTML
    return f'<img src="data:image/png;base64,{image_base64_1}" />', f'<img src="data:image/png;base64,{image_base64_2}" />'

class PitchTracker:
    def __init__(self):
        self.pt_model = BasicPitchTorch()
        self.pt_model.load_state_dict(torch.load(os.path.join(basic_pitch_dir, 'assets', 'basic_pitch_pytorch_icassp_2022.pth')))
        self.pt_model.eval()

    def cals_pitch_metric(self, y1: torch.Tensor, y2: torch.Tensor, metric: str = "dtw", plot: bool = False, window_size: int = 0, specs_back=False, pair=None) -> list:
        with torch.no_grad():
            output_pt1 = self.pt_model(y1.cpu())
            note_pt1 = output_pt1['note'].cpu().numpy()

            output_pt2 = self.pt_model(y2.cpu())
            note_pt2 = output_pt2['note'].cpu().numpy()

        distances = []
        note_pt1_trimmed = aggregate_to_semitones(note_pt1)
        note_pt2_trimmed = aggregate_to_semitones(note_pt2)
        dominating_melody1_batch = extract_dominating_melody(note_pt1_trimmed, window_size=window_size)
        dominating_melody2_batch = extract_dominating_melody(note_pt2_trimmed, window_size=window_size)
        
        for i in range(y1.shape[0]):
            dominating_melody1 = dominating_melody1_batch[i]
            dominating_melody2 = dominating_melody2_batch[i]

            
            if metric == "dtw":
                distance = dtw_distance(dominating_melody1, dominating_melody2)
            elif metric == "jaccard":
                distance = jaccard_distance(dominating_melody1, dominating_melody2)
            elif metric == "both":
                dtw_distance_value = dtw_distance(dominating_melody1, dominating_melody2)
                jaccard_distance_value = jaccard_distance(dominating_melody1, dominating_melody2)
                distance = (dtw_distance_value, jaccard_distance_value)
            else:
                raise ValueError("Invalid metric. Choose between 'dtw', 'jaccard' or 'both'.")

            if plot and i==0:
                label1 = ''
                label2 = ''
                if pair:
                    label1 = label1 + f' {pair[0]}'
                    label2 = label2 + f' {pair[1]}'
                    imgs_pair = plot_dominating_melodies(dominating_melody1, note_pt1_trimmed, label1, dominating_melody2, note_pt2_trimmed, label2)
                
            distances.append(distance)
        if specs_back:
            return distances, note_pt1_trimmed, note_pt2_trimmed
        if plot:
            return distances, imgs_pair
        return distances
