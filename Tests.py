import torch
from Tools import *

def test_model_with_dataloader(model, spec_list, labels):
    model.eval()

    with torch.no_grad():

        tensor_list = [torch.tensor(spec).float() for spec in spec_list]
        inputs = torch.stack(tensor_list).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        accuracy = torch.sum(predictions == labels).item() / len(labels)

    return accuracy, predictions.tolist()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load('model.pt').to(device)

root_dir = 'C:/Proxectos/MNIST_audio_classifier/00'
dataset = load_audio_dataset(root_dir)

spec_list = []
for path in dataset[0]:
    waveform, sample_rate = librosa.load(path, sr=None)
    waveform_sh = adjust_audio_shape(waveform)
    waveform_adp = adjust_audio_duration(waveform_sh, sample_rate, target_duration=0.9)
    spec = convert_mel_spectrogram(waveform_adp, sample_rate)
    spec_list.append(spec)

accuracy, pred_labels = test_model_with_dataloader(model, spec_list, dataset[1])
print(f"Accuracy: {accuracy:.2f}")