import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def gpu_info():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("Verfügbare GPUs: ")
        for i in range(device_count):
            print(torch.cuda.get_device_name(i))
    else:
        print("Keine CUDA-fähigen GPUs gefunden.")
