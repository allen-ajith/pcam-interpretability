import torch
import gc
import time
from utils.pcam_dataloader import get_pcam_loaders
from models.resnet50 import create_resnet50
from models.dino_vit import create_dino_vit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu():
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2) 

def test_batch_size(model_fn, model_name, model_type, start=16, max_batch=1024, step=16):
    print(f"\n===== Testing {model_name} =====")
    for batch_size in range(start, max_batch + 1, step):
        clear_gpu()
        try:
            print(f"Trying batch size: {batch_size}")
            model = model_fn(pretrained=True).to(device)
            loader, _, _ = get_pcam_loaders(batch_size=batch_size, model_type=model_type, seed=42)

            images, labels = next(iter(loader))
            images, labels = images.to(device), labels.to(device)
            
            with torch.cuda.amp.autocast(): 
                outputs = model(images)
            print(f"Batch size {batch_size} successful!")

            del model
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM at batch size {batch_size}")
                break
            else:
                print(f"Other error: {e}")
                break

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True 

    test_batch_size(create_resnet50, "ResNet-50", model_type="resnet", start=16, step=16)
    test_batch_size(create_dino_vit, "DINO ViT-S/16", model_type="vit", start=16, step=16)
