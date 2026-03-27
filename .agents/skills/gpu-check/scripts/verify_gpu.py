import torch

def main():
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is AVAILABLE. Verified hardware: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT AVAILABLE. Hardware fallback: CPU")
        
if __name__ == "__main__":
    main()
