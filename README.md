# Face + Emotion + AI Panel
## Installation Guide (Windows, NVIDIA GPU)

This project uses:

- YOLOv8-Face (GPU via PyTorch CUDA)
- FER+ Emotion model (ONNX Runtime GPU)
- OpenCV (camera + UI)
- Gemini API
- Groq API

### 1. Install GPU-Enabled PyTorch (CUDA)

    pip uninstall -y torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Verify GPU works (in CMD):

    py
    import torch
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

### 2. Install ONNX Runtime (GPU version)

    pip install onnxruntime-gpu

Verify (in py):

    import onnxruntime as ort
    print(ort.get_available_providers())

You should see: `['CUDAExecutionProvider', 'CPUExecutionProvider']`

### 3. Install Libraries

    pip install ultralytics
    pip install opencv-python
    pip install numpy
    pip install requests

### 4. Set API Keys

Set API keys in `api_keys.json` file in project folder. The format is:

    {
        "gemini_api_key": "YOUR_GEMINI_KEY",
        "groq_api_key": "YOUR_GROQ_KEY",
        "gemini_model": "gemini-2.5-flash-lite",
        "groq_model": "llama-3.3-70b-versatile"
    }



