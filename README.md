# Stable Diffusion v1.5 VRAM Calculator

A research assignment to derive an analytical equation for estimating peak GPU VRAM usage during inference for `stable-diffusion-v1-5/stable-diffusion-v1-5` model for arbitrary input image sizes.

## 📋 Objective

Derive a formula to estimate peak vRAM usage during diffusion model inference, accounting for:
- Model weights (fixed)
- Intermediate activations (vary with image dimensions and prompt length)
- Framework overhead (CUDA kernels, workspace buffers)
- Attention mechanism memory scaling (O(N²) with sequence length)

## 🧠 My Understanding of Stable Diffusion Architecture

The pipeline follows this flow:

```
Input = {Prompt, Image}
   │
   ├─→ Image → VAE Encoder → Latent Image
   │                              │
   │                              ↓
   │                   Latent Image + Noise → Noisy Latent
   │                                              │
   │                                              ↓
   │                   Noisy Latent → UNet Encoders → Downsampling + Feature Extraction
   │                                                         │
   │                                                         ↓
   └─→ Prompt → Text Encoder (Word2Vec) → Embeddings ─────→ ┼
                                                            │
                              UNet Decoder ← Output + Embeddings
                                    │
                                    ↓
                          Predicted Noise (Upsampling)
                                    │
                                    ↓
              Noisy Latent - Predicted Noise = New Latent Image (Denoising)
                                    │
                                    ↓
                    New Latent Image → VAE Decoder → Final Generated Image
```

## 🔬 VRAM Calculation Methodology

### Phase 1: Base VRAM (Model Weights)

**Concept:** All model parameters get loaded into VRAM.

- Calculate total parameters for UNet, VAE, and Text Encoder
- Since we use FP16 (float16), each parameter takes **2 bytes**
- **Base VRAM = Total Parameters × 2 bytes**

```python
def get_base_vram():
    unet_params = sum(p.numel() for p in pipeline.unet.parameters())
    vae_params = sum(p.numel() for p in pipeline.vae.parameters())
    text_encoder_params = sum(p.numel() for p in pipeline.text_encoder.parameters())
    
    total_params = unet_params + vae_params + text_encoder_params
    return total_params * 2  # FP16 bytes
```

### Phase 2: Activation Memory (Variable)

Memory that scales with input dimensions.

#### 2.1 VAE Encoder Memory

- **Input Memory:** H × W × 3 × 2 bytes
- **Temporary Tensors:** ~3× input memory (intermediate processing)
- **Latent Output:** (H/8) × (W/8) × 4 × 2 bytes
  - Compression ratio: 8× spatial (512→64)
  - Channels change from 3 to 4 for essential information

```python
def vae_encoder_memory(height, width):
    input_memory = height * width * 3 * 2
    temp_memory = input_memory * 3
    latent_memory = (height // 8) * (width // 8) * 4 * 2
    return input_memory + temp_memory + latent_memory
```

#### 2.2 UNet Memory

**For the image:**
- Input is latent values from VAE encoder
- Initial layers consume the most memory
- As we go deeper, spatial dimensions reduce but channels increase
- Using average channel count across layers: (320 + 640 + 1280) / 3 ≈ **745**

**For the prompt:**
- Attention matrix maps words to image regions
- Memory scales with prompt_length × latent_size

```python
def unet_memory(height, width, prompt_length):
    latent_h = height // 8
    latent_w = width // 8
    
    channel_activations = latent_h * latent_w * 745 * 2
    channel_memory = channel_activations * 3  # Temp tensors
    
    attention_matrix = latent_h * latent_w * prompt_length * 2
    
    return channel_memory + attention_matrix
```

#### 2.3 VAE Decoder Memory

- **Input:** Latent image from UNet
- **Output:** Full-resolution image (H × W × 3)
- **Temporary Memory:** ~3× output memory

```python
def vae_decoder_memory(height, width):
    latent_memory = (height // 8) * (width // 8) * 4 * 2
    output_memory = height * width * 3 * 2
    temp_memory = output_memory * 3
    return latent_memory + output_memory + temp_memory
```

### Final Formula

```python
def total_vram(H, W, prompt_length):
    base_mem = get_base_vram()
    
    encoder_mem = vae_encoder_memory(H, W)
    unet_mem = unet_memory(H, W, prompt_length)
    decoder_mem = vae_decoder_memory(H, W)
    
    # Peak memory is the maximum of the three stages
    peak_mem = max(encoder_mem, unet_mem, decoder_mem)
    
    # 20% overhead for CUDA kernels, workspace buffers
    overhead_factor = 1.2
    
    total = (base_mem + peak_mem) * overhead_factor
    return total / (1024**3)  # Convert to GB
```

### 📊 Mathematical Summary

$$VRAM_{total} = (VRAM_{base} + max(VRAM_{VAE\_enc}, VRAM_{UNet}, VRAM_{VAE\_dec})) \times 1.2$$

Where:
- **VRAM_base** = (UNet_params + VAE_params + TextEncoder_params) × 2
- **VRAM_VAE_enc** = H×W×3×2 + (3 × H×W×3×2) + (H/8)×(W/8)×4×2
- **VRAM_UNet** = (H/8)×(W/8)×745×2×3 + (H/8)×(W/8)×prompt_length×2
- **VRAM_VAE_dec** = (H/8)×(W/8)×4×2 + H×W×3×2 + (3 × H×W×3×2)

## 🛠️ Setup Instructions

### Local Setup
```bash
# Create virtual environment
python -m venv env

# Activate (Windows)
env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the file
python main.py
```

### Google Colab Setup (with T4 GPU) (Recommended)
1. Open the notebook in Google Colab
2. Go to Runtime → Change runtime type → Select GPU (T4)
3. Run all cells
4. Use the main_colab.ipynb file


## 📚 Key Assumptions

1. **FP16 Precision:** All calculations assume 2 bytes per parameter
2. **No CPU Offloading:** Model fully loaded on GPU
3. **Peak Memory:** Formula captures maximum memory allocation stage
4. **No Gradient Storage:** Inference only, no training-related memory
5. **20% Overhead:** Accounts for CUDA workspace, kernels, and framework internals

## 🔗 References

- [Stable Diffusion v1.5 on Hugging Face](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
- [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
