# Stable Diffusion v1.5 Architecture

This document provides a detailed technical breakdown of the Stable Diffusion architecture and how VRAM is consumed during image-to-image inference.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              STABLE DIFFUSION v1.5 PIPELINE                             │
│                                    (Image-to-Image)                                     │
└─────────────────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────┐
                    │    INPUT     │
                    │   PROMPT     │
                    │  (String)    │
                    └──────┬───────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │      CLIP TEXT ENCODER       │
            │  ─────────────────────────   │
            │  • Tokenizes prompt text     │
            │  • Creates word embeddings   │
            │  • Output: (77, 768) tensor  │
            │  • Params: ~123M             │
            │  • Memory: ~246 MB (FP16)    │
            └──────────────┬───────────────┘
                           │
                           │ Text Embeddings
                           ▼
┌───────────────┐    ┌─────────────────────────────────────────────────────────────┐
│  INPUT IMAGE  │    │                          U-NET                              │
│  (H × W × 3)  │    │  ─────────────────────────────────────────────────────────  │
└───────┬───────┘    │                                                             │
        │            │  ┌─────────────────────────────────────────────────────┐   │
        ▼            │  │              ENCODER (Downsampling)                 │   │
┌───────────────┐    │  │  ─────────────────────────────────────────────────  │   │
│  VAE ENCODER  │    │  │  Layer 1: 320 channels  →  Extract features        │   │
│ ─────────────-│    │  │  Layer 2: 640 channels  →  Compress spatial        │   │
│ Compresses to │    │  │  Layer 3: 1280 channels →  Dense representations   │   │
│ latent space  │    │  └─────────────────────────────────────────────────────┘   │
│               │    │                          │                                  │
│ H/8 × W/8 × 4 │───▶│                          ▼                                  │
│               │    │  ┌─────────────────────────────────────────────────────┐   │
│ +Noise added  │    │  │            CROSS-ATTENTION LAYERS                   │   │
│               │    │  │  ─────────────────────────────────────────────────  │   │
│ Params: ~84M  │    │  │  Attention(Q=image, K=text, V=text)                 │   │
│ Memory: ~168MB│    │  │  Maps: which words influence which regions          │   │
└───────────────┘    │  │  Memory: O(latent_size × prompt_length)             │   │
                     │  └─────────────────────────────────────────────────────┘   │
                     │                          │                                  │
                     │                          ▼                                  │
                     │  ┌─────────────────────────────────────────────────────┐   │
                     │  │              DECODER (Upsampling)                   │   │
                     │  │  ─────────────────────────────────────────────────  │   │
                     │  │  Layer 3: 1280 → 640 channels + Skip connections    │   │
                     │  │  Layer 2: 640 → 320 channels  + Skip connections    │   │
                     │  │  Layer 1: 320 channels        → Predicted noise     │   │
                     │  └─────────────────────────────────────────────────────┘   │
                     │                                                             │
                     │  Params: ~860M    |    Memory: ~1.72 GB (FP16)              │
                     └──────────────────────────────┬──────────────────────────────┘
                                                    │
                               Predicted Noise      │
                                                    ▼
                     ┌─────────────────────────────────────────────────────────────┐
                     │                    DENOISING LOOP                           │
                     │  ─────────────────────────────────────────────────────────  │
                     │  For each timestep t in [T, T-1, ..., 1]:                   │
                     │      noisy_latent = noisy_latent - scheduler(noise_pred)    │
                     │                                                             │
                     │  Default: 40 iterations                                     │
                     └─────────────────────────────────────────────────────────────┘
                                                    │
                               Denoised Latent      │
                                                    ▼
                     ┌─────────────────────────────────────────────────────────────┐
                     │                       VAE DECODER                           │
                     │  ─────────────────────────────────────────────────────────  │
                     │  • Input: H/8 × W/8 × 4 latent                              │
                     │  • Output: H × W × 3 RGB image                              │
                     │  • Upsamples 8× spatially                                   │
                     │  • Params: ~84M (shared with encoder)                       │
                     │  • Memory: ~168 MB (FP16) + 3× temp tensors                 │
                     └─────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
                     ┌─────────────────────────────────────────────────────────────┐
                     │                      OUTPUT IMAGE                           │
                     │                       (H × W × 3)                            │
                     └─────────────────────────────────────────────────────────────┘
```

## Memory Breakdown by Component

### Component Summary

| Component | Parameters | FP16 Memory | Role |
|-----------|------------|-------------|------|
| **UNet** | ~860M | ~1.72 GB | Noise prediction, image understanding |
| **VAE** | ~84M | ~168 MB | Encode/decode between pixel & latent space |
| **Text Encoder** | ~123M | ~246 MB | Convert text prompts to embeddings |
| **Safety Checker** | ~308M | ~608 MB | NSFW content filtering |
| **Total Base** | ~1.37B | ~2.74 GB | Model weights on GPU |

### Dynamic Memory (Scales with Input)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MEMORY SCALING FACTORS                              │
├─────────────────────────┬───────────────────────────────────────────────┤
│  Input Dimensions       │  Memory Impact                                │
├─────────────────────────┼───────────────────────────────────────────────┤
│  Image Size (H × W)     │  • VAE input: H × W × 3 × 2 bytes             │
│                         │  • Latent: (H/8) × (W/8) × 4 × 2 bytes        │
│                         │  • VAE output: H × W × 3 × 2 bytes            │
├─────────────────────────┼───────────────────────────────────────────────┤
│  Prompt Length (tokens) │  • Attention: latent_size × prompt_len × 2   │
│                         │  • Max 77 tokens (CLIP limit)                 │
├─────────────────────────┼───────────────────────────────────────────────┤
│  Batch Size             │  • Multiplies all activation memory           │
│                         │  • Default: 1                                 │
└─────────────────────────┴───────────────────────────────────────────────┘
```

## UNet Channel Structure

The UNet processes features through multiple resolution levels:

```
                      ENCODER                           DECODER
                         │                                  │
   Input (64×64×4) ─────►│                                  │──────► Output (64×64×4)
                         ▼                                  ▲
              ┌──────────────────┐              ┌──────────────────┐
              │   320 channels   │──── skip ───►│   320 channels   │
              │   64 × 64        │              │   64 × 64        │
              └────────┬─────────┘              └────────▲─────────┘
                       │                                 │
                       ▼                                 │
              ┌──────────────────┐              ┌──────────────────┐
              │   640 channels   │──── skip ───►│   640 channels   │
              │   32 × 32        │              │   32 × 32        │
              └────────┬─────────┘              └────────▲─────────┘
                       │                                 │
                       ▼                                 │
              ┌──────────────────┐              ┌──────────────────┐
              │  1280 channels   │──── skip ───►│  1280 channels   │
              │   16 × 16        │              │   16 × 16        │
              └────────┬─────────┘              └────────▲─────────┘
                       │                                 │
                       └────────────► Bottleneck ────────┘
                                    (1280 channels)
                                      (8 × 8)
```

**Average Channel Calculation:**
```
Average = (320 + 640 + 1280) / 3 ≈ 745 channels
```

This approximation is used in our VRAM formula for estimating UNet activation memory.

## Latent Space Compression

```
┌────────────────────────────────────────────────────────────────────────┐
│                        VAE COMPRESSION RATIOS                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   Original Image              Latent Representation                    │
│   ─────────────               ─────────────────────                    │
│   512 × 512 × 3               64 × 64 × 4                              │
│   = 786,432 values            = 16,384 values                          │
│                                                                        │
│   Compression: 48× fewer values!                                       │
│                                                                        │
│   Spatial: 512 → 64 (8× compression per dimension, 64× total)          │
│   Channels: 3 → 4 (stores mean & variance for reconstruction)          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## VRAM Formula Derivation

### Stage 1: Base Memory (Fixed)
```
VRAM_base = (UNet_params + VAE_params + TextEncoder_params) × 2 bytes
          ≈ (860M + 84M + 123M) × 2
          ≈ 2.13 GB
```

### Stage 2: VAE Encoder Activation
```
VRAM_vae_enc = input_mem + temp_mem + latent_mem
             = (H × W × 3 × 2) + (3 × H × W × 3 × 2) + ((H/8) × (W/8) × 4 × 2)
             = H × W × 24 + (H/8) × (W/8) × 8
```

### Stage 3: UNet Activation (Peak)
```
VRAM_unet = channel_mem + attention_mem
          = ((H/8) × (W/8) × 745 × 2 × 3) + ((H/8) × (W/8) × prompt_len × 2)
          = (H/8) × (W/8) × (4470 + 2 × prompt_len)
```

### Stage 4: VAE Decoder Activation
```
VRAM_vae_dec = latent_mem + output_mem + temp_mem
             = ((H/8) × (W/8) × 4 × 2) + (H × W × 3 × 2) + (3 × H × W × 3 × 2)
             = (H/8) × (W/8) × 8 + H × W × 24
```

### Final Formula
```
VRAM_total = (VRAM_base + max(VRAM_vae_enc, VRAM_unet, VRAM_vae_dec)) × 1.2
```

The 1.2 overhead factor accounts for:
- CUDA workspace buffers
- PyTorch memory allocator fragmentation
- Kernel compilation caches
- Framework internal structures

## Test Images Used

| Image | Resolution | Type | Prompt Tokens |
|-------|------------|------|---------------|
| balloon--low-res.jpeg | Low | Hot air balloon scene | ~50 |
| bench--high-res.jpg | High | Valley landscape | ~55 |
| groceries--low-res.jpg | Low | Shopping scene | ~45 |
| truck--high-res.jpg | High | Vehicle scene | ~40 |

## Inference Pipeline Steps

```
1. Load Model          → Allocate base VRAM (~2.7 GB)
2. Tokenize Prompt     → Text encoder processes prompt
3. Encode Image        → VAE compresses to latent (peak: VAE encoder)
4. Add Noise           → Apply noise schedule
5. Denoise Loop ×40    → UNet predicts noise each step (peak: UNet)
6. Decode Latent       → VAE reconstructs image (peak: VAE decoder)
7. Safety Check        → Optional NSFW filter
8. Return Image        → Final output
```

