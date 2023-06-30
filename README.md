# DL_framework

## Supported tasks
### Classification
- [x] MLP
- [x] Simple CNN
- [x] Torchvision ResNet
- [x] SimpleViT
- [x] MLP-Mixer
- [x] ConvMixer

### Dense Prediction
- [x] Noise2Noise (Gaussian noise)
- [x] Noise2Noise (Text overlay, results not very good)

### Generative Model
- [x] Vanilla VAE
- [x] Simplified DDPM (UNet)
- [x] NCSN v2 (The EMA resume may not correctly handled)
- [ ] More complexed diffusion models
- [ ] GAN
- [ ] Normalizing flow


## Other Components
- [ ] wandb support
- [ ] RandAugment
- [ ] AugMix
- [ ] mixup
- [ ] linear warmup
- [ ] save ckpt by metric (save the best model)


## How to run
To start training, run the following command:

```shell
./train.sh
```

