from models.image_generation.ddpm import ddpm


def load_ddpm():
    model = ddpm.UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
    sampler = ddpm.GaussianDiffusionSampler(model, beta_1=0.0001, beta_T=0.02, T=1000, img_size=32,
                                            mean_type='epsilon', var_type='fixedlarge')
    return sampler, model
