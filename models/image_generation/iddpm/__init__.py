from models.image_generation.iddpm.script_util import *


def load_iddpm():
    # MODEL_FLAGS = "--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
    # DIFFUSION_FLAGS = "--diffusion_steps 4000 --noise_schedule cosine"
    # TRAIN_FLAGS = "--lr 1e-4 --batch_size 128"
    model, diffusion = create_model_and_diffusion(image_size=32,
                                                  num_channels=128,
                                                  num_res_blocks=3,
                                                  num_heads=4,
                                                  num_heads_upsample=-1,
                                                  attention_resolutions="16,8",
                                                  dropout=0.3,
                                                  learn_sigma=True,
                                                  sigma_small=False,
                                                  class_cond=False,
                                                  diffusion_steps=1000,
                                                  noise_schedule="cosine",
                                                  timestep_respacing="",
                                                  use_kl=False,
                                                  predict_xstart=False,
                                                  rescale_timesteps=True,
                                                  rescale_learned_sigmas=True,
                                                  use_checkpoint=False,
                                                  use_scale_shift_norm=True)
    sample_fn = diffusion.p_sample_loop
    return sample_fn, model
