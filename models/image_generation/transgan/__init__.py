from models.image_generation.transgan.models import *
from models.image_generation.transgan.utils import *


def load_transgan():
    generator = models.Generator(depth1=5, depth2=4, depth3=2, initial_size=8, dim=384, heads=4, mlp_ratio=4,
                                 drop_rate=0.5)
    generator.apply(inits_weight)
    return generator
