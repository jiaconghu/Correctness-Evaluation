from models.image_generation.dcgan import dcgan


def load_dcgan():
    return dcgan.Generator(ngpu=1)
