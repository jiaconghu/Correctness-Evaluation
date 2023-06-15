from models.image_generation import ddpm, nvae, transgan, iddpm, dcgan


def load_ddpm():
    return ddpm.load_ddpm()


def load_transgan():
    return transgan.load_transgan()


def load_nvae():
    return nvae.load_nvae()


def load_iddpm():
    return iddpm.load_iddpm()


def load_dcgcn():
    return dcgan.load_dcgan()
