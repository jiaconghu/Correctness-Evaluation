import argparse
import io
import os
import pickle
import time
from tqdm import tqdm, trange
import numpy as np
import torch

import models
# import metrics
# from torcheval import metrics
# from torchmetrics import image
from metrics import plcv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('DEVICE:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA DIR:', args.data_dir)
    print('-' * 50)

    # ----------------------------------------
    # test configuration
    # ----------------------------------------

    since = time.time()

    evaluates = {
        'KL': [
            # plcv.FID(data_name=args.data_name, data_dir=args.data_dir, feature=2048, normalize=True).to(device),
            # plcv.InceptionScore(normalize=True).to(device)
        ],
        'K': [plcv.SWD(data_name=args.data_name, data_dir=args.data_dir)],
        'L': [plcv.NDB(data_name=args.data_name, data_dir=args.data_dir)]
    }

    if args.model_name == 'DDPM':
        sampler, model = models.load_model(model_name=args.model_name)
        # model.load_state_dict(torch.load(args.model_path)['net_model'])
        model.load_state_dict(torch.load(args.model_path)['ema_model'])
        sampler.to(device)
        model.to(device)
        scores = test_ddpm(sampler, model, evaluates, device)
    # elif args.model_name == 'StyleGAN3':
    #     test_stylegan3(args, device)
    elif args.model_name == 'NVAE':
        model = models.load_model(model_name=args.model_name)
        model.to(device)
        model.eval()
        scores = test_nvae(model, evaluates)
    elif args.model_name == 'TransGAN':
        model = models.load_model(model_name=args.model_name)
        model.load_state_dict(torch.load(args.model_path)['generator_state_dict'])
        model.to(device)
        model.eval()
        scores = test_transgan(model, evaluates)
    elif args.model_name == 'ImprovedDDPM':
        sampler, model = models.load_model(model_name=args.model_name)
        model.load_state_dict(torch.load(args.model_path))
        # model.load_state_dict(torch.load(args.model_path)['ema_model'])
        model.to(device)
        model.eval()
        scores = test_iddpm(sampler, model, evaluates)
    elif args.model_name == 'DCGAN':
        model = models.load_model(model_name=args.model_name)
        model.load_state_dict(torch.load(args.model_path))
        model.to(device)
        model.eval()
        scores = test_dcgcn(model, evaluates, device)

    save_path = os.path.join(args.save_dir, '{}_{}.npy'.format(args.model_name, args.data_name))
    print('===>', save_path)
    np.save(save_path, scores)

    print('-' * 50)
    print('TIME CONSUMED', time.time() - since)


def test_dcgcn(model, evaluates, device):
    with torch.no_grad():
        # images = []
        for i in trange(0, 50000, 32, desc="generating images"):  # num_images, batch_size
            batch_size = min(32, 50000 - i)  # batch_size, num_images
            latent_size = 100
            noise = torch.randn(batch_size, latent_size, 1, 1).to(device)
            imgs = model(noise)
            imgs = torch.reshape(imgs, (batch_size, 3, 32, 32))
            imgs = (imgs + 1.0) / 2.0

            # print(imgs.shape)
            # print(torch.min(imgs), torch.max(imgs), torch.mean(imgs))
            # print('-' * 20)
            for values in evaluates.values():
                for evaluate in values:
                    evaluate.update(imgs)

        # calculate result
        scores = {}
        for key, value in zip(evaluates.keys(), evaluates.values()):
            scores[key] = []
            for evaluate in value:
                score = evaluate.compute()
                scores[key].append(score)
        print(scores)

        return scores


def test_iddpm(sampler, model, evaluates):
    with torch.no_grad():
        # images = []
        for i in trange(0, 50000, 32, desc="generating images"):  # num_images, batch_size
            batch_size = min(32, 50000 - i)  # batch_size, num_images
            imgs = sampler(model,
                           (batch_size, 3, 32, 32),
                           clip_denoised=True,
                           model_kwargs={})
            imgs = (imgs + 1.0) / 2.0

            print(imgs.shape)
            print(torch.min(imgs), torch.max(imgs), torch.mean(imgs))
            print('-' * 20)
            for values in evaluates.values():
                for evaluate in values:
                    evaluate.update(imgs)

        # calculate result
        scores = {}
        for key, value in zip(evaluates.keys(), evaluates.values()):
            scores[key] = []
            for evaluate in value:
                score = evaluate.compute()
                scores[key].append(score)
        print(scores)

        return scores


def test_transgan(model, evaluates):
    num_img = 50000
    val_batch_size = 256
    latent_dim = 1024

    with torch.no_grad():
        # eval mode
        generator = model.eval()

        eval_iter = num_img // val_batch_size
        for _ in tqdm(range(eval_iter), desc='sample images'):
            noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (val_batch_size, latent_dim)))
            imgs = generator(noise)
            imgs = (imgs + 1.0) / 2.0

            # print(imgs.shape)
            # print(torch.min(imgs), torch.max(imgs), torch.mean(imgs))
            # print('-' * 20)
            for values in evaluates.values():
                for evaluate in values:
                    evaluate.update(imgs)

        # calculate result
        scores = {}
        for key, value in zip(evaluates.keys(), evaluates.values()):
            scores[key] = []
            for evaluate in value:
                score = evaluate.compute()
                scores[key].append(score)
        print(scores)

        return scores


def test_nvae(model, evaluates):
    batch_size = 256
    num_total_samples = 50000
    num_iters = int(np.ceil(num_total_samples / batch_size))
    with torch.no_grad():
        for i in tqdm(range(num_iters)):
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            imgs = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()

            # print(img.shape)
            # print(torch.min(img), torch.max(img), torch.mean(img))
            # print('-' * 20)
            for values in evaluates.values():
                for evaluate in values:
                    evaluate.update(imgs)

        # calculate result
        scores = {}
        for key, value in zip(evaluates.keys(), evaluates.values()):
            scores[key] = []
            for evaluate in value:
                score = evaluate.compute()
                scores[key].append(score)
        print(scores)

        return scores


def test_ddpm(sampler, model, evaluates, device):
    model.eval()
    with torch.no_grad():
        # images = []
        for i in trange(0, 50000, 512, desc="generating images"):  # num_images, batch_size
            '''
            copy from https://github.com/w86763777/pytorch-ddpm
            '''
            batch_size = min(512, 50000 - i)  # batch_size, num_images
            x_T = torch.randn((batch_size, 3, 32, 32))  # img_size, img_size
            batch_images = sampler(x_T.to(device))
            imgs = (batch_images + 1) / 2

            # print(batch_images.shape)
            # print(torch.min(batch_images), torch.max(batch_images), torch.mean(batch_images))
            # print('-' * 10)

            for values in evaluates.values():
                for evaluate in values:
                    evaluate.update(imgs)

        # calculate result
        scores = {}
        for key, value in zip(evaluates.keys(), evaluates.values()):
            scores[key] = []
            for evaluate in value:
                score = evaluate.compute()
                scores[key].append(score)
        print(scores)

        return scores


def test_stylegan3(args, device):
    # with dnnlib.util.open_url(network_pkl) as f:

    f = open(args.model_path, 'rb')
    # print(f)
    f = pickle.load(f, encoding='utf-8')
    f = io.BytesIO(f)
    G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    # os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    seeds = [2]

    def make_transform(translate, angle):
        m = np.eye(3)
        s = np.sin(angle / 360.0 * np.pi * 2)
        c = np.cos(angle / 360.0 * np.pi * 2)
        m[0][0] = c
        m[0][1] = s
        m[0][2] = translate[0]
        m[1][0] = -s
        m[1][1] = c
        m[1][2] = translate[1]
        return m

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform((0, 0), 0)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=1, noise_mode='const')
        print(img.shape)
        print(torch.min(img), torch.max(img), torch.mean(img))
        # img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


if __name__ == '__main__':
    main()
