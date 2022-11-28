"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time
import numpy as np
import torch as th
import torch.distributed as dist
import matplotlib.pyplot as plt
from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    truck = [555, 569, 717, 864, 867]
    ship = [724,510]
    i = 0
    while len(all_images) * args.batch_size < args.num_samples:
        cur_batch = []
        cur_labels = []
        model_kwargs = {}
        if args.class_cond:
            # classes = th.randint(
            #     low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            # )
            classes = th.cuda.LongTensor(np.random.choice([8], args.batch_size)) # Example to generate Crane (Class 242 from image net)
            model_kwargs["y"] = classes
            # print("Model",model_kwargs['y'])
            print("Classes",classes)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        cur_batch.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            # all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            cur_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        #logger.log(f"created {len(all_images) * args.batch_size} samples")
        logger.log(f"created {len(cur_batch) * args.batch_size} samples")

        # arr = np.concatenate(all_images, axis=0)
        arr = np.concatenate(cur_batch, axis=0)
        arr = arr[: args.num_samples]
        tmp = arr[0]
        plt.imshow(tmp); plt.show()
        if args.class_cond:
            # label_arr = np.concatenate(all_labels, axis=0)
            label_arr = np.concatenate(cur_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
            print("Cond Label",label_arr)
            print("Num Samples:",args.num_samples)
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            TIMESTR = time.strftime("%Y%m%d-%H%M%S")
            out_path = os.path.join("./cifar_unc_samples", f"samples_{i}_{shape_str}_{TIMESTR}.npz")
            # label_arr = np.concatenate(all_labels, axis=0)

            # out_path = os.path.join(logger.get_dir(), f"samples_555_{i}_{shape_str}_{TIMESTR}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
        i += 1
        # print('samples saved in ', out_path)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    th.cuda.empty_cache()
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        batch_size=1,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()