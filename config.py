import argparse
import os
from pathlib import Path
import ruamel.yaml as yaml


def get_caption_config():
    config={
    'image_root': '/home/ubuntu/datasets/coco/images/',
    'ann_root': 'annotation/',
    'coco_gt_root': 'annotation/coco_gt',
    'image_size': 384,
    'max_length': 20,
    'min_length': 5,
    'num_beams': 3,
    'prompt': 'a picture of ',

    'vit': 'base',
    'vit_grad_ckpt': False,
    'vit_ckpt_layer': 0,
    'pretrained': 'pretrained/model_base_caption_capfilt_large.pth',
    }
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default='CompVis/stable-diffusion-v1-4',)
    parser.add_argument( "--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.",)
    parser.add_argument("--dataset_name",type=str,default=None)
    parser.add_argument("--image_column", type=str, default="image", help="The column of the dataset containing an image.")
    parser.add_argument("--caption_column", type=str, default="text",help="The column of the dataset containing a caption or a list of captions.",)
    parser.add_argument("--max_train_samples",type=int,default=None,)
    parser.add_argument("--cache_dir",type=str,default=None,help="The directory where the downloaded models and datasets will be stored.",)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--center_crop",action="store_true",help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",)
    parser.add_argument("--random_flip",action="store_true",help="whether to randomly flip images horizontally",)
    parser.add_argument("--logging_dir",type=str,default="logs")
    parser.add_argument("--mixed_precision",type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--max_train_steps",type=int,default=None, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",)

    parser.add_argument("--train_data_dir",type=str, default='/home/ubuntu/datasets/coco/captions/train/')
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--resolution",type=int,default=512,)

    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--log_every_steps', type=int, default=1000)

    #unet training
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=8)
    parser.add_argument("--num_train_epochs", type=int, default=1)

    parser.add_argument("--learning_rate",type=float,default=1e-4,help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--scale_lr",action="store_true",default=False,help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    parser.add_argument("--lr_scheduler",type=str,default="constant",help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'' "constant", "constant_with_warmup"]'),)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")

    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    #blip architecture
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--token_transform_logit', action='store_true')

    #blip training
    parser.add_argument('--blip_text_decoder', type=str, default=None)
    parser.add_argument('--unfreeze_blip', type=str, default='xattn_q')
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--init_lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-4)
    parser.add_argument('--max_epoch', type=int, default=5)

    parser.add_argument('--unfreeze_unet', type=str, default='xattn_q')
    parser.add_argument('--lambda_sd', type=float, default=1.0)
    parser.add_argument('--batch_sd', type=int, default=1)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    args.config=get_caption_config()
    args.config.update(dict(
        init_lr=args.init_lr,min_lr=args.min_lr,
        weight_decay=args.weight_decay,max_epoch=args.max_epoch))

    args.result_dir = os.path.join(args.output_dir, 'result')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(vars(args), open(os.path.join(args.output_dir, 'config.yaml'), 'w'))
    return args


if __name__=='__main__':
    args=parse_args()
