import logging
import math
import os
import random
from tqdm.auto import tqdm
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torchvision import transforms

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

from models.blip import blip_decoder
from utils import cosine_lr_schedule
from data.utils import pre_caption
from torchvision.transforms.functional import InterpolationMode
from transform.randaugment import RandomAugment

from config import parse_args
from utils_model import save_model, load_blip_model
from evaluate_blip_lib import evaluate_blip_batch as evaluate_blip


logger = get_logger(__name__)


def unfreeze_layers_blip(blip, condition):
    conditions={
        'xattn_kv': lambda name: 'crossattention' in name and ('key' in name or 'value' in name),
        'xattn_qkv': lambda name: 'crossattention' in name and ('key' in name or 'value' in name or 'query' in name),
        'xattn_q': lambda name: 'crossattention' in name and 'query' in name,
        'none': lambda x: False
    }
    is_condition=conditions[condition]

    for name, param in blip.text_decoder.named_parameters():
        if is_condition(name):
            param.requires_grad_(True)
    print("Num trainable params blip: ", sum(p.numel() for p in blip.parameters() if p.requires_grad))
    return blip

def unfreeze_layers_unet(unet, condition):
    conditions={
        'xattn_kv': lambda name: 'attn2' in name and ('to_k' in name or 'to_v' in name),
        'xattn_qkv': lambda name: 'attn2' in name and ('to_k' in name or 'to_v' in name or 'to_q' in name),
        'xattn_q': lambda name: 'attn2' in name and 'to_q' in name,
        'none': lambda x: False
    }
    is_condition=conditions[condition]

    for name, param in unet.named_parameters():
        if is_condition(name):
            param.requires_grad_(True)
    print("Num trainable params unet: ", sum(p.numel() for p in unet.parameters() if p.requires_grad))
    return unet


def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        logging_dir=logging_dir,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config = args.config

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet = unfreeze_layers_unet(unet, args.unfreeze_unet)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    args.batch_sd = args.train_batch_size * args.batch_sd #todo:double check

    print("Creating BLIP model")
    blip = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])

    blip = load_blip_model(blip, args.blip_text_decoder)
    blip.requires_grad_(False)
    blip = unfreeze_layers_blip(blip, args.unfreeze_blip)

    softer_layer=torch.load('vocabs/reindex.pth')

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    noise_scheduler = DDPMScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    optimizer_blip = torch.optim.AdamW(params=blip.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
    column_names = dataset["train"].column_names

    dataset_name_mapping = {}
    dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    def tokenize_captions_blip(examples, is_train=True):
        max_words=30
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                caption = config['prompt']+pre_caption(caption, max_words) 
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                caption = random.choice(caption) if is_train else caption[0]
                caption = config['prompt']+pre_caption(caption, max_words)
                captions.append(caption)
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        return captions

    train_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    train_transforms_blip = transforms.Compose(
        [                        
            transforms.RandomResizedCrop(config['image_size'],scale=(0.5, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,5,isPIL=True,augs=['Identity','AutoContrast','Brightness','Sharpness','Equalize',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    tfm=transforms.Resize((config['image_size'], config['image_size']), interpolation=transforms.InterpolationMode.BILINEAR)
    def decode_latents(latents):
        latents = 1 / 0.18215 * latents
        image = vae.decode(latents).sample
        return image

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        examples["pixel_values_blip"] = [train_transforms_blip(image) for image in images]
        examples["input_captions"] = tokenize_captions_blip(examples)
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values_blip = torch.stack([example["pixel_values_blip"] for example in examples])
        pixel_values_blip = pixel_values_blip.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        input_captions = [example["input_captions"] for example in examples]
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
            "pixel_values_blip": pixel_values_blip,
            "input_captions": input_captions,
        }

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, num_workers=4,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    unet, blip, optimizer, optimizer_blip, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, blip, optimizer, optimizer_blip, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    blip.to(accelerator.device)
    softer_layer = softer_layer.to(accelerator.device)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers("finetune", config={k:v for k,v in vars(args).items() if k!='config'})

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    
    prompt_length=blip.prompt_length-1
    bos_tok=F.one_hot(torch.tensor(tokenizer.bos_token_id, device=accelerator.device), tokenizer.vocab_size)[None,None,...]
    eos_tok=F.one_hot(torch.tensor(tokenizer.eos_token_id, device=accelerator.device), tokenizer.vocab_size)[None,None,...]

    def tokenizer_transform_blip_to_sd_prob(logits):
        logits = logits[:,prompt_length:,:]
        logits=torch.softmax(logits,-1)
        logits_sd=logits[...,softer_layer]
        logits_sd=torch.cat([bos_tok.expand(logits_sd.size(0),-1,-1),logits_sd,eos_tok.expand(logits_sd.size(0),-1,-1)],1)
        return logits_sd
    def tokenizer_transform_blip_to_sd_logits(logits):
        logits = logits[:,prompt_length:,:]
        logits_sd=logits[...,softer_layer]
        logits_sd=torch.softmax(logits_sd, -1)
        logits_sd=torch.cat([bos_tok.expand(logits_sd.size(0),-1,-1),logits_sd,eos_tok.expand(logits_sd.size(0),-1,-1)],1)
        return logits_sd
    tokenizer_transform_blip_to_sd = tokenizer_transform_blip_to_sd_logits if args.token_transform_logit else tokenizer_transform_blip_to_sd_prob

    best = dict(ciderb4=0)
    best_epoch = dict(ciderb4=0)

    for epoch in range(args.num_train_epochs):
        unet.train()
        blip.train()
        train_loss = 0.0
        train_loss_lm = 0.0
        train_loss_sd = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate([unet,blip]):
                if step%2==0:
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss_sd = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    pred_latents = noise_scheduler.step(model_pred, timesteps.item(), latents).prev_sample

                    pixel_values_blip=tfm(decode_latents(pred_latents)).to(accelerator.device)                   
                    loss_lm=blip(pixel_values_blip,batch['input_captions'])['loss']

                    loss = loss_sd + loss_lm

                else:
                    blip_output=blip(batch['pixel_values_blip'],batch['input_captions'])
                    loss_lm,decoded_caption_states=blip_output['loss'],blip_output['logits']

                    text_gen_dist=tokenizer_transform_blip_to_sd(decoded_caption_states/args.temperature)
                    text_gen_dist=text_gen_dist.type(weight_dtype)
                    encoder_hidden_states = text_encoder(text_gen_dist)[0]

                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215
                    latents = latents.expand(args.batch_sd, *latents.size()[1:])

                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    encoder_hidden_states = encoder_hidden_states.expand(args.batch_sd, *encoder_hidden_states.size()[1:])
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss_sd = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    loss = args.lambda_sd * loss_sd + loss_lm

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_loss_lm += loss_lm.item() / args.gradient_accumulation_steps
                train_loss_sd += loss_sd.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                optimizer_blip.step()
                cosine_lr_schedule(optimizer_blip, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])
                optimizer_blip.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if (step+1)%args.gradient_accumulation_steps!=0: 
                    ratio=args.gradient_accumulation_steps/((step+1)%args.gradient_accumulation_steps)
                    train_loss = train_loss*ratio
                    train_loss_lm = train_loss_lm*ratio
                    train_loss_sd = train_loss_sd*ratio
                accelerator.log({"train_loss": train_loss, "train_loss_lm": train_loss_lm, "train_loss_sd": train_loss_sd}, step=global_step)
                train_loss = 0.0
                train_loss_lm = 0.0
                train_loss_sd = 0.0

            logs = {"step_loss": loss.detach().item(), "lr_sd": lr_scheduler.get_last_lr()[0], "lr_blip":  optimizer_blip.param_groups[0]['lr']}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

            if not args.skip_evaluation and accelerator.sync_gradients and (global_step)%args.log_every_steps==0:
                coco_eval_result = evaluate_blip(blip, 'coco_val', args.output_dir+'/result/coco_epoch%d_%d.json'%(epoch, global_step))
                accelerator.log({"val_coco":coco_eval_result['CIDEr']}, step=global_step)
                save_model(blip, args.output_dir+'/blip_text_decoder.pth')

                if coco_eval_result['CIDEr']+coco_eval_result['Bleu_4'] > best['ciderb4']:
                    best['ciderb4'] = coco_eval_result['CIDEr']+coco_eval_result['Bleu_4']
                    best_epoch['ciderb4'] = (epoch, global_step)
                    save_model(blip, args.output_dir+'/blip_text_decoder_best_ciderb4.pth')

                log_stats = {**{f'val_{k}': v for k, v in coco_eval_result.items()},
                             'epoch': epoch,
                             'step': global_step,
                             'best_epoch': best_epoch,
                            }
                with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_model(unet, args.output_dir+'/unet.pth')


        if not args.skip_evaluation:
            coco_eval_result = evaluate_blip(blip, 'coco_val', args.output_dir+'/result/coco_epoch%d_%d.json'%(epoch, global_step))
            accelerator.log({"val_coco":coco_eval_result['CIDEr']}, step=global_step)
            save_model(blip, args.output_dir+'/blip_text_decoder.pth')

            if coco_eval_result['CIDEr']+coco_eval_result['Bleu_4'] > best['ciderb4']:
                best['ciderb4'] = coco_eval_result['CIDEr']+coco_eval_result['Bleu_4']
                best_epoch['ciderb4'] = (epoch, global_step)
                save_model(blip, args.output_dir+'/blip_text_decoder_best_ciderb4.pth')

            log_stats = {**{f'val_{k}': v for k, v in coco_eval_result.items()},
                            'epoch': epoch,
                            'step': global_step,
                            'best_epoch': best_epoch,
                        }
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")

            save_model(unet, args.output_dir+'/unet.pth')

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        save_model(unet, args.output_dir+'/unet.pth')
    accelerator.end_training()
    save_model(blip, args.output_dir+'/blip_text_decoder.pth')


if __name__ == "__main__":
    main()
