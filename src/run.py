import argparse
import os
import swanlab
import torch
import warnings
import numpy as np
from tqdm import tqdm
import accelerate
from torch.utils.data import DataLoader
from accelerate import Accelerator
from model_gpt2 import PromptGPT2forCRS
from model_prompt import ErniePromptModel
from crsdataset import CRSDataset, KGInfo, CRSDataCollator
from evaluator import RecEvaluator, ConvEvaluator
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, AutoModel
warnings.filterwarnings("ignore")


def train_eval(task, num_train_epoch, batch_size, learning_rate):
    if num_train_epoch==0:
        return
    # seed init.
    global new_tag, load_model_path
    accelerate.utils.set_seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    accelerator = Accelerator(device_placement=False)
    device = accelerator.device

    encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
    encoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<movie>'], })
    encoder_model = AutoModel.from_pretrained(encoder_model_path)
    encoder_model.resize_token_embeddings(len(encoder_tokenizer))
    encoder_model = encoder_model.to(device)
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_model_path)
    decoder_tokenizer.add_special_tokens({'pad_token': '<pad>', 'additional_special_tokens': ['<movie>'], })
    decoder_model = PromptGPT2forCRS.from_pretrained(decoder_model_path)
    decoder_model.resize_token_embeddings(len(decoder_tokenizer))
    decoder_model.config.pad_token_id = decoder_tokenizer.pad_token_id
    decoder_model = decoder_model.to(device)

    kg_info = KGInfo(data_path=data_path, device=device).get_kg_info()
    prompt_model = ErniePromptModel(kg_info, add_user=args.add_user, add_align=args.add_align, add_tree=args.add_tree, n_prefix_rec=args.n_prefix_rec, n_prefix_conv=args.n_prefix_conv).to(device)

    fix_modules = [encoder_model, decoder_model]
    for module in fix_modules:
        module.requires_grad_(False)

    modules = [prompt_model]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for model in modules for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.01, },
        {"params": [p for model in modules for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0, }, ]

    if load_model_path != "" and new_tag != "":
        prompt_model.load_model(load_model_path, new_tag)

    train_dataset = CRSDataset(data_path=data_path, split="train", task=task, entity_pad_idx=kg_info['entity_pad_idx'], add_tree=args.add_tree, ary1=args.ary1, ary2=args.ary2)
    test_dataset = CRSDataset(data_path=data_path, split="test", task=task, entity_pad_idx=kg_info['entity_pad_idx'], add_tree=args.add_tree, ary1=args.ary1, ary2=args.ary2)
    collator = CRSDataCollator(task=task, device=device, entity_pad_idx=kg_info['entity_pad_idx'], add_tree=args.add_tree)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collator, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collator)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, len(train_dataloader), num_train_epoch * 2 * len(train_dataloader))
    prompt_model, train_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(prompt_model, train_dataloader, test_dataloader, optimizer, lr_scheduler)
    if task == 'rec' or task == 'rec_pre':
        evaluator = RecEvaluator()
    elif task == 'conv' or task == 'conv_pre':
        evaluator = ConvEvaluator(decoder_tokenizer, output_dir)
    else:
        assert False
    progress_bar = tqdm(range(num_train_epoch * len(train_dataloader)), disable=not accelerator.is_local_main_process)
    print(f"[-1/{num_train_epoch}]--------------------------begin {task} task--------------------------")
    for epoch in range(num_train_epoch):
        prompt_model.train()
        train_losses = []
        for data_batch in train_dataloader:
            with torch.no_grad():
                data_batch["prompt"]["token_embeds"] = encoder_model(**data_batch["encoder"]).last_hidden_state
                if args.add_tree:
                    data_batch["prompt"]["tree_embeds"] = encoder_model(**data_batch["tree"]).last_hidden_state
            if task == 'rec' or task == 'rec_pre':
                prompt_embeds, user_loss, align_loss = prompt_model(**data_batch["prompt"], output_entity=True, use_rec_prefix=task == 'rec')
                data_batch["decoder"]["prompt_embeds"] = prompt_embeds
                data_batch["decoder"]["entity_embeds"] = prompt_model.get_entity_embeds()
                outputs = decoder_model(**data_batch["decoder"], rec=True)
                loss = outputs.rec_loss
                if args.add_user:
                    loss += args.alpha_a * user_loss
                if args.add_align:
                    loss += args.alpha_b * align_loss
            elif task == 'conv' or task == 'conv_pre':
                prompt_embeds, user_loss, align_loss = prompt_model(**data_batch["prompt"], output_entity=False, use_conv_prefix=task == 'gen')
                data_batch["decoder"]["prompt_embeds"] = prompt_embeds
                data_batch["conv_decoder"]["prompt_embeds"] = prompt_embeds
                loss = decoder_model(**data_batch["decoder"], conv=True).conv_loss
                if args.add_user:
                    loss += args.alpha_a * user_loss
                if args.add_align:
                    loss += args.alpha_b * align_loss
            else:
                assert False
            accelerator.backward(loss)
            train_losses.append(float(loss))
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            run.log({f'train_{task}/step': progress_bar.n, f'train_{task}/loss': np.mean(train_losses)})
        new_tag = prompt_model.save_model(output_dir, new_tag, task)
        load_model_path = output_dir
        del train_losses
        prompt_model.eval()
        eval_losses = []
        for data_batch in enumerate(test_dataloader):
            with torch.no_grad():
                data_batch["prompt"]["token_embeds"] = encoder_model(**data_batch["encoder"]).last_hidden_state
                if args.add_tree:
                    data_batch["prompt"]["tree_embeds"] = encoder_model(**data_batch["tree"]).last_hidden_state
                if task == 'rec' or task == 'rec_pre':
                    prompt_embeds, _, _ = prompt_model(**data_batch["prompt"], output_entity=True, use_rec_prefix=task == 'rec')
                    data_batch["decoder"]["prompt_embeds"] = prompt_embeds
                    data_batch["decoder"]["entity_embeds"] = prompt_model.get_entity_embeds()
                    outputs = decoder_model(**data_batch["decoder"], rec=True)
                    loss = outputs.rec_loss
                    eval_losses.append(float(loss))
                    if task == 'rec':
                        logits = outputs.rec_logits[:, kg_info['movie_idxs']]
                        ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                        ranks = [[kg_info['movie_idxs'][rank] for rank in batch_rank] for batch_rank in ranks]
                    else:
                        logits = outputs.rec_logits
                        ranks = torch.topk(logits, k=50, dim=-1).indices.tolist()
                    labels = data_batch['decoder']['rec_labels']
                    evaluator.evaluate(ranks, labels)
                elif task == 'conv' or task == 'conv_pre':
                    prompt_embeds, _, _ = prompt_model(**data_batch["prompt"], output_entity=False, use_conv_prefix=True)
                    data_batch["decoder"]["prompt_embeds"] = prompt_embeds
                    data_batch["gen_decoder"]["prompt_embeds"] = prompt_embeds
                    loss = decoder_model(**data_batch['decoder'], conv=True).conv_loss
                    eval_losses.append(float(loss))
                    gen_seqs = accelerator.unwrap_model(decoder_model).generate(**data_batch['gen_decoder'], max_new_tokens=50, no_repeat_ngram_size=3)
                    gen_resp_ids = []
                    for gen_seq, length in zip(gen_seqs, data_batch["gen_decoder_input_ids_len"]):
                        gen_seq = [token_id for token_id in gen_seq if token_id != decoder_tokenizer.pad_token_id]
                        gen_resp_ids.append(gen_seq[length:])
                    evaluator.evaluate(gen_resp_ids, data_batch["gen_decoder_label_ids"], log=True)
                else:
                    assert False
        report = accelerator.gather(evaluator.report())
        if task == 'rec' or task == 'rec_pre':
            eval_report = {f'eval_{task}/{k}': v.sum().item() / report['count'] for k, v in report.items() if k != 'count'}
        elif task == 'conv' or task == 'conv_pre':
            eval_report = {f'eval_{task}/{k}': v for k, v in report.items() if k != 'count'}
        else:
            assert False
        eval_report.update({f'eval_{task}/epoch': epoch, f'eval_{task}/loss': np.mean(eval_losses)})
        run.log(eval_report)
        evaluator.reset_metric()
        del eval_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="inspired")
    parser.add_argument("--add_user", type=bool, default=True)
    parser.add_argument("--add_tree", type=bool, default=True)
    parser.add_argument("--add_align", type=bool, default=True)
    parser.add_argument("--alpha_a", type=float, default=0.02)
    parser.add_argument("--alpha_b", type=float, default=0.002)
    parser.add_argument("--ary1", type=int, default=5)
    parser.add_argument("--ary2", type=int, default=5)
    parser.add_argument("--n_prefix_rec", type=int, default=10)
    parser.add_argument("--n_prefix_conv", type=int, default=20)
    parser.add_argument("--rec_pre_num_train_epoch", type=int, default=0)
    parser.add_argument("--rec_pre_learning_rate", type=float, default=6e-4)
    parser.add_argument("--rec_pre_batch_size", type=int, default=64)
    parser.add_argument("--rec_num_train_epoch", type=int, default=0)
    parser.add_argument("--rec_learning_rate", type=float, default=1e-4)
    parser.add_argument("--rec_batch_size", type=int, default=64)
    parser.add_argument("--conv_pre_num_train_epoch", type=int, default=0)
    parser.add_argument("--conv_pre_learning_rate", type=float, default=6e-4)
    parser.add_argument("--conv_pre_batch_size", type=int, default=8)
    parser.add_argument("--conv_num_train_epoch", type=int, default=0)
    parser.add_argument("--conv_learning_rate", type=float, default=1e-4)
    parser.add_argument("--conv_batch_size", type=int, default=8)
    args = parser.parse_args()
    print(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    data_path = f"../data/{args.dataset}/"
    logdir = "../swanlab/"
    os.makedirs(logdir, exist_ok=True)
    run = swanlab.init(project="crs-prompt-" + args.dataset + "-X", logdir=logdir)
    output_dir = logdir + str(run)
    encoder_model_path = "../model/roberta-base"
    decoder_model_path = "../model/dialogpt-small"
    load_model_path, new_tag = "", ""
    train_eval(task='rec_pre', num_train_epoch=args.rec_pre_num_train_epoch, batch_size=args.rec_pre_batch_size, learning_rate=args.rec_pre_learning_rate)
    train_eval(task='rec', num_train_epoch=args.rec_num_train_epoch, batch_size=args.rec_batch_size, learning_rate=args.rec_learning_rate)
    train_eval(task='conv_pre', num_train_epoch=args.conv_pre_num_train_epoch, batch_size=args.conv_pre_batch_size, learning_rate=args.conv_pre_learning_rate)
    train_eval(task='conv', num_train_epoch=args.conv_num_train_epoch, batch_size=args.conv_batch_size, learning_rate=args.conv_learning_rate)