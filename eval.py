import numpy as np
import argparse, time
import os

import torch
from torch.utils.data import DataLoader

from transformers import get_linear_schedule_with_warmup
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    set_seed,
)

from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

from utils import compute_metrics, print_result, load_tokenizer, load_model, load_optimizer, load_scheduler, format_time
from dataset import DepressionDataset

from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()

    # initialization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument('--gpu_id', type=str, default="1")
    parser.add_argument("--task_name", type=str, default="depression")
    parser.add_argument('--model_path', type=str, default='./output_models/{}/')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--data_path", type=str, default="./dataset/{}/{}.json")
    parser.add_argument('--num_labels', type=int, default=2)
    
    # model related
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-cased")
    #parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--logging_steps', type=int, default=100)

    return parser.parse_args()



def main(args):

    print(args)
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('  *** Device: ', device)
    print('  *** Current cuda device:', args.gpu_id)


    model_path = args.model_path.format('checkpoint-16_0_2297')
    print(' *** Test run by model: {}'.format(model_path))

    tokenizer = load_tokenizer(model_path)

    model = load_model(model_path)

    device = torch.device("cuda")
    model.cuda()

    
    test_dataset = DepressionDataset(
        args = args,
        mode = 'test',
        tokenizer=tokenizer,
    )


    test_dl = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )


    model.eval()
    all_preds = []
    all_labels = []
    t0 = time.time()

    
    for step, data in enumerate(tqdm(test_dl, desc='test', mininterval=0.01, leave=True), 0):
        inputs = {
                "input_ids": data['input_ids'].to(device),
                "attention_mask": data['attention_mask'].to(device),
            }
        labels = data['labels'].to(device)

        with torch.no_grad():
            outputs = model(inputs, labels=labels)


        logits = outputs[1]

        preds = logits.argmax(-1)

        if len(all_preds)==0:
            all_preds = preds.detach().cpu().clone().numpy()
            all_labels = labels.detach().cpu().clone().numpy()
        else:
            all_preds = np.append(all_preds, preds.detach().cpu().clone().numpy())
            all_labels = np.append(all_labels, labels.detach().cpu().clone().numpy())


    test_result = compute_metrics(labels=all_labels, preds=all_preds)
    print_result(test_result)
    print("  Test took: {:}".format(format_time(time.time() - t0)))




if __name__ == '__main__':
    from eval import get_args
    args = get_args()
    main(args)
