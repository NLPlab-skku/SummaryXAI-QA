# -*- coding: utf-8 -*-"""

from util import *
from transformers import AdamW, get_cosine_with_hard_restarts_schedule_with_warmup, AutoConfig, AutoTokenizer
from model import Model, Model_copy

logger = logging.getLogger(__name__)


def Metatrain(args, model, source_dataset_name_list, target_dataset_name):
    """ MetaTrain the model """
    outer_optimizer = AdamW(model.parameters(), lr=args.outer_learning_rate, eps=args.adam_epsilon)
    batch_tasks = make_support_query_tensordataset(args, source_dataset_name_list)
    num_task = len(batch_tasks)
    best_loss=999999.99

    loss_function = nn.NLLLoss()
    for epoch in tqdm(range(args.outer_epoch)):
        model.train()
        sum_gradients = []
        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query = task[1]

            fast_model = deepcopy(model)
            fast_model.to(args.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support), batch_size=args.inner_batch_size, collate_fn=collate_fn)

            inner_optimizer = AdamW(fast_model.parameters(), lr=args.inner_learning_rate, eps=args.adam_epsilon)
            fast_model.train()

            for i in range(0, args.num_inner_update_step):
                all_loss = []
                batch = iter(support_dataloader).next()
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "decoder_input_ids": batch[2],
                    "decoder_attention_mask": batch[3],
                }

                outputs = torch.log(fast_model(**inputs)+1e-7)

                loss = loss_function(outputs.view(-1, outputs.shape[-1]), batch[4].view(-1))
                loss.backward()

                for num, params in enumerate(fast_model.parameters()):
                    if num < args.total_num_of_grad - args.num_of_grad:
                        params.grad = None

                inner_optimizer.step()
                inner_optimizer.zero_grad()

                if i % 3 == 0:
                    print(f"{task_id} Inner Loss: ", loss.item())

            query_dataloader = DataLoader(query, sampler=RandomSampler(query), batch_size=args.outer_batch_size, collate_fn=collate_fn)
            query_batch = iter(query_dataloader).next()
            query_batch = tuple(t.to(args.device) for t in query_batch)
            query_inputs = {
                "input_ids": query_batch[0],
                "attention_mask": query_batch[1],
                "decoder_input_ids": query_batch[2],
                "decoder_attention_mask": query_batch[3],
            }
            q_outputs = torch.log(fast_model(**query_inputs)+1e-7)

            q_loss = loss_function(q_outputs.view(-1, q_outputs.shape[-1]), query_batch[4].view(-1))
            q_loss.backward()

            for i, params in enumerate(fast_model.parameters()):
                if i < args.total_num_of_grad - args.num_of_grad:
                    params.grad = None

            fast_model.to('cpu')
            for i, params in enumerate(fast_model.parameters()):
                if task_id == 0:
                    if params.grad is None:
                        sum_gradients.append(None)
                    else:
                        sum_gradients.append(deepcopy(params.grad))
                else:
                    if params.grad is None:
                        pass
                    else:
                        sum_gradients[i] += deepcopy(params.grad)

            del fast_model, inner_optimizer
            torch.cuda.empty_cache()

        # Average gradient across tasks
        for i in range(0, len(sum_gradients)):
            if sum_gradients[i] is not None:
                sum_gradients[i] = sum_gradients[i] / float(num_task)

        # Assign gradient for original model, then using optimizer to update its weights
        for i, params in enumerate(model.parameters()):
            params.grad = sum_gradients[i]

        outer_optimizer.step()
        outer_optimizer.zero_grad()
        del sum_gradients

        if (epoch+1)%100 == 0:
            validation_model = deepcopy(model)
            torch.cuda.empty_cache()
            results = meta_validation(args, validation_model, target_dataset_name)

            if best_loss > results:
                best_loss = results

                logging.info("Saving model checkpoint!")
                torch.save(args, args.output_dir/"training_args.bin")
                torch.save(model.state_dict(), args.output_dir/"pytorch_model.bin")
                torch.save(outer_optimizer.state_dict(), args.output_dir/"optimizer.pt")
                
            if (epoch+1) == 6000:
                logging.info("Saving model checkpoint!")
                torch.save(args, args.output_dir/"training_args_last.bin")
                torch.save(model.state_dict(), args.output_dir/"pytorch_model_last.bin")
                torch.save(outer_optimizer.state_dict(), args.output_dir/"optimizer_last.pt")

            validation_model.to('cpu')
            del validation_model
            torch.cuda.empty_cache()

        gc.collect()

def meta_validation(args, model, data_name):
    """ Train the model """
    dataset_file = Path(f"{args.output_dir}/meta_validation_train_{args.seed}")
    if not dataset_file.exists():
        train_dataset = load_and_cache_examples(args, data_name, mode="meta_validation_train")
        torch.save(train_dataset,dataset_file)
    else:
        train_dataset = torch.load(dataset_file)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=5, collate_fn=collate_fn)

    optimizer = AdamW(model.parameters(), lr=args.train_learning_rate, eps=args.adam_epsilon)
    model.to(args.device)
    loss_function = nn.NLLLoss()
    train_iterator = trange(int(args.num_meta_val_train_epochs), desc="Epoch...")
    model.train()

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, position=0, leave=True, desc="Iteration")
        for batch in epoch_iterator:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "decoder_input_ids": batch[2],
                "decoder_attention_mask": batch[3],
            }

            outputs = torch.log(model(**inputs)+1e-7)

            loss = loss_function(outputs.view(-1, outputs.shape[-1]), batch[4].view(-1))
            loss.backward()

            for i, params in enumerate(model.parameters()):
                if i < args.total_num_of_grad - args.num_of_grad:
                    params.grad = None

            optimizer.step()
            optimizer.zero_grad()

            print(f"{epoch}'th Loss: ", loss.item())

    model.to('cpu')
    torch.cuda.empty_cache()
    results = validation(args, model, data_name, mode="meta_validation")
    return results

def validation(args, model, data_name,mode):
    validation_dataset = load_and_cache_examples(args, data_name, mode=mode)
    validation_sampler = SequentialSampler(validation_dataset)
    validation_dataloader = DataLoader(validation_dataset, sampler=validation_sampler, batch_size=args.validation_batch_size,collate_fn=collate_fn)
    loss_function = nn.NLLLoss()
    total_loss = 0.0
    model.to(args.device)
    model.eval()

    for batch in tqdm(validation_dataloader):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "decoder_input_ids": batch[2],
            "decoder_attention_mask": batch[3],
        }

        outputs = torch.log(model(**inputs)+1e-7)

        loss = loss_function(outputs.view(-1, outputs.shape[-1]), batch[4].view(-1))
        loss=loss.detach()
        total_loss+=loss
        model.zero_grad()
        torch.cuda.empty_cache()

    logger.info("***** {} loss *****".format(mode))
    logger.info("Total loss = {}".format(total_loss))
    return total_loss


def fine_tune(args, model, data_name, num_of_sampled_examples):
    """ Train the model """
    train_dataset = load_and_cache_examples(args, data_name, mode=f"fine_tuning_{num_of_sampled_examples}")
    train_sampler = RandomSampler(train_dataset)

    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.fine_tuning_learning_rate, eps=args.adam_epsilon)
    loss_function = nn.NLLLoss()
    train_iterator = trange(int(args.num_fine_tuning_epochs), desc="Epoch...")

    best_loss =99999.99
    best_f1 = 0.0

    for epoch in train_iterator:
        if num_of_sampled_examples == 100:
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.fine_tuning_batch_size, collate_fn=collate_fn)
        elif num_of_sampled_examples == 10:
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=5, collate_fn=collate_fn)
        elif num_of_sampled_examples == 1000:
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=5, collate_fn=collate_fn)
            args.num_fine_tuning_epochs = 10
        epoch_iterator = tqdm(train_dataloader, position=0, leave=True, desc="Iteration")
        model.to(args.device)
        model.train()
        for batch in epoch_iterator:
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "decoder_input_ids": batch[2],
                "decoder_attention_mask": batch[3],
            }

            outputs = torch.log(model(**inputs)+1e-7)
                  
            loss = loss_function(outputs.view(-1, outputs.shape[-1]), batch[4].view(-1))
            loss.backward()
            for i, params in enumerate(model.parameters()):
                if i < args.total_num_of_grad - args.num_of_grad:
                    params.grad = None
            optimizer.step()
            optimizer.zero_grad()

            print(f"{epoch}'th Loss: ", loss.item())
  
        results = validation(args, model, data_name, mode="meta_validation")

        if best_loss > results:
            best_loss = results

            logging.info("Saving model checkpoint!")
            torch.save(args, args.output_dir / f"training_args_{num_of_sampled_examples}.bin")
            torch.save(model.state_dict(), args.output_dir / f"pytorch_model_{num_of_sampled_examples}.bin")
            torch.save(optimizer.state_dict(), args.output_dir / f"optimizer_{num_of_sampled_examples}.pt")

    model.to('cpu')
    del model
    torch.cuda.empty_cache()

    return results

def evaluate(args, model, data_name, mode):
    eval_dataset, labels = load_and_cache_examples(args, data_name, mode)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = []
    model.to(args.device)
    model.eval()
    count = 0
    if mode == "test":
        index = 0
        json_dict = {}
    for batch in tqdm(eval_dataloader, position=0, leave=True, desc="Evaluating..."):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            input_ids = batch[0]
            inputs = {
                "input_ids": input_ids,
                "max_length": args.generate_max_length,
                "min_length": args.generate_min_length,
                "num_beams": args.generate_num_beams,
                "length_penalty": args.generate_length_penalty,
                "no_repeat_ngram_size": args.generate_no_repeat_ngram_size,
                "early_stopping": True
            }
            summary_ids = model.generative_step(**inputs)

            for i in range(len(summary_ids)):
                pred = tokenizer.decode(summary_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                preds.append(pred)
                print(pred)
                if mode == "test":
                    tmp_dict={}
                    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    tmp_dict["input_text"]=input_text
                    tmp_dict["label"]=labels[index]
                    tmp_dict["pediction"]=pred
                    tmp_dict["score"]=calculate_rouge([pred], [labels[index]])
                    json_dict[f"{index}"]=tmp_dict
                    index+=1
            if index ==100:
                break

    results = calculate_rouge(preds, labels)

    logger.info("***** Eval results *****")
    logger.info("rouge1:  {:.4f}%; rouge2:  {:.4f}%; rougeL:  {:.4f}%"
                .format(results["rouge1"], results["rouge2"], results["rougeL"]))

    if mode == "validation" or mode == "meta_validation":
        return results
    elif mode == "test":
        if not Path(f"{args.output_dir}/output_{data_name}_10.json").exists():
            with open(f"{args.output_dir}/output_{data_name}_10.json", "w") as json_file:
                json.dump(json_dict, json_file,indent=4)
        else:
            with open(f"{args.output_dir}/output_{data_name}_100.json", "w") as json_file:
                json.dump(json_dict, json_file,indent=4)            
        return results, preds

def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None and pad_token_id != -100:
        return input_ids[:, keep_column_mask]
    elif attention_mask is None and pad_token_id == -100:
        prefix = torch.tensor([-100]).unsqueeze(0).repeat(input_ids.shape[0],1)
        return torch.cat([input_ids[:, keep_column_mask],prefix],dim=1)
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def collate_fn(batch) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([x[0] for x in batch])
    attention_mask = torch.stack([x[1] for x in batch])
    decoder_input_ids = torch.stack([x[2] for x in batch])
    decoder_attention_mask = torch.stack([x[3] for x in batch])
    label_ids = torch.stack([x[4] for x in batch])
    pad_token_id = 1
    trim_input_ids, trim_attention_mask = trim_batch(input_ids, pad_token_id, attention_mask=attention_mask)
    trim_decoder_input_ids, trim_decoder_attention_mask = trim_batch(decoder_input_ids, pad_token_id, attention_mask=decoder_attention_mask)
    trim_label_ids = trim_batch(label_ids, -100)
    batch = [trim_input_ids,trim_attention_mask,trim_decoder_input_ids,trim_decoder_attention_mask,trim_label_ids]
    return batch

def calculate_rouge(output_lns, reference_lns, use_stemmer=True):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

def main():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--layers", default=12, type=int, help="The number of encoder or decoder layers.")
    parser.add_argument("--d_model", default=1024, type=int, help="Dimension of output state.")
    parser.add_argument("--n_heads", default=16, type=int, help="Number of multi heads.")
    parser.add_argument("--d_k", default=64, type=int, help="Dimension of a head.")
    parser.add_argument("--d_ff", default=4096, type=int, help="Dimension of ffn.")
    parser.add_argument("--preseqlen", default=200, type=int, help="Length of prefix.")
    parser.add_argument("--mid_dim", default=800, type=int, help="Dimension of mid.")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout probability.")
    # util
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--data_dir",
        default="./corpus",
        type=Path,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    # main
    parser.add_argument(
        "--save_model",
        type=str,
        required=True,
        help="The output directory name where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--model_name",
        default="facebook/bart-large",
        type=str,
        help="Shortcut name select"
    )
    parser.add_argument("--outer_learning_rate", default=5e-5, type=float,
                        help="The initial outer learning rate for AdamW.")#5e-5
    parser.add_argument("--inner_learning_rate", default=5e-5, type=float,
                        help="The initial inner learning rate for Adam.")
    parser.add_argument("--train_learning_rate", default=5e-5, type=float,
                        help="The initial train learning rate for Adam.")
    parser.add_argument("--fine_tuning_learning_rate", default=5e-5, type=float,
                        help="The initial train learning rate for Adam.")
    parser.add_argument("--total_num_of_grad", default=526, type=int, help="Total number of gradient.")
    parser.add_argument("--num_of_grad", default=15, type=int, help="Number of gradient we use.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--outer_epoch", default=6000, type=int, help="Total epoch of outer loop.")
    parser.add_argument("--inner_batch_size", default=4, type=int, help="Inner batch size for training.")
    parser.add_argument("--outer_batch_size", default=4, type=int, help="Outer batch size for training.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="train batch size for training.")
    parser.add_argument("--fine_tuning_batch_size", default=4, type=int, help="fine tuning batch size for training.")
    parser.add_argument("--validation_batch_size", default=4, type=int, help="validation batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int, help="evaluation batch size for training.")
    parser.add_argument("--num_inner_update_step", default=4, type=int,
                        help="Number of inner update step for training.")
    parser.add_argument("--num_meta_val_train_epochs", default=4, type=int, help="Total number of meta validation training epochs to perform.")
    parser.add_argument("--num_fine_tuning_epochs", default=20, type=int, help="Total number of fine tuning epochs to perform.")

    parser.add_argument("--generate_max_length", default=512, type=int, help="The maximum length of the sequence to be generated.")
    parser.add_argument("--generate_min_length", default=10, type=int, help="The minimum length of the sequence to be generated.")
    parser.add_argument("--generate_num_beams", default=8, type=int, help="Number of beams for beam search. 1 means no beam search.")
    parser.add_argument("--generate_repetition_penalty", default=1.0, type=float, help="The parameter for repetition penalty. 1.0 means no penalty.")
    parser.add_argument("--generate_length_penalty", default=1.0, type=float, help="A value > 1.0 in order to encourage the model to produce longer sequences.")
    parser.add_argument("--generate_no_repeat_ngram_size", default=3, type=int, help="If set to int > 0, all ngrams of that size can only occur once.")
    parser.add_argument("--data_name", default="", type=str, help="If set to int > 0, all ngrams of that size can only occur once.")
    parser.add_argument("--num_data", default=0, type=int, help="The number of few shot data when finetuning")

    args = parser.parse_args()

    if not args.do_train and not args.do_test:
        raise ValueError("You must select do_train or do_test")
    #"aeslc", "billsum", "gigaword", "multi_news", "newsroom", "reddit_tifu", "arxiv", "pubmed", "wikihow", "bigpatent","xsum","cnn_dailymail"
    for task in [args.data_name]:
        args.output_dir = Path(f"./models/{args.save_model}_{task}")
        target_dataset_name = task
        source_dataset_name_list=[]
        if task == "aeslc":
            source_dataset_name_list = ["wikihow", "reddit_tifu", "xsum"]
            # args.generate_max_length = 32
            args.generate_min_length = 5
        elif task == "billsum":
            source_dataset_name_list = ["bigpatent","wikihow", "pubmed"]
            # args.generate_max_length = 256
            args.generate_min_length = 64
            args.generate_length_penalty = 2.0
        elif task == "gigaword":
            source_dataset_name_list = ["xsum", "aeslc", "reddit_tifu"]
            # args.generate_max_length = 32
            # args.generate_min_length = 8
        elif task == "multi_news":
            source_dataset_name_list = ["cnn_dailymail", "xsum", "reddit_tifu"]
            # args.generate_max_length = 256
            # args.generate_min_length = 128
        elif task == "newsroom":
            source_dataset_name_list = ["cnn_dailymail", "xsum", "reddit_tifu"]
            # args.generate_max_length = 256
            # args.generate_min_length = 20
        elif task == "reddit_tifu":
            source_dataset_name_list = ["wikihow", "xsum", "cnn_dailymail"]
            # args.generate_max_length = 128
        elif task == "arxiv":
            source_dataset_name_list = ["bigpatent", "pubmed", "wikihow"]
            # args.generate_max_length = 256
            args.generate_min_length = 64
            args.generate_length_penalty = 2.0
        elif task == "pubmed":
            source_dataset_name_list = ["bigpatent", "arxiv", "multi_news"]
            # args.generate_max_length = 256
            args.generate_min_length = 64
            args.generate_length_penalty = 2.0
        elif task == "wikihow":
            source_dataset_name_list = ["reddit_tifu", "xsum", "cnn_dailymail"]
            # args.generate_max_length = 256
            # args.generate_min_length = 24
            # args.generate_length_penalty = 1.0
        elif task == "bigpatent":
            source_dataset_name_list = ["pubmed", "arxiv", "wikihow"]
            # args.generate_max_length = 256
            args.generate_min_length = 64
            args.generate_length_penalty = 2.0
        elif task == "xsum":
            source_dataset_name_list = ["reddit_tifu", "cnn_dailymail", "wikihow"]
            args.generate_max_length = 64
            args.generate_min_length = 11
            # args.generate_length_penalty = 1.0
        elif task == "cnn_dailymail":
            source_dataset_name_list = ["multi_news", "wikihow","xsum"]
            args.generate_max_length = 142
            args.generate_min_length = 56
            args.generate_length_penalty = 2.0
        if not args.output_dir.exists():
            args.output_dir.mkdir()

        # Setup CUDA, GPU
        args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        # Setup logging
        if args.do_train:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
                filename=args.output_dir / "train_logging.log",
                filemode="w"
            )
        elif args.do_test and args.num_data==0:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
                filename=args.output_dir / "test_logging.log",
                filemode="w"
            )
        elif args.do_test and args.num_data!=0:
            logging.basicConfig(
                format="%(asctime)s - %(levelname)s - %(message)s",
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
                filename=args.output_dir / f"test_logging_{args.num_data}.log",
                filemode="w"
            )

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

        logger.warning(
            "======[Device]: %s, [n_gpu]: %s ======",
            args.device,
            args.n_gpu
        )

        # Set seed
        set_seed(args)
        logger.info("Training/evaluation parameters %s", args)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

        # Training
        if args.do_train:
            logging.info("Loading model...")
            model = Model_copy(args)
            logging.info("Loading is done!")
            Metatrain(args, model, source_dataset_name_list, target_dataset_name)

        # Evaluation
        if args.do_test:
            if Path(args.output_dir / "pytorch_model.bin").exists() and args.num_data==0:

                model = Model_copy(args)
                logging.info("Loading model...")
                model.load_state_dict(torch.load(Path(args.output_dir / "pytorch_model.bin")))
                logging.info("Loading is done!")
                fine_tune(args, model, target_dataset_name,10)

                model = Model_copy(args)
                logging.info("Loading model...")
                model.load_state_dict(torch.load(Path(args.output_dir / "pytorch_model.bin")))
                logging.info("Loading is done!")
                fine_tune(args, model, target_dataset_name,100)
                
            elif Path(args.output_dir / f"pytorch_model_{args.num_data}.bin").exists():
                model = Model(args)
                logging.info("Loading model...")
                model.load_state_dict(torch.load(Path(args.output_dir / f"pytorch_model_{args.num_data}.bin")))
                logging.info("Loading is done!")
                evaluate(args, model, target_dataset_name, "test")    

            else:
                raise ValueError("Checkpoint dose not exist")

if __name__ == "__main__":
    main()
