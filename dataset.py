import torch
import json
from torch.utils.data import Dataset


def load_data(path):
    with open(path, "r") as fp:
        data = fp.read().strip().split("\n")
    return data


def print_dataset_example(input_input_ids, label_input_ids, tokenizer):
    print("input_ids", input_input_ids)
    print("input_tokens", tokenizer.convert_ids_to_tokens(input_input_ids))
    print("inputs", tokenizer.decode(input_input_ids))
    print("label_ids", label_input_ids)
    print("label_tokens", tokenizer.convert_ids_to_tokens(label_input_ids))
    print("labels", tokenizer.decode(label_input_ids))


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)

IGNORE_INDEX = -100


class NerCollate:
    def __init__(self, args, tokenizer):
        self.instruct_column = args.instruct_column
        self.query_column = args.query_column
        self.response_column = args.response_column
        self.history_column = None
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length

    def collate_fn(self, batch):
        sources = []
        targets = []
        prompt = PROMPT_TEMPLATE
        for example in batch:
            if isinstance(example, str):
                example = json.loads(example)
            instruction = example[self.instruct_column]
            input = example[self.query_column]
            output = example[self.response_column]
            if input is not None and input != "":
                instruction = instruction + '\n' + input
            source = prompt.format_map({'instruction': instruction})
            target = f"{self.tokenizer.bos_token}{output}{self.tokenizer.eos_token}"

            # print(json.dumps(source, ensure_ascii=False), json.dumps(target, ensure_ascii=False))
            sources.append(source)
            targets.append(target)

        tokenized_sources = self.tokenizer(sources, return_attention_mask=False, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, return_attention_mask=False, add_special_tokens=False)
        # print(tokenized_sources)
        # print(tokenized_targets)

        # print(self.tokenizer.convert_ids_to_tokens(tokenized_sources["input_ids"][0]))
        # print(self.tokenizer.convert_ids_to_tokens(tokenized_targets["input_ids"][0]))

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_ids = (s + t)[:self.max_seq_length]
            labels = ([IGNORE_INDEX] * len(s) + t)[:self.max_seq_length]
            assert len(input_ids) == len(labels)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(input_ids))
            labels = labels + [IGNORE_INDEX] * (self.max_seq_length - len(labels))
            # print(input_ids)
            # print(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

            # print(self.tokenizer.decode(input_ids))
            # print(labels)
        results = {'input_ids': torch.tensor(all_input_ids), 'labels': torch.tensor(all_labels)}
        return results


if __name__ == "__main__":
    class Args:
        max_seq_length = 128 + 64
        instruct_column = "instruct"
        query_column = "query"
        response_column = "answer"
        train_path = "data/msra/instruct_data/train.txt"


    args = Args()
    from transformers import LlamaTokenizer

    tokenizer = LlamaTokenizer.from_pretrained("./model_hub/chinese-alpaca-7b", trust_remote_code=True)
    data = load_data(args.train_path)[:10]
    data = [
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
         "query": "文本：我们是受到郑振铎先生、阿英先生著作的启示，从个人条件出发，瞄准现代出版史研究的空白，重点集藏解放区、国民党毁禁出版物。", "answer": "郑振铎_人名\n阿英_人名\n国民党_机构名"},
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
         "query": "文本：去年，我们又被评为“北京市首届家庭藏书状元明星户”。", "answer": "北京市_地名"},
        {"instruct": "你现在是一个实体识别模型，你需要提取文本里面的人名、地名、机构名，如果存在结果，返回'实体_实体类型'，不同实体间用\n分隔。如果没有结果，回答'没有'。",
         "query": "文本：藏书家、作家姜德明先生在1997年出版的书话专集《文林枝叶》中以“爱书的朋友”为题，详细介绍了我们夫妇的藏品及三口之家以书为友、好乐清贫的逸闻趣事。", "answer": "姜德明_人名"},
    ]
    data = [data[1]]
    print(data)

    ner_collate = NerCollate(args, tokenizer)

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(data,
                                  batch_size=1,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=0,
                                  collate_fn=ner_collate.collate_fn)
    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        print(input_ids.shape, labels.shape)
        break

    # train_dataset = ner_collate.collate_fn(data)
    # print(train_dataset["input_ids"][0])
