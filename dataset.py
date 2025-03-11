import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict
import torch
from torch.utils.data import DataLoader



class IterableParquetDataset(IterableDataset):
    def __init__(
        self,
        parquet_file: str,
        tokenizer,
        sequence_length: int,
        bos_token_id: int = 1
    ):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.bos_token_id = bos_token_id
        self.current_index = 0
        self.token_buffer = []

    def __iter__(self):
        # Reset buffer and index when starting a new iteration
        self.token_buffer = []
        self.current_index = 0
        return self

    def __next__(self):
        # Keep filling a buffer until we have enough tokens for a new sample.
        # Mask the loss for each token following the BoS token using -100 index.
        if self.current_index >= self.real_length:
            raise StopIteration
        
        while len(self.token_buffer) < self.sequence_length:
            
            self.token_buffer.append(self.bos_token_id)
            sample_str = str(self.parquet_ds["text"][self.current_index % self.real_length])
            tokens = self.tokenizer.encode_plus(sample_str, max_length=self.sequence_length, padding=False, truncation=True)['input_ids']
            self.token_buffer.extend(tokens)
            self.current_index += 1
        
        this_sequence = self.token_buffer[:self.sequence_length]
        self.token_buffer = self.token_buffer[self.sequence_length:]

        inputs = torch.LongTensor(this_sequence[:-1]).clone()
        labels = torch.LongTensor(this_sequence[1:])

        # For padding tokens, mask the loss
        mask = labels == self.bos_token_id
        mask = torch.roll(mask, shifts=1, dims=0)
        labels[mask] = -100    

        return inputs, labels


@dataclass
class CollatorForCLM:
    sequence_length: int
    pad_token_id: int
    
    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.LongTensor([examples[i]["input_ids"] for i in range(len(examples))])  # (b, s+1)
        inputs = input_ids[:, :-1].clone()
        labels = input_ids[:, 1:]
        
        # For padding tokens, mask the loss
        labels[labels == self.pad_token_id] = -100
        
        assert inputs.shape[1] == labels.shape[1] == self.sequence_length
        assert inputs.shape == labels.shape
        
        return inputs, labels


class ParquetDataset(Dataset):
    def __init__(self, parquet_file: str, tokenizer, sequence_length: int, training_samples: int):
        self.parquet_ds = pq.read_table(parquet_file, memory_map=True)
        self.real_length = len(self.parquet_ds)
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.training_samples = training_samples

    def __len__(self):
        return self.training_samples

    def __getitem__(self, idx: int):
        sample_str = str(self.parquet_ds["text"][idx % self.real_length])
        return self.tokenizer.encode_plus(sample_str, max_length=self.sequence_length + 1, padding='max_length', truncation=True, padding_side="right")


if __name__ == "__main__":
    # Step 1: Create a dummy Parquet file with sample text data
    # data = pa.Table.from_pydict({"text": ["Hello world!", "This is a test.", "PyArrow is great!", "Transformers are powerful."]})
    num_samples = 1000
    texts = [f"Sample text number {i}. This is a randomly generated sentence." for i in range(num_samples)]
    data = pa.Table.from_pydict({"text": texts})
    parquet_file = "dummy.parquet"
    pq.write_table(data, parquet_file)
    # Step 2: Define a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
    
    # Step 4: Test the IterableParquetDataset
    dataset = IterableParquetDataset(parquet_file, tokenizer, sequence_length=10)
    # dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    # Fetch and print the first batch
    for sample in dataset:
        inputs, labels = sample
        print("Inputs:", inputs)
        print("Labels:", labels)
        decoded  = tokenizer.decode(inputs)
        print("Decoded:", decoded)
        print(inputs.shape, labels.shape)
        break  # Only print one batch for verification