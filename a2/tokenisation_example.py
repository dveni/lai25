from transformers import AutoTokenizer
from dataset import IterableParquetDataset, ParquetDataset, CollatorForCLM
from torch.utils.data import DataLoader

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/Mistral-Nemo-Base-2407-bnb-4bit")
# Example of converting text to tokens and back
text = "Hello, I am a language model."
# Convert text to tokens (IDs)
tokens = tokenizer.encode(text)
print(f"Tokens: {tokens}")
# Convert tokens back to text
decoded_text = tokenizer.decode(tokens)
print(f"Decoded text: {decoded_text}")


# Create dataset instance
dataset_path = "/capstor/store/cscs/ethz/large-sc/datasets/train_data.parquet"
sequence_length = 4096
# Create dataset (only requesting 1 sample)
dataset = ParquetDataset(
parquet_file=dataset_path,
tokenizer=tokenizer,
sequence_length=sequence_length,
training_samples=128
)
# Get the first sample
sample = dataset[0]

print(sample.keys())

decoded = tokenizer.decode(sample['input_ids'][:200])
print(decoded)


# Create collator
collator = CollatorForCLM(sequence_length=sequence_length, pad_token_id=tokenizer.pad_token_id)
batch_size = 32
# Create dataloader
dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
# Get a batch using a for loop
for batch_inputs, batch_labels in dataloader:
    # Print shapes
    print(f"Input shape: {batch_inputs.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    # Count ignored tokens in the loss calculation
    ignored_count = (batch_labels == -100).sum().item()
    total_label_tokens = batch_labels.numel()
    print(f"Ignored tokens in loss: {ignored_count} out of {total_label_tokens} ({ignored_count/total_label_tokens*100:.2f}%)")
    # Only process the first batch
    break
