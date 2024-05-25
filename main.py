from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import Accelerator

# Initialize the Accelerator
accelerator = Accelerator()

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("AIFS/Prometh-222B")
model = AutoModelForCausalLM.from_pretrained("AIFS/Prometh-222B")

# Tell Accelerate to prepare the model
model, tokenizer = accelerator.prepare(model, tokenizer)

# Prepare your text data
prompt = "The future of AI in healthcare is"
input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Use the model and tokenizer to generate text
# Note: We directly use model and tokenizer without pipeline for better control and efficiency in multi-GPU setups.
generated_ids = accelerator.unwrap_model(model).generate(**input_ids, max_length=50, num_return_sequences=3)

# Decode generated ids to text
for g in generated_ids:
    print(tokenizer.decode(g, skip_special_tokens=True))
