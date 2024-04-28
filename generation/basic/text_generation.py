from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", load_in_4bit=True)

#text = input("Text: ")

tokenizer = AutoTokenizer.from_pretrained(MODEL)#padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(["1, 2, 3", "Portugal is"], return_tensors="pt", padding=True).to("cuda")
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, do_sample=True)
generated_sentence = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_ids)
print(generated_sentence)
