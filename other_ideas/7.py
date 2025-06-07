from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Φόρτωση μοντέλου και tokenizer
model_name = "Vasmi/t5_parapgrase_Paws"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Αρχικό κείμενο (χωρισμένο σε προτάσεις για καλύτερα αποτελέσματα)
sentences = [
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor?",
    "Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation.",
    "We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think.",
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again.",
    "Because I didn’t see that part final yet, or maybe I missed, I apologize if so.",
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
]

# Συνάρτηση παραφράσης
def paraphrase(text):
    input_text = "paraphrase: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(
        input_ids,
        max_length=256,
        num_beams=5,
        num_return_sequences=1,
        temperature=1.5,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Εφαρμογή παραφράσης σε κάθε πρόταση
for i, sentence in enumerate(sentences):
    new_sentence = paraphrase(sentence)
    print(f"📝 Original: {sentence}")
    print(f"🔁 Paraphrased: {new_sentence}\n")
