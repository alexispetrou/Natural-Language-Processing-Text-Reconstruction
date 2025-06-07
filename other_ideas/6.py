from transformers import pipeline
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

text = """During our final discuss, I told him about the new submission — the one we were waiting since 
last autumn, but the updates was confusing as it not included the full feedback from reviewer or 
maybe editor?
 Anyway, I believe the team, although bit delay and less communication at recent days, they really 
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance 
and efforts until the Springer link came finally last week, I think.
 Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before 
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
 Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future 
targets"""

sentences = sent_tokenize(text)


models = {
    "Grammar Correction (vennify)": pipeline("text2text-generation", model="vennify/t5-base-grammar-correction"),
    "Grammar Correction (prithivida)": pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1"),
    "Text Simplification (distilbart)": pipeline("text2text-generation", model="sshleifer/distilbart-cnn-12-6"),
}


for model_name, model_pipeline in models.items():
    print(f"\n--- {model_name} ---\n")
    for sentence in sentences:
        result = model_pipeline(sentence, max_length=256, do_sample=False)[0]["generated_text"]
        print(f"Original: {sentence}")
        print(f"→ Transformed: {result}\n")
