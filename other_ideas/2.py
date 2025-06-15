from transformers import pipeline
 
# You can swap in "t5-base" or any other text2text-generation checkpoint
paraphraser = pipeline(
    "text2text-generation", 
    model="Vamsi/T5_Paraphrase_Paws", 
    tokenizer="Vamsi/T5_Paraphrase_Paws",
    device=0  # set to -1 for CPU
)
 
def paraphrase(text: str, max_length: int = 256) -> str:
    """
    Generates a single paraphrase of the input text.
    """
    prompt = f"paraphrase: {text} </s>"
    outputs = paraphraser(prompt, max_length=max_length, num_beams=5, num_return_sequences=1)
    return outputs[0]['generated_text']
 
if __name__ == "__main__":
    raw2 = """During our final discuss, I told him about the new submission — the one we were waiting since
last autumn, but the updates was confusing as it not included the full feedback from reviewer or
maybe editor?
Anyway, I believe the team, although bit delay and less communication at recent days, they really
tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance
and efforts until the Springer link came finally last week, I think.
Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before
he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so.
Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future
targets"""
    print("=== Original ===")
    print(raw2)
    print("\n=== Paraphrased ===")
    print(paraphrase(raw2))