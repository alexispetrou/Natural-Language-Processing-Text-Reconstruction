import spacy
from transformers import pipeline
import language_tool_python
 
 
nlp = spacy.load("en_core_web_sm")
tool = language_tool_python.LanguageTool('en-US')
paraphraser = pipeline("text2text-generation", model="t5-base", tokenizer="t5-base", device=-1)
 
def reconstruct(text: str) -> str:
    doc = nlp(text)
    reconstructed_sents = []
    for sent in doc.sents:
        corr = language_tool_python.utils.correct(sent.text, tool.check(sent.text))
        prompt = f"paraphrase: {corr} </s>"
        out = paraphraser(prompt, max_length=128, num_beams=4, num_return_sequences=1)[0]['generated_text']
        reconstructed_sents.append(out)
    return " ".join(reconstructed_sents)
 
raw1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""
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

combined = raw1 + "\n\n" + raw2
print(reconstruct(combined))