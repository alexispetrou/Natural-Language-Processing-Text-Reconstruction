from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")

paraphraser = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

new_text1 = paraphraser(f"paraphrase: {str(text1)}", max_length=512, do_sample=True)
print(str(new_text1[0]["generated_text"]))

