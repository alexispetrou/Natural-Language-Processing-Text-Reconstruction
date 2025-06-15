import fitz  # PyMuPDF

# Άνοιγμα του PDF
doc = fitz.open(r"BONUS\data_minning\ast_code.pdf")


all_text = ""
skip_articles = {"1113", "1114"}

for page in doc:
    text = page.get_text()
    all_text += text + "\n"

doc.close()

# Διαχωρισμός σε άρθρα
import re

articles = re.split(r'\bΑρθρο: (\d+)\b', all_text)
formatted_articles = []

# Παράλειψη άρθρων 1113 και 1114
for i in range(1, len(articles), 2):
    article_number = articles[i]
    article_body = articles[i + 1]

    if article_number not in skip_articles:
        formatted_articles.append(f"Άρθρο {article_number}\n{article_body.strip()}")

# Αποθήκευση σε αρχείο
with open("data.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(formatted_articles))

print("Το αρχείο data.txt δημιουργήθηκε χωρίς τα άρθρα 1113 και 1114.")
