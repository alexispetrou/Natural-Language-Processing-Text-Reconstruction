import language_tool_python
 
def correct_text(text: str) -> str:
    """
    Uses LanguageTool to find and apply suggested corrections.
    """
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    # Apply all non-overlapping corrections:
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected
 
if __name__ == "__main__":
    raw1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""
 
    print("=== Original ===")
    print(raw1)
    print("\n=== Corrected ===")
    print(correct_text(raw1))