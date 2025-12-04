# src/replies.py

REPLY_TEMPLATES = {
    "praise": "Thank you so much for your kind words and appreciation!",
    "support": "Thanks for the encouragement. We really appreciate your support.",
    "constructive_criticism": "Thank you for the honest feedback. We’ll review this and work on improving.",
    "hate_abuse": "Your feedback has been noted. Let’s try to keep the conversation respectful.",
    "threat": "We take your concern seriously and will review this situation carefully.",
    "emotional": "Thank you for sharing how this made you feel. We’re glad it resonated with you.",
    "spam_irrelevant": "This comment seems unrelated to the post and may not be addressed.",
    "question_suggestion": "Thanks for the suggestion! We’ll consider creating content on this topic."
}

def generate_reply(label: str) -> str:
    """
    Return a reply template for a given label.
    """
    return REPLY_TEMPLATES.get(label, "Thank you for your comment.")
