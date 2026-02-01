from models.analyzer import get_analyzer

analyzer = get_analyzer()

test_text = "The radical left is destroying our country with their socialist agenda! We need to fight back against the woke mob."
neutral_text = "The weather today is sunny and mild. I think I'll go for a walk."

print("\n--- Analytics Test ---")
print(f"Text: {test_text}\n")

# Sentiment
sentiment = analyzer.analyze_sentiment(test_text)
print("Sentiment:")
print(f"  Positive: {sentiment['positive_pct']}%")
print(f"  Negative: {sentiment['negative_pct']}%")
print(f"  Score: {sentiment['overall_score']}")

# Bias
bias = analyzer.analyze_bias(test_text)
print("\nBias:")
print(f"  Score: {bias['score']}")
print(f"  Level: {bias['level']}")
print(f"  Flags: {bias['flags_count']}")
for flag in bias['flags']:
    print(f"   - Found '{flag['phrase']}' ({flag['category']})")

print("\n--- Neutral Text Test ---")
sentiment_n = analyzer.analyze_sentiment(neutral_text)
print(f"Sentiment Score: {sentiment_n['overall_score']}")
