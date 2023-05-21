from nltk.sentiment import SentimentIntensityAnalyzer
import text2emotion as te

pozytywna = "It was very close to the center and the apartment was beautiful. Great to have a quick dinner or breakfast in the apartment. Very clean and comfy."
negatywna = "The apartment is very small, old and not maintained as expected (nothing like the pictures in Booking). It does not fit for a family of 5. The bath is disaster. No place to sit and eat. Beds are very uncomfortable."

sia = SentimentIntensityAnalyzer()

pozytywna_info = sia.polarity_scores(pozytywna)
print("Pozytywna opinia:", pozytywna_info)
positive_emotions = te.get_emotion(pozytywna)
print("Positive Opinion Emotions:", positive_emotions)

negatywna_info = sia.polarity_scores(negatywna)
print("Pozytywna opinia:", negatywna_info)
negative_emotions = te.get_emotion(negatywna)
print("Negative Opinion Emotions:", negative_emotions, "\n\n")

pozytywna = "It was very close to the center and the apartment was beautiful. Great to have a quick dinner or breakfast in the apartment. Very clean and comfy. Beautiful place to be. Nice landscape. Good prices. Recommended"
negatywna = "The apartment is very small, old and not maintained as expected (nothing like the pictures in Booking). WHAT THE FUCK, I'm ANGRY It does not fit for a family of 5. The bath is disaster. No place to sit and eat. Beds are very uncomfortable."

pozytywna_info = sia.polarity_scores(pozytywna)
print("Pozytywna opinia:", pozytywna_info)
positive_emotions = te.get_emotion(pozytywna)
print("Positive Opinion Emotions:", positive_emotions)

negatywna_info = sia.polarity_scores(negatywna)
print("Pozytywna opinia:", negatywna_info)
negative_emotions = te.get_emotion(negatywna)
print("Negative Opinion Emotions:", negative_emotions, "\n\n")
