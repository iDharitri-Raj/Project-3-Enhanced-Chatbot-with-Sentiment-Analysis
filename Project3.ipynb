import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')

[nltk_data] Downloading package vader_lexicon to
[nltk_data]     C:\Users\idhar\AppData\Roaming\nltk_data...
[nltk_data] Downloading package punkt to
[nltk_data]     C:\Users\idhar\AppData\Roaming\nltk_data...
[nltk_data]   Package punkt is already up-to-date!
[nltk_data] Downloading package wordnet to
[nltk_data]     C:\Users\idhar\AppData\Roaming\nltk_data...
[nltk_data]   Package wordnet is already up-to-date!
True

warnings.filterwarnings('ignore')

f = open('chatbot.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()

# Tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
​
    # TF-IDF Vectorization
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
​
    # Cosine similarity
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
​
    if req_tfidf == 0:
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response

def sentiment_analysis(user_response):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(user_response)
    if score['compound'] >= 0.05:
        return "positive"
    elif -0.05 < score['compound'] < 0.05:
        return "neutral"
    else:
        return "negative"

flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while flag:
    user_response = input()
    user_response = user_response.lower()
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            if greeting(user_response) is not None:
                print("ROBO: " + greeting(user_response))
            else:
                print("ROBO: ", end="")
                sentiment = sentiment_analysis(user_response)
                if sentiment == "positive":
                    print("That sounds great! ", end="")
                elif sentiment == "negative":
                    print("I'm sorry to hear that. ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! Take care..")

ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!
hello
ROBO: hi there
What is a chatbot?
ROBO: design
the chatbot design is the process that defines the interaction between the user and the chatbot.the chatbot designer will define the chatbot personality, the questions that will be asked to the users, and the overall interaction.it can be viewed as a subset of the conversational design.
That sounds interesting!
ROBO: That sounds great! I am sorry! I don't understand you
Can chatbots understand emotions?
ROBO: in order to properly understand a user input in a free text form, a natural language processing engine can be used.the second task may involve different approaches depending on the type of the response that the chatbot will generate.
Okay, thanks for the information.
ROBO: That sounds great! most people prefer to engage with programs that are human-like, and this gives chatbot-style techniques a potentially useful role in interactive systems that need to elicit information from users, as long as that information is relatively straightforward and falls into predictable categories.
bye
ROBO: Bye! Take care..
