from os import walk
from os.path import join
import pandas as pd

EXAMPLE_FILE = 'practice_email.txt'

SPAM_1_PATH = 'spam_1'
SPAM_2_PATH = 'spam_2'
EASY_NONSPAM_1_PATH = 'easy_ham_1'
EASY_NONSPAM_2_PATH = 'easy_ham_2'

VOCAB_SIZE = 2500



def email_body_generator(path):
    
    for root, dirnames, filenames in walk(path):
        for file_name in filenames:
            
            filepath = join(root, file_name)
            
            stream = open(filepath, encoding='latin-1')

            is_body = False
            lines = []

            for line in stream:
                if is_body:
                    lines.append(line)
                elif line == '\n':
                    is_body = True

            stream.close()

            email_body = '\n'.join(lines)
            
            yield file_name, email_body
def df_from_directory(path, classification):
    rows = []
    row_names = []
    
    for file_name, email_body in email_body_generator(path):
        rows.append({'MESSAGE': email_body, 'CATEGORY': classification})
        row_names.append(file_name)
        
    return pd.DataFrame(rows, index=row_names)




SPAM_CAT = 1
HAM_CAT = 0
spam_emails = df_from_directory(SPAM_1_PATH, 1)
spam_emails = spam_emails.append(df_from_directory(SPAM_2_PATH, 1))


ham_emails = df_from_directory(EASY_NONSPAM_1_PATH, HAM_CAT)
ham_emails = ham_emails.append(df_from_directory(EASY_NONSPAM_2_PATH, HAM_CAT))
ham_emails.shape

data = pd.concat([spam_emails, ham_emails])


data['MESSAGE'].isnull().values.all()

(data.MESSAGE.str.len() == 0).sum()

#Locate empty emails
data[data.MESSAGE.str.len() == 0].index

#data[4608:4611]

data.drop(['cmds'], inplace=True)


#Add Document IDs to Track Emails in Dataset
document_ids = range(0, len(data.index))
data['DOC_ID'] = document_ids

data['FILE_NAME'] = data.index
data.set_index('DOC_ID', inplace=True)


data.CATEGORY.value_counts()


amount_of_spam = data.CATEGORY.value_counts()[1]

amount_of_ham = data.CATEGORY.value_counts()[0]



import matplotlib.pyplot as plt

category_names = ['Spam Mail', 'Ham Mail']
sizes = [amount_of_spam, amount_of_ham]
custom_colours = ['#ff7675', '#74b9ff']

plt.figure(figsize=(2, 2), dpi=227)
plt.pie(sizes, labels=category_names, textprops={'fontsize': 6}, startangle=90, autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.1])

#Natural Language Processing
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
'''
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('gutenberg')
nltk.download('shakespeare')
'''




#Removing HTML tags from Emails
from bs4 import BeautifulSoup

def clean_msg_no_html(message, stemmer=PorterStemmer(), 
                 stop_words=set(stopwords.words('english'))):
    
    # Remove HTML tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    # Converts to Lower Case and splits up the words
    words = word_tokenize(cleaned_text.lower())
    
    filtered_words = []
    
    for word in words:
        # Removes the stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(stemmer.stem(word))
#             filtered_words.append(word) 
    
    return filtered_words
clean_msg_no_html(data['MESSAGE'].all())




# use apply() on all the messages in the dataframe
#nested_list = data.MESSAGE.apply(clean_msg_no_html)

stemmed_nested_list = data.MESSAGE.apply(clean_msg_no_html)
flat_stemmed_list = [item for sublist in stemmed_nested_list for item in sublist]
unique_words = pd.Series(flat_stemmed_list).value_counts()
print('Nr of unique words', unique_words.shape[0])
unique_words.head()

VOCAB_SIZE = 2500
frequent_words = unique_words[0:VOCAB_SIZE]
print('Most common words: \n', frequent_words[:10])

#Create Vocabulary DataFrame with a WORD_ID
word_ids = list(range(0, VOCAB_SIZE))
vocab = pd.DataFrame({'VOCAB_WORD': frequent_words.index.values}, index=word_ids)
vocab.index.name = 'WORD_ID'
vocab.head()
vocab.to_csv('I:/YesInfotechBatch-2/MAchineLearning/ML-Day-5-NaiveBayes/WORDS.csv', index_label=vocab.index.name, header=vocab.VOCAB_WORD.name)



from wordcloud import WordCloud
from PIL import Image
plt.figure(figsize=(10,20))
word_cloud = WordCloud().generate(str(frequent_words[:25]))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis('off')



data.tail()

data.shape

data.sort_index(inplace=True)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
all_features = vectorizer.fit_transform(data.MESSAGE)

all_features.shape


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_features, data.CATEGORY, 
                                                   test_size=0.3, random_state=88)


from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()


classifier.fit(X_train, y_train)

nr_correct = (y_test == classifier.predict(X_test)).sum()

print(f'{nr_correct} documents classfied correctly')

nr_incorrect = y_test.size - nr_correct

print(f'Number of documents incorrectly classified is {nr_incorrect}')

fraction_wrong = nr_incorrect / (nr_correct + nr_incorrect)
print(f'The (testing) accuracy of the model is {1-fraction_wrong:.2%}')


classifier.score(X_test, y_test)



from sklearn.metrics import recall_score, precision_score, f1_score
recall_score(y_test, classifier.predict(X_test))
precision_score(y_test, classifier.predict(X_test))
f1_score(y_test, classifier.predict(X_test))
