from hazm import Normalizer, Lemmatizer, POSTagger #import the hazm library
from hazm import word_tokenize
import pandas as pd
import csv
import math
df=pd.read_csv('books_train.csv')#read the data
normalizer = Normalizer()
lemmatizer = Lemmatizer()
posTagger = POSTagger(model = 'pos_tagger.model',  universal_tag = True)

def replace_bad_characters(s):
    bad_characters = [';', ':', '!', "*", ")", "(", "؛", "\n", ",", "-", ".", "،", "»", "«", "…","[", "]", "\'" ,"?" ]
    persian_zero_unicode = ord('۰')
    bad_characters += [chr(code) for code in range(persian_zero_unicode, persian_zero_unicode + 10)]
    for bad_character in bad_characters:
        s = s.replace(bad_character, ' ') 
    return s
def remove_extra_words(words):#a functio that remove all sentence components except nouns and verbs
    word_tags=posTagger.tag(tokens = words)
    result=[]
    for word_tag in word_tags:
        if word_tag[1]=='NOUN' or word_tag[1]=='VERB':
            result.append(word_tag[0])
    return result

def clean_csv_data(df, use_lemmatizer=True, remove_frequent_words=True):
    t=[]
    for i in range(len(df['title'])):
        title = normalizer.normalize(replace_bad_characters(df["title"][i]))
        description = normalizer.normalize(replace_bad_characters(df["description"][i]))
        t.append( [title + " " + description, df["categories"][i]] )

    result=[]
    for text, category in t:
        words=word_tokenize(text)
        clean_words=[]
        if remove_frequent_words==True:
            words=remove_extra_words(words) 


        for word in words:
            if use_lemmatizer==True:
                word=lemmatizer.lemmatize(word)

            clean_words.append(word)

        result.append( [clean_words , category] )
    return result

train_data=clean_csv_data(df, True, True)

all_words=set()
for i in range(len(train_data)):
    item=train_data[i]
    item_words=item[0]
    for j in range(len(item_words)):
        item_word=item_words[j]
        all_words.add(item_word)
    
all_words = list(all_words)

all_categories = list(set(df["categories"]))

def replace_bad_characters(s):
    bad_characters = [';', ':', '!', "*", ")", "(", "؛", "\n", ",", "-", ".", "،", "»", "«", "…","[", "]", "\'" ,"?" ]
    persian_zero_unicode = ord('۰')
    bad_characters += [chr(code) for code in range(persian_zero_unicode, persian_zero_unicode + 10)]
    for bad_character in bad_characters:
        s = s.replace(bad_character, ' ') 
    return s
def remove_extra_words(words):#a functio that remove all sentence components except nouns and verbs
    word_tags=posTagger.tag(tokens = words)
    result=[]
    for word_tag in word_tags:
        if word_tag[1]=='NOUN' or word_tag[1]=='VERB':
            result.append(word_tag[0])
    return result

def clean_csv_data(df, use_lemmatizer=True, remove_frequent_words=True):
    t=[]
    for i in range(len(df['title'])):
        title = normalizer.normalize(replace_bad_characters(df["title"][i]))
        description = normalizer.normalize(replace_bad_characters(df["description"][i]))
        t.append( [title + " " + description, df["categories"][i]] )

    result=[]
    for text, category in t:
        words=word_tokenize(text)
        clean_words=[]
        if remove_frequent_words==True:
            words=remove_extra_words(words) 


        for word in words:
            if use_lemmatizer==True:
                word=lemmatizer.lemmatize(word)

            clean_words.append(word)

        result.append( [clean_words , category] )
    return result

train_data=clean_csv_data(df, True, True)

all_words=set()
for i in range(len(train_data)):
    item=train_data[i]
    item_words=item[0]
    for j in range(len(item_words)):
        item_word=item_words[j]
        all_words.add(item_word)
    
all_words = list(all_words)
all_categories = list(set(df["categories"]))

def find_index(item, array):#a function that fine the index of a item in array
    for i in range(len(array)):
        if item == array[i]:
            return i
    return -1

def make_2d_array(n, m):#a function that make a n*m matrix with zero values
    result=[]
    for i in range(n):
        a=[]
        for j in range(m):
            a.append(0)
        result.append(a)
    return result
#making a dictionary that cotain word with its index
word_indexes = {}
for i in range(len(all_words)):
    word = all_words[i]
    word_indexes[word] = i
#making the bow matrix
bow=make_2d_array(len(all_categories), len(all_words))
for book in train_data:
    category=book[1]
    category_index=find_index(category, all_categories)
    for word in book[0]:
        word_index=word_indexes[word]
        bow[category_index][word_index]+=1

probibility_bow = make_2d_array(len(all_categories) , len(all_words) + 1)
for i in range(len(probibility_bow)) :
    sum_row = 0
    
    #calculate sum_row
    for j in range(len(all_words)):
        sum_row += bow[i][j]

    for j in range(len(all_words)):
        probibility_bow[i][j]=math.log((bow[i][j]+1)/(len(all_words)+1+sum_row)) # Additive Smoothing algorithm
    probibility_bow[i].append(math.log(1/(len(all_words)+1+sum_row))) # Additive Smoothing for words that does not exist


# probibility_bow = make_2d_array(len(all_categories) , len(all_words) + 1)
# for i in range(len(probibility_bow)) :
#     sum_row = 0
    
#     #calculate sum_row
#     for j in range(len(all_words)):
#         sum_row += bow[i][j]

#     for j in range(len(all_words)):
#         if bow[i][j] != 0:
#             probibility_bow[i][j]=math.log((bow[i][j])/(sum_row)) 

def category_probability(category):
    tekrar = 0
    for book in train_data:
        if book[1] == category:
            tekrar += 1
    return tekrar / len(train_data)


def word_probability_in_category(category, word):#a functio that return the probability of word in given category from probability_bow
    category_index= all_categories.index(category)#find category index in all_category array
    if word in word_indexes:
        word_index=word_indexes[word]#find word index in all_words dictionary
        return probibility_bow[category_index][word_index]
    else:
        return probibility_bow[category_index][-1]#when the word does not exist in train file
        
def find_book_category_probability(book , category):#find the probability of each category in each book of test file
    probability=0
    for word in book[0]:
        probability+=word_probability_in_category(category , word )
    return probability+math.log(category_probability(category))

def find_best_category(book):#a function that find the category with highest probability
    best_category = ''
    best_probability=None
    for i in range(len(all_categories)):
        category_probability=find_book_category_probability(book , all_categories[i])
        if  best_probability==None or category_probability > best_probability:
            best_probability = category_probability
            best_category=all_categories[i]
    return best_category

def run_test(test_data):#this function show the probability of correctness
    true_valu=0
    for book in test_data:
        best_category=find_best_category(book)
        if best_category == book[1]:
            true_valu +=1
            
    print("the probability of correctness is:",true_valu/(len(test_data)))
    
def show_guesses(test_data):
    for book in test_data:
        print(find_best_category(book))
        print(book[1])
        print("______________________________")



df_test=pd.read_csv('books_test.csv') 
    
test_data=clean_csv_data(df_test, True , True)
# show_guesses(test_data)
run_test(test_data)