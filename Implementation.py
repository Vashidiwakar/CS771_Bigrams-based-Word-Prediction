from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
import numpy as np

# You may define any new functions, variables, classes here
# For example, classes to create the Tree, Nodes etc
vectorizer = None
all_bigrams = None

################################
# Non Editable Region Starting #
################################
def my_fit(words):
################################
# Non Editable Region Ending  #
################################
    global vectorizer, all_bigrams
    
    def get_bigrams(word, lim=5):
        
        bigrams = sorted(set(''.join(x) for x in zip(word, word[1:])))
        return tuple(bigrams[:lim])  
    
    
    bigrams = [get_bigrams(word) for word in words]
    
    
    all_bigrams = sorted(set(bigram for sublist in bigrams for bigram in sublist))
    
    
    vectorizer = DictVectorizer(sparse=False)
    
    
    X = vectorizer.fit_transform([{bigram: (bigram in word_bigrams) for bigram in all_bigrams} for word_bigrams in bigrams])
    
    
    y = np.array(words)
    
    
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)
    
    return model
################################
# Non Editable Region Starting #
################################
def my_predict(model, bigrams):
################################
# Non Editable Region Ending #
################################
    global vectorizer, all_bigrams
    
    def create_feature_vector(bigrams, all_bigrams):
        
        return {bigram: (bigram in bigrams) for bigram in all_bigrams}
    
    
    X_test = vectorizer.transform([create_feature_vector(bigrams, all_bigrams)])
    
    
    prediction_probs = model.predict_proba(X_test)
    
    
    top_index = np.argmax(prediction_probs[0])  
    
    
    top_guess = model.classes_[top_index]
    
    return [top_guess]  
