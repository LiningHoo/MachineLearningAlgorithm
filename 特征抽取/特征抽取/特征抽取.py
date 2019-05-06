from sklearn.feature_extraction import DictVectorizer
import tensorboard

def dict_vectorizer(x,sparse):
    if sparse == True:
        dict_vector = DictVectorizer()
        data = dict_vector.fit_transform(x)
    else:
        dict_vector = DictVectorizer(sparse = False)
        data = dict_vector.fit_transform(x)
    return data

if __name__ == '__main__':
    dict = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'深圳','temperature':30}]
    data = dict_vectorizer(dict,True)
    print(data)
    print('\n')
    data = dict_vectorizer(dict,False)
    print(data)
    
