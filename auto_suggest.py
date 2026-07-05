import numpy as np
import pandas as pd

data = pd.read_json("../data/mrbilit_search.json")
data = data[(data["ServiceType"] == "bus") | (data["ServiceType"] == "taxi")]
print(data.shape)
data.head()

cities = pd.read_csv("../data/iran_cities.csv" , usecols=["City EN","City FA"])
enfa = pd.read_csv("../data/typo_char.csv")
test = pd.read_json("../data/test_data.json")


s = data.apply(lambda x: pd.Series(x['TypedStrings']), axis=1).stack().reset_index(level=1, drop=True)
s.name = 'TypedStrings'
train = data.drop('TypedStrings', axis=1).join(s)
train['TypedStrings'] = pd.Series(train['TypedStrings'])
train.reset_index(inplace=True , drop=True)


train["#char"] = train["TypedStrings"].str.len()


def detect_word_language(word):
    if not word:
        return 'unknown'
    
    en_count = 0
    fa_count = 0
    
    for char in word:
        if 'a' <= char.lower() <= 'z':
            en_count += 1
        elif '\u0600' <= char <= '\u06FF':
            fa_count += 1
    
    if en_count > fa_count:
        return 'en'
    elif fa_count > en_count:
        return 'fa'
    else:
        return 'mixed'
    


en_to_fa = dict(zip(enfa['EN'], enfa['FA']))

def convert_en_to_fa(word):
    if not word or word == '-':
        return word
    
    converted_chars = []
    for char in word:
        converted_chars.append(en_to_fa.get(char, char))
    
    return ''.join(converted_chars)

char_df = pd.DataFrame(columns=[str(i) for i in range(12)])  # ستون‌های 0 تا 11

results = []
for idx, row in train.iterrows():
    clm = row["#char"]
    typed_str = row["TypedStrings"]

    new_row = pd.Series("-", index=char_df.columns)
    
    typed_str = convert_en_to_fa(typed_str) if detect_word_language(typed_str) == 'en' else typed_str

    if clm <= 10:
        if clm < len(char_df.columns):
            new_row[str(clm)] = typed_str
    else:
        new_row["11"] = typed_str
    
    results.append(new_row)

char_df = pd.DataFrame(results, index=train.index)
char_df = char_df.fillna("-")
char_df.to_csv("../data/char_df.csv", index=False)


from sklearn.preprocessing import OrdinalEncoder
ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
df = ord_encoder.fit_transform(char_df)


from sklearn.preprocessing import OrdinalEncoder

ord_encoder_typed = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
ord_encoder_accepted = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)

train["TypedStrings"] = ord_encoder_typed.fit_transform(train["TypedStrings"].to_numpy().reshape(-1,1))
train["AcceptString"] = ord_encoder_accepted.fit_transform(train["AcceptString"].to_numpy().reshape(-1,1))


from sklearn.model_selection import train_test_split
x_train, x_validation, y_train, y_validation = train_test_split(train[["TypedStrings","#char"]],train["AcceptString"].to_numpy().reshape(-1,1))

from sklearn.model_selection import train_test_split
x_train_df, x_validation_df, y_train_df, y_validation_df = train_test_split(df,train["AcceptString"].to_numpy().reshape(-1,1))

from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier

tree = DecisionTreeClassifier()
tree.fit(x_train_df,y_train_df)

predicted_validatin_df = tree.predict(x_validation_df)

from sklearn.metrics import f1_score ,accuracy_score
score_df = f1_score(y_validation_df,predicted_validatin_df,average='weighted')
print("f1 score:", score_df)
print("accuracy_score: " , accuracy_score(y_validation_df,predicted_validatin_df))

top_cities = pd.DataFrame(ord_encoder_accepted.inverse_transform(train["AcceptString"].to_numpy().reshape(-1,1)))[0].value_counts().head(5).index.to_list()


def suggestion_creator(city):
    import Levenshtein
    lev_distance = Levenshtein
    name_ls = []
    distance_ls = []
    for name in suggestion_names: # type: ignore

        if (lev_distance.distance(name,city) < 3):
            name_ls.append(name)
            distance_ls.append(lev_distance.distance(name,city))
            # print(name ," ",  city ," ", lev_distance.distance(name,city))
    sugg_frame = pd.DataFrame({"name": name_ls,
                    "distance": distance_ls})
    return sugg_frame.set_index("distance").sort_index()["name"].to_list()


mask = test["Typed"].apply(detect_word_language) == 'en'
test.loc[mask, "Typed"] = test.loc[mask, "Typed"].apply(convert_en_to_fa)
sample = test["Typed"].to_list()

submission = pd.DataFrame(columns=['Suggestion0','Suggestion1','Suggestion2','Suggestion3','Suggestion4'])

for i in range(len(sample)):
    len(sample)
    suggestion = pd.DataFrame(columns=["0","1","2","3","4","5","6","7","8","9","10","11"])
    if len(sample[i]) >10:
        suggestion.loc[0,"11"] = sample[i]
    else:
        suggestion.loc[0,str(len(sample[i]))] = sample[i]
    suggestion = suggestion.fillna("-")
    X = ord_encoder.transform(suggestion)
    city = ord_encoder_accepted.inverse_transform((tree.predict(X)).reshape(-1,1))[0][0]

    row_result = suggestion_creator(city)

    ls_size = 5-len(row_result)     
    for city in top_cities:
        if ls_size > 0:
            if city not in row_result:
                row_result.append(city)
                ls_size -= 1
                
    if ls_size < 0:
        for j in range(np.abs(ls_size)):
            row_result.pop()
        
    print(i , row_result)
    submission.loc[i] = row_result

import zipfile
import joblib

def compress(file_names):
    print("File Paths:")
    print(file_names)
    compression = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile("result.zip", mode="w") as zf:
        for file_name in file_names:
            zf.write('./' + file_name, file_name, compress_type=compression)

submission.to_csv('submission.csv', index=False)
file_names = ['auto_suggest.ipynb', 'submission.csv']
compress(file_names)
