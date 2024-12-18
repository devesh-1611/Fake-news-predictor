import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv(r"C:\Users\DELL\Desktop\Projects\NLP_DL\train.csv")
df.head()
df.shape
df.isnull().sum()
df=df.fillna(" ")

df.isnull().sum()
df["content"]=df["author"]+" "+df["title"]
df

df["content"][20798]





import re

def stemming(content):
    # Ensure content is a string before processing
    if not isinstance(content, str):
        return content  # Return unchanged if not a string

    stemmed_content = re.sub("[^a-zA-Z]", " ", content)  # Pass the string as the third argument
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    # Apply stemming logic here if needed
    return " ".join(stemmed_content)

# Apply the function to the 'content' column
df["content"] = df["content"].apply(stemming)



x=df['content'].values
y=df['label'].values

vector=TfidfVectorizer()
vector.fit(x)
x=vector.transform(x)

print(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


model=LogisticRegression()
model.fit(x_train,y_train)


x_train.shape

x_test.shape

train_y_pred=model.predict(x_train)
print("train accuracy:",accuracy_score(train_y_pred,y_train))

test_y_pred=model.predict(x_test)
print("test accuracy:",accuracy_score(test_y_pred,y_test))

# Ensure the proper index or data is passed to x_test
input_data = x_test[0]  # Replace '0' with the specific index you want to test

# Ensure the data is in the correct shape for the model
input_data = input_data.reshape(1, -1)  # Reshape to match model input requirements

# Perform prediction
prediction = model.predict(input_data)

# Interpret and print the result
if prediction[0] == 1:
    print("Fake news")
else:
    print("Real news")

    



#Frontend
import streamlit as st
st.title("Fake News Predictor")
input_text=st.text_input("Enter news article")
