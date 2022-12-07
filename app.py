


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
import streamlit as st
import altair as alt
import plotly.graph_objects as go
README.md


s = pd.read_csv('social_media_usage.csv')


# #### Q2 Define Function clean_sm

# In[10]:


def clean_sm(x):
    x = np.where(x == 1, 1,0)
    return x


ss = pd.DataFrame({
"sm_li":clean_sm(s['web1h']),
"income": np.where(s['income'] >9,np.nan,s['income']),
"education": np.where(s['educ2']>8,np.nan,s['educ2']),
"parent": np.where(s['par']==1,1,0),
"married": np.where(s['marital']==1,1,0),
"female": np.where(s['gender']==2,1,0),
"age": np.where(s['age']>98,np.nan,s['age'])})
ss = ss.dropna()


# #### Q4 Create Target vector and feature set
y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]

# #### Q5 Split Data

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,      
                                                    test_size=0.2,   
                                                    random_state=1123)


lr = LogisticRegression(class_weight = 'balanced')


lr.fit(X_train,y_train)

st.header("LinkedIn User Prediction")

age_slide = st.number_input("How old are you?",min_value = 0, max_value=130)

female_check = st.selectbox(
    'What is your gender?',
    ('Female', 'Male', 'other'))

if (female_check == 'Female'):
                   female_select= 1
else:
                    female_select=0

married_slide = st.selectbox('Are you married?',
                            ('Yes','No'))
if (married_slide == 'Yes'):
                   married_select= 1
else:
                    married_select=0

parent_check = st.selectbox('Do you have a child/dependent under 18 living with you?',
                            ('Yes','No'))
if (parent_check == 'Yes'):
    parent_select = 1
else:
    parent_select = 0    

income_slide = st.number_input("What is your annual income?",min_value = 0, value = 10000,step=5000)
if income_slide >=150000:
    income_res = 9
elif income_slide <150000 and income_slide >= 100000:
    income_res = 8
elif income_slide <100000 and income_slide >= 75000:
    income_res = 7
elif income_slide <75000 and income_slide >= 50000:
    income_res = 6
elif income_slide <50000 and income_slide >= 40000:
    income_res = 5
elif income_slide <40000 and income_slide >= 30000:
    income_res = 4
elif income_slide <30000 and income_slide >= 20000:
    income_res = 3
elif income_slide <20000 and income_slide >= 10000:
    income_res = 2
elif income_slide <10000:
    income_res = 1
else:
    income_res = 1


educ_select = st.selectbox("What is the highest level of school/degree you have completed",
                        ('Less than High School','High School (No Degree)',
                        'High School Graduate','Some College (No Degree)',
                        'Associate Degree','Four-Year/Bachelors','Some Post Graduate (No Degree)',
                        'Post Graduate or Professional Degree'))
if educ_select =='Less than High School':
    educ3 = 1
elif educ_select == 'High School (No Degree)':
    educ3 =2
elif educ_select == 'High School Graduate':
    educ3 =3
elif educ_select == 'Some College (No Degree)':
    educ3 =4
elif educ_select == 'Associate Degree':
    educ3 =5
elif educ_select == 'Four-Year/Bachelors':
    educ3 =6
elif educ_select == 'Some Post Graduate (No Degree)':
    educ3 =7
else:
    educ3 = 8

person=[income_res,educ3,parent_select,female_select,married_select,age_slide]
predicted_class = lr.predict([person])
probs = lr.predict_proba([person])
probability = round(probs[0][1]*100,2)
if predicted_class == 1:
            result=('Predicted LinkedIn User')
else:
            result=('Not a Predicted LinkedIn User')








fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    title = {'text': f"Results: {result}<br> There is a {probability}% chance you are a LinkedIn user"}, 
    value = probability,
    gauge = {"axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 50], "color":"red"},
                {"range": [50, 100], "color":"green"}
            ],
            "bar":{"color":"orange"}}
))
st.plotly_chart(fig)