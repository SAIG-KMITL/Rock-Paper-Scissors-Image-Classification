import snscrape.modules.twitter as sntwitter
import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request
import PySimpleGUI as sg

def twitter(name):
    # Created a list to append all tweet attributes(data)
    attributes_container = []

    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f"from:{name}").get_items()):
        if i>10:
            break
        attributes_container.append([tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])
        
    # Creating a dataframe from the tweets list above 
    tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Number of Likes", "Source of Tweet", "Tweets"])

    return tweets_df["Tweets"]

def preprocess(text):
    new_text = []
 
    for t in text.split(" ")
        t = '' if t.startswith('@') and len(t) > 1 else t
        new_text.append(t)
    return " ".join(new_text)

sg.theme('DarkTeal12')
backgroundColor = '#c8d2dc'
textColor = '#1c587a'
colorlist = ['green','yellow','red']


tab1_layout = [  
            [sg.Image(filename='saiglogo.png',background_color=backgroundColor)],
            # [sg.Text('SAIG',background_color=backgroundColor,text_color=textColor,font=('Courier',30,'bold'))],
            # [sg.Text('',background_color=backgroundColor)],
            [sg.Text('Sentiment-Analysis',size=(0,1),background_color=backgroundColor,text_color=textColor,font=('Courier',20,'bold'))],           
            [sg.Multiline(size=(38, 5), font=('Courier 15'),key = '-TEXTINPUT-')],
            [sg.Button('COMPUTE', font=('Courier 15'),button_color = textColor, key = '-COMPUTE-')],
            [sg.Text('',background_color=backgroundColor)],
            [sg.Text('---',font=('Courier',15,'bold'),text_color='#1c587a',background_color=backgroundColor, key = '-OUTPUT1-')],
            [sg.Text('---',font=('Courier',15,'bold'),text_color='#1c587a',background_color=backgroundColor, key = '-OUTPUT2-')],
            [sg.Text('---',font=('Courier',15,'bold'),text_color='#1c587a',background_color=backgroundColor, key = '-OUTPUT3-')],
            [sg.Text('',background_color=backgroundColor, size=(0,3))]
        ]

tab2_layout = [[sg.T('This is inside tab 2')],
               [sg.In(key='in')]]

layout = [[sg.TabGroup([
            [ sg.Tab('Tab 1', tab1_layout,background_color=backgroundColor,element_justification='c'), 
              sg.Tab('Tab 2', tab2_layout)]])]]

window = sg.Window('Sentiment-Analysis',background_color=backgroundColor, layout = layout, size=(600,600), element_justification='c')


#------------------------- MODEL ------------------------------
task ='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)

labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)



listTW = []
for i in range(5):
    textTW = twitter("elonmusk")  
    text = textTW[i]
    text = preprocess(text)
    # print(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    listTW.append([text,scores[2], scores[1], scores[0]])

print(listTW)

#---------------------------------------------------------------

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED: 
        break
    # ------------     
    if event == '-COMPUTE-':
        text = values['-TEXTINPUT-']
        text = preprocess(text)
        # print(text)
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]  

        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]]
            # print(f"{i+1}) {l} {np.round(float(s), 4)}")
            if(l == 'positive'):                  
                window[f'-OUTPUT{i+1}-'].update(f"{l} {np.round(float(s), 4)}", text_color=colorlist[0],background_color=backgroundColor)
            elif(l == 'neutral'):
                window[f'-OUTPUT{i+1}-'].update(f"{l} {np.round(float(s), 4)}", text_color=colorlist[1],background_color=backgroundColor)
            elif(l == 'negative'):
                window[f'-OUTPUT{i+1}-'].update(f"{l} {np.round(float(s), 4)}", text_color=colorlist[2],background_color=backgroundColor)                  

window.close()   