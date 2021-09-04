## Import all the required libraries
import pandas as pd
import string 
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
import warnings
import speech_recognition as sr
import pyttsx3
import requests
import aiml
from keras.models import load_model
import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
warnings.filterwarnings('ignore')
from nltk.sem import Expression
from nltk.inference import ResolutionProver
import sys
read_expr = Expression.fromstring

import os
from azure.cognitiveservices.language.textanalytics import TextAnalyticsClient
from msrest.authentication import CognitiveServicesCredentials
import IPython
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
from azure.cognitiveservices.speech import SpeechConfig, SpeechSynthesizer, AudioConfig
import azure.cognitiveservices.speech as speechsdk

#==================Uncomment the following three lines of code for first-time use only=============
#nltk.download('punkt', quiet=True)
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

#============Initialise Azure Key and Endpoint============
cog_key = 'b3982e131b7f4db2b621540df34e12b8'
cog_endpoint = 'https://mymultilingualtranslator.cognitiveservices.azure.com/'
cog_region = 'uksouth'

#============Load model===================================
model = load_model('C:\\Users\milad\OneDrive\Documents\Year3_ComputerScience\Artificial_Intelligence\stage_3\Stage3\model-010.model')
img_size = 100;

#===========Read the CSV file=====================
corpus=pd.read_csv('covidQA.csv',header=None)
corpus.columns=['Question','Response']
question=corpus['Question']
response=corpus['Response']

question = list(question)
bot_response = list(response)

#================= Initialise Knowledgebase ====================
kb=[]
data = pd.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]

#================= check for conradiction ======================
answer=ResolutionProver().prove("", kb, verbose=False)
if answer:
    print("There is condradiction in the knowledge base")
    sys.exit()

#=======get appropiate 'part-of-speach tag================
#some code used from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
def get_pos(word):
    the_pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    valid_tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return valid_tag_dict.get(the_pos_tag, wordnet.NOUN)


#=================Remove punctuation ====================
def remove_punctuation(myString):
    for x in myString:
        if(x in string.punctuation):
            myString = myString.replace(x, '')
    return myString

#==========================Get chatbot similarity based response===============================================
#some code used from https://github.com/BoulahiaAhmed/Retrieval-Based-chatbot-
didntUnderstand = "I am sorry! I don't understand you"
def response(user_query):
    chatbot_response=''
    question.append(user_query)
    
    lemmatizer = WordNetLemmatizer()
    lemmatised = list()
    for sentence in question:
        cleared_sentence = remove_punctuation(sentence)
        #Lemmatise each word of a Sentence with the appropriate POS tag
        lemmatised_sent = [lemmatizer.lemmatize(w, get_pos(w)) for w in nltk.word_tokenize(cleared_sentence)]
        
        lemmatised.append(str(lemmatised_sent).lower())
    
    
    TfidfVec = TfidfVectorizer(stop_words='english')
    tfidf = TfidfVec.fit_transform(lemmatised)

    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        chatbot_response=chatbot_response+didntUnderstand
        return chatbot_response
    else:
        chatbot_response = chatbot_response+bot_response[idx]
        return chatbot_response

#=====================Covid-19 API=========================
url = "https://rapidapi.p.rapidapi.com/v1/total"

#===================== Initialise AIML agent ==============
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#================Speak bot response=======================
#++++++++++++++++This speach recogniser was used for stage 1, 2 and 3 of the coursework+++++++++++++++++++
#some code used from https://github.com/BoulahiaAhmed/Retrieval-Based-chatbot-
# def speak(response_message):
#     speak_it= pyttsx3.init()
#     speak_it.say('{}'.format(response_message))
#     speak_it.runAndWait()

def speak(response_message, language_locale):
    speech_config = SpeechConfig(cog_key, cog_region)
    # we can use speech_config.speech_synthesis_language = "locale" for speech to text for non-english languages
    
    speech_config.speech_synthesis_language = language_locale

    speech_synthesizer = SpeechSynthesizer(speech_config)
    result = speech_synthesizer.speak_text(response_message)
    

def languageDetector(question_txt):

    user_question = []
    question = {"id": "None", "text": question_txt}
    user_question.append(question)
        
    text_analytics_client = TextAnalyticsClient(endpoint=cog_endpoint,
                                                credentials=CognitiveServicesCredentials(cog_key))
    
    
    language_analysis = text_analytics_client.detect_language(documents=user_question)
    
    lang = language_analysis.documents[0].detected_languages[0]
    return lang.iso6391_name


# Create a function that makes a REST request to the Text Translation service
def translate_text(text, from_lang='en', to_lang='en'):
    import requests, uuid, json

    # Create the URL for the Text Translator service REST request
    path = 'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0'
    params = '&from={}&to={}'.format(from_lang, to_lang)
    constructed_url = path + params

    # Prepare the request headers with Cognitive Services resource key and region
    headers = {
        'Ocp-Apim-Subscription-Key': cog_key,
        'Ocp-Apim-Subscription-Region':cog_region,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    # Add the text to be translated to the body
    body = [{
        'text': text
    }]

    # Get the translation
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    return response[0]["translations"][0]["text"]
   
#========= Remember expression if no condriction =========
def remember(expr):
    answer=ResolutionProver().prove("", kb, verbose=True)
    if answer:
        kb.remove(expr)
        print("BOT: sorry it condradicts to what I know my friend")
    else:
        print('BOT: OK, I will remember that')

#========= Check for one of 3 answers ====================
def check(expr):
    answer=ResolutionProver().prove(expr, kb, verbose=True)
    if answer:
       print('BOT: Correct')
    else:               
       expr=read_expr('-' + str(expr))
       answer=ResolutionProver().prove(expr, kb, verbose=False)
       if answer:
           print("BOT: Incorrect")
       else:
           print("BOT: Sorry I don't know") 
    
#==============Live camera face mask detection============
#Some code for this section is gotten from https://github.com/aieml/face-mask-detection-keras/blob/master/3.0%20detecting%20Masks.ipynb
def liveFaceMaskDetection():
    face_clsfr = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    source=cv2.VideoCapture(0)
    
    labels_dict={0:'With Mask',1:'Without Mask'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    
    while(True):
    
        ret,img=source.read()
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces=face_clsfr.detectMultiScale(imgRGB,1.3,5)  
    
        for (x,y,w,h) in faces:
        
            face_img=imgRGB[y:y+w,x:x+w]
            resized=cv2.resize(face_img,(img_size,img_size))
            
            img_array = keras.preprocessing.image.img_to_array(resized)
            img_array = tf.expand_dims(img_array, 0)
            result = model.predict(img_array)
            score = tf.nn.softmax(result[0])
    
            label=np.argmax(result,axis=1)[0]
          
            cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
            cv2.rectangle(img,(x,y-60),(x+w,y),color_dict[label],-1)
            cv2.putText(img, labels_dict[label], (x+10, y-40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            cv2.putText(img," {}%".format(round(100 * np.max(score), 2)), (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
            
            
        cv2.imshow('LIVE',img)
        key=cv2.waitKey(1)
        
        if(key==27):
            break
    
    cv2.destroyAllWindows()
    source.release()

#Open folder to select and open an image
categories = ['WithMask', 'WithoutMask']
        
def openImage():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(initialdir = "/", title = "Select a File", filetypes = (("JPG", "*.jpg*"), ("PNG", "*.png"))) 
    img = keras.preprocessing.image.load_img(img_path,target_size=(img_size, img_size))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    return ("The person in the image is {}, i am {} precent confident.".format(categories[np.argmax(score)], round(100 * np.max(score),2)))

not_understood = "I did not get that, please try again"
not_udrstnd = ""
#communication between the chatbot and user
#code used for speech to text from https://www.youtube.com/watch?v=K_WbsFrPUCk
def communication_point(user_choice):
    global not_udrstnd;
    lang_code=""
    detected_language=""
    userInput = ""
    while True:
        if(user_choice == 1):
            try:
                #This Azure's speech recogniser 
                speech_config = SpeechConfig(cog_key, cog_region)
                audio_config = AudioConfig(use_default_microphone=True)

                auto_detect_source_language_config = \
                speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=["en-GB", "fr-FR", "es-ES"])
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, 
                        auto_detect_source_language_config=auto_detect_source_language_config, 
                        audio_config=audio_config)
                
                try:
                    speech = speech_recognizer.recognize_once()
                    result = speechsdk.AutoDetectSourceLanguageResult(speech)
                    detected_language = result.language
                    userInput = speech.text
                    lang_code = languageDetector(userInput)
                    print("detected language ", detected_language)
                    print("You: " + userInput)
                    userInput = translate_text(userInput, to_lang='en', from_lang=lang_code)
                except:
                    print("Sorry I did not understand what you said!")
            except (KeyboardInterrupt, EOFError) as e:
                print("Bye!")
                break
            
            #++++++++++++++++++++++++This is the previous speech recogniser that was used in stage 1, 2 and 3 +++++++++++++++
            # try:
            #     r = sr.Recognizer()
            #     with sr.Microphone() as source:
            #         print("Ask me: ")
            #         audio = r.listen(source)
            #         try:
            #             userInput = r.recognize_google(audio)
            #             print("You: " + userInput)
            #         except:
            #             print("Sorry I did not understand what you said!")
            # except (KeyboardInterrupt, EOFError) as e:
            #     print("Bye!")
            #     break
        else:
            try:
                userInput = input("You: ")
                userInput = userInput.lower()
                lang_code = languageDetector(userInput)
                userInput = translate_text(userInput, to_lang='en', from_lang=lang_code)
                
            except (KeyboardInterrupt, EOFError) as e:
                print("Bye!")
                break
        
        if(userInput == ""):
            print("Please enter something, 'bye' for quit")
            continue
        
        previousInput = userInput
        #remove - from user input to make suitable for KB
        userInput = userInput.replace('-', '')
        answer = kern.respond(userInput)
        if answer[0] == '#':
            params = answer[1:].split('$')
            cmd = int(params[0])
            if cmd == 0:
                params[1] = translate_text(params[1], to_lang=lang_code, from_lang="en")
                if(user_choice == 1):
                    print("BOT: " + params[1])
                    speak(params[1], detected_language)
                else:
                    print("BOT: " + params[1])
                break
            elif cmd == 1:
                
                querystring = {"country": params[1].title()}
                headers = {
                    'x-rapidapi-key': "d3bbc7dc10mshb55dcc8fcb3a055p11b3e4jsn661a77770122",
                    'x-rapidapi-host': "covid-19-coronavirus-statistics.p.rapidapi.com"
                    }
                
                api_response = requests.request("GET", url, headers=headers, params=querystring)
                
                full_response = "The number of deaths in " + str(api_response.json()["data"]["location"]) + " are "+ str(api_response.json()["data"]["deaths"])
                full_response = translate_text(full_response, to_lang=lang_code, from_lang="en")
                if(user_choice == 1):
                    print("BOT: " + full_response)
                    speak(full_response,detected_language)
                else:
                    print("BOT: " + full_response)
            elif cmd == 99:
                not_udrstnd = "I did not get that, please try again"
            elif cmd == 2:
                bot_command = params[1]
                bot_command = translate_text(bot_command, to_lang=lang_code, from_lang="en")
                if(user_choice == 1):
                    print("BOT: " + bot_command)
                    speak(bot_command,detected_language)
                    classifier_result = openImage();
                    print("BOT: " + classifier_result)
                    speak(classifier_result,detected_language)
                else:
                    print("BOT: " + bot_command)
                    classifier_result = openImage();
                    print("BOT: " + classifier_result)
            elif cmd == 4:
                bot_command2 = params[1]
                bot_command2 = translate_text(bot_command2, to_lang=lang_code, from_lang="en")
                if(user_choice == 1):
                    print("BOT: " + bot_command2)
                    speak(bot_command2,detected_language)
                else:
                    print("BOT: " + bot_command2)
                liveFaceMaskDetection()

            # Here are the processing of the new logical component:
            elif cmd == 31:
                subject,predicate=params[1].split(' is ')
    
                expr=read_expr(predicate + '(' + subject + ')')
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
                    
            elif cmd == 32:
                subject,predicate=params[1].split(' is ')
                expr=read_expr(predicate + '(' + subject + ')')
                check(expr);
                
            elif cmd == 33:
                subjectOne,sentenceOne = params[1].split(' was ')
                predicateOne,subjectTwo = sentenceOne.split(' in ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
    
            elif cmd == 34:
                subjectOne,sentenceOne = params[1].split(' is a ')
                predicateOne,subjectTwo = sentenceOne.split(' in ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
                    
            elif cmd == 35:
                subjectOne,sentenceOne = params[1].split(' is a ')
                predicateOne,subjectTwo = sentenceOne.split(' in ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')            
                check(expr);
                    
            elif cmd == 36:
                subjectOne,sentenceOne = params[1].split(' was ')
                predicateOne,subjectTwo = sentenceOne.split(' in ')
        
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')            
                check(expr);
    
            elif cmd == 37:
                subjectOne,sentenceOne = params[1].split(' is ')
                predicateOne,subjectTwo = sentenceOne.split(' by ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
                    
            elif cmd == 38:
                subjectOne,sentenceOne = params[1].split(' is ')
                predicateOne,subjectTwo = sentenceOne.split(' by ')
        
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')            
                check(expr);
                
            elif cmd == 39:
                subjectOne,sentenceOne = params[1].split(' is ')
                predicateOne,subjectTwo = sentenceOne.split(' with ')
        
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')            
                check(expr);
                
            elif cmd == 40:
                subjectOne,sentenceOne = params[1].split(' can be ')
                predicateOne,subjectTwo = sentenceOne.split(' with ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')') 
                check(expr);
                       
            elif cmd == 41:
                subjectOne,sentenceOne = params[1].split(' can ')
                predicateOne,subjectTwo = sentenceOne.split(' to ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)                
    
            elif cmd == 43:
                subjectOne,sentenceOne = params[1].split(' can ')
                predicateOne,subjectTwo = sentenceOne.split(' to ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')') 
                check(expr);
    
            elif cmd == 44:
                subjectOne,sentenceOne = params[1].split(' is in ')
                predicateOne,subjectTwo = sentenceOne.split(' with ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
                    
            elif cmd == 45:
                subjectOne,sentenceOne = params[1].split(' is at ')
                predicateOne,subjectTwo = sentenceOne.split(' serious illness ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')') 
                check(expr);
                       
            elif cmd == 46:
                subjectOne,sentenceOne = params[1].split(' can ')
                predicateOne,subjectTwo = sentenceOne.split(' through ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')')
                print(expr)
                if expr not in kb:
                    kb.append(expr)
                remember(expr)
                    
            elif cmd == 47:
                subjectOne,sentenceOne = params[1].split(' can ')
                predicateOne,subjectTwo = sentenceOne.split(' through ')
                
                expr=read_expr(predicateOne + '(' + subjectOne + ',' + subjectTwo + ')') 
                check(expr);
            
        else:
            answer = translate_text(answer , to_lang=lang_code, from_lang="en")
            if(user_choice == 1):
                print("BOT: " + answer)
                speak(answer,detected_language)
            else:
                print("BOT: " + answer)
        
        if(not_understood == not_udrstnd):
            bot_res = response(previousInput)
            bot_res = translate_text(bot_res, to_lang=lang_code, from_lang="en")
            if(user_choice == 1):
                print("BOT: " + bot_res)
                speak(bot_res,detected_language)
            else:
                print("BOT: " + bot_res)
            not_udrstnd = ""
            question.remove(previousInput)
#===============Get user choice===================================================================================
def options():
    print("Do you want to speak or type?")
    print("1 - Speak")
    print("2 - Type")
    
    flag = True
    while(flag == True):
        
        try:
            usrInput = input("Choose an option: ")
            print("=============================================")
        except (KeyboardInterrupt, EOFError) as e:
            print("Bye Bye!")
            break
        
        try:
            usr_input = int(usrInput)
            if(usr_input < 1 or usr_input > 2):
                print("****************=> Please enter a valid option number <=**************************")
            elif(usr_input == 1 or usr_input == 2):
                flag = False
                communication_point(usr_input)
        except (ValueError):
            print("****************=> Please enter a valid input <=**************************")
        
options()