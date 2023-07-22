"""

Krushi_Jethe

Installs required dependencies
Loads models for deployment
Gives output summary from articles using the models trained


"""


import subprocess

subprocess.run(['pip3', 'install', 'newspaper3k'])
subprocess.run(['pip', 'install', 'transformers'])

import warnings

warnings.filterwarnings('ignore')


import textwrap
import newspaper
import nltk
from transformers import pipeline
from datetime import datetime
nltk.download('punkt')

summarizer_abs = pipeline("summarization", model="KrushiJethe/Final_T5_summarization")
summarizer_ext = pipeline("summarization", model="KrushiJethe/Final_BART_summarization")

class article():
    
    
    """
    
    Interactive Class
    
    Usage :  obj = article()
             obj.summarize(type_of_summary)
             
    Class :  article
    
    attributes : web_dict -- stores newspaper websites
    
    methods    : summarize -- interactive function for summarizing articles
             
    It will ask you which newspaper you would like to refer and from that newspaper
    which article to summarize
    
    """
    
    def __init__(self):
        
        self.web_dict = {1:"http://www.cnn.com",2:"https://www.hindustantimes.com/",3:"https://www.news18.com/",4:"https://www.dw.com/en/"}
    
    def summarize(self , type_of_summary):
        
        want_to_read = 'yes'
        
        while want_to_read == 'yes':
            
                    temp = int(input("""Which newspaper website do you want to check for news?
                                     1. CNN
                                     2. Hindustan times
                                     3. News18
                                     4. Deutsche Welle English
                                     Select 1 , 2 , 3 or 4\n\n\n"""))
                    
                    paper = newspaper.build(self.web_dict[temp])
            
                    for i,article in enumerate(paper.articles):
                         print(i ,article.url)
                         if i >= 50:
                             break
                         
                    temp_2 = int(input('\n\n\nPick an article number to summarize\n\n\n'))
            
            
                    article = paper.articles[temp_2]
                    article.download()
                    article.parse()
                    article.nlp()
                        
                    print('\n\n***************************************\n\n')
                    print('\n\nSummary produced by the newspaper library\n\n',textwrap.fill(article.summary))
                    print('\n\n***************************************\n\n')
                    
                    inputs = article.text
            
                    if type_of_summary == 'abstractive':
                        
                        
                        start3 = datetime.now() 
                        x = summarizer_abs(inputs, do_sample=False)
                        wrapped_text = textwrap.fill(x[0]['summary_text'], width=100)
                        stop3 = datetime.now()
                        print('\n\n This is the abstractive summary you requested for , Uses T5 small pretrained on multi_news dataset\n\n')
                        print(wrapped_text)
                    
                    
                    elif type_of_summary == 'extractive':
                        
                       
                        start3 = datetime.now() 
                        x = summarizer_ext(inputs, do_sample=False)
                        wrapped_text = textwrap.fill(x[0]['summary_text'], width=100)
                        stop3 = datetime.now()
                        print('\n\n This is the extractive summary you requested for , Uses BART pretrained on CNN news dataset\n\n')
                        print(wrapped_text)
                        print('\n\n\nExecution time : ' , stop3 - start3)
                    
                    want_to_read = str(input('\n\nWould you like to continue reading newspaper summaries ? (yes/no)\n\n'))
                    if want_to_read == 'yes':
                        type_of_summary = str(input('\n\n Type of summary ? (extractive/abstractive)\n\n'))
        
    