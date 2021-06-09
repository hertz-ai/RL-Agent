# -*- coding: utf-8 -*-

import argparse
import ast
import itertools
import json
import pickle
import pprint
import random
import re
import string
import sys
import nltk
import numpy as np
import pke
import requests
import spacy
from flashtext import KeywordProcessor
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from collections import Counter, defaultdict
from nltk.tokenize import sent_tokenize
from pywsd.lesk import adapted_lesk, cosine_lesk, simple_lesk
from pywsd.similarity import max_similarity
from transformers import *
import sqlalchemy as db
from sqlalchemy import create_engine,MetaData, Table, String, Column, Text, DateTime, Boolean, Integer,Float,insert,ForeignKey,select,desc
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import pandas as pd


#Giri - flask endpoint for QA gen
from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import time

app = Flask(__name__)

# nltk.download('stopwords')
# nltk.download('popular')

def keyword_extractor(summarized_text,num_extractor=20):

    try:

        def get_nouns_multipartite(text):
            out=[]

            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=text)
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'PROPN'}
            #pos = {'VERB', 'ADJ', 'NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            extractor.candidate_selection(pos=pos, stoplist=stoplist)
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=int(num_extractor))

            for key in keyphrases:
                out.append(key[0])
            return out

        keywords = get_nouns_multipartite(summarized_text)
        # print (keywords)
        filtered_keys=[]
        for keyword in keywords:
            if keyword.lower() in summarized_text.lower():
                filtered_keys.append(keyword)

        # print (filtered_keys)
        return filtered_keys
    except:
        return []

def split_Para2Sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

# Distractors from Wordnet
def get_distractors_wordnet(syn,word):
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0:
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_wordsense(sent,word):
    word= word.lower()

    if len(word.split())>0:
        word = word.replace(" ","_")


    synsets = wn.synsets(word,'n')
    if synsets:
        wup = max_similarity(sent, word, 'wup', pos='n')
        adapted_lesk_output =  adapted_lesk(sent, word, pos='n')
        lowest_index = min (synsets.index(wup),synsets.index(adapted_lesk_output))
        return synsets[lowest_index]
    else:
        return None

# Distractors from http://conceptnet.io/
def get_distractors_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = []
    try:
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
        obj = requests.get(url).json()
        # return obj
        for edge in obj['edges']:
            link = edge['end']['term']

            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
            obj2 = requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
    except Exception as Expt:
        print(Expt,"Warning: From concept-net api")

    return distractor_list

def get_para_for_question_generation(para,sent):
    # for question generation
    st_point = para.find(sent)
    end_point = st_point + len(sent)+1
    start=para.find(sent)
    if st_point == -1:
        ## if cant find return 0 to consider whole para
        return start,end_point
    # loop till finding the end of sentence
    while True:
        if para[st_point] == ".":
            return st_point+1,end_point
        elif st_point <1:
            return start,end_point
        else:
            st_point -=1



def _getQuestion(text,answer):
    """Question generator"""

    ## qgen allows 512 token size ,if exceeds it will get truncated.
    # if len(answer.split())>70:
    #     print("Truncating the answer given for Question generation as it exceeds the 70 words")
    # answer = answer.split()[:70]
    # num_text = len(text.split())
    # num_ans = len(answer)
    # text = text.split()[:200-num_ans] ##450

    # data = " ".join(text)+"[SEP]"+" ".join(answer)


    q_url="http://localhost:9875/predict"
    q_data = {"input":str(text)+"[SEP]"+str(answer)}
    headers = {'Content-Type': 'application/json'}
    r = requests.post(q_url, data=json.dumps(q_data), headers=headers)

    return str(list(r.text.split("[split]"))[0])


def _getAnswer(text,question,include_noanswer=False):
    """Answer generator"""
    headers = {'Content-Type': 'application/json'}
    a_url="http://localhost:5000/predict"
    if include_noanswer:
        threshold=0.0
        best_size=1
        max_answer_length=30
        a_data = {"passage": str(text),"questions":[str(question)],"threshold":threshold,"best_size":best_size,"max_answer_length":max_answer_length}
    else:
        a_data = {"passage": str(text),"questions":[str(question)]}

    a = requests.post(a_url, data=json.dumps(a_data), headers=headers)
    res = ast.literal_eval(str(a.text))
    return res[0][0] if res[0]!="" else ""


def _Summarizer(context):
    """
    summarize the context to single sentence"""

    headers = {'Content-Type': 'application/json'}
    data_creation={"context":context}
    try:
        r = requests.post("http://localhost:5012", data=json.dumps(data_creation), headers=headers)
    except ConnectionRefusedError:
        print("PEGASUS SUMMARIZER IS DOWN. CHECK DOCKER LOGS")
    if r.status_code == 200:
        return (json.loads(r.text))["response"]
    else:
        raise RuntimeError

def _Paraphrasing(context,num_duplicates=5):
    """
     It creates the similar sentences. """
    context = str(context)

    if len(context.split(' ')) > 60:
        print("WARNING: Paraphrasing supports less than 60 words, Truncating the excess words.")

    headers = {'Content-Type': 'application/json'}
    data_creation={"context":str(context),"num_duplicates":num_duplicates}
    try:
        r = requests.post("http://localhost:5011", data=json.dumps(data_creation), headers=headers)
    except ConnectionRefusedError:
        print("PEGASUS PARAPHRASING IS DOWN. CHECK DOCKER LOGS")

    return (json.loads(r.text))["response"]

# def get_last_paraid_from_db():
#     # data =  json.dumps({
#     # "para_id":paraid,
#     # "paragraph":para,
#     # "is_active":True
#     # })
#     # headers = {'Content-Type': 'application/json'}
#     r = requests.get("http://0.0.0.0:6005/get_last_paraid")#, headers=headers,data=data)
#     print(r.text)
#     return (json.loads(r.text))["para_id"]

def put_para_tracker_in_db(para):
    data =  json.dumps({
    "paragraph":para,
    "is_active":True
    })
    headers = {'Content-Type': 'application/json'}
    r = requests.post("https://mailer.hertzai.com/create_paragraph_tracker", headers=headers,data=data)
    # r = requests.post("http://0.0.0.0:6006/create_paragraph_tracker", headers=headers,data=data)
    print(r.text)
    return r.text

def put_qa_pair_in_db(question,answer,assessment_name,para_id,options=[],question_type="OBJECTIVE",is_active=True):
    data=json.dumps({
                "question": str(question[:3]),
                "question_type": question_type,
                "answer": answer,
                "assessment_name": assessment_name,
                "is_active": True,
                "para_id":para_id,
                "options":str(options[:5])
                    })
    print("QA to be inserted :: ", data)

    # #Giri - Workaround to avoid question duplication
    # data=json.dumps({
    #             "question": str(question[:1][0]),
    #             "question_type": question_type,
    #             "answer": answer,
    #             "assessment_name": assessment_name,
    #             "is_active": True,
    #             "para_id":para_id,
    #             "options":str(options[:5])
    #                 })
    # print("QA to be inserted :: ", data)

    headers = {'Content-Type': 'application/json'}
    r = requests.post("http://0.0.0.0:6006/create_question_answers", headers=headers,data=data)
    print(r.text)
    # exit()
    return r.text

# class DB():
#     def __init__(self):
#         self.engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
#         self.cnx = self.engine.connect()
#         self.meta = MetaData(engine,reflect=True)
#         self.db_para = Table('paragraph_tracker', self.meta,
#                 Column('paraID',Integer,nullable=False,primary_key = True),
#                 Column('para', Text, nullable=False),
#                 Column('created_on', DateTime(), default=datetime.now),
#                 Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now)
#             )
#         self.db_qa = Table("qa", self.meta,
#             Column('questionID', Integer, primary_key=True, nullable=False),
#             Column('created_on', DateTime(), default=datetime.now),
#             Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now),
#             Column('paraID',ForeignKey(self.db_para.c.paraID),nullable=False),
#             Column('topic',String(500),nullable=False),
#             Column('question',String(2000),nullable=False),
#             Column('groundtruth',String(1000),nullable=False),
#             Column('predicted',String(1000)),
#             Column('is_mcq',Boolean,nullable=False,default=False),
#             Column('options',String(1000),default=[])
#             )
#         self.base = declarative_base()

#     def create_table(self):
#         # engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
#         # # cnx = engine.connect()
#         # metadata = MetaData(engine)
#         # # if not engine.dialect.has_table(engine, "paragraph_tracker"):  # If table don't exist, Create.
#         # db_para = Table('paragraph_tracker', metadata,
#         #         Column('paraID',Integer,nullable=False,primary_key = True),
#         #         Column('para', Text, nullable=False),
#         #         Column('created_on', DateTime(), default=datetime.now),
#         #         Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now)
#         #     )
#         # # if not engine.dialect.has_table(engine, "qa"):  # If table don't exist, Create.
#         #     # Create a table with the appropriate Columns
#         # db_qa = Table("qa", metadata,
#         #     Column('questionID', Integer, primary_key=True, nullable=False),
#         #     Column('created_on', DateTime(), default=datetime.now),
#         #     Column('updated_on', DateTime(), default=datetime.now, onupdate=datetime.now),
#         #     Column('paraID',ForeignKey(db_para.c.paraID),nullable=False),
#         #     Column('topic',String(500),nullable=False),
#         #     Column('question',String(2000),nullable=False),
#         #     Column('groundtruth',String(1000),nullable=False),
#         #     Column('predicted',String(1000)),
#         #     Column('is_mcq',Boolean,nullable=False,default=False),
#         #     Column('options',String(1000),default=[])
#         #     )


#         self.meta.create_all(self.engine)

#     def drop_table(self,table_name):
#         if not isinstance(table_name,str):
#             raise ValueError
#         if table_name not in ["qa","paragraph_tracker"]:
#             raise ValueError
#         # engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
#         # # cnx = engine.connect()
#         # base = declarative_base()
#         # metadata = MetaData(engine,reflect=True)
#         table = self.meta.tables.get(table_name)
#         if table is not None:
#             # logging.info(f'Deleting {table_name} table')
#             self.base.metadata.drop_all(self.engine, [table], checkfirst=True)

#     def insert_in_qa(self,value):
#         # engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
#         # conn = engine.connect()
#         # base = declarative_base()
#         # metadata = MetaData(engine,reflect=True)
#         if not isinstance(value,list(dict)):
#             raise ValueError
#         table = get_table("qa")
#         ins = insert(table)
#         r = self.conn.execute(ins,value)
#         print("no of values inserted ",r.rowcount)


#     def insert_in_para(self,value):
#         if not isinstance(value,list(dict)):
#             raise ValueError
#         table = get_table("paragraph_tracker")
#         ins = insert(table)
#         r= self.conn.execute(ins,value)
#         print("no of paragraph inserted ",r.rowcount)

#     def get_table(self,tablename):
#         return self.meta.tables.get(tablename)

#     def get_qa_ids_and_para_ids_for_topic(self,topic):
#         question_table = get_table("qa")
#         qury = select([question_table.c.questionID,question_table.c.paraID]).where(question_table.c.topic ==topic)
#         r = self.cnx.execute(qury)
#         # IDS = r.fetchall()
#         # if not isinstance(IDS,list):
#         #     raise ValueError
#         result = r.fetchall() #
#         # colums = r.keys() # gets all the columns name
#         qno=[]
#         pno=[]
#         ## changing the format of list to dict ## easy usabality
#         for i,x in result:
#             qno.append(i)
#             if x not in pno:
#                 pno.append(x)

#         return qno,pno

#         # return IDS.insert(0,r.keys())

#     def get_qa_for_questionID(self,questionID):
#         table = get_table("qa")
#         # creating a query to get all the qa for the topic
#         pairs = select([table]).where(table.c.questionID == questionID)

#         r = self.cnx.execute(pairs)

#         result = r.first() #
#         colums = r.keys() # gets all the columns name
#         # output=[]
#         ## changing the format of list to dict ## easy usabality
#         # for i in range(len(result)):
#         li = {}
#         for head,body in zip(colums,result):
#             if head in ["question","options"]:## as sqlalchemy doesnt support array ## we changed to str
#                 body = ast.literal_eval(body) # converting str into list
#             li[head]=body
#             # output.append(li)

#         return li
#     def get_all_topics(self):
#         table = get_table("qa")
#         cmd = select([table.c.topic]).distinct()
#         r = self.cnx.execute(cmd)
#         if not isinstance(r,list(tuple)):
#             raise ValueError
#         return [x for y in r for x in y]

#     def get_all_qa_for_topic(self,topic):
#         table = get_table("qa")
#         # creating a query to get all the qa for the topic
#         pairs = select([table]).where(table.c.topic == topic)

#         r = self.cnx.execute(pairs)

#         result = r.fetchall() #
#         colums = r.keys() # gets all the columns name
#         output=[]
#         ## changing the format of list to dict ## easy usabality
#         for i in range(len(result)):
#             li = {}
#             for head,body in zip(colums,result[i]):
#                 if head in ["question","options"]:## as sqlalchemy doesnt support array ## we changed to str
#                     body = ast.literal_eval(body) # converting str into list
#                 li[head]=body
#             output.append(li)

#         return output

#     def last_question_no(self):
#         table = get_table("qa")
#         pairs = select([table.c.questionID]).order_by(desc(table.c.questionID))

#         r = self.cnx.execute(pairs)
#         output = r.first()
#         if not isinstance(output,int):
#             raise ValueError
#         return output


#     def last_para_id(self):

#         table = get_table("paragraph_tracker")
#         pairs = select([table.c.paraID]).order_by(desc(table.c.paraID))

#         r = self.cnx.execute(pairs)
#         output = r.first()
#         if not isinstance(output,int):
#             raise ValueError
#         return output

#     def get_para_for_question(self,questionID):

#         question_table = get_table("qa")
#         qury = select([question_table.c.paraID]).where(question_table.c.questionID ==questionID)
#         r = self.cnx.execute(qury)
#         paraID = r.first()
#         if not isinstance(paraID,int):
#             raise ValueError

#         para_table = get_table("paragraph_tracker")

#         qury = select([para_table.c.para]).where(para_table.c.paraID == paraID)
#         r = self.cnx.execute(qury)
#         para = r.first()
#         if not isinstance(para,int):
#             raise ValueError
#         return para

#     def get_all_para_for_topic(self,topic):

#         question_table = get_table("qa")
#         qury = select([question_table.c.paraID]).where(question_table.c.topic ==topic)
#         r = self.cnx.execute(qury)
#         paraID = r.fetchall()
#         if not isinstance(paraID,list(int)):
#             raise ValueError

#         para_table = get_table("paragraph_tracker")

#         qury = select([para_table.c.para]).where((min(paraID)<para_table.c.paraID< max(paraID)))
#         r = self.cnx.execute(qury)
#         para = r.fetchall()
#         if not isinstance(para,list(str)):
#             raise ValueError

#         return " ".join(para)



class QA_main():
    def __init__(self):
        # self.topic = ""
        self.paragraphs_defined_type = {"dense":False,"sparse":True}
        self.engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
        self.cnx = self.engine.connect()
        self.meta = MetaData(self.engine,reflect=True)
        # self.para_id = 0#DB.last_para_id()
        # self.question_id =0# DB.last_question_no()

        # self.SQLALCHEMY_DATABASE_URL = "mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb"
        # self.engine = db.create_engine(self.SQLALCHEMY_DATABASE_URL)
        # self.connection = self.engine.connect()
        # self.metadata = db.MetaData()

        # self.paragraphs_id_map = []
        # self.QA_buffer=[]


    def dict_template(self,question,groundanswer,predictedanswer,para_id,question_id,questiontype="Normal",options=[]):
        print(question_id)
        if questiontype=="MCQ":
            questiontype = True
        else:
            questiontype = False

        return {"topic":self.topic.lower(),"question": str(question), "groundtruth": groundanswer, "predicted": predictedanswer,"questionID":question_id,"paraID":para_id,"is_mcq": questiontype, "options": str(options)}

    # def get_last_paraid(self):
    #     # self.engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
    #     # self.cnx = self.engine.connect()
    #     # self.meta = MetaData(engine,reflect=True)
    #     table = self.meta.tables.get("paragraphs")
    #     pairs = select([table.c.paraID]).order_by(desc(table.c.paraID))

    #     r = self.cnx.execute(pairs)
    #     output = r.first()
    #     return output

    def handle_MCQ(self,assessment_name,para,para_id,num_mcq_per_para):
        spacy_nlp = spacy.load('en_core_web_sm')
        keywords=[]
        spacy_para = spacy_nlp(para)
        num_buff_keywords = 10 ## extra keywords
        for entity in spacy_para.ents:
            if not entity.text in keywords and len(keywords)<num_mcq_per_para+num_buff_keywords:
                keywords.append(entity.text)

        if len(keywords) < num_mcq_per_para+num_buff_keywords:## calling second keyword extractor
            required_num_keywords = num_mcq_per_para - len(keywords)
            second_extractor = keyword_extractor(para,num_mcq_per_para - len(keywords)+num_buff_keywords)

            for ext in second_extractor:
                # if len(keywords) >= num_mcq_per_para:
                #     break
                if ext not in keywords:
                    keywords.append(ext)

        keywords.sort(key=lambda x:len(x),reverse=True)## sorting the keywords  based on the size of keyword

        # if len(keywords)>num_mcq_per_para:
        #     keywords = keywords[:num_mcq_per_para]

        list_sentences = split_Para2Sentences(para)  ## or just split by period
        keyword_sentence_mapping = get_sentences_for_keyword(keywords,list_sentences)

        key_distractor_list = {}

        for keyword in keyword_sentence_mapping:
            ## check ## no of distractor generated
            if len(key_distractor_list)>= num_mcq_per_para:
                break
            ## if corrsponding sentences less than 1 are neglected
            if len(keyword_sentence_mapping[keyword])<1:
                continue
            wordsense = get_wordsense(keyword_sentence_mapping[keyword][0],keyword)
            if wordsense:
                distractors = get_distractors_wordnet(wordsense,keyword)
                if len(distractors) ==0:
                    distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors
            else:

                distractors = get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    key_distractor_list[keyword] = distractors

        fillups=[]
        for each in keywords:
        # for each in key_distractor_list:
            if each in key_distractor_list:

                pattern = re.compile(each, re.IGNORECASE)
                # sentence = keyword_sentence_mapping[each][0]
                # output = pattern.sub( " _______ ", sentence)
                question = []
                para_for_ans_ext=""

                for sentence in keyword_sentence_mapping[each]:
                    question.append(pattern.sub( " _______ ", sentence))
                    ## joining thesentences to form paragraph
                    para_for_ans_ext +=sentence+" "

                ## Due to truncation of important parts ## passing extracted sentence as paragraph instead of para variable
                # paragraphs needs atleast 20 words to generate relavent que
                if len(keyword_sentence_mapping[each])<2 or len(para_for_ans_ext.split())<20:
                    #getting para of the first sentence in keyword_sentence_mapping
                    if len(keyword_sentence_mapping[each])!=0:
                        # st_pos = para.find(keyword_sentence_mapping[each][0])
                        # end_pos = st_pos+len(keyword_sentence_mapping[each][0])
                        # para_for_ans_ext = para[st_pos-50:end_pos+100]
                        st,ed = get_para_for_question_generation(para,keyword_sentence_mapping[each][0])
                        para_for_ans_ext = para[st:]
                    else:
                        #search for first occurence of the word and getting the surrounding paragraph
                        st = para.find(each) if para.find(each) >45 else 0
                        para_for_ans_ext = para[st-50:]
                question.insert(1,_getQuestion(para_for_ans_ext,each)) #### create normal qa for mcq
                random.shuffle(key_distractor_list[each])
                choices = [each.capitalize()] + key_distractor_list[each]

                # top4choices = choices
                # random.shuffle(top4choices)
                put_qa_pair_in_db(question,answer=each,assessment_name=assessment_name,para_id=para_id,options=choices,question_type="MCQ",is_active=True)
                # self.QA_buffer.append(self.dict_template(question=output,groundanswer=each,predictedanswer=each,para_id=self.para_id,question_id=self.question_id,questiontype="MCQ",options=choices))
                # self.question_id +=1
            else:
                fillups.append(each)
                # self.QA_buffer.append(dict_template(question="",groundanswer=each,predictedanswer=each,para_id=self.para_id,question_id=self.question_id,questiontype="fillups",options=[]))

        return fillups,keyword_sentence_mapping

    def sparse_QA(self,assessment_name,paragraphs,paragraphs_split,total_num_para,max_num_QA,need_MCQ,need_NormalQA):
        #how many  questions need to generate for each paragraph based on the requirement.
        num_mcq_per_para = max_num_QA["MCQ"]//total_num_para if max_num_QA["MCQ"]//total_num_para > 0 else 1
        num_normalqa_per_para = max_num_QA["NormalQA"]//total_num_para if max_num_QA["NormalQA"]//total_num_para > 0 else 1

        #Looping throught each para
        for idx,para in enumerate(paragraphs_split):
            # keeping track of each para with id
            # para_id = get_last_paraid_from_db()+1#self.get_last_paraid() ## or use get get_last_paraid_from_db()
            para_id = json.loads(put_para_tracker_in_db(para))["id"]

            if need_MCQ:
                fillups,keyword_sentence_mapping = self.handle_MCQ(assessment_name,para,para_id,num_mcq_per_para)
            else:
                fillups,keyword_sentence_mapping = [],[]

            if need_NormalQA:
                ## to create required no of qa, need to split the para
                # as summarizer creates single sentence out of para
                step = len(para)//num_normalqa_per_para
                summarized_text = []
                blocks=[] ## to track the summarized sentence from where they generated.
                for i in range(0,len(para),step):

                    ## block to search end of sentence
                    ## to use end of sentence # uncomment the following code
                    last_idx = i+step
                    # for k in para[last_idx:]:
                    #     last_idx+=1
                    #     if k == ".":
                    #         break
                    #     if last_idx >= len(para):
                    #         break
                    block = para[i:last_idx]
                    if len(block) < 20:
                        if len(para)-step == i:
                            # ## removing end of para sentence which is less the 20 character
                            continue
                        else:
                            print("Reduce the max no of question for normal qa")
                    ## call summarizer
                    summarized_text.append((_Summarizer(block))[0])
                    blocks.append(block)

                for each_word in fillups:
                    ## adding the keywords to normalqa which distractor are insufficient
                    summarized_text.append(each_word)
                    blocks.append("NULL") ##padding blocks

                for i,para_block in zip(summarized_text,blocks):
                    if para_block=="NULL": #to discriminate between summarized and entities
                        para_block=" ".join(keyword_sentence_mapping[i])

                        #if the para_block doesnt have enought text
                        if len(keyword_sentence_mapping[i]) in [0,1] or len(para_block)<20:
                            if len(keyword_sentence_mapping[i])==0:
                                # st = para.find(i) if para.find(i) >45 else 0
                                # para_block = para[st-50:]
                                # para_block = para[(para.find(i))-50:]
                                st,ed = get_para_for_question_generation(para,i)
                                para_block = para[st:]
                            else:
                                # st_pos = para.find(keyword_sentence_mapping[i][0])
                                # end_pos = st_pos+len(keyword_sentence_mapping[i][0])
                                # para_block = para[st_pos-50:end_pos+100]
                                st,ed = get_para_for_question_generation(para,keyword_sentence_mapping[i][0])
                                para_block = para[st:]
                    question = []
                    ##call question gen  ## using the blocks used for summarization
                    created_que = _getQuestion(para_block,i)

                    if created_que.strip() == "":
                        continue
                    question.append(created_que)

                    ## call paraphrasing to create duplicate questions
                    dup_questions =_Paraphrasing(created_que,num_duplicates=3)
                    for dup in dup_questions:
                        if dup not in question and any(dup.startswith(x.capitalize()) for x in ["wh","how","was","is"]):

                            question.append(dup)

                    ## call answer gen ## change include_noanswer to False if you want answer for all type of question even it is out of scope
                    # predicted_answer = _getAnswer(para,created_que,include_noanswer=True)

                    put_qa_pair_in_db(question,answer=i,assessment_name=assessment_name,para_id=para_id,options=[],question_type="OBJECTIVE",is_active=True)


    def dense_QA(self,assessment_name,paragraphs,paragraphs_split,total_num_para,max_num_QA,need_MCQ,need_NormalQA):

        #how many  questions need to generate for each paragraph based on the requirement.
        num_mcq_per_para = max_num_QA["MCQ"]//total_num_para if max_num_QA["MCQ"]//total_num_para > 0 else 1
        num_normalqa_per_para = max_num_QA["NormalQA"]//total_num_para if max_num_QA["NormalQA"]//total_num_para > 0 else 1

        for para in paragraphs_split:
            # splitting the dense paragraph into list of sentences
            list_sentences = split_Para2Sentences(para)  ## or just split by period// \n\n

            # tracking paragraph with id
            # para_id =get_last_paraid_from_db()+1# self.get_last_paraid() ## or use get get_last_paraid_from_db()
            # put_para_tracker_in_db(para_id,para)

            para_id = json.loads(put_para_tracker_in_db(para))["id"]

            if need_MCQ:
                ## call mcq
                fillups,keyword_sentence_mapping = self.handle_MCQ(assessment_name,para,para_id,num_mcq_per_para)
            else:
                fillups,keyword_sentence_mapping=[],[]

            if need_NormalQA:

                ## position embedding  ##for mapping entities and list_sentences
                list_sentences_pos = [0 for _ in list_sentences]

                ## add fillups to normal qa if len(list_sentence) doesnt met the max num of qa
                if len(list_sentences) <= num_normalqa_per_para:
                    for each_word in fillups:
                        list_sentences.append(each_word)
                        list_sentences_pos.append(1)

                for i,idx in zip(list_sentences,list_sentences_pos):
                    question = []

                    if idx == 0:
                        # for sentences
                        start,end = get_para_for_question_generation(para,i)
                        para_for_que = para[start:]


                        # idx_from_para = para.find(i)
                        # if idx_from_para != -1:
                        #     ### adding particular paragraph for that sentences
                        #     para_for_que = get_para_for_question_generation(para,)
                        #     # para_for_que = para[idx_from_para-50:] if idx_from_para>50 else para
                        # else:
                        #     ## as i not found in para
                        #     para_for_que = para
                    else:
                        #for entities
                        para_for_que = " ".join(keyword_sentence_mapping[i])
                        if len(keyword_sentence_mapping[i]) in [0,1] or len(para_for_que)<20:
                            if len(keyword_sentence_mapping[i])==0:
                                start,end=get_para_for_question_generation(para,i)
                                para_for_que = para[start:]
                            else:
                                # st_pos = para.find(keyword_sentence_mapping[i][0])
                                # end_pos = st_pos+len(keyword_sentence_mapping[i][0])
                                # para_for_que = para[st_pos-50:end_pos+100]
                                start,end=get_para_for_question_generation(para,keyword_sentence_mapping[i][0])
                                para_for_que = para[start:]


                    ##call question gen
                    created_que = _getQuestion(para_for_que,i)

                    if created_que.strip() == "":
                        continue
                    question.append(created_que)

                    ## call paraphrasing to create duplicate questions
                    dup_questions =_Paraphrasing(created_que,num_duplicates=3)
                    for dup in dup_questions:
                        if dup not in question and any(dup.startswith(x.capitalize()) for x in ["wh","how","was","is"]):
                            question.append(dup)

                    ## call answer gen ## change include_noanswer to False if you want answer for all type of question even it is out of scope
                    predicted_answer = _getAnswer(para,created_que,include_noanswer=True)
                    if predicted_answer.strip() =="":
                        continue
                    put_qa_pair_in_db(question,answer=predicted_answer,assessment_name=assessment_name,para_id=para_id,options=[],question_type="OBJECTIVE",is_active=True)

            #         self.QA_buffer.append(self.dict_template(question=question,groundanswer=i,predictedanswer=predicted_answer,para_id=self.para_id,question_id=self.question_id,questiontype="NormalQA",options=[]))
            #         self.question_id +=1
            # self.para_id +=1

    def start_creating(self,assessment_name,paragraphs,paragraphs_type = None,mx_num_mcq =1,mx_num_normalqa=1):
        #TODO
        '''
        Create assessment page
        Schedule assessment option
        Take page_rage, and fetch text from database
            get from each page/layout wise text
                |-- text
                | -- if the num of paragrafs > 1000words, then we will do sparse qa gen which requires summarizer
                | -- if less num of para..., then dense qa gen
                | -- once qa are generated, a db entry is made with
                                                                | -- question, answer, assessment_id/name, (fileId, layout_num, page_num)~composite-keys or need to crate primary key in publay table.



        '''
        if not paragraphs_type:
            if len(paragraphs.split()) > 1000:
                paragraphs_type = "sparse"
            else:
                paragraphs_type = "dense"

        need_MCQ= True if mx_num_mcq>0 else False
        need_NormalQA = True if mx_num_normalqa>0 else False

        max_num_QA = {"MCQ":mx_num_mcq,"NormalQA":mx_num_normalqa}
        print(" **** 834")

        # if not isinstance(self.question_id,int):
        #     self.question_id =0

        # if not isinstance(self.para_id,int):
        #     self.para_id = 0

        use_summarizer = self.paragraphs_defined_type[paragraphs_type]
        print("**** 844 ")

        # preparing the paragraphs and splitted para's
        if isinstance(paragraphs,str):
            paragraphs_split = list(paragraphs.split("\n\n"))
        elif isinstance(paragraphs,list):
            paragraphs_split = paragraphs
            paragraphs = "\n\n".join(paragraphs_split)
        else:
            raise ValueError("Expected string or list for paragraph, instead got"+type(paragraphs))

        ## total no of para's
        total_num_para = len(paragraphs_split)
        if use_summarizer:
            self.sparse_QA(assessment_name,paragraphs,paragraphs_split,total_num_para,max_num_QA,need_MCQ,need_NormalQA)

        else:
            print("** dense ")
            self.dense_QA(assessment_name,paragraphs,paragraphs_split,total_num_para,max_num_QA,need_MCQ,need_NormalQA)


        ## filter



paragrs="""Python is an interpreted, high-level and general-purpose programming language. Created by Guido van Rossum and first released in 1991, Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.[28]

Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured (particularly, procedural), object-oriented, and functional programming. Python is often described as a "batteries included" language due to its comprehensive standard library.[29]

Python was created in the late 1980s as a successor to the ABC language. Python 2.0, released in 2000, introduced features like list comprehensions and a garbage collection system with reference counting.

Python 3.0, released in 2008, was a major revision of the language that is not completely backward-compatible, and much Python 2 code does not run unmodified on Python 3.

The Python 2 language was officially discontinued in 2020 (first planned for 2015), and "Python 2.7.18 is the last Python 2.7 release and therefore the last Python 2 release."[30] No more security patches or other improvements will be released for it.[31][32] With Python 2's end-of-life, only Python 3.5.x[33] and later are supported.

Python interpreters are available for many operating systems. A global community of programmers develops and maintains CPython, a free and open-source[34] reference implementation. A non-profit organization, the Python Software Foundation, manages and directs resources for Python and CPython development."""

full_text1 = """The Greek historian knew what he was talking about. The Nile River fed Egyptian civilization for hundreds of years. The Longest River the Nile is 4,160 miles long—the world’s longest river. It begins near the equator in Africa and flows north to the Mediterranean Sea. In the south the Nile churns with cataracts. A cataract is a waterfall. Near the sea the Nile branches into a delta. A delta is an area near a river’s mouth where the water deposits fine soil called silt. In the delta, the Nile divides into many streams. The river is called the upper Nile in the south and the lower Nile in the north. For centuries, heavy rains in Ethiopia caused the Nile to flood every summer. The floods deposited rich soil along the Nile’s shores. This soil was fertile, which means it was good for growing crops. Unlike the Tigris and Euphrates, the Nile River flooded at the same time every year, so farmers could predict when to plant their crops. Red Land, Black Land The ancient Egyptians lived in narrow bands of land on each side of the Nile. They called this region the black land because of the fertile soil that the floods deposited. The red land was the barren desert beyond the fertile region. Weather in Egypt was almost always the same. Eight months of the year were sunny and hot. The four months of winter were sunny but cooler. Most of the region received only an inch of rain a year. The parts of Egypt not near the Nile were a desert. Isolation The harsh desert acted as a barrier to keep out enemies. The Mediterranean coast was swampy and lacked good harbors. For these reasons, early Egyptians stayed close to home. Each year, Egyptian farmers watched for white birds called ibises, which flew up from the south. When the birds arrived, the annual flood waters would soon follow. After the waters drained away, farmers could plant seeds in the fertile soil. Agricultural Techniques By about 2400 B.C., farmers used technology to expand their farmland. Working together, they dug irrigation canals that carried river water to dry areas. Then they used a tool called a shaduf to spread the water across the fields. These innovative, or new, techniques gave them more farmland. Egyptian Crops Ancient Egyptians grew a large variety of foods. They were the first to grind wheat into flour and to mix the flour with yeast and water to make dough rise into bread. They grew vegetables such as lettuce, radishes, asparagus, and cucumbers. Fruits included dates, figs, grapes, and watermelons. Egyptians also grew the materials for their clothes. They were the first to weave fibers from flax plants into a fabric called linen. Lightweight linen cloth was perfect for hot Egyptian days. Men wore linen wraps around their waists. Women wore loose, sleeveless dresses. Egyptians also wove marsh grasses into sandals. Egyptian Houses Egyptians built houses using bricks made of mud from the Nile mixed with chopped straw. They placed narrow windows high in the walls to reduce bright sunlight. Egyptians often painted walls white to reflect the blazing heat. They wove sticks and palm trees to make roofs. Inside, woven reed mats covered the dirt floor. Most Egyptians slept on mats covered with linen sheets. Wealthy citizens enjoyed bed frames and cushions. Egyptian nobles had fancier homes with tree-lined courtyards for shade. Some had a pool filled with lotus blossoms and fish. Poorer Egyptians simply went to the roof to cool off after sunset. They often cooked, ate, and even slept outside. Egypt’s economy depended on farming. However, the natural resources of the area allowed other economic activities to develop too. The Egyptians wanted valuable metals that were not found in the black land. For example, they wanted copper to make tools and weapons. Egyptians looked for copper as early as 6000 B.C. Later they learned that iron was stronger, and they sought it as well. Ancient Egyptians also desired gold for its bright beauty. The Egyptian word for gold was nub. Nubia was the Egyptian name for the area of the upper Nile that had the richest gold mines in Africa. Mining minerals was difficult. Veins (long streaks) of copper, iron, and bronze were hidden inside desert mountains in the hot Sinai Peninsula, east of Egypt. Even during the cool season, chipping minerals out of the rock was miserable work. Egyptians mined precious stones too. They were probably the first people in the world to mine turquoise. The Egyptians also mined lapis lazuli. These beautiful blue stones were used in jewelry.The Nile had fish and other wildlife that Egyptians wanted. To go on the river, Egyptians made lightweight rafts by binding together reeds. They used everything from nets to harpoons to catch fish. One ancient painting even shows a man ready to hit a catfish with a wooden hammer. More adventurous hunters speared hippopotamuses and crocodiles along the Nile. Egyptians also captured quail with nets. They used boomerangs to knock down flying ducks and geese. (A boomerang is a curved stick that returns to the person who threw it.) Eventually, Egyptians equipped their reed boats with sails and oars. The Nile then became a highway. The river’s current was slow, so boaters used paddles to go faster when they traveled north with the current. Going south, they raised a sail and let the winds that blew in that direction push them. The Nile provided so well for Egyptians that sometimes they had surpluses, or more goods than they needed. They began to trade with each other. Ancient Egypt had no money, so people exchanged goods that they grew or made. This method of trade is called bartering. Egypt prospered along the Nile. This prosperity made life easier and provided greater opportunities for many Egyptians. When farmers produce food surpluses, the society’s economy begins to expand. Cities emerge as centers of culture and power, and people learn to do jobs that do not involve agriculture. For example, some ancient Egyptians learned to be scribes, people whose job was to write and keep records. As Egyptian civilization grew more complex, people took on jobs other than that of a farmer or scribe. Some skilled artisans erected stone or brick houses and temples. Other artisans made pottery, incense, mats, furniture, linen clothing, sandals, or jewelry. A few Egyptians traveled to the upper Nile to trade with other Africans. These traders took Egyptian products such as scrolls, linen, gold, and jewelry. They brought back exotic woods, animal skins, and live beasts. As Egypt grew, so did its need to organize. Egyptians created a government that divided the empire into 42 provinces. Many officials worked to keep the provinces running smoothly. Egypt also created an army to defend itself. One of the highest jobs in Egypt was to be a priest. Priests followed formal rituals and took care of the temples. Before entering a temple, a priest bathed and put on special linen garments and white sandals. Priests cleaned the sacred statues in temples, changed their clothes, and even fed them meals. Together, the priests and the ruler held ceremonies to please the gods. Egyptians believed that if the gods were angry, the Nile would not flood. As a result, crops would not grow, and people would die. So the ruler and the priests tried hard to keep the gods happy. By doing so, they hoped to maintain the social and political order. Slaves were at the bottom of society. In Egypt, people became slaves if they owed a debt, committed a crime, or were captured in war. Egyptian slaves were usually freed after a period of time. One exception was the slaves who had to work in the mines. Many died from the exhausting labor. Egypt was one of the best places in the ancient world to be a woman. Unlike other ancient African cultures, in Egyptian society men and women had fairly equal rights. For example, they could both own and manage their own property. The main job of most women was to care for their children and home, but some did other jobs too. Some women wove cloth. Others worked with their husbands in fields or workshops. Some women, such as Queen Tiy, even rose to important positions in the government. Children in Egypt played with toys such as dolls, animal figures, board games, and marbles. Their parents made the toys from wood or clay. Boys and girls also played rough physical games with balls made of leather or reeds. Boys and some girls from wealthy families went to schools run by scribes or priests. Most other children learned their parents’ jobs. Almost all Egyptians married when they were in their early teens. As in many ancient societies, much of the knowledge of Egypt came about as priests studied the world to find ways to please the gods. Other advances came about because of practical discoveries. Egyptian priests studied the sky as part of their religion. About 5,000 years ago, they noticed that a star now called Sirius appeared shortly before the Nile began to flood. The star returned to the same position in 365 days. Based on that, Egyptians developed the world’s first practical calendar. The Egyptians developed some of the first geometry. Each year the Nile’s floods washed away land boundaries. To restore property lines, surveyors measured the land by using ropes that were knotted at regular intervals. Geometric shapes such as squares and triangles were sacred to Egyptians. Architects used them in the design of royal temples and monuments. Egyptian doctors often prepared dead bodies for burial, so they knew the parts of the body. That knowledge helped them perform some of the world’s first surgery. Some doctors specialized in using medicines made of herbs. Egyptian medicine was far from perfect. Doctors believed that the heart controlled thought and the brain circulated blood, which is the opposite of what is known now. Some Egyptian treatments would raise eyebrows today. One “cure” for an upset stomach was to eat a hog’s tooth crushed inside sugar cakes! Beginning about 3000 B.C., Egyptians developed a writing system using hieroglyphs. Hieroglyphs Hieroglyphs are pictures that stand for different words or sounds. Early Egyptians created a hieroglyphic system with about 700 characters. Over time the system grew to include more than 6,000 symbols. The Egyptians also developed a paperlike material called papyrus papyrus from a reed of the same name. Egyptians cut the stems into strips, pressed them, and dried them into sheets that could be rolled into scrolls. Papyrus scrolls were light and easy to carry. With them, Egyptians created some of the first books. Legend says a king named Narmer united Upper and Lower Egypt. Some historians think Narmer actually represents several kings who gradually joined the two lands. After Egypt was united, its ruler wore the Double Crown. It combined the red Crown of Lower Egypt with the white Crown of Upper Egypt. The first dynasty of the Egyptian empire began about 2925 B.C. A dynasty is a line of rulers from the same family. When a king died, one of his children usually took his place as ruler. The order in which members of a royal family inherit a throne is called the succession. More than 30 dynasties ruled ancient Egypt. Historians divide ancient Egyptian dynasties into the Old Kingdom, the Middle Kingdom, and the New Kingdom. The Old Kingdom started about 2575 B.C., when the Egyptian empire was gaining strength. The king of Egypt became known as the pharaoh pharaoh. The word pharaoh meant “great house,” and it was originally used to describe the king’s palace. Later it became the title of the king himself. The pharaoh ruled from the capital city of Memphis. The ancient Egyptians thought the pharaoh was a child of the gods and a god himself. Egyptians believed that if the pharaoh and his subjects honored the gods, their lives would be happy. If Egypt suffered hard times for a long period, the people blamed the pharaoh for angering the gods. In such a case, a rival might drive him from power and start a new dynasty. Because the pharaoh was thought to be a god, government and religion were not separate in ancient Egypt. Priests had much power in the government. Many high officials were priests. The first rulers of Egypt were often buried in an underground tomb topped by mud brick. Soon, kings wanted more permanent monuments. They replaced the mud brick with a small pyramid of brick or stone. A pyramid is a structure shaped like a triangle, with four sides that meet at a point. About 2630 B.C., King Djoser built a much larger pyramid over his tomb. It is called a step pyramid because its sides rise in a series of giant steps. It is the oldest-known large stone structure in the world. About 80 years later, a pharaoh named Khufu decided he wanted a monument that would show the world how great he was. He ordered the construction of the largest pyramid ever built. Along its base, each side was about 760 feet long. The core was built from 2.3 million blocks of stone. Building the Great Pyramid was hard work. Miners cut the huge blocks of stone using copper saws and chisels. These tools were much softer than the iron tools developed later. Other teams of workers pulled the stone slabs up long, sloping ramps to their place on the pyramid. Near the top of the pyramid, the ramps ended. Workers dragged each heavy block hundreds of feet and then set it in place. Farmers did the heavy labor of hauling stone during the season when the Nile flooded their fields. Skilled stonecutters and overseers worked year-round. The Great Pyramid took nearly 20 years to build. An estimated 20,000 Egyptians worked on it. A city called Giza was built for the pyramid workers and the people who fed, clothed, and housed them. Eventually, Egyptians stopped building pyramids. One reason is that the pyramids drew attention to the tombs inside them. Grave robbers broke into the tombs to steal the treasure buried with the pharaohs. Sometimes they also stole the mummies. Egyptians believed that if a tomb was robbed, the person buried there could not have a happy afterlife. During the New Kingdom, pharaohs began building more secret tombs in an area called the Valley of the Kings. The burial chambers were hidden in mountains near the Nile. This way, the pharaohs hoped to protect their bodies and treasures from robbers. Both the pyramids and later tombs had several passageways leading to different rooms. This was to confuse grave robbers about which passage to take. Sometimes relatives, such as the queen, were buried in the extra rooms. Tombs were supposed to be the palaces of pharaohs in the afterlife. Mourners filled the tomb with objects ranging from food to furniture that the mummified pharaoh would need. Some tombs contained small statues that were supposed to be servants for the dead person. Egyptian artists decorated royal tombs with wall paintings and sculptures carved into the walls. Art was meant to glorify both the gods and the dead person. A sculpture of a dead pharaoh had “perfect” features, no matter how he really looked. Artists also followed strict rules about how to portray humans. Paintings showed a person’s head, arms, and legs from the side. They showed the front of the body from the neck down to the waist. Wall paintings showed pharaohs enjoying themselves so they could have a happy afterlife. One favorite scene was of the pharaoh fishing in a papyrus marsh. Warlike kings were often portrayed in battle. Scenes might also show people providing for the needs of the dead person. Such activities included growing and preparing food, caring for animals, and building boats. As hard as the pharaohs tried to hide themselves, robbers stole the treasures from almost every tomb. Only a secret tomb built for a New Kingdom pharaoh was ever found with much of its treasure untouched. The dazzling riches found in this tomb show how much wealth the pharaohs spent preparing for the afterlife. By about 2130 B.C., Egyptian kings began to lose their power to local rulers of the provinces. For about 500 more years, the kings held Egypt together, but with a much weaker central government. This period of Egyptian history is called the Middle Kingdom. Rulers during the Middle Kingdom also faced challenges from outside Egypt. A nomadic people called the Hyksos invaded Egypt from the northeast. Their army conquered by using better weapons and horse-drawn chariots, which were new to Egyptians. After about 100 years, the Egyptians drove out the Hyksos and began the New Kingdom.
"""


bio_text="""Photosynthesis
Photosynthesis is the process by which plants, some bacteria and some protistans use the energy
from sunlight to produce glucose from carbon dioxide and water.
 This glucose can be converted into
pyruvate which releases adenosine triphosphate (ATP) by cellular respiration. Oxygen is also formed.

The conversion of usable sunlight energy into chemical energy is associated with the action of the
green pigment chlorophyll.
Chlorophyll is a complex molecule. Several modifications of chlorophyll occur among plants and other
photosynthetic organisms.
 All photosynthetic organisms have chlorophyll

"""
"""
Chlorophyll
All chlorophylls have:
• a lipid-soluble hydrocarbon tail (C20H39 -)
• a flat hydrophilic head with a magnesium ion at its centre; different chlorophylls have different
side-groups on the head
The tail and head are linked by an ester bond.
Leaves and leaf structure
Plants are the only photosynthetic organisms to have leaves (and not all plants have leaves). A leaf
may be viewed as a solar collector crammed full of photosynthetic cells.
The raw materials of photosynthesis, water and carbon dioxide, enter the cells of the leaf, and the
products of photosynthesis, sugar and oxygen, leave the leaf.
Water enters the root and is transported up to the leaves through specialized plant cells known as
xylem vessels. Land plants must guard against drying out and so have evolved specialized structures
known as stomata to allow gas to enter and leave the leaf. Carbon dioxide cannot pass through the
protective waxy layer covering the leaf (cuticle), but it can enter the leaf through the stoma (the
singular of stomata), flanked by two guard cells. Likewise, oxygen produced during photosynthesis
can only pass out of the leaf through the opened stomata. Unfortunately for the plant, while these
gases are moving between the inside and outside of the leaf, a great deal of water is also lost.
Cottonwood trees, for example, will lose 100 gallons (about 450 dm3) of water per hour during hot
desert days.
The structure of the chloroplast and photosynthetic membranes
The thylakoid is the structural unit of photosynthesis. Both photosynthetic prokaryotes and eukaryotes
have these flattened sacs/vesicles containing photosynthetic chemicals. Only eukaryotes have
chloroplasts with a surrounding membrane.
Thylakoids are stacked like pancakes in stacks known collectively as grana. The areas between
grana are referred to as stroma. While the mitochondrion has two membrane systems, the chloroplast
has three, forming three compartments.
Structure of a chloroplast
Stages of photosynthesis
When chlorophyll a absorbs light energy, an electron gains energy and is 'excited'. The excited
electron is transferred to another molecule (called a primary electron acceptor). The chlorophyll
molecule is oxidized (loss of electron) and has a positive charge. Photoactivation of chlorophyll a
results in the splitting of water molecules and the transfer of energy to ATP and reduced nicotinamide
adenine dinucleotide phosphate (NADP).
The chemical reactions involved include:
• condensation reactions - responsible for water molecules splitting out, including
phosphorylation (the addition of a phosphate group to an organic compound)
• oxidation/reduction (redox) reactions involving electron transfer
Photosynthesis is a two stage process.
The light dependent reactions, a light-dependent series of reactions which occur in the grana, and
require the direct energy of light to make energy-carrier molecules that are used in the second
process:
• light energy is trapped by chlorophyll to make ATP (photophosphorylation)
• at the same time water is split into oxygen, hydrogen ions and free electrons:
2H2O 4H+ + O2 + 4e- (photolysis)
• the electrons then react with a carrier molecule nicotinamide adenine dinucleotide phosphate
(NADP), changing it from its oxidised state (NADP+) to its reduced state (NADPH):
NADP+ + 2e- + 2H+ NADPH + H+
The light-independent reactions, a light-independent series of reactions which occur in the stroma
of the chloroplasts, when the products of the light reaction, ATP and NADPH, are used to make
carbohydrates from carbon dioxide (reduction); initially glyceraldehyde 3-phosphate (a 3-carbon atom
molecule) is formed.
The light-dependent reactions
When light energy is absorbed by a chlorophyll molecule its electrons gain energy and move to higher
energy levels in the molecule (photoexcitation). Sufficient energy ionises the molecule, with the
electron being 'freed' leaving a positively charged chlorophyll ion. This is called photoionisation.
In whole chloroplasts each chlorophyll molecule is associated with an electron acceptor and
an electron donor. These three molecules make up the core of a photosystem. Two electrons from
a photoionised chlorophyll molecule are transferred to the electron acceptor. The positively charged
chlorophyll ion then takes a pair of electrons from a neighbouring electron donor such as water.
An electron transfer system (a series of chemical reactions) carries the two electrons to and fro across
the thylakoid membrane. The energy to drive these processes comes from two photosystems:
• Photosystem II (PSII) (P680)
• Photosystem I (PSI) (P700)
It may seem confusing, but PSII occurs before PSI. It is named because it was the second to be
discovered and hence named second.
The energy changes accompanying the two sets of changes make a Z shape when drawn out. This is
why the electron transfer process is sometimes called the Z scheme. Key to the scheme is that
sufficient energy is released during electron transfer to enable ATP to be made from ADP and
phosphate.
A condensation reaction has led to phosphorylation.
Non-cyclic phosphorylation (the Z scheme)
Both adenosine triphosphate (ATP) and NADPH are produced.
In the first photosystem (Photosystem II, PSII):
• photoionisation of chlorophyll transfers excited electrons to an electron acceptor
• photolysis of water (an electron donor) produces oxygen molecules, hydrogen ions and
electrons, and the latter are transferred to the positively-charged chlorophyll
• the electron acceptor passes the electrons to the electron transport chain; the final acceptor is
photosystem PSI
• further absorbed light energy increases the energy of the electrons, sufficient for the reduction
of NADP+ to NADPH
The oxidised form of nicotinamide adenine dinucleotide phosphate (NADP+)
The reduced form of nicotinamide adenine dinucleotide phosphate (NADPH)
Chemiosmosis and ATP synthesis
The components of non-cyclic phosphorylation are found in the thylakoid membranes of the
chloroplast. Electrons passing through the transport chain provide energy to pump H+ ions from the
stroma, across the thylakoid membrane into the thylakoid compartment. H+ ions are more
concentrated in the thylakoid compartment than in the stroma. We say there is an electrochemical
gradient. H+ ions diffuse from the high to the low regions of concentration. This drives the production
of ATP.
Chemiosmosis as it operates in photophosphorylation within a chloroplast
Cyclic phosphorylation
The net effect of non-cyclic phosphorylation is to pass electrons from water to NADP. Energy
released enables the production of ATP. But much more ATP is needed to drive the light-independent
reactions.
This extra energy is obtained from cyclic phosphorylation. This involves only Photosystem I which
generates excited electrons. These are transferred to the electron transport chain between PSII and
PSI, rather than to NADP+ and so no NADPH is formed. The cycle is completed by electrons being
transported back to PSI by the electron transport system.
The light-independent reactions
In the Light-Independent Process (the Dark reaction) carbon dioxide from the atmosphere (or water
for aquatic/marine organisms) is captured and modified by the addition of hydrogen to form
carbohydrates. The incorporation of carbon dioxide into organic compounds is known as carbon
fixation. The energy for this comes from the first phase of the photosynthetic process. Living systems
cannot directly utilize light energy, but can, through a complicated series of reactions, convert it into
C-C bond energy that can be released by glycolysis and other metabolic processes.
Carbon dioxide combines with a five-carbon sugar, ribulose 1,5-biphosphate (RuBP). A six-carbon
sugar forms but is unstable. Each molecule breaks down to form two glycerate 3-phosphate (GP)
molecules.
These glycerate 3-phosphate (GP) molecules are phosphorylated by ATP into glycerate diphosphate
molecules.
These are reduced by NADPH to two molecules of glyceraldehyde 3-phosphate (GALP).
Of each pair of GALP molecules produced:
• one molecule is the initial end product of photosynthesis; it is quickly converted to glucose
and other carbohydrates, lipids or amino acids
• one molecule forms RuBP through a series of chemical reactions
The first steps in the Calvin cycle
The first stable product of the Calvin Cycle is phosphoglycerate (PGA), a 3-C chemical. The energy
from ATP and NADPH energy carriers generated by the photosystems is used to phosphorylate the
PGA. Eventually there are 12 molecules of glyceraldehyde phosphate (also known as
phosphoglyceraldehyde or PGAL, a 3-C), two of which are removed from the cycle to make a
glucose. The remaining PGAL molecules are converted by ATP energy to reform six RuBP
molecules, and thus start the cycle again.
Summary of stages of photosynthesis
Factors affecting the rate of photosynthesis
The main factors are light intensity, carbon dioxide concentration and temperature, known as limiting
factors.
As light intensity increases, the rate of the light-dependent reaction, and therefore photosynthesis
generally, increases proportionately. As light intensity is increased however, the rate of
photosynthesis is eventually limited by some other factor. Chlorophyll a is used in both photosystems.
The wavelength of light is also important. PSI absorbs energy most efficiently at 700 nm and PSII at
680 nm. Light with a high proportion of energy concentrated in these wavelengths will produce a high
rate of photosynthesis.
An increase in the carbon dioxide concentration increases the rate at which carbon is incorporated
into carbohydrate in the light-independent reaction and so the rate of photosynthesis generally
increases until limited by another factor.
Photosynthesis is dependent on temperature. It is a reaction catalysed by enzymes. As the enzymes
approach their optimum temperatures the overall rate increases. Above the optimum temperature the
rate begins to decrease until it stops.
Test your knowledge
Take quiz on Photosynthesis"""
# print(len(paragrs))

# test = QA_main()?
# print(_getQuestion("deep learning is a part of machine learning","machine"))
# import time
# start = time.time()
# engine = create_engine("mysql+pymysql://dbausermcg:om21m24s20d#@@0.0.0.0:3306/hzdb")
# cnx = engine.connect()
# meta = MetaData()
# print(meta.tables.keys())

# # # drop_table("qa")

# print(engine.table_names())
# import pandas as pd
# # print(time.time()-start)
# for table in engine.table_names():
#     # table_df = pd.read_sql_table(table,con=engine)
#     # print(table_df.info())
#     print("==========================")
#     print(table)
#     print(pd.read_sql_query("select * from %s"%table,engine))


# Giri - created rest end point
@app.route('/quesAnws', methods=['POST'])
def quesAnws():
    print("Entered quesAnws")
    content = request.get_json()
    print("content :: ", content)
    # assessment info
    assessment_name = content['assessment']['name']
    course_id = content['assessment']['course_id']
    number_of_questions = content['assessment']['number_of_questions']
    assessment_type = content['assessment']['assessment_type']
    is_active = content['assessment']['is_active']

    # source
    text_corpus = content['text_corpus']

    try:
        #initialize mysql db connection
        connection = mysql.connector.connect(host='localhost',
                                            database='hzdb',
                                            user='dbausermcg',
                                            password='om21m24s20d#@')
        cursor = connection.cursor()
        print("MySQL connection is created")

        #verify the assessment is not created already
        select_stmt = "SELECT * FROM assessment WHERE name = %(assessment_name)s"
        cursor.execute(select_stmt, { 'assessment_name': assessment_name })
        records = cursor.fetchall()
        print("The records -> " , records)
        if len(records) > 0:
            return jsonify({"error": "Assessment - {}, already created".format(assessment_name)}), 400

        # insert assessment record
        sql_update_query ="""insert into assessment (name, course_id, number_of_questions, assessment_type, is_active) value(%s,%s,%s,%s,%s);"""
        inputData = (assessment_name, course_id, number_of_questions, assessment_type, True)
        cursor.execute(sql_update_query, inputData)
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
    finally:
        if (connection.is_connected()):
            connection.commit()
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    main_qas = QA_main()
    print("***** qmai working")
    main_qas.start_creating(assessment_name,text_corpus,paragraphs_type = "dense",mx_num_mcq =2,mx_num_normalqa=1)
    # fetch QA from db api
    url = "http://0.0.0.0:6006/get_all_QAs_by_assessment_name?assessment_name=" + assessment_name + "&skip=0&limit=100"
    print("url : ", url)
    payload={}
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload)
    print(" -- questions and answers fetched from database -- ")
    print(response.text)
    return jsonify({"status" : "Completed","QA":response.json(), "error":""}), 201

def convertTuple(tup):
    if all(tup):
        str =  ''.join(tup)
        return str
    else:
        return ""
@app.route('/quesAnws2', methods=['POST'])
def quesAnws2():
    print("Entered quesAnws2")
    content = request.get_json()
    print("content :: ", content)

    # assessment info
    assessment_name = content['assessment']['name']
    course_id = content['assessment']['course_id']
    number_of_questions = content['assessment']['number_of_questions']
    assessment_type = content['assessment']['assessment_type']
    is_active = content['assessment']['is_active']

    try:
        #initialize mysql db connection
        connection = mysql.connector.connect(host='localhost',
                                            database='hzdb',
                                            user='dbausermcg',
                                            password='om21m24s20d#@')

        cursor = connection.cursor()

        #verify the assessment is not created already
        select_stmt = "SELECT * FROM assessment WHERE name = %(assessment_name)s"
        cursor.execute(select_stmt, { 'assessment_name': assessment_name })
        records = cursor.fetchall()
        print("The records -> " , records)
        if len(records) > 0:
            return jsonify({"error": "Assessment - {}, already created".format(assessment_name)}), 400
        # create assessment
        sql_update_query ="""insert into assessment (name, course_id, number_of_questions, assessment_type, is_active) value(%s,%s,%s,%s,%s);"""
        inputData = (assessment_name, course_id, number_of_questions, assessment_type, True)
        cursor.execute(sql_update_query, inputData)

    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
    finally:
        if (connection.is_connected()):
            connection.commit()
            cursor.close()
            connection.close()
            print("MySQL connection is closed")
    print("Created assessment entry in database, now fetching the text corpus from publay table..")
    file_id = content['file_id']
    page_start = content['page_start']
    page_end = content['page_end']
    data = ""
    try:
        connection = mysql.connector.connect(host='localhost',
                                             database='hertzocr',
                                             user='dbausermcg',
                                             password='om21m24s20d#@')
        if connection.is_connected():
            db_Info = connection.get_server_info()
            print("Connected to MySQL Server version ", db_Info)
            cursor = connection.cursor()
            select_query = "select passage from publay_table where Page_number>=%s and Page_number <=%s and file_id=%s;"
            cursor.execute(select_query,(page_start, page_end, file_id))
            records = cursor.fetchall()
            for record in records:
               print("record ->  " , record)
               data = data + "\n" + convertTuple(record).replace("\n","")
            print("Collected data from publay table :: " + data)
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None
    finally:
        if (connection.is_connected()):
            connection.commit()
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    text_corpus = data
    #text_corpus = content['text_corpus']
    main_qas = QA_main()
    print("***** qmai working")
    main_qas.start_creating(assessment_name,text_corpus,paragraphs_type = "dense",mx_num_mcq =2,mx_num_normalqa=1, )
    return "working .."

if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.run(debug = False, host = '0.0.0.0', port=8086)
# Giri - created rest end point


# if __name__ =="__main__":
#     main_qas = QA_main()
#     assessment_name = "Bio-photosynthesis-demo"
#     print("***** qmai working")
#     main_qas.start_creating(assessment_name,bio_text,paragraphs_type = "dense",mx_num_mcq =2,mx_num_normalqa=1)
