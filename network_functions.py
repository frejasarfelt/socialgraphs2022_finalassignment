# Import libraries
import re
import os
import math
import nltk
# import codecs
# import pathlib
# import operator
import requests
# import xmltodict
import numpy as np
import networkx as nx
import pandas as pd  # 1.5.1
from nltk import FreqDist
from fa2 import ForceAtlas2
# from statistics import mode
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
# from operator import itemgetter
# from wordcloud import WordCloud
# from collections import Counter
from nltk.corpus import stopwords

# import xml.etree.ElementTree as ET
# import matplotlib.colors as mcolors
# import community #install python-louvain
# from nltk.stem import WordNetLemmatizer
from urllib.parse import urljoin
# import webbrowser
from form_extractor import get_all_forms, get_form_details, session

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
pd.set_option('max_colwidth', 1000)
seed = 17


def convert_course_id(courseid):
    """
    Converting a course id to a course code
    """
    link = "http://kurser.dtu.dk/coursewebservicev2/course.asmx/GetCourseFromCourseId?courseId={}".format(courseid)
    response = requests.request("GET", link)
    course_code = pd.read_xml(response.text, converters={'CourseCode': str})['CourseCode'][0]

    if len(course_code) == 4:
        course_code = '0' + course_code

    return course_code


def get_course_prerequisites(course_code):
    """
    Collecting all prerequisites for a course given a course code
    """
    url = "https://kurser.dtu.dk/coursewebservicev2/course.asmx/GetCourse?courseCode=" + course_code + "&yearGroup=2022/2023"

    response = requests.request("GET", url)
    try:
        df = pd.read_xml(response.text, xpath=".//Course/Qualified_Prerequisites/DTU_CoursesTxt")
        return df["Txt"].to_string(index=False)
    except ValueError:
        return ""


def create_histogram(graph, degree=None):
    """
    Plotting function of either in- or out-degrees or all degrees
    for given network.
    """
    if degree == 'in':
        degree_sequence = sorted((d for n, d in graph.in_degree()), reverse=True)
    elif degree == 'out':
        degree_sequence = sorted((d for n, d in graph.out_degree()), reverse=True)
    elif degree == None:
        degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

    dmax = max(degree_sequence)
    dmin = min(degree_sequence)
    v = range(dmin, dmax + 1, 1)

    # creating hist with bins=v
    hist = np.histogram(degree_sequence, bins=v)
    # plt.hist(degree_sequence, bins = v)

    y = list(hist[0])
    x = list(hist[1][:-1])

    return x, y, degree_sequence, v


def get_course_info(dep_id, G, dtu_department):
    """
    Adds department and course type/programme attribute to nodes in network G.
    It does not return anything, but G is updated
    """
    url = "https://kurser.dtu.dk/coursewebservicev2/course.asmx/GetCoursesByDepartment?department=" + dep_id + "&catalogversion=2022/2023"  # 02809
    response = requests.request("GET", url)
    try:
        df = pd.read_xml(response.text, xpath=".//CourseList")
        df = df.reset_index()
        courses = df['CourseCode']
        programmes = df['programme']
        points = df['Point']

        # Add department and programme attributes to nodes in G
        department, programme, point = "", "", ""
        i = 0
        no_attributes = []
        for c in courses:
            c = str(c)

            try:
                # Department attribute
                department = list(dtu_department.keys())[list(dtu_department.values()).index(dep_id)]
                if department != "":
                    G.nodes()[c]['Department'] = department
                else:
                    print(course, "it not related to any department.")
                    G.nodes()[c]['Department'] = None
            except KeyError as e:
                no_attributes.append(c)
                # print("Could not add department attribute to course", e, sep=":")

            try:
                # Programme attribute
                programme = programmes[i]
                if programme != "":
                    G.nodes()[c]['Course type'] = programme
                else:
                    print(course, "it not related to any programme.")
                    G.nodes()[c]['Course type'] = None
            except KeyError as e:
                no_attributes.append(c)
                # print("Could not add course type attribute to course", e, sep=":")

            try:
                # Course size
                point = points[i]
                if point != "":
                    G.nodes()[c]['Points(ECTS)'] = point
                else:
                    print(course, "does not have points described.")
                    G.nodes()[c]['Points(ECTS)'] = None
            except KeyError as e:
                no_attributes.append(c)
                # print("Could not add points attribute to course", e, sep=":")

            i += 1
        if no_attributes:
            return no_attributes

    except ValueError:
        pass_ = 1


def merge_txts(df):
    """
    Merging data with lists of text and remove unnecessary text.
    Return: text
    """
    txt_list = list(df.iloc[0])
    # Removing uninteresting text
    txt_list = [txt for txt in txt_list if (txt != 'en-GB' and txt != 'See the Danish version')]

    # Remove Nan from list elements
    txt_list = [x for x in txt_list if str(x) != 'nan']
    txt = ''.join(filter(lambda x: x if x is not None else '', txt_list))

    return txt


def remove_punct(txt):
    """
    Helper function removing white space and new lines and lowercase words.
    Return: text
    """
    txt = txt.lower()
    txt = txt.replace("\n", " ")
    txt = re.sub(r'[^\w\s]', ' ', txt)

    return txt


def get_course_txt(course_code):
    """
    Takes a course code as input, extracts course content using API.
    Return: Saves content as .txt file
    """
    path_for_saving = 'course_content/{}.txt'.format(course_code)

    if not os.path.exists(path_for_saving):
        url = "https://kurser.dtu.dk/coursewebservicev2/course.asmx/GetCourse?courseCode=" + course_code + "&yearGroup=2022/2023"

        try:
            response = requests.request("GET", url)
        except (ConnectionError, requests.exceptions.SSLError) as e:  # This is the correct syntax
            print(e)
            r = "No response"

        try:
            df = pd.read_xml(response.text, xpath=".//Course/Txt")
            df = df[df['Lang'] == 'en-GB']
            txt = merge_txts(df)
            txt = remove_punct(txt)

            # Save text file
            with open(path_for_saving, 'w', encoding="utf8") as f:
                f.write(txt)
            return txt
        except ValueError:
            return ""


def sentiment(G, node, node_content, lab_MT, dict_labMT):
    """
    Takes names of node e.g. course code and it's name of .txt file with content
    and adds sentiment score as an attribute to node in graph.
    """
    # If content already downloaded via API
    if node in node_content:
        # Save text file
        path_for_open = 'course_content/{}.txt'.format(node)
        with open(path_for_open, 'r', encoding='utf-8') as f:
            text = f.read().rstrip()

    # Else use API to read URL
    else:
        text = get_course_txt(node)
        print("Getting content from URL for course", node)

    text_dict = dict(FreqDist(w for w in text.split()))
    labMT_words = lab_MT['word']

    words_total = 0
    happiness_sum = 0
    for word in labMT_words:
        if word in text_dict:
            words_total += text_dict[word]
            happiness_sum += text_dict[word] * dict_labMT[word]

    # Error handling if amount of words is zero
    if words_total != 0:
        avg_happiness = happiness_sum / words_total
        # Append sentiment score to node as attribute
        G.nodes()[node]['Sentiment'] = round(avg_happiness, 3)
    else:
        avg_happiness = None
        # Append sentiment score to node as attribute
        G.nodes()[node]['Sentiment'] = avg_happiness


# return round(avg_happiness, 3)


def semester_evaluations(study_id, course_period):
    """
    Code inspiration from https://www.thepythoncode.com/code/extracting-and-submitting-web-page-forms-in-python
    Write .html file with course evaluation content.
    """

    if not os.path.exists('Evaluation_semesters/{}.html'.format(study_id)):
        # Get the URL
        url = "https://evaluering.dtu.dk/CourseSearch"
        all_forms = get_all_forms(url)
        # Get the first form (edit this as you wish)
        # first_form = get_all_forms(url)[0]
        for i, f in enumerate(all_forms, start=1):
            form_details = get_form_details(f)

        choice = 2
        # Extract all form details
        form_details = get_form_details(all_forms[choice - 1])
        # The data body we want to submit
        data = {}
        for input_tag in form_details["inputs"]:
            if input_tag["type"] == "hidden":
                # If it's hidden, use the default value
                data[input_tag["name"]] = input_tag["value"]
            elif input_tag["type"] == "select":
                choice = course_period
                try:
                    choice = int(choice)
                except:
                    # Choice invalid, take the default
                    value = input_tag["value"]
                else:
                    value = input_tag["values"][choice - 1]
                data[input_tag["name"]] = value
            elif input_tag["type"] != "submit":
                # All others except submit
                value = study_id
                data[input_tag["name"]] = value

        # Join the url with the action (form request URL)
        url = urljoin(url, form_details["action"])
        if form_details["method"] == "post":
            res = session.post(url, data=data)
        elif form_details["method"] == "get":
            res = session.get(url, params=data)

        # The below code is only for replacing relative URLs to absolute ones
        soup = BeautifulSoup(res.content, "html.parser")
        for link in soup.find_all("link"):
            try:
                link.attrs["href"] = urljoin(url, link.attrs["href"])
            except:
                pass
        for script in soup.find_all("script"):
            try:
                script.attrs["src"] = urljoin(url, script.attrs["src"])
            except:
                pass
        for img in soup.find_all("img"):
            try:
                img.attrs["src"] = urljoin(url, img.attrs["src"])
            except:
                pass
        for a in soup.find_all("a"):
            try:
                a.attrs["href"] = urljoin(url, a.attrs["href"])
            except:
                pass

        # write the page content to a file
        name_of_page = "Evaluation_semesters/ev_{}.html".format(study_id)

        f = open(name_of_page, "w")
        f.write(str(soup))
        f.close()


def get_evaluation(file):
    """
    Extract course evaluations.
    Return: Match from regex search.
    """
    soup = BeautifulSoup(open(file), "html.parser")
    try:
        last_option = soup.find_all("div", {"class": "ResultsPublicRow"})[-1].a
        result = re.search(r'href="(.+)"', str(last_option))
        get_group = result.group(1)

    except:
        get_group = "no evaluations!"

    return get_group


def average_happiness(url):
    """
    Calculates average evaluation happiness scores and add as an attribute to node.
    """
    if url != 'no evaluations!':

        # url = 'https://evaluering.dtu.dk/kursus/01018/242429'
        response = requests.request("GET", url)
        soup = BeautifulSoup(response.text, 'html.parser')
        evaluations = soup.find_all("div", {"class": "ResultCourseModelWrapper grid_6 clearmarg"})

        happy_scores = []

        for box in evaluations:

            judgement = box.find_all("div", {"class": "FinalEvaluation_Result_OptionColumn grid_1 clearmarg"})
            votes = box.find_all("div", {"class": "Answer_Result_Background"})
            all_votes = box.find_all("div", {"class": "CourseSchemaResultFooter grid_6 clearmarg"})[0]

            # find total votes
            result = re.search(r'<span>(\d+).+<\/span>', str(all_votes))
            get_total_votes = int(result.group(1))

            good_votes = 0
            for i in range(len(judgement)):
                try:
                    result = re.search(r'<div.+>(.+)<\/div>', str(judgement[i]))
                    get_judge = result.group(1)

                    # find good votes
                    if get_judge in ["Helt enig", "Enig"]:
                        result = re.search(r'<span>(.+)<\/span>', str(votes[i]))
                        get_vote = result.group(1)
                        good_votes += int(get_vote)
                except:
                    good_votes = 0

            # only look at happiness for evaluation questions of type: 'Jeg synes, at…'
            if good_votes != 0:
                # find percentage of happiness
                happy = good_votes / get_total_votes * 100
                happy_scores.append(happy)

        if len(happy_scores) != 0:
            average_happy = sum(happy_scores) / len(happy_scores)
            return average_happy


def fa(graph, node_color, node_department="", export=""):
    """
    Creates a plot of the network with ForceAtlas2
    """

    # Force Atlas plot colored by attribute: 'Department'
    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,
        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED
        # Tuning
        scalingRatio=2.0,
        strongGravityMode=False,
        gravity=1.0,
        # Log
        verbose=True)
    positions = forceatlas2.forceatlas2_networkx_layout(graph, pos=None, iterations=2000)

    # Set node size by edges
    degrees = nx.degree(graph)

    # plt.figure(figsize=(10, 10))
    if node_department == "":
        nx.draw_networkx_nodes(graph, positions, node_size=[v * 10 for v in dict(degrees).values()], node_color=node_color,
                               alpha=0.3)
    else:
        nx.draw_networkx_nodes(graph, positions, node_size=[v * 10 for v in dict(degrees).values()], node_color=node_color,
                               alpha=0.3, label = node_department)
    nx.draw_networkx_edges(graph, positions, edge_color="grey", alpha=0.4, width=0.75)  # , arrows = True
    plt.axis('off')
    if export != "":
        plt.savefig(export, bbox_inches='tight')
        plt.show()
        plt.close()
    else:
        # plt.show()
        pass_ = 0


def computeTF(wordDict, community):
    """
    Computes TF for each word in community
    """
    tfDict = {}
    communityCount = len(community)
    for word, count in wordDict.items():
        tfDict[word] = round(count / float(communityCount), 5)
    return tfDict


def computeIDF(documents):
    """
    Computes the IDF for each word in the community
    """
    N = len(documents)

    idfDict = documents  # dict.fromkeys(documents.keys(), 0)
    # for document in documents:
    for word, val in documents.items():
        if val > 0:
            try:
                idfDict[word] += 1
            except KeyError as e:
                key = e.args[0]

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict


def computeTFIDF(tfBagOfWords, idfs):
    """
    Compute TF-IDF scores for all words in the community
    """
    tfidf = {}
    for word, val in tfBagOfWords.items():
        try:
            tfidf[word] = val * idfs[word]
        except KeyError as e:
            key = e.args[0]
    return tfidf
