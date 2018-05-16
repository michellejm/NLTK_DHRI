## Introduction to Text Analysis with Python and the Natural Language ToolKit (NLTK)
### Digital Humanities Research Institute
### The Graduate Center at CUNY | June 11, 2018
### Michelle A. McSweeney, PhD and Rachel Rakov

### Contents:

* Overview
* Setup and Installation
	* Python 
	* NLTK
	* Data
* Text As Data
* NLTK Methods with the NLTK Corpus
* Built-in Python Methods
* Making your own corpus
	* Data Cleaning
		* Types vs. Tokens
	* Input/Output
* Part-of-Speech Tagging


### Overview
This tutorial will give a brief overview of the considerations and tools involved in basic text analysis with Python. By completing this tutorial, you will have a general sense of how to turn text into data using the Python package, NLTK. You will also be able to take publicly available text files and transform them into a corpus that you can perform your own analysis on. Finally, you will have some insight into the types of questions that can be addressed with text analysis. 

### Setup and Installation
If you have not already installed the [Anaconda](https://www.anaconda.com/download/) distribution of Python 3, please do so. You will also need nltk and matplotlib to complete this tutorial. Both packages come installed with Anaconda. To check to be sure you have them, open a new Jupyter Notebook (or any IDE to run Python).

Find Anaconda on your computer, Launch a Jupyter Notebook. 

![jupyter](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/jupyter.png)

It will open in the browser. All of the directories (folders) in your home directory will appear - we'll get to that later. For now, select 'New' >> Python3 in the upper right corner.

![jupyter](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/jupyter1.png)

A blank page with an empty box should appear.

![jupyter](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/jupyter2.png)


In the box, type:

`import nltk`

`import matplotlib`

Press **Shift** + **Enter** to run the cell (or click run at the top of the page). Don't worry too much about what this is doing - that will be explained later in this tutorial. For now, we just want to make sure the packages we will need are installed.

![jupyter](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/jupyter3.png)

If nothing happens, they are installed and you are ready to move on! If you get an error message, either you have a typo or they are not installed. If it is the later, open the command line and type:

`pip install nltk`

`pip install matplotlib`


Now we need to install the nltk corpus. This is very large and may take some time if you are on a weak connection. 

In the next cell, type:

`nltk.download()` and run the cell.

The NLTK downloader should appear. Please install all of the packages. If you are short on time, focus on 'book' for this tutorial, and come back to this step. 

Yours will look a little different, but the same interface. Click on the 'all' option and then 'Download'. Once they all trun green, you can close the Downloader dialogue box.

![nltk downloader](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/nltk.png)

Return to your Jupyter Notebook and type:

`from nltk.book import *`

A list of books should appear. If this happens great! If not, return to the downloader to make sure everything is ok.

Close this Notebook without saving - the only purpose was to check if we have the appropriate packages installed.


### Text As Data

When we think of 'data', we often think of numbers, things that can be summarized, statisticized, and graphed. Rarely when I ask people "what is data?" do they respond "Moby Dick!" And yet, more and more, text is data. Whether it is Moby Dick, or every romance novel written since 1750, or today's newspaper or twitter feed, we are able to transform written (and spoken) language into data that can be quantified and visualized. 

The first step in gathering insights from texts is to create a corpus. A corpus is a collection of texts that are somehow related to each other. For example, [Donald Trump's Tweets LINK], [text messages](www.byts.commons.gc.cuny.edu) sent by bilingual young adults, newspapers from the 1850's, or [books](www.gutenberg.com) in the public domain are all corpora. There are infinitely many corpora, but often, you will want to make your own that best fits your research question. 

The route you take from here will depend on your research question. Let's say, for example, that you want to examine gender differences in writing style. Based on previous linguistic research, you suspect that male authors use more definitives than female ones. So you collect two corpora: one written by males, one written by females, and you count the number of *the*s, *this*s, and *that*s compared to the number of *a*s, *an*s, and *one*s. Maybe you find a difference, maybe you don't. We can already see that this is a relatively crude way of going about answering this question, but it is a start (more likely you'd use a *supervised classification task*, which you will learn about in the Machine Learning Tutorial). 

Maybe you want to know if newspapers contain more grammatical structure than tweets. One way to do this would be to use Part-of-Speech tagging. If newspaper articles have a higher ratio of function words (prepositions, auxiliaries, determiners, etc.) to semantic words, than tweets, then your hypothesis is confirmed. Since newspaper articles are much longer than tweets, they will clearly have higher raw numbers, but the ratios can be compared. 

Once we have a corpus (whether that is one text or millions), usually, we want to clean and normalize it. Language is messy, and created for and by people, not computers. There is a lot of grammatical information in a sentence that a computer cannot use. For example, I could say to you `The house is burning.` and you would understand me. You would also understand if I say `house burn`. The first has more information about tense, and which house in particular, but the sentiment is the same either way. 

Usually, the computer works better with the second version (depending, of course, on your research question). In going from the first to the second, we removed the stop words (*the* and *is*), and normalized (removed punctuation and case) and lemmatized what was left (*burning* becomes *burn* - though we might have stemmed this, its impossible to tell from the example). 

Now that we have a transformed corpus, we can ask the computer to count things and return 

### NLTK Methods with the NLTK Corpus

Return to the Jupyter Home Tab in your Browser (or Launch the Jupyter Notebook again), and start a New Python3 Notebook using the 'New' button in the upper right corner. 

Start by importing the NLTK library by typing 

`import nltk`

Libraries are sets of instructions that Python can use to perform specialist functions. The Python programming language is very simple by design, and can be used for a wide variety of functions. It allows users to customize it to their needs by importing packages. Packages are written by individuals and groups who see the need for a simple set of commands to do routine functions. The Natural Langauge ToolKit (NLTK) is one such library. As the name suggests, its focus is on language processing. 


We will also need the matplotlib library to help us visualize a graph

`import matplotlib`

Finally, because of a quirk of Jupyter notebooks, we need to specify that matplotlib should display its graphs in the notebook (as opposed to in a separate window), so we type this command (this is technically a Jupyter command, not Python):

`%matplotlib inline`

All three of these commands can be written in the same cell and run all at once (**Shift** + **Enter**) or in different cells. 

![imports](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/imports.png)

If nothing happens, it's all correct. 

Next we need to load all of the nltk corpora into our program. Even though we downloaded them to our computer, we need to tell Python we want to use them every time we want to use them. 

`from nltk.book import *`

All of the texts should load again. These are pre-formatted datasets that come with NLTK. We will still have to do some minor processing, but having the data in this format will save us a few steps. However, at the end of this tutorial, we will load in our own data, but these will get us started. 


* Built-in Python Methods
* Making your own corpus
	* Data Cleaning
		* Types vs. Tokens
	* Input/Output
* Part-of-Speech Tagging
