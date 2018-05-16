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

In the box that appears, type:

`import nltk`
`import matplotlib`

Press **Shift** + **Enter** to run the cell (or click run at the top of the page).

![jupyter](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/jupyter2.png)

If nothing happens, they are installed and you are ready to move on! If you get an error message, either you have a typo or they are not installed. If it is the later, open the command line and type:

`pip install nltk`
`pip install matplotlib`

