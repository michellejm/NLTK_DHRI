## Introduction to Text Analysis with Python and the Natural Language ToolKit (NLTK)
### Digital Humanities Research Institute
#### The Graduate Center at CUNY | June 11, 2018
#### Michelle A. McSweeney, PhD and Rachel Rakov


### Contents:

* Overview
* Setup and Installation
* Text As Data
* NLTK Methods with the NLTK Corpus
* Built-in Python Methods
* Making your own corpus

* Part-of-Speech Tagging

*Before we get started, please clone or download the [GitHub Repository](https://github.com/michellejm/NLTK_DHRI/) for this tutorial*
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

#### Corpora

The first step in gathering insights from texts is to create a corpus. A corpus is a collection of texts that are somehow related to each other. For example, the [Corpus of Contemporary American English](https://corpus.byu.edu/coca/), [Donald Trump's Tweets](http://www.trumptwitterarchive.com/), [text messages](www.byts.commons.gc.cuny.edu) sent by bilingual young adults, [digitized newspapers](https://chroniclingamerica.loc.gov/newspapers/), or [books](https://www.gutenberg.org/) in the public domain are all corpora. There are infinitely many corpora, and sometimes, you will want to make your own that best fits your research question. 

The route you take from here will depend on your research question. Let's say, for example, that you want to examine gender differences in writing style. Based on previous linguistic research, you suspect that male authors use more definitives than female ones. So you collect two corpora: one written by males, one written by females, and you count the number of *the*s, *this*s, and *that*s compared to the number of *a*s, *an*s, and *one*s. Maybe you find a difference, maybe you don't. We can already see that this is a relatively crude way of going about answering this question, but it is a start (more likely you'd use a *supervised classification task*, which you will learn about in the Machine Learning Tutorial). 

There has been some research about how the [linguistic complexity of written language](http://science.sciencemag.org/content/sci/331/6014/176.full.pdf) has decreased, and we want to know if short-form platforms are emblematic of the problem. One way to do this would be to use Part-of-Speech tagging. Part-of-Speech tagging identifies the category of words. NLTK uses the [Penn Tree Bank Tag Set](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). This is a very detailed tag list that goes far beyond just nouns, verbs, and adjectives, but gives insight into different types of nouns, prepositions, and verbs as well. Virtually all POS taggers will create a list of (word, POS) pairs. If newspaper articles have a higher ratio of function words (prepositions, auxiliaries, determiners, etc.) to semantic words (nouns, verbs, adjectives), than tweets, then your hypothesis is confirmed. It's important to note here that either ratios or otherwise normalized data should be considered. Because of the way that language works (function words are often repeated, etc.), a sample of 100 words will have more unique words than a sample of 1,000. Therefore, to compare different data types (articles vs. tweets), this fact should be taken into account. If we are only comparing function words 

#### Data Cleaning 

Generally, however, our questions are more about topics rather than writing style. So, once we have a corpus (whether that is one text or millions), we usually want to clean and normalize it. Language is messy, and created for and by people, not computers. There is a lot of grammatical information in a sentence that a computer cannot use. For example, I could say to you `The house is burning.` and you would understand me. You would also understand if I say `house burn`. The first has more information about tense, and which house in particular, but the sentiment is the same either way. 

Usually, the computer works better with the second version (depending, of course, on your research question). In going from the first to the second, we removed the stop words (*the* and *is*), and normalized (removed punctuation and case) and lemmatized what was left (*burning* becomes *burn* - though we might have stemmed this, its impossible to tell from the example). This results in what is essentially a "bag of words", or a corpus of words without any structure. Again, this will be covered more in depth in the Machine Learning Tutorial, but for the time being, we just need to know that there is "clean" and "dirty" data. Sometimes our questions are about the clean data, but sometimes our questions are in the "dirt."

#### Words into Numbers

In the next section, we are going to go through a series of methods that come built-in to NLTK that allow us to turn our words into numbers and visualizations. This is just scratching the surface, but should give you an idea of what is possible beyond just counting. 

### NLTK Methods with the NLTK Corpus

*All of the code for this section is in a Jupyter Notebook in the GitHub Repository. I encourage you to follow along by retyping all of the code, but if you get lost, or want another reference, the code is there as well. To open it in a Jupyter Notebook, Launch the Notebook and then through the file list that appears in your browser, navigate to where you saved the file.*

Return to the Jupyter Home Tab in your Browser (or Launch the Jupyter Notebook again), and start a New Python3 Notebook using the 'New' button in the upper right corner. 

Start by importing the NLTK library by typing:

`import nltk`

Libraries are sets of instructions that Python can use to perform specialist functions. The Python programming language is very simple by design, and can be used for a wide variety of functions. It allows users to customize it to their needs by importing packages. Packages are written by individuals and groups who see the need for a simple set of commands to do routine functions. The Natural Langauge ToolKit (NLTK) is one such library. As the name suggests, its focus is on language processing. 


We will also need the matplotlib library later on, so import it now:

`import matplotlib`

Finally, because of a quirk of Jupyter notebooks, we need to specify that matplotlib should display its graphs in the notebook (as opposed to in a separate window), so we type this command (this is technically a Jupyter command, not Python):

`%matplotlib inline`

All three of these commands can be written in the same cell and run all at once (**Shift** + **Enter**) or in different cells. 

![imports](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/imports.png)

If nothing happens, it's all correct. 

Next we need to load all of the nltk corpora into our program. Even though we downloaded them to our computer, we need to tell Python we want to use them every time we want to use them. 

`from nltk.book import *`

The pre-loaded NLTK texts should appear again. These are pre-formatted datasets. We will still have to do some minor processing, but having the data in this format saves us a few steps. At the end of this tutorial, we will make our own corpus. This is a special type of python object specific to NLTK (it isn't a string, list, or dictionary per se). Sometimes it will behave like a string, and sometimes like a list of words. How it is behaving is noted for each function below.

![imports](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/nltkbook.png)

Let's start by analyzing Moby Dick, which is text1 for NLTK. 

#### Searching for words

The first function we will look at is concordance. Concordance in this context means the characters on either side of the word. Our text is behaving like a string. As discussed in the [Python tutorial LINK](), Python does not *evaluate* strings, so it just counts the number of characters on either side. By default, this is 25 characters on either side of our target word (including spaces). 

In the Jupyter Notebook, type:

`text1.concordance("whale")`

The output shows us the 25 characters on either side of the word, "whale" in Mody Dick. Let's try this with another word, maybe "love". Just replace the word, "whale" with love, and we get the contexts in which Melville uses "love" in Moby Dick. Concordance is used (behind the scenes) for several other functions (including "similar" and "common_contexts")

Let's now see which words appear in similar contexts as the word "love". NLTK has a built-in function for this as well, 'similar'.

`text1.similar("love")`

Behind the scenes, Python found all the contexts where the word, "love" appears, and similar environments, and then what words were common among the similar contexts. This gives a sense of what other words appear in similar contexts. This is somewhat interesting, but more interesting if we can compare it to something else. Let's take a look at another text. What about *Sense and Sensibility*. Let's see what words are similar to "love" for Jane Austen. In the next cell, type:

`text2.similar("love")`

We can compare the two and see immediately that Melville and Austen have write about "love" differently.

Let's expand from novels for a minute and take a look at the Chat Corpus. In chats, text messages, and other digital communication platforms, "lol" is exceedingly common. We know it doesn't really mean "laughing out loud" - maybe the "similar" function can provide some insight into what it does mean.

`text5.similar("lol")`

The resulting list is a lot of greetings, indicating that "lol" probably has more of a phatic function than a semantic one. 

If you are really interested in this type of analysis, see the "common_contexts" function in the [NLTK book](https://www.nltk.org/book/) or in the [NLTK docs])https://www.nltk.org/).

#### Positioning Words

In many ways, concordance and similar are heightened word searches that tell us something about what is happening near the target words. Another metric we can use is to visualize where the words appear in the text. In the case of Moby Dick, we want to compare where "whale" and "monster" appear throughout the text. In this case, the text is functioning as a list of words, and will make a mark where each word appears, offset from the first word. We will *pass* this *function* a *list* of *strings* to plot. This will likely help us develop a visual of the story - where the whale goes from being a whale to being a monster to being a whale again. In the next cell, type:

`text1.dispersion_plot(["whale", "monster"])`

A graph should appear with a tick mark everywhere that "whale" appears and everywhere that "monster" appears. Knowing the story, we can interpret this graph and align it to what we know of how the narrative progresses. If we did not know the story, this could give us a picture of the narrative arc. 

Try this with text2, *Sense and Sensibility* Some relevant words are "marriage", "love", "home", "mother", "husband", "sister", "wife". Pick a few to compare (you can compare an unlimited amount, but it's easier to read a few at a time). 

NLTK has many more functions built-in, but some of the most powerful functions are related to cleaning, POS tagging, and other stages in the text analysis pipeline (aside from actually doing the analysis).

### Built-in Python Methods

We will now turn our attention away from the NLTK library and work with our text using the built-in Python functions. 

#### Types vs. Tokens

First, let's find out how many times a given word appears in the corpus.  In this case (and all cases going forward), our text will be treated as a list of words. Therefore, we will use the 'count' function. We could just as easily do this with a text editor, but performing this in Python allows us to save it to a variable and then utilize this statistic in other calculations (for example, if we want to know what percentage of words in a corpus are 'lol', we would need a count of the 'lol's. In the next cell, type:

 `text1.count("whale")`
 
We see that "whale" occurs 906 times, that seems a little low. Let's check on "Whale" and see how often that appears:
 
 `text1.count("Whale")`
 
"Whale", with a capital "W" appears 282 times. This is a problem for us, we actually want them to be collapsed into one word - since "whale" and "Whale" really are the same for our purposes. We will deal with that in a moment. But, for the time being, we will accept that we have two entries for "whale". 

This gets at a **type/token** distinction. "Whale" and "whale" are different types (as of now) because they do not match identically. Every instance of "whale" in the corpus is another **token** - it is an instance of the type, "whale." Therefore, there are 906 tokens of "whale" in our corpus. 

Let's fix this by making all of the words lowercase. We will make a new list of words, and call it text1_tokens. We will fill this list with all the words in text1, but in their lowercase form. In this same step, we are going to do a tricky move, and only keep the words that are alphabetical (so no punctuation or numbers), and pass over anything, effectively just not adding it to the list. Type the following code (the tabs are necessary):

```text1_tokens = []
for t in text1:
	if t.isalpha:
		t.lower()
		text1_tokens.append(t)
	else:
		pass

```

![code](https://github.com/michellejm/NLTK_DHRI/blob/master/Images/code.png)
	
Another way to type this (more efficiently) is:

`text1_tokens= [t.lower() for t in text1 if t.isalpha()]` 

Great! Now text1_tokens is a list of all of the tokens in our corpus, with the punctuation removed, and all the words in lowercase

Now we want to know how many words there are in our corpus (how many tokens total). Therefore, we want to ask, what is the length of that list of words. Python has a built-in 'len' function that allows you to find out the length of anything. Pass it a list (as we will), and it will tell you how many items are in the list. Pass it a string, and it will tell you how many characters in the string. Pass it a dictionary, it returns how many entries are in the dictionary, etc. In the next cell, type:

`len(text1_tokens)`

Just for comparison, check out how many words were in "text1" - before we removed the punctuation and the numbers. 

`len(text1)`

We see there are over 218,000 words in Moby Dick (including metadata). But this is the number of words total - we want to know the number of unique words. We want to know how many *types* - not just how many tokens. 

We will make a set from the list. Sets in Python work just like they do in math, it's all the unique values with no overlap. So let's find out the length of our set. just like in math, we can also embed. So, rather than saying `x=set(text1_tokens)` and then finding the length of x, we can do it all in one step. 

`len(set(text1_tokens))`

Great! Now we can calculate the lexical density of Moby Dick. Statistical studies have shown that lexical density (the number of unique words per total words) is a [good metric to approximate lexical diversity](http://www.pjos.org/index.php/LWPL/article/viewFile/2273/1848) (the range of vocabulary an author uses). For our first pass at lexical density, we will simply divide the number of unique words by the total number of words:

`len(set(text1_tokens))/len(text1_tokens)`

If we want to use this metric to compare texts, we immediately notice a problem. Lexical density is dependent upon the length of a text and therefore is strictly a comparative measure. It is possible to compare 100 words from one text to 100 words from another, but because language is finite and repetitive, it is not possible to compare 100 words from one to 200 words from another. Even with these restrictions, lexical density is a useful metric in grade level estimations, [vocabulary use](http://www.mdpi.com/2226-471X/2/3/7) and genre classification, and a reasonable proxy for lexical diversity. 

Let's take this constraint into account by working with only the first 10,000 words of our text. First we need to slice our list, returning the words in position 0 to position 9,999 (we'll actually write is as "up to, but not including" 10,000). 

`text1_slice = text1_tokens[0:10000]`

Now we can do the same calculation we did above:

`len(set(text1_slice))/len(text1_slice)`

This is a much higher number, though the number itself is arbitrary, when comparing different texts, this step is essential to get an accurate measure. 

If we wanted to perform the same set of steps with *Sense and Sensibility* 
1. make all the words lowercase and remove punctuation, 
2. make a slice of the first 10,000 words 
3. calculate lexical density by dividing the length of the set of the slice by the length of the slice

We could compare Melville's and Austen's range of vocabulary. 

#### Clean the Corpus

Thus far, we have been asking questions that take stopwords and grammatical features into account. For the most part, we want to exclude these features since they don't actually contribute very much semantic content to our models. Therefore, we will:
1. Remove capitalization and punctuation (DONE)
2. Remove stop words
3. Lemmatize (or stem) our words.

We already completed step 1, and are now working with our text1_tokens. Remember, this *variable* contains a *list* of *strings* that we will work with. We want to remove the stop words from that list. The NLTK library comes with fairly comprehensive lists of stop words for many languages. Stop words are function words that contribute very little semantic meaning, they most often have grammatical functions. Usually, these are function words such as determiners, prepositions, auxiliaries, and others. 

To use NLTK's stop words, we need to import the list of words from the corpus (we could have done this at the beginning of our program, and in more fully developed code, we would put it up there, but this works, too). In the next cell, type:

`from nltk.corpus import stopwords`

We need to specify the English list, and save it into its own variable that we can use in the next step, so type:

`stops = stopwords.words('english')`

Now we want to go through all of the words in our text, and if that word is in the stop words list, remove it from our list. Otherwise, skip it. The code below is VERY slow (there's a faster option beneath it). The way we write this in Python is:

```for t in text1_tokens:
    if t in stops:
        text1_tokens.remove(t)
    else:
        pass
```
        
Faster option: `text1_tokens = [t for t in text1_tokens if t not in stops]`

Now that we removed our stop words, let's see how many words are left:

`len(text1_tokens)`

You should get a much lower number.

For reference, let's also check how many unique words there are.

`len(set(text1_tokens))`

The next step is to stem or lemmatize the remaining words. This means that we will strip off the grammatical structure from the words. For example, cats --> cat, and walked --> walk. If that was all we had to do, we could stem the corpus and achieve the correct result, because stemming (as the name implies) really just means cutting off affixes to find the root (or the stem). Very quickly, this gets very complicated, though as men --> man and sang --> sing. Lemmatization deals with this by looking up the word in a dictionary of sorts and finding the appropriate root (though still is not 100% accurate). Lemmatization therefore takes longer. NLTK comes with pre-built stemmers and lemmatizers.

We will use the WordNet Lemmatizer from the NLTK Stem library, so let's import that now: 

`from nltk.stem import WordNetLemmatizer`

Because of the way that it is written "under the hood", an instance of the lemmatizer needs to be called (we know this from reading [the docs](https://www.nltk.org/))

`wordnet_lemmatizer = WordNetLemmatizer()`

Now we will lemmatize the words in the list. This time, we will only use the faster version because it takes a long time. 

 `text1_clean = [wordnet_lemmatizer.lemmatize(t) for t in t1_tokens]`

Let's check now to see how long our final, cleaned version of the data is is and then the unique set of words. 

`len(text1_clean)`

`len(set(text1_clean))`

This set should be much smaller than the set before we lemmatized. Now if we were to calculate lexical density, we would be looking at how many word stems with semantic content are represented in Moby Dick, which gets at a different question than our first analysis of lexical density. 

Now let's have a look at the words Melville uses in Moby Dick. We'd like to look at all of the *types*, but not necessarily all of the *tokens.* We will order this set so that it is in an order we can handle. In the next cell, type:

`sorted(set(text1_tokens))`

A list of all the words in Moby Dick should appear. The list begins with 'a', which we might have expected to be removed in the stemming process, and some words we wouldn't have expected, such as "abbreviate" and "abbreviation". We can try this with a stemmer instead (I recommend Porter, but there are many), but we end up with a lot of unrecoverable words. We will stick with the output of the Lemmatizer for now. The code for Porter is below:

```
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
t1_porter = [porter_stemmer.stem(t) for t in t1_tokens]
sorted(set(t1_porter))
```

 Now let's visualize the word frequency. First we will create a frequency distribution. This is a special type of NLTK object that is kind of like a dictionary, where the words are the keys, and the counts are the values. (The NLTK-ness of our text1 is still retained with text1_clean)
 
 `my_dist=FreqDist(text1_clean)`

If nothing happened, that is normal, check to make sure it is there by calling for the type of the "my_dist" object.


`type(my_dist)`

It should say it is an nltk probability distribution. It doesn't matter too much right now what it is, only that it worked. We can now plot this with the matplotlib function, "plot". We want to plot the first 50 entries of the my_dist object, but we don't want it to be cumulative (if we were working with financial data, we might want cumulative).

`my_dist.plot(50,cumulative=False)`

We've made a nice image here, but it might be easier to comprehend as a list, because this is a special probability distribution object, we can call the "most common" on this, too. Let's find the 20 most common words:

`my_dist.most_common(20)`

What about if we are interested in a list of specific words - maybe to identify texts that have biblical references. Let's make a (short) list of words that might suggest a biblical reference and see if they appear in Moby Dick. Set this list equal to a variable.

`b_words = ['god', 'apostle', 'angel']`

Then we will loop through the words in our cleaned corpus, and see if any of them are in our list of biblical words, and save just those words that appear in both in another list. 

``` my_list = []
for word in b_words:
    if word in text1_clean:
        my_list.append(word)
    else:
        pass```

And then we will print the results.

`print(my_list)`

You can obviously do this with much larger lists and compare entire novels if you wish (though it would take a while with this syntax). You can use this to get similarity measures and answer related questions. 


### Make Your Own Corpus

Now that we have seen and implemented a series of text analysis techniques, let's go to the Internet to find a new text. You could just as easily use a txt file that is on your computer (say you have a txt copy of the [Hunger Games LINK TO GIT](LINK), for example. Or historic newspapers, or Supreme Court proceedings, etc. Here, we will use [Project Gutenberg](http://www.gutenberg.org). Project Gutenberg is an archive of public domain written works, available in a wide variety of formats, including .txt. You can download these to your computer or access them via the url. We'll use the url method. We found Don Quixote in the archive, and will work with that.

The Python package, urllib comes installed with Python, but is inactive by default, so we still need to import it to utilize the functions. Since we are only going to use the urlopen function, we will just import that one.

In the next cell, type:

`from urllib.request import urlopen`

The urlopen function allows your program to interact with files on the internet by opening them. It does not read them, however - they are just available to be read in the next line. This is the default behavior any time a file is opened and read by Python. One reason is that you might want to read a file in different ways. For example, if you have a **really** big file (think big data), you might want to read line-by-line rather than the whole thing at once. 

Now let's specify which url we are going to use. Though you might be able to find Don Quixote in the Project Gutenberg files, please type this in so that we are all using the same format (there are multiple .txt files on the site, one with utf-8 encoding, another with ascii encoding). We want the utf-8 encoded one. The difference between these is beyond the scope of this tutorial, check out this [introduction to character encoding](https://www.w3.org/International/questions/qa-what-is-encoding) from The World Wide Web Consortium (W3C). 

Set the url we want to a variable:

`my_url = "http://www.gutenberg.org/cache/epub/996/pg996.txt"`

We still need to open the file and read the file. You will have to do this with files stored locally as well. (in which case, you would type the path to the file (i.e., "data/texts/mytext.txt") in place of `my_url`)


`file = urlopen(my_url)`

`raw = file.read()`

This file is in bytes, so we need to decode it into a string. In the next cell, type:

`don=raw.decode()`

Now let's check on what kind of object we have in the "don" variable. Type:

`type(don)`

This should be a string. Great! We have just read in our first file and now we are going to transform that string into a text that we can perform NLTK functions on. Since we already imported nltk at the beginning of our program, we don't need to import it again, we can just use its functions by specifying 'nltk' before the function. The first step is to tokenize the words, transforming the giant string into a list of words. A simple way to do this would be to break on spaces, and that would probably be fine, but we are going to use the NLTK tokenizer to ensure that edge cases are captured (i.e., "don't" is made into 2 words: do and n't). In the next cell, type:

`don_tokens = nltk.word_tokenize(don)`

You can check out the type of "don_tokens" to make sure it worked, but you don't have to (it should be a list). Let's see how many words there are in our novel:

`len(don_tokens)`

Since this is a list, we can look at any slice of it that we want. Let's inspect the first ten words: 

`don_tokens[:10]`

That looks like metadata - not what we want to analyze. We will strip this off before proceeding. If you were doing this to many texts, you would want to use Regular Expressions. Regular Expressions are an extremely powerful way to match text in a document. However, we are just using this text, so we could either guess, or cut and paste the text into a text reader and identify the position of the first content (i.e., how many words in is the first word). That is the route we are going to take. We found that the first word of the story begins at word 120, so let's make a slice of the text from word position 120 to the end. 

`dq_text = don_tokens[120:]`

Finally, if we want to use the NLTK specific functions:
* concordance
* similar
* dispersion plot 
* etc., *see the [NLTK book](https://www.nltk.org/book/)*

We would have to make a specific NLTK Text object. 

`dq_nltk_text = nltk.Text(dq_text)`

If we wanted to use the built-in Python functions, we can just stick with our list of words in `dq_text`. Since we've already covered all of those functions, we are going to move ahead with cleaning our text. 

Just as we did earlier, we are going to remove the stopwords based on a list provided by NLTK, remove punctuation, and capitalization, and lemmatize the words. The code for each step follows:

1. Remove stop words

```mystops = stopwords.words('english')
dq_clean = [w for w in dq_text if w not in mystops]
```

2. Lowercase and remove punctuation

`dq_clean = [t.lower() for t in dq_clean if t.isalpha()]`

3. Lemmatize

 
```from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
dq_clean = [wordnet_lemmatizer.lemmatize(t) for t in dq_clean]
```
From here, you could perform all of the operations that we did after cleaning our text in the previous session. Instead, we will perform another type of analysis: Part-of-Speech (POS) tagging. 


### Part-of-Speech Tagging

**We are going to use the pre-cleaned, dq_text object for this section** 

POS tagging is going through a text and identifying which part of speech each word belongs to (i.e., Noun, Verb, or Adjective). Every word belongs to a part of speech, but some words can be confusing.

* Floyd is happy 
* Happy is a state of being
* Happy has five letters
* I'm going to Happy Cat tonight

Therefore, part of speech is as much related to the word itself as its relationship to the words around it. A good part of speech tagger takes this into account, but there are some impossible cases as well:

* Wanda was entertaining last night

Part of Speech tagging can be done very simply: with a very small Tag Set, or in a very complex way: with a much more elaborate tag set. We are going to implement a compromise, and use a neither small nor large tag set, the [Penn Tree Bank POS Tag Set](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html).

This is the tag set that is pre-loaded into NLTK. When we call the tagger, we expect it to return an object with the word and the tag associated. Because POS tagging is dependent upon the stop words, we have to use a text that includes the stop words. Therefore, we will go back to using the **dq_text** object for this section. Let's try it out. Type:

`dq_tagged = nltk.pos_tag(dq_text)`

Let's inspect what we have:

`print(dq_tagged[:10])`

This is a list of ordered tuples. Each element in the list is a pairing of (word, POS-tag). This is great, but it is very detailed, I would like to know how many Nouns, Verbs, and Adjectives I have. First, I'll make an epty dictionary to hold my results. Then I will go through this list of tuples and count the number of times each tag appears. Every time I encounter a new tag, I'll add it to a dictionary and then increment by one every time I encounter that tag again. Let's see what that looks like in code:

```tag_dict = {}
#for every word/tag combo in my list, 
for (word, tag) in dq_text:
    if tag in tag_dict: 
        tag_dict[tag]+=1
    else:
        tag_dict[tag] = 1```

Now let's see what we got:

`tag_dict`

This would be better with some order to it, so let's organize our dictionary to find out what the most common tag is. We need the OrderedDict function. Just like with the url.request library, the 'collections' package comes built-in with Python, but is inactive by default. Therefore, to use the functions from it, we need to activate it by importing that set of functions. We will only need the OrderedDict, so that's all we will import. Then we will pass the OrderedDict function our dictionary with a set of parameters, to tell it exactly how we want it to be ordered, and in which direction, etc. We know what to do for this function because we [READ THE DOCS](https://docs.python.org/3/library/collections.html#collections.OrderedDict)

```from collections import OrderedDict
tag_dict = OrderedDict(sorted(tag_dict.items(), key=lambda t: t[1], reverse=True))```

Now check out what we have. It looks like NN is the most common tag, we can look up what that is back at the [Penn Tree Bank](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html). Looks like that is a Noun, singular or mass. Great! This information will likely help us with genre classification (as you will do in the Machine Learning tutorial), or identifying the author of a text, or a variety of other functions. 

### Conclusion

At this point, you should have a familiarity with what is possible with text analysis, and some of the most important functions (i.e., cleaning and part-of-speech tagging). Yet, this tutorial has only scratched the surface of what is possible with text analysis and natural language processing. It is a rapidly growing field, if you are interested, be sure to work through the NLTK Book as well as peruse the resources in the Zotero Library. 


______________________________________________________________________________________________________________

Tutorial written by [Michelle McSweeney](www.michelleamcsweeney.com), for *Digital Humanities Research Institute*, an intensive 10 day workshop on Digital Humanities methods and applications at the Graduate Center at CUNY June 11-20. More information about the institue is available [here](http://dhinstitutes.org/).
