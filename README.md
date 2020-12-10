# Unbiased News

## Motivation  

We live in a world today that has become increasingly more divisive and vitriolic. People hold certain viewpoints and oft consider that view to be the complete & objective truth. But unfortunately, it is rarely so. 
Taking a step back, a huge driver for these divisive stands are news organizations that thrive on reporting with certain forms of bias built into their coverage. They would like to intentionally drive the narratives to their vested foregone conclusion.
If we the people, want to seek a wholesome understanding, it is expected that we read perhaps 10+ new sources to get to the source what’s actually happening in earnestness and what is being spewed as conjecture. As part of my passion project, I aim to use Data Science to make it easier for those who seek for an objective comprehension.

## Objective  

Using news articles written on the same topic by outlets who are on different ranges of political bias spectrum, I would like to create a model that can combine and summarize these articles to display what’s commonly being said. This will serve as a springboard objectives towards an array of downstream developments.

## Data Source: 
[All Sides](www.allsides.com)   
Fox News, Washington Times, New York Times, Washington Post, HuffPost

## Approach:

1. Allsides contains news headlines & links to a left leaning, right leaning and center leaning news organization for that headline. Hence, all 2000+ news headlines from All Sides were scrapped.
1. Individual news sites were then scrapped to obtain the entire news articles. To allow for progress, only 5 news organization were focused on at present - as each news organization required its own scrapping & cleaning steps.
1. The articles were arranged in a giant corpus and at first LDA/NMF Topic Modeling, Doc2Vec & Top2Vec techniques were attempted. The resultant scores were experimented with to see if sentence pairs could be created, to no avail.
1. Sentence embeddings from Sentence Transformers were ultimately used, employing the power of transfer learning. The resultant embeddings were used to create best matching pairs, using cosine similarity.
1. The pairs were ultimately summarized to give a digestible read of what's commonly being said across the political spectrum on the provided news headlines & their associated articles

## Result:

The primary objective of the project was accomplished. The deployed streamlit based app allows for providing links to any two articles (from the selected 5 news organizations) and receive a live summary of common points. The original articles are also presented below, with highlighting for sentences that were used to make the summary.

## Future Steps:
1. Identify ways to scope out more articles on the same topic (beyond the 2 used for ‘Step 1’) and include in the model for combining on common points & summarizing.
1. Extend functionality to be able to enter a news article link on an interactive interface, which can automatically seek out articles with opposing bias on the same topic & combine/summarize them.
1. Extend functionality to include viewpoints that have been completely omitted (or presented in an opposing manner) onto the combined article. 
