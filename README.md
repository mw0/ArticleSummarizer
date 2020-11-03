## Apps based upon HuggingFace API

### Contents

* [Summarization](#summarization)
* []()

### Summarization

Applies Facebook's BART model, as implemented by HuggingFace, to summarize articles from the NY Times.

<table>
<tr valign="top">
<td>This is a streamlit app that does several things:

* uses the <em>NY Times</em> Top Stories API to get current metadata
* creates sidebar dropdown from top 5 URLs and titles
* when user selects a title:
  * fetches the article from <em>Times</em> website
  * extracts body of article using BeautifulSoup
  * article is truncated to 720 words maximum<sup>&dagger;</sup>
  * applies Bart summarizer model
  * displays summary, full article, profiling info
* Streamlit's caching capabilities obviate repeating steps &mdash; e.g., fetching and extracting text from an article already parsed

<sup>&dagger;</sup>Summarizer can fail if text is too long.

See [streamlitSummarizer.py](https://github.com/mw0/MLnotebooks/blob/master/HuggingFace/python/streamlitSummarizer.py) for source code.
</td><td width="700"><img src="SummarizerAppScreenshot1.png" width="700" height="1215"</td>
</tr>
</table>
