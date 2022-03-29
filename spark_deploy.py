from pyspark import SparkConf, SparkContext
from flask import Flask, render_template, request
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
import re
import string
from pyspark.sql.types import FloatType
from textblob import TextBlob

sc = SparkContext(master="local", appName="SparkDemo")

app = Flask(__name__)


@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        message = request.form['message']

        # remove non ASCII characters
        def strip_non_ascii(data_str):
            ''' Returns the string without non ASCII characters'''
            stripped = (c for c in data_str if 0 < ord(c) < 127)
            return ''.join(stripped)

        # setup pyspark udf function
        strip_non_ascii_udf = udf(strip_non_ascii, StringType())

        def fix_abbreviation(data_str):
            data_str = data_str.lower()
            data_str = re.sub(r'\bthats\b', 'that is', data_str)
            data_str = re.sub(r'\bive\b', 'i have', data_str)
            data_str = re.sub(r'\bim\b', 'i am', data_str)
            data_str = re.sub(r'\bya\b', 'yeah', data_str)
            data_str = re.sub(r'\bcant\b', 'can not', data_str)
            data_str = re.sub(r'\bdont\b', 'do not', data_str)
            data_str = re.sub(r'\bwont\b', 'will not', data_str)
            data_str = re.sub(r'\bid\b', 'i would', data_str)
            data_str = re.sub(r'wtf', 'what the fuck', data_str)
            data_str = re.sub(r'\bwth\b', 'what the hell', data_str)
            data_str = re.sub(r'\br\b', 'are', data_str)
            data_str = re.sub(r'\bu\b', 'you', data_str)
            data_str = re.sub(r'\bk\b', 'OK', data_str)
            data_str = re.sub(r'\bsux\b', 'sucks', data_str)
            data_str = re.sub(r'\bno+\b', 'no', data_str)
            data_str = re.sub(r'\bcoo+\b', 'cool', data_str)
            data_str = re.sub(r'rt\b', '', data_str)
            data_str = data_str.strip()
            return data_str

        def remove_features(data_str):
            # compile regex
            url_re = re.compile('https?://(www.)?\w+\.\w+(/\w+)*/?')
            punc_re = re.compile('[%s]' % re.escape(string.punctuation))
            num_re = re.compile('(\\d+)')
            mention_re = re.compile('@(\w+)')
            alpha_num_re = re.compile("^[a-z0-9_.]+$")
            # convert to lowercase
            data_str = data_str.lower()
            # remove hyperlinks
            data_str = url_re.sub(' ', data_str)
            # remove @mentions
            data_str = mention_re.sub(' ', data_str)
            # remove puncuation
            data_str = punc_re.sub(' ', data_str)
            # remove numeric 'words'
            data_str = num_re.sub(' ', data_str)
            # remove non a-z 0-9 characters and words shorter than 1 characters
            list_pos = 0
            cleaned_str = ''
            for word in data_str.split():
                if list_pos == 0:
                    if alpha_num_re.match(word) and len(word) > 1:
                        cleaned_str = word
                    else:
                        cleaned_str = ' '
                else:
                    if alpha_num_re.match(word) and len(word) > 1:
                        cleaned_str = cleaned_str + ' ' + word
                    else:
                        cleaned_str += ' '
                list_pos += 1
            # remove unwanted space, *.split() will automatically split on
            # whitespace and discard duplicates, the " ".join() joins the
            # resulting list into one string.
            return " ".join(cleaned_str.split())

        # setup pyspark udf function
        remove_features_udf = udf(remove_features, StringType())

        def sentiment_analysis(text):
            return TextBlob(text).sentiment.polarity

        sentiment_analysis_udf = udf(sentiment_analysis, FloatType())

        def condition(r):
            if (r >= 0.1):
                label = "positive"
            elif (r <= -0.1):
                label = "negative"
            else:
                label = "neutral"
            return label

        sentiment_udf = udf(lambda x: condition(x), StringType())

        df['text'] = message
        df = df.withColumn('text_non_asci', strip_non_ascii_udf(df['text']))

        df = df.withColumn('fixed_abbrev', fix_abbreviation_udf(df['text_non_asci']))
        df = df.withColumn('removed', remove_features_udf(df['fixed_abbrev']))
        df = df.withColumn("sentiment_score", sentiment_analysis_udf(df['removed']))
        df = df.withColumn("sentiment", sentiment_udf(df['sentiment_score']))

        result = df['sentiment']

        return render_template('prediction.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
