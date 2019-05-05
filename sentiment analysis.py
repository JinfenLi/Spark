from __future__ import division
from pyspark.sql import SparkSession
from pyspark.ml import feature, regression, evaluation, Pipeline
from pyspark.sql import functions as fn, Row
import matplotlib.pyplot as plt
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
import numpy as np

# dataframe functions
from pyspark.sql import functions as fn
documents_df = spark.sparkContext.parallelize([
        [1, 'cats are cute', 0],
        [2, 'dogs are playfull', 0],
        [3, 'lions are big', 1],
        [4, 'cars are fast', 1]]).toDF(['doc_id', 'text', 'user_id'])

from pyspark.ml.feature import Tokenizer
# the tokenizer object
tokenizer = Tokenizer().setInputCol('text').setOutputCol('words')

from pyspark.ml.feature import CountVectorizer
count_vectorizer_estimator = CountVectorizer().setInputCol('words').setOutputCol('features')
count_vectorizer_transformer = count_vectorizer_estimator.fit(tokenizer.transform(documents_df))
from pyspark.ml import Pipeline
pipeline_cv_estimator = Pipeline(stages=[tokenizer, count_vectorizer_estimator])
pipeline_cv_transformer = pipeline_cv_estimator.fit(documents_df)
pipeline_cv_transformer.transform(documents_df).show()
import pandas as pd
sentiments_df = spark.read.parquet('/datasets/sentiments.parquet')
# a sample of positive words
sentiments_df.where(fn.col('sentiment') == 1).show(5)
sentiments_df.groupBy('sentiment').agg(fn.count('*')).show()

imdb_reviews_df = spark.read.parquet('/datasets/imdb_reviews_preprocessed.parquet')
from pyspark.ml.feature import RegexTokenizer
#\p{L}+ means that it will extract letters without accents setGaps means that it will keep applying the rule until it can't extract new words
tokenizer = RegexTokenizer().setGaps(False)\
  .setPattern("\\p{L}+")\
  .setInputCol("review")\
  .setOutputCol("words")
review_words_df = tokenizer.transform(imdb_reviews_df)
print(review_words_df)

review_word_sentiment_df = review_words_df.\
    select('id', fn.explode('words').alias('word')).\
    join(sentiments_df, 'word')
review_word_sentiment_df.show(5)

simple_sentiment_prediction_df = review_word_sentiment_df.\
    groupBy('id').\
    agg(fn.avg('sentiment').alias('avg_sentiment')).\
    withColumn('predicted', fn.when(fn.col('avg_sentiment') > 0, 1.0).otherwise(0.))
simple_sentiment_prediction_df.show(5)

# we obtain the stop words from a website
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("words")\
  .setOutputCol("filtered")


from pyspark.ml.feature import CountVectorizer

# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")
# we now create a pipelined transformer
cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(imdb_reviews_df)
# now we can make the transformation between the raw text and the counts
cv_pipeline.transform(imdb_reviews_df).show(5)

from pyspark.ml.feature import IDF
idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(imdb_reviews_df)
tfidf_df = idf_pipeline.transform(imdb_reviews_df)
training_df, validation_df, testing_df = imdb_reviews_df.randomSplit([0.6, 0.3, 0.1], seed=0)
from pyspark.ml.classification import LogisticRegression
lambda_par = 0.02
alpha_par = 0.3
en_lr = LogisticRegression().\
        setLabelCol('score').\
        setFeaturesCol('tfidf').\
        setRegParam(lambda_par).\
        setMaxIter(100).\
        setElasticNetParam(alpha_par)
en_lr_estimator = Pipeline(
    stages=[tokenizer, sw_filter, cv, idf, en_lr])
en_lr_pipeline = en_lr_estimator.fit(training_df)
en_lr_pipeline.transform(validation_df).select(fn.avg(fn.expr('float(prediction = score)'))).show()
en_weights = en_lr_pipeline.stages[-1].coefficients.toArray()
en_coeffs_df = pd.DataFrame({'word': en_lr_pipeline.stages[2].vocabulary, 'weight': en_weights})
en_coeffs_df.sort_values('weight').head(15)
print(en_coeffs_df.query('weight == 0.0').shape[0]/en_coeffs_df.shape[0])
from pyspark.ml.tuning import ParamGridBuilder
en_lr_estimator.getStages()
grid = ParamGridBuilder().\
    addGrid(en_lr.regParam, [0., 0.01, 0.02]).\
    addGrid(en_lr.elasticNetParam, [0., 0.2, 0.4]).\
    build()

all_models = []
for j in range(len(grid)):
    print("Fitting model {}".format(j+1))
    model = en_lr_estimator.fit(training_df, grid[j])
    all_models.append(model)
accuracies = [m.\
    transform(validation_df).\
    select(fn.avg(fn.expr('float(score = prediction)')).alias('accuracy')).\
    first().\
    accuracy for m in all_models]
