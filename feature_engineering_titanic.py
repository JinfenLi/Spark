from pyspark.sql import SparkSession
from pyspark.ml import feature
from pyspark.ml import classification
from pyspark.sql import functions as fn
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, \
    MulticlassClassificationEvaluator, \
    RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql import SparkSession


spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
titanic_df = spark.read.csv('/datasets/titanic_original.csv', header=True, inferSchema=True)
titanic_df.limit(10).toPandas()

# some basic cleanup
drop_cols = ['boat', 'body']
new_titanic_df = titanic_df.\
    drop(*drop_cols).\
    withColumnRenamed('home.dest', 'home_dest').\
    fillna('O').\
    dropna(subset=['pclass', 'age', 'sibsp', 'parch', 'fare', 'survived'])
training, test = new_titanic_df.randomSplit([0.8, 0.2], 0)

model0 = Pipeline(stages=[feature.VectorAssembler(inputCols=['pclass', 'age', 'sibsp', 'parch', 'fare'],
                                        outputCol='features'),
                 classification.LogisticRegression(labelCol='survived', featuresCol='features')])

model2_pipeline = Pipeline(feature.VectorAssembler(inputCols=['pclass', 'age', 'sibsp', 'parch', 'fare']),
              feature.StandardScaler(withMean=True),
             classification.LogisticRegression(labelCol='survived'))

model0_fitted = model0.fit(training)
evaluator = BinaryClassificationEvaluator(labelCol='survived')
evaluator.evaluate(model0_fitted.transform(test))
model1 = Pipeline(stages=[feature.VectorAssembler(inputCols=['pclass', 'age', 'sibsp', 'parch', 'fare'],
                                        outputCol='features'),
                          feature.StringIndexer(inputCol='sex', outputCol='encoded_sex'),
                          feature.VectorAssembler(inputCols=['features', 'encoded_sex'], outputCol='final_features'),
                 classification.LogisticRegression(labelCol='survived', featuresCol='final_features')])

model1_fitted = model1.fit(training)
evaluator.evaluate(model1_fitted.transform(test))
feature.Bucketizer(splits=[0, 20, 50, 100, 400, 800], inputCol='fare').transform(new_titanic_df).toPandas().iloc[:, -1].hist()
plt.xticks([-1, 0, 1, 2, 3, 4, 5])
plt.xlabel('Fare bucket')
feature.QuantileDiscretizer(numBuckets=4, inputCol='fare').fit(new_titanic_df).transform(new_titanic_df).toPandas().iloc[:, -1].hist()
plt.xticks([-1, 0, 1, 2, 3, 4, 5])
plt.xlabel('Fare quantiles')


gender_pipe = feature.StringIndexer(inputCol='sex', handleInvalid='skip')
titles_list = " Capt  Col  Don  Dona  Dr  Jonkheer  Lady  Major  Master  Miss  Mlle  Mme  Mr  Mrs  Ms  Rev  Sir".lower().split()
title_pipe = Pipeline(feature.RegexTokenizer(pattern="\\b(" + ("|".join(titles_list)) + ")\\b",
                       gaps=False,
                      inputCol='name'),
                  feature.CountVectorizer())

embarked_pipe = Pipeline(feature.StringIndexer(inputCol='embarked', handleInvalid='skip'), feature.OneHotEncoder())



cabin_pipe = Pipeline(stages=[feature.SQLTransformer(statement='select *, substring(cabin,1,1) as cabin_col from __THIS__'),
                              feature.StringIndexer(inputCol='cabin_col', outputCol='cabin_col2', handleInvalid='skip'),
                              feature.OneHotEncoder(inputCol='cabin_col2')])
numerical_features = Pipeline(feature.VectorAssembler(inputCols=['pclass', 'age', 'sibsp', 'parch']),
                          feature.StandardScaler())
all_features = Pipeline((numerical_features, feature.QuantileDiscretizer(inputCol='fare', numBuckets=4),
                     gender_pipe, title_pipe, embarked_pipe, cabin_pipe), feature.VectorAssembler())
lr = classification.LogisticRegression(labelCol='survived')
final_model_pipeline = Pipeline(all_features, lr)

paramGrid = ParamGridBuilder() \
    .addGrid(lr.elasticNetParam, [0., 0.01, 0.1]) \
    .addGrid(lr.regParam, [0.1, 0.01, 0.001, 0.0001]) \
    .build()

evaluator = BinaryClassificationEvaluator(labelCol=lr.getLabelCol(), rawPredictionCol=lr.getRawPredictionCol())
crossval = CrossValidator(estimator=final_model_pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)
final_model_fitted = crossval.fit(training)
evaluator.evaluate(final_model_fitted.transform(test))