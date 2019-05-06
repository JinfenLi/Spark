# dataframe functions
from pyspark.sql import functions as fn
import os
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import pandas as pd

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
credit_score_df = spark.read.parquet('/datasets/cs-training.parquet')
# change type to double, drop NAs, and change dependent variable name to label
credit_score_df = credit_score_df.withColumnRenamed('SeriousDlqin2yrs', 'label')
from pyspark.ml.feature import VectorAssembler
training_df, validation_df, testing_df = credit_score_df.randomSplit([0.6, 0.3, 0.1])
# build a pipeline for analysis
va = VectorAssembler().setInputCols(training_df.columns[2:]).setOutputCol('features')
lr = LogisticRegression(regParam=0.1)
lr_pipeline = Pipeline(stages=[va, lr]).fit(training_df)
rf = RandomForestClassifier()
rf_pipeline = Pipeline(stages=[va, rf]).fit(training_df)
bce = BinaryClassificationEvaluator()
bce.evaluate(lr_pipeline.transform(validation_df))
bce.evaluate(rf_pipeline.transform(validation_df))
lr_model = lr_pipeline.stages[-1]
pd.DataFrame(list(zip(credit_score_df.columns[2:], lr_model.coefficients.toArray())),
            columns = ['column', 'weight']).sort_values('weight')
rf_model = rf_pipeline.stages[-1]
pd.DataFrame(list(zip(credit_score_df.columns[2:], rf_model.featureImportances.toArray())),
            columns = ['column', 'weight']).sort_values('weight')