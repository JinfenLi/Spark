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
import numpy as np

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

def display_first_as_img(df):
    plt.imshow(df.first().raw_pixels.toArray().reshape([60,40]), 'gray', aspect=0.5);
    display()

caltech101_df = spark.read.parquet('/datasets/caltech101_60_40_ubyte.parquet')
caltech101_df.printSchema()
caltech101_df.select(fn.countDistinct('category')).show()
caltech101_df.groupby('category').agg(fn.count('*').alias('n_images')).orderBy(fn.desc('n_images')).show()
display_first_as_img(caltech101_df.where(fn.col('category') == "Motorbikes").sample(True, 0.1))
display_first_as_img(caltech101_df.where(fn.col('category') == "Motorbikes").sample(True, 0.1))

training_df, validation_df, testing_df = caltech101_df.\
    where(fn.col('category').isin(['airplanes', 'Faces_easy', 'Motorbikes'])).\
    randomSplit([0.6, 0.2, 0.2], seed=0)

validation_df.groupBy('category').agg(fn.count(*)).show()

category_to_number_model = feature.StringIndexer(inputCol='category', outputCol='label').\
    fit(training_df)
category_to_number_model.transform(training_df).show()

list(enumerate(category_to_number_model.labels))
mlp = classification.MultilayerPerceptronClassifier(seed=0).\
    setStepSize(0.2).\
    setMaxIter(200).\
    setFeaturesCol('raw_pixels')
mlp = mlp.setLayers([60*40, 100,3])
mlp_simple_model = Pipeline(stages=[category_to_number_model, mlp]).fit(training_df)
mlp_simple_model.transform(validation_df).show(10)
from pyspark.ml import evaluation
evaluator = evaluation.MulticlassClassificationEvaluator(metricName="accuracy")
evaluator.evaluate(mlp_simple_model.transform(validation_df))

evaluation_info = []

for training_size in [0.1, 0.5, 1.]:
    for n_neurons in [1, 3, 10, 20]:
        print("Training size: ", training_size, "; # Neurons: ", n_neurons)
        training_sample_df = training_df.sample(False, training_size, seed=0)
        mlp_template = classification.MultilayerPerceptronClassifier(seed=0).\
            setStepSize(0.2).\
            setMaxIter(200).\
            setFeaturesCol('raw_pixels').\
            setLayers([60*40, n_neurons, 3])
        mlp_template_model = Pipeline(stages=[category_to_number_model, mlp_template]).fit(training_sample_df)
        # append training performance
        evaluation_info.append({'dataset': 'training',
                                'training_size': training_size,
                                'n_neurons': n_neurons,
                                'accuracy': evaluator.evaluate(mlp_template_model.transform(training_sample_df))})
        evaluation_info.append({'dataset': 'validation',
                                'training_size': training_size,
                                'n_neurons': n_neurons,
                                'accuracy': evaluator.evaluate(mlp_template_model.transform(validation_df))})

evaluation_df = pd.DataFrame(evaluation_info)
for training_size in sorted(evaluation_df.training_size.unique()):
    fig, ax = plt.subplots(1, 1);
    evaluation_df.query('training_size == ' + str(training_size)).groupby(['dataset']).\
        plot(x='n_neurons', y='accuracy', ax=ax);
    plt.legend(['training', 'validation'], loc='upper left');
    plt.title('Training size: ' + str(int(training_size*100)) + '%');
    plt.ylabel('accuracy');
    plt.ylim([0, 1]);
    display()

for n_neurons in sorted(evaluation_df.n_neurons.unique()):
    fig, ax = plt.subplots(1, 1);
    evaluation_df.query('n_neurons == ' + str(n_neurons)).groupby(['dataset']).\
        plot(x='training_size', y='accuracy', ax=ax);
    plt.legend(['training', 'validation'], loc='upper left');
    plt.title('# Neurons: ' + str(n_neurons));
    plt.ylabel('accuracy');
    plt.ylim([0, 1]);
    display()

from PIL import Image
import requests
from io import BytesIO
# response = requests.get("http://images.all-free-download.com/images/graphicthumb/airplane_311727.jpg")
# response = requests.get("https://www.tugraz.at/uploads/pics/Alexander_by_Kanizaj_02.jpg")

# face
response = requests.get("https://www.sciencenewsforstudents.org/sites/default/files/scald-image/350_.inline2_beauty_w.png")
# motorbujke
response = requests.get("https://www.cubomoto.co.uk/img-src/_themev2-cubomoto-1613/theme/panel-1.png")
img = Image.open(BytesIO(response.content))

# convert to grayscale
gray_img = np.array(img.convert('P'))
plt.imshow(255-gray_img, 'gray')

shrinked_img = np.array((img.resize([40, 60]).convert('P')))
plt.imshow(shrinked_img, 'gray')
from pyspark.ml.linalg import Vectors
new_image = shrinked_img.flatten()
new_img_df = spark.createDataFrame([[Vectors.dense(new_image)]], ['raw_pixels'])

#********************************************
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

df = spark.createDataFrame([
    (0, "a"),
    (1, "b"),
    (2, "c"),
    (3, "a"),
    (4, "a"),
    (5, "c")
], ["id", "category"])

stringIndexer = StringIndexer(inputCol="category", outputCol="categoryIndex")
model = stringIndexer.fit(df)
indexed = model.transform(df)

encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
encoded = encoder.transform(indexed)
encoded.show()
VectorAssembler(inputCols=['categoryIndex', 'categoryVec']).transform(encoded).show()