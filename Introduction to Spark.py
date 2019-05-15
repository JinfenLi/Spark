from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('spark-intro').getOrCreate()
sc = spark.sparkContext
rdd = sc.parallelize(range(20))
rdd.first()
rdd.take(2)
rdd.collect()
def less_than_10(x):
    if x < 10:
        return True
    else:
        return False
# show that it is lazy evaluation
rdd.filter(less_than_10)
rdd.filter(less_than_10).collect()
rdd.filter(less_than_10).count()
def square(x):
    return x*x # x**2
rdd.map(square).collect()

# read from hdfs
sotu_rdd = sc.textFile('./datasets/shakespeare.txt')

#map_reduce
example_dataset = [
['JAN', 'NY', 3.],
['JAN', 'PA', 1.],
['JAN', 'NJ', 2.],
['JAN', 'CT', 4.],
['FEB', 'PA', 1.],
['FEB', 'NJ', 1.],
['FEB', 'NY', 2.],
['FEB', 'VT', 1.],
['MAR', 'NJ', 2.],
['MAR', 'NY', 1.],
['MAR', 'VT', 2.],
['MAR', 'PA', 3.]]
dataset_rdd = sc.parallelize(example_dataset)
def map_func(row):
    return [row[0], row[2]]
dataset_rdd.map(map_func).take(5)

def reduce_func(value1, value2):
    return value1 + value2

dataset_rdd.map(map_func).reduceByKey(reduce_func).collect()

from pyspark.sql import Row
raw_data = [Row(state='NY', month='JAN', orders=3),
            Row(state='NJ', month='JAN', orders=4),
            Row(state='NY', month='FEB', orders=5),
           ]
data_df = spark.createDataFrame(raw_data)