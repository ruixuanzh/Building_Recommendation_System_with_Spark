# Author: Ruixuan Zhang 

from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# data was stored in amazon web service S3 
f1 = "s3n://ruixuanrecommendation/ratings_Books.csv"

#Create SparkContext. 
conf = SparkConf().setMaster("yarn-client").setAppName("RuixuanProject")

# conf.set("spark.executor.memory", "8G")
# conf.set("spark.driver.memory", "20G")
# conf.set("spark.executor.cores", "4")
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)


num_partition = 10
rating = sc.textFile(f1)
rating_rows = rating.map(lambda x: x.split(",")).map(lambda x: [x[0], x[1], x[2]]).coalesce(num_partition)

# Since the algorithm only take numerical 'itemId' and 'personId' as input
# we created on unique numerical id for each reviewer
rating1 = rating_rows.map(lambda x: x[0]).distinct().zipWithUniqueId().partitionBy(num_partition)
# similarly, create one unique numerical id for each item
rating2 = rating_rows.map(lambda x: x[1]).distinct().zipWithUniqueId().partitionBy(num_partition)

# convert 3 RDD to Dataframe
def toFloatSafe(inval):
    try:
        return float(inval)
    except ValueError:
        return None

def person(r):
    return Row(
    r[0],
    float(r[1]))

def book(r):
    return Row(
    r[0],
    float(r[1]))
    
def rating(r):
    return Row(
    r[0],
    r[1],
    toFloatSafe(r[2]))

personSchema = StructType([
        StructField('reviewerID', StringType(), False),
        StructField('userID', FloatType(), False)
    ])

BookSchema = StructType([
        StructField("asin", StringType(), False),
        StructField('itemID', FloatType(), False)
    ])

RatingSchema = StructType([
  StructField('reviewerID', StringType(), False),
  StructField("asin", StringType(), False),
  StructField("rating", FloatType(), True)
  ])

personRDD = rating1.map(lambda x : person(x))
personDF = sqlContext.createDataFrame(personRDD, personSchema)
            
bookRDD = rating2.map(lambda x : book(x))
bookDF = sqlContext.createDataFrame(bookRDD, BookSchema)

ratingRDD = rating_rows.map(lambda x: rating(x)) 
ratingDF = sqlContext.createDataFrame(ratingRDD, RatingSchema)


# join three DF together
tmpDF1 = ratingDF.join(personDF, 'reviewerID')
DF_final = tmpDF1.join(bookDF, 'asin')
df = DF_final.select('userID','itemID','rating')

# sample
splits = df.randomSplit([0.8, 0.2])
train = splits[0].cache()
test = splits[1].cache() # cache the training and set set into memory


# Build an recommendation engine with ALS
als = ALS(rank=20, maxIter=200, regParam=0.1, userCol="userID", itemCol="itemID", ratingCol="rating")
model = als.fit(train)
# predict on the test set
prediction = model.transform(test)
prediction_noNaN = prediction.filter(prediction.prediction != 'NaN' )
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(prediction_noNaN)
print("Root-mean-square error = " + str(rmse))

# cross-validation, parameter tunning
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [10, 30, 40]) \
    .addGrid(als.regParam, [0.1]) \
    .addGrid(als.maxIter, [20]) \
    .build()
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator= RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction"),
                          numFolds=4) # use 5 fold cross-validation


# Run cross-validation, and choose the best set of parameters.
cvModel = crossval.fit(train)
prediction_2 = cvModel.transform(test)
prediction_2_withoutNaN = prediction_2.filter(prediction_2.prediction != 'NaN' )
rmse = evaluator.evaluate(prediction_2_withoutNaN)
print crossval
print rmse

sc.stop()





