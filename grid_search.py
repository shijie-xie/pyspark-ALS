#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import pickle
from pyspark.sql import SparkSession

# $example on$
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import gc
# $example off$
def start_spark():
    spark = SparkSession\
        .builder\
        .appName("ALS")\
        .config("spark.driver.extraJavaOptions","-Xss300M")\
        .config("spark.driver.memory", "15g")\
	.getOrCreate()
    #spark.executor.memory=12g

    # $example on$
    lines = spark.read.text("data/data.txt").rdd
    parts = lines.map(lambda row: row.value.split(","))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    (training, test) = ratings.randomSplit([0.75, 0.25])
    return training,test,spark




def calculate(training,test,test_rank = 10,test_iter= 50,test_reg = 0.1):
    

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    import time

    time_start=time.time()
    als = ALS(rank= test_rank,maxIter=test_iter, regParam=test_reg, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop",intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK')
    model = als.fit(training)
    time_end=time.time()
    print('time cost',time_end-time_start,'s')
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse)+'\n\n')

    # Generate top 10 movie recommendations for each user
    #userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    #movieRecs = model.recommendForAllItems(10)

    # Generate top 10 movie recommendations for a specified set of users
    #users = ratings.select(als.getUserCol()).distinct().limit(3)
    #userSubsetRecs = model.recommendForUserSubset(users, 10)
    # Generate top 10 user recommendations for a specified set of movies
    #movies = ratings.select(als.getItemCol()).distinct().limit(3)
    #movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    # $example off$
    #userRecs.show()
    #movieRecs.show()
    #userSubsetRecs.show()
    #movieSubSetRecs.show()
    return rmse,time_end-time_start

training,test,spark = start_spark()
test_reg = 0.01
test_rank = 5
test_iter = 50
for test_rank in [1,2,3,4,5]:
    for test_reg in [0.01,0.03,0.003]:
        for test_iter in [5,10,20,30,40]:
            print(test_rank,test_reg,test_iter)
            rmse,times = calculate(training,test,test_rank ,test_iter,test_reg )
            with open('./grid_test/'+str(test_reg)+'_' + str(test_rank) + '_' + str(test_iter)+'.txt','w') as f:    #设置文件对象
                f.write(str(rmse)+'\n'+str(times)) 
        
            
        
spark.stop()

