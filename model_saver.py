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
from pyspark import SparkContext
# $example on$
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import gc
# $example off$
def start_spark(spark):
    
    #spark.executor.memory=12g

    # $example on$
    lines = spark.read.text("data/full_data.txt").rdd
    parts = lines.map(lambda row: row.value.split(","))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                         rating=float(p[2])))
    ratings = spark.createDataFrame(ratingsRDD)
    
    return ratings




def calculate(spark,training,test_rank = 10,test_iter= 50,test_reg = 0.1):
    

    # Build the recommendation model using ALS on the training data
    # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    import time

    time_start=time.time()
    als = ALS(rank= test_rank,maxIter=test_iter, regParam=test_reg, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop",intermediateStorageLevel='MEMORY_AND_DISK', finalStorageLevel='MEMORY_AND_DISK')
    model = als.fit(training)
    model.save("./myCollaborativeFilter")
    time_end=time.time()
    print('time cost',time_end-time_start,'s')

    return time_end-time_start

spark = SparkSession\
        .builder\
        .appName("ALS")\
        .config("spark.driver.extraJavaOptions","-Xss300M")\
        .config("spark.driver.memory", "110g")\
	    .getOrCreate()

training = start_spark(spark)
for test_rank in [1]:
    for test_reg in [0.01]:
        for test_iter in [25]:
            print(test_rank,test_reg,test_iter)
            times = calculate(spark,training,test_rank ,test_iter,test_reg )
            print('time cost:'+ str(times))
        

        
spark.stop()

