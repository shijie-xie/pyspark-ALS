from fastparquet import ParquetFile
import snappy
import pandas as pd
import numpy as np
import time
import pickle
import os
def snappy_decompress(data, uncompressed_size):
    return snappy.decompress(data)

def read_single(n,type):
    #print(type)
    if type == 'item':
        pf = ParquetFile('/itemFactors/part-0000'+str(n)+'-bb0e8317-d384-4c08-824c-0b2a8661846f-c000.snappy.parquet')
        return pf.to_pandas()
    elif type =='user':
        pf = ParquetFile('/userFactors/part-0000'+str(n)+'-e7a03551-5ae9-4231-b614-549034330d20-c000.snappy.parquet')
        return pf.to_pandas()            
    return -1

def read_df(type):
    df = read_single(0,type)
    for i in range(1,10):
        dfnew = read_single(i,type)
        df = pd.concat([df,dfnew],axis=0)
    #df.rename(columns={'features':type},inplace=True)
    return df

def process(type):
    x = read_df(type)
    x = x.sort_values(['id'])
    tem = x.features.tolist()
    #print("resth")
    size = len(tem)
    item = np.zeros(size)
    for i in range (0,size):
        item[i] = tem[i][0]
    #item = np.array(list(sum(tem, [])))
    return item



def data_generator():
    itemlist = np.random.randint(0,9999,size = (1000000))
    userlist = np.random.randint(0,6999999,size = (1000000))

    ll = []
    for i in range(0,100000): 
        ll.append([itemlist[i],userlist[i]])
    #print(ll)
    pickle.dump(ll,open('data.pkl','wb'))
    os.system('tar -czvf data.tar.gz data.pkl ')
    os.system('rm data.pkl')


def time_test():
    item = process('item')
    user = process('user')
    print(item.shape)
    print(user.shape)
    result = np.zeros(100000)
    time1 = time.time()
    os.system('tar -xzf data.tar.gz')
    predict_list = pickle.load(open('data.pkl','rb') )  
    for i,uu in enumerate(predict_list):
        result[i] = item[uu[0]]*user[uu[1]]
    pickle.dump(result,open('timer.pkl','wb'))
    os.system('tar -czf timer.tar.gz timer.pkl ')
    time2 = time.time()
    print(time2-time1)
    os.system('rm data.pkl')

item = process('item')
user = process('user')
u1 = user[0:4000000]
u2 = user[4000000:7000000]
print(u1.shape)
print(u2.shape)
pickle.dump(item,open('./last_item.pkl','wb'))
pickle.dump(user,open('./last_u_Q1.pkl','wb'))
pickle.dump(user,open('./last_u_Q2.pkl','wb'))
