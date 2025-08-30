import pandas as pd

df = pd.DataFrame({
    'Caffeine(mg)' : [94.2,93.7,95.5,93.9,94.0,95.2,94.7,93.5,92.8,94.4,
                      93.8,94.6,93.3,95.1,94.3,94.9,93.9,94.8,95.0,94.2,
                      93.7,94.4,95.1,94.0,93.6]
})

print(df.head())

from scipy import stats

#1
print(df['Caffeine(mg)'].mean())

#2

from scipy import stats

print(stats.shapiro(df['Caffeine(mg)']).pvalue) # 정규성 만족 O

#3

from scipy.stats import ttest_1samp

print(ttest_1samp(df['Caffeine(mg)'] , 95 , alternative='less').statistic) #


#4
print(ttest_1samp(df['Caffeine(mg)'] , 95 , alternative='less').pvalue) #

#5
#기각

from scipy.stats import ttest_ind

df = pd.DataFrame({
    '충전기' : ['New']*10 + ['Old']*10 ,
    '충전시간' : [1.5,1.6,1.4,1.7,1.5,1.6,1.7,1.4,1.6,1.5,
                 1.7,1.8,1.7,1.9,1.8,1.7,1.8,1.9,1.7,1.6]
 })

print(df.head())

#1

cond = df['충전기'] == 'New'

cond2 = df['충전기'] == 'Old'

print(ttest_ind( df[cond]['충전시간'], df[cond2]['충전시간'], alternative='less'))

#1

#-4.582575694955849

#2

#0.00011546547787696304

#3
#기각


df = pd.DataFrame({
    'User' : list(range(1,11)),
    '기존방법' : [60.4,60.7,60.5,60.3,60.8,60.6,60.2,60.5,60.7,60.4],
    '새로운방법' : [59.8,60.2,60.1,59.9,59.7,58.4,57.0,60.3,59.6,59.8]
})

from scipy.stats import ttest_rel

#1

print((df['새로운방법']-df['기존방법']).mean())

#2
print(ttest_rel(df['새로운방법'] , df['기존방법'] , alternative='less').statistic)

#3
print(ttest_rel(df['새로운방법'] , df['기존방법'] , alternative='less').pvalue)

#4

print('기각')

from statsmodels.formula.api import ols

df= pd.read_csv('math.csv')

print(df.head())

from scipy import stats
cond = df['groups'] == 'group_A'
cond2 = df['groups'] == 'group_B'
cond3 = df['groups'] == 'group_C'
cond4 = df['groups'] == 'group_D'
#1

print(stats.shapiro(df[cond]['scores']))
print(stats.shapiro(df[cond2]['scores']))
print(stats.shapiro(df[cond3]['scores']))
print(stats.shapiro(df[cond4]['scores']))

##pvalue=0.41357170430459295

#2

print(stats.levene(df[cond]['scores'] ,df[cond2]['scores']  ,df[cond3]['scores']  ,df[cond4]['scores']  ))


#3
model = ols('scores ~ groups',df).fit()
#print(dir(model))
print(model.summary())

from statsmodels.stats.anova import anova_lm

print(anova_lm(model))

#기각

df= pd.read_csv('tomato2.csv')

print(df.head())

#1

from statsmodels.formula.api import ols

model = ols('수확량 ~ C(비료유형) *C(물주기)',df).fit()

from statsmodels.stats.anova import anova_lm
print(model.summary())
print(anova_lm(model))

#1
##3.184685

#2
#0.059334

#3
#채택

#4
model = ols('수확량 ~ 물주기',df).fit()

from statsmodels.stats.anova import anova_lm

print(anova_lm(model))

#7.447557

#5
## 0.009984

#6
#기각

#7
model = ols('수확량 ~ 비료유형 + 물주기 + 비료유형:물주기',df).fit()

from statsmodels.stats.anova import anova_lm

print(anova_lm(model))

#1.301171

#8
#0.287135

#9
#채택


from scipy.stats import chisquare

observed = [550,250,100,70,30]
expected = [1000*0.60 , 1000*0.25 , 1000*0.08 , 1000*0.05 , 1000*0.02]

print(chisquare(observed,expected))

#1

print(30/1000)
#2

##22.166666666666668


#3
##pvalue=0.00018567620386641427

#4
##기각

df = pd.DataFrame({
    '캠프' : ['빅분기'] * 80 + ['정처기'] * 100,
    '등록여부' : ['등록'] *50 + ['등록안함']*30 + ['등록']*60 + ['등록안함']*40
})

print(df.head())
df = pd.crosstab(df['캠프'],df['등록여부'])
# cond = df['캠프'] == '빅분기'
# cond2 = df['캠프'] == '정처기'

from scipy.stats import chi2_contingency

print(chi2_contingency(df))

#1

##statistic=0.03535714285714309

#2
##pvalue=0.8508492527705047

#3
##채택



#1
df = pd.DataFrame({
    '할인율' : [28,24,13,0,27,30,10,16,6,5,7,11,11,30,25,4,7,24,19,21,6,10,26,13,15,6,12,6,20,2],
    '온도' : [15,34,15,22,29,30,14,17,28,29,19,19,34,10,29,28,12,25,32,28,22,16,30,11,16,18,16,33,12,22],
    '광고비' : [342,666,224,764,148,499,711,596,797,484,986,347,146,362,642,591,846,260,560,941,469,309,730,305,892,147,887,526,525,884],
    '주문량' : [635,958,525,25,607,872,858,732,1082,863,904,686,699,615,893,830,856,679,918,951,789,583,988,631,866,549,910,946,647,943]
})

print(round(df['온도'].corr(df['할인율']),2))

from statsmodels.formula.api import ols

model = ols('주문량 ~ 온도+광고비+할인율',df).fit()

print(model.summary())


#2
##  0.40

#3

#온도 9.4798
#광고비  0.4148 
#주문량 4.2068

#4
print(round( -89.7587,4))

#5

## 0.029 유의함

#6

data = pd.DataFrame({
    '할인율' : [10],
    '온도' : [20] ,
    '광고비' : [500]}) 

result = model.predict(data)

print(result[0])

#7

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['주문량'] , model.predict(df))

print(mse * len(df))

#8

print(mse)

#9
print(model.conf_int(0.1))

## 2.490702   16.468984

#10

data = pd.DataFrame({
    '할인율' : [15],
    '온도' : [25] ,
    '광고비' : [300]}) 

result = model.predict(data)

print(result)

#11

print(model.summary())



df = pd.read_csv('customer_travel.csv')

print(df.head())
ldf = int(len(df)/2)
a = df.iloc[:ldf]
df2 = df.iloc[ldf:]


print(df2)

#1

from statsmodels.formula.api import logit

model = logit('target ~ age+ service + social + booked ',a).fit()

print(model.summary())

## 2

#2
model = logit('target ~ age  + booked ',a).fit()

print(model.summary())
## age

#3
## booked

#4
## -211.43

#5

print(-211.43*-2)

#7
print( 2.4581 -0.1025-0.9461  )

#8

result2 = model.predict(df2)
result2 = (result2>0.5).astype(int)
print(result2)

from sklearn.metrics import accuracy_score

print(accuracy_score(result2,df2['target']))

#9

print(1- accuracy_score(result2,df2['target']))