# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 12:42:45 2021

@author: Clauber
"""
'1.0 Descrevendo os dados'
import pandas as pd
import numpy as np
import inflection
import math
import seaborn as sns
import matplotlib.pyplot as plt


df_sales_raw = pd.read_csv('train.csv', low_memory=False)
df_store_raw = pd.read_csv('store.csv', low_memory=False)
#1.1 merge
df_raw = pd.merge(df_sales_raw, df_store_raw, how = 'left', on = 'Store')
#1.2 cópia de segurança
df1 = df_raw.copy()
#1.3 preparando para renomear as colunas usando o método underscore da função snekecase da biblioteca inflection
cols_old = ['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
       'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
       'CompetitionDistance', 'CompetitionOpenSinceMonth',
       'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
       'Promo2SinceYear', 'PromoInterval']
snekecase = lambda x: inflection.underscore(x)
#1.4 a função map faz o mapeamento de todas as palavras da lista cols_old
cols_new = list(map(snekecase, cols_old))
'1.1 #1.5 rename'
df1.columns = cols_new
'1.2#1.6 Data Dimensions'
print('Number of Rowns: {}'.format(df1.shape[0]))
print('Number of Cols: {}'.format(df1.shape[1]))
'1.3#1.7 Data Types'
df1['date'] = pd.to_datetime(df1['date'])
df1.dtypes
'1.4#1.8 Check NAN'
df1.isna().sum()
#1.5 Fillout NAN
#var competition_distance  (a loja que faz competição mais prox) 
#distancia maior entre a loja e o competidor mais prox é 75860.0m
df1['competition_distance'].max()
#max_value if math.isnan(df1['competition_distance'])
df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000 if math.isnan(x) else x)
df1.isna().sum()
#var competition_open_since_month(qual o mês e ano que o competidor mais prox foi aberto)    
df1['competition_open_since_month'] = df1.apply(lambda x: x['date'].month if math.isnan(x['competition_open_since_month']) else x['competition_open_since_month'], axis = 1)
     
#var competition_open_since_year em ano
df1['competition_open_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['competition_open_since_year']) else x['competition_open_since_year'], axis = 1)                                
#promo2_since_week(continuação de uma promoção para algumas lojas 0 n ta part se for 1 ta part )     
df1['promo2_since_week'] = df1.apply(lambda x: x['date'].week if math.isnan(x['promo2_since_week']) else x['promo2_since_week'], axis = 1)          
#promo2_since_year   em anos      
df1['promo2_since_year'] = df1.apply(lambda x: x['date'].year if math.isnan(x['promo2_since_year']) else x['promo2_since_year'], axis = 1)               
#promo_interval meses da promoção
month_map =  {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
df1['promo_interval'].fillna(0, inplace=True) #substitui na por zero
df1['month_map'] = df1['date'].dt.month.map(month_map)
df1.sample(5).T 
df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x ['promo_interval'].split(',') else 0, axis = 1)            
#1.6 Change type
df1.dtypes 
df1['competition_open_since_month'] = df1['competition_open_since_month'].astype('int64')
df1['competition_open_since_year'] = df1['competition_open_since_year'].astype('int64')
df1['promo2_since_week'] = df1['promo2_since_week'].astype('int64')
df1['promo2_since_year'] = df1['promo2_since_year'].astype('int64')
#1.7 Descriptive Statistical Types
#separação das var num e categóricas
num_attributes = df1.select_dtypes(include = ['int64','float64'])
cat_attributes = df1.select_dtypes(exclude = ['int64','float64', 'datetime64[ns]'])
cat_attributes.sample(2)
num_attributes.sample(2)
#1.7.1 Numerical attributes
#central tendency - mean, median
ct1 = pd.DataFrame(num_attributes.apply(np.mean)).T #T transpondo 
ct2 = pd.DataFrame(num_attributes.apply(np.median)).T
#dispercion - std(desvio padrão), min, max, range, skew, kurtosis 
d1 = pd.DataFrame(num_attributes.apply(np.std)).T
d2 = pd.DataFrame(num_attributes.apply(min)).T
d3 = pd.DataFrame(num_attributes.apply(max)).T
d4 = pd.DataFrame(num_attributes.apply(lambda x: x.max() - x.min())).T
d5 = pd.DataFrame(num_attributes.apply(lambda x: x.skew())).T
d6 = pd.DataFrame(num_attributes.apply(lambda x: x.kurtosis)).T
#concatenate
m = pd.concat([d2,d3,d4,ct1,ct2,d1,d5,d6]).T.reset_index()
m.columns = ['attributes','min','max','range', ',mean', 'median', 'std', 'skew','kurtosis']
sns.distplot(df1['competition_distance'])
#1.7.2 Categorical attributes
#quantos níveis cada var categórica tem shape para pegar os valores unicos
cat_attributes.apply(lambda x: x.unique().shape)
#elaboração do boxplot comparação delas com as vendas, outlier é o desvio padrão x3
aux1 = df1[(df1['state_holiday'] !='0')] & (df1['sales'] > 0)
plt.subplot(1,3,1)
sns.boxplot(x='store_holiday', y='sales',data=aux1)
plt.subplot(1,3,2)
sns.boxplot(x='store_type', y='sales',data=aux1)
plt.subplot(1,3,3)
sns.boxplot(x='store_assortment', y='sales',data=aux1)
