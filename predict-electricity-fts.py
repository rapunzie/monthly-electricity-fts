#!/usr/bin/env python
# coding: utf-8

# ## Predict Electricity Production using Fuzzy Time Series
# 

# ### 1. Data Preprocessing

# In[97]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
from pylab import rcParams
rcParams['figure.figsize'] = 10, 7


# In[98]:


dataset = pd.read_csv("MES_0223.csv",delimiter=",", encoding='cp1252')


# In[99]:


dataset


# In[100]:


df = dataset.dropna()


# In[101]:


df


# In[102]:


balance = df["Balance"].unique()
print(balance)


# In[103]:


product = df["Product"].unique()
print(product)


# In[104]:


df = df.query("Balance == 'Net Electricity Production'")


# In[105]:


df


# In[106]:


balance = df["Balance"].unique()
print(balance)


# In[107]:


df = df.query("Product == 'Electricity'")


# In[108]:


df


# In[109]:


df = df.apply(lambda row: row[df['Country'].isin(['Australia', 'Austria', 'Belgium', 'Canada', 'Chile', 'Colombia', 'Costa Rica',
 'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany',
 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Japan', 'Korea', 'Latvia',
 'Lithuania', 'Luxembourg', 'Mexico', 'Netherlands', 'New Zealand', 'Norway',
 'Poland', 'Portugal', 'Slovak Republic', 'Slovenia', 'Spain', 'Sweden',
 'Switzerland', 'Republic of Turkiye', 'United Kingdom', 'United States', 'Argentina', 'Brazil', 'Bulgaria', "People's Republic of China",
 'Croatia', 'Cyprus', 'India', 'Malta', 'North Macedonia', 'Romania', 'Serbia'])])


# In[110]:


df


# In[111]:


country = df["Country"].unique()
print(country)


# In[112]:


##2 Cluster negara dengan 3 cluster: rendah, sedang, tinggi


# In[113]:


total = df.groupby("Country")["Value"].sum().reset_index()


# In[114]:


# Menentukan persentil atau titik potong
low_threshold = total["Value"].quantile(1/3)
high_threshold = total["Value"].quantile(2/3)

# Mengelompokkan negara-negara menjadi rendah, sedang, dan tinggi berdasarkan total produksi listrik
total["Group"] = pd.cut(total["Value"], bins=[0, low_threshold, high_threshold, float("inf")], labels=["Rendah", "Sedang", "Tinggi"])


# In[115]:


# Menyimpan dataframe ke dalam file CSV
total.to_csv("cluster.csv", index=False)


# In[116]:


# Membaca dataset
df2 = pd.read_csv("cluster_electricity.csv")

# Menghitung jumlah negara dalam setiap kelompok
group_counts = total["Group"].value_counts()


# In[117]:


# Membuat bar plot
plt.bar(group_counts.index, group_counts.values)
plt.xlabel("Indeks")
plt.ylabel("Negara")
plt.title("Cluster of Electricity Production")
plt.show()


# ## 2. Menentukan centroid dari setiap kelas

# In[118]:


# Menentukan centroid cluster
centroid_rendah = df2[df2["Group"] == "Rendah"]["Value"].mean()
centroid_sedang = df2[df2["Group"] == "Sedang"]["Value"].mean()
centroid_tinggi = df2[df2["Group"] == "Tinggi"]["Value"].mean()


# In[119]:


rb_centroid_rendah = centroid_rendah - 50
ra_centroid_rendah = centroid_rendah + 50


# In[120]:


centroid_rendah


# In[121]:


# Negara dengan produksi listrik kelas rendah
negara_rendah_df = df2[df2['Group'] == 'Rendah']
negara_rendah_df


# In[122]:


centroid_rendah


# In[123]:


# Negara dengan produksi listrik kelas rendah
negara_rendah = negara_rendah_df[negara_rendah_df['Country'] == 'Slovenia']
negara_rendah


# In[124]:


centroid_sedang


# In[125]:


# Negara dengan produksi listrik kelas sedang
negara_sedang_df = df2[df2['Group'] == 'Sedang']
negara_sedang_df


# In[126]:


negara_sedang = negara_sedang_df[negara_sedang_df['Country'] == 'Chile']
negara_sedang


# In[127]:


centroid_tinggi


# In[128]:


# Negara dengan produksi listrik kelas tinggi
negara_tinggi_df = df2[df2['Group'] == 'Tinggi']
negara_tinggi_df


# In[129]:


centroid_tinggi


# In[130]:


# Negara dengan produksi listrik kelas tinggi
negara_tinggi = negara_tinggi_df[negara_tinggi_df['Country'] == 'Japan']
negara_tinggi


# ## 3. FTS

# In[131]:


df


# In[132]:


from pyFTS.partitioners import Grid


# ## A. Negara dengan Kelas Produksi Rendah

# In[133]:


slovenia = df[df['Country'] == 'Slovenia']['Value'].values
time_slovenia = df[df['Country'] == 'Slovenia']['Time'].values


# In[134]:


# create plot for country_data for every category
import plotly.express as px

fig = px.line(slovenia, x=time_slovenia, y=slovenia)
fig.show()


# In[135]:


from tabulate import tabulate

# Membuat data
data = {'slovenia': slovenia, 'time_slovenia': time_slovenia}

# Mengonversi data menjadi bentuk yang sesuai untuk tabel
table_data = list(zip(data['slovenia'], data['time_slovenia']))

# Menampilkan tabel
table = tabulate(table_data, headers=['slovenia', 'time_slovenia'], tablefmt='pretty')
print(table)


# In[136]:


from tabulate import tabulate

# Membuat data
data = {'slovenia': slovenia, 'time_sloveia': time_slovenia}

# Mengonversi data menjadi bentuk yang sesuai untuk tabel
table_data = list(zip(data['slovenia'], data['time_slovenia']))

# Menghasilkan string HTML untuk tabel
html_table = tabulate(table_data, headers=['slovenia', 'time_slovenia'], tablefmt='html')

# Mencetak tabel HTML
print(html_table)


# In[27]:


df.to_html('table.html', index=False)


# In[137]:


# make fuzzy set
fs_rendah = Grid.GridPartitioner(
    data=df[df['Country'] == 'Slovenia']['Value'].values, npart=10)

# make fuzzy set plotly from fs
# fig = go.Figure()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25, 10])

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, len(df[df['Country'] == 'Slovenia']['Value'].values))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

fs_rendah.plot(ax)
ax.set_title("Slovenia; Produksi Listrik Kelas Rendah", fontsize=30)

plt.show()


# In[138]:


print(fs_rendah)


# In[139]:


from pyFTS.models import chen

model_rendah = chen.ConventionalFTS(partitioner=fs_rendah)
model_rendah.fit(df[df['Country'] == 'Slovenia']['Value'].values)
print(model_rendah)


# In[140]:


prediction_rendah = model_rendah.predict(df[df['Country'] == 'Slovenia']['Value'].values)


# In[141]:


fts_dates1 = df[df['Country'] == 'Slovenia']['Time'].values
data_rendah = pd.DataFrame({
    'date': fts_dates1,
    'actual': df[df['Country'] == 'Slovenia']['Value'].values,
    'forecast': prediction_rendah
})


# In[142]:


# Plot the data using Plotly
import plotly.graph_objects as go
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=data_rendah['date'], y=data_rendah['actual'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(
    x=data_rendah['date'], y=data_rendah['forecast'], mode='lines', name='Forecast'))

# Set layout
fig.update_layout(
    xaxis=dict(
        tickangle=45,
        tickfont=dict(size=12),
        tickformat='%Y-%m-%d'
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=20)
    ),
    legend=dict(
        x=1,
        y=0,
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12)
    )
)

# Add vertical lines with alternating styles
for i in range(len(fts_dates1)):
    if i % 2 == 0:
        fig.add_shape(type='line', x0=fts_dates1[i], y0=0, x1=fts_dates1[i], y1=1, line=dict(
            color='black', width=1, dash='solid'))
    else:
        fig.add_shape(type='line', x0=fts_dates1[i], y0=0, x1=fts_dates1[i], y1=1, line=dict(
            color='black', width=1, dash='dash'))

fig.show()


# In[143]:


# show model performance
from pyFTS.benchmarks import Measures

print("RMSE : ", Measures.rmse(data_rendah['actual'], data_rendah['forecast']))
print("MAPE : ", Measures.mape(data_rendah['actual'], data_rendah['forecast']))


# ## B. Negara dengan Kelas Produksi Listrik Sedang

# In[158]:


chile = df[df['Country'] == 'Chile']['Value'].values
time_chile = df[df['Country'] == 'Chile']['Time'].values


# In[159]:


# create plot for country_data for every category
import plotly.express as px

fig = px.line(chile, x=time_chile, y=chile)
fig.show()


# In[160]:


# make fuzzy set
fs_sedang = Grid.GridPartitioner(
    data=df[df['Country'] == 'Chile']['Value'].values, npart=10)

# make fuzzy set plotly from fs
# fig = go.Figure()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25, 10])

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, len(df[df['Country'] == 'Chile']['Value'].values))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

fs_sedang.plot(ax)
ax.set_title("Chile; Produksi Listrik Kelas Sedang", fontsize=30)

plt.show()


# In[161]:


print(fs_sedang)


# In[162]:


from pyFTS.models import chen

model_sedang = chen.ConventionalFTS(partitioner=fs_sedang)
model_sedang.fit(df[df['Country'] == 'Chile']['Value'].values)
print(model_sedang)


# In[163]:


prediction = model_sedang.predict(df[df['Country'] == 'Chile']['Value'].values)


# In[164]:


fts_sedang = df[df['Country'] == 'Chile']['Time'].values
data_sedang = pd.DataFrame({
    'date': fts_sedang,
    'actual': df[df['Country'] == 'Chile']['Value'].values,
    'forecast': prediction
})


# In[167]:


# Plot the data using Plotly
import plotly.graph_objects as go
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=data_sedang['date'], y=data_sedang['actual'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(
    x=data_sedang['date'], y=data_sedang['forecast'], mode='lines', name='Forecast'))

# Set layout
fig.update_layout(
    xaxis=dict(
        tickangle=45,
        tickfont=dict(size=12),
        tickformat='%Y-%m-%d'
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=20)
    ),
    legend=dict(
        x=1,
        y=0,
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12)
    )
)

# Add vertical lines with alternating styles
for i in range(len(fts_sedang)):
    if i % 2 == 0:
        fig.add_shape(type='line', x0=fts_sedang[i], y0=0, x1=fts_sedang[i], y1=1, line=dict(
            color='black', width=1, dash='solid'))
    else:
        fig.add_shape(type='line', x0=fts_sedang[i], y0=0, x1=fts_sedang[i], y1=1, line=dict(
            color='black', width=1, dash='dash'))

fig.show()


# In[168]:


# show model performance
from pyFTS.benchmarks import Measures

print("RMSE : ", Measures.rmse(data_sedang['actual'], data_sedang['forecast']))
print("MAPE : ", Measures.mape(data_sedang['actual'], data_sedang['forecast']))


# ### Negara dengan Kelas Produksi Listrik Tinggi

# In[186]:


japan = df[df['Country'] == 'Japan']['Value'].values
time_japan = df[df['Country'] == 'Japan']['Time'].values


# In[188]:


# create plot for country_data for every category
import plotly.express as px

fig = px.line(japan, x=time_japan, y=japan)
fig.show()


# In[190]:


# make fuzzy set
fs_tinggi = Grid.GridPartitioner(
    data=df[df['Country'] == 'Japan']['Value'].values, npart=10)

# make fuzzy set plotly from fs
# fig = go.Figure()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=[25, 10])

ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, len(df[df['Country'] == 'Japan']['Value'].values))
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

fs_tinggi.plot(ax)
ax.set_title("Japan; Produksi Listrik Kelas Tinggi", fontsize=30)

plt.show()


# In[191]:


print(fs_tinggi)


# In[192]:


from pyFTS.models import chen

model_tinggi = chen.ConventionalFTS(partitioner=fs_tinggi)
model_tinggi.fit(df[df['Country'] == 'Japan']['Value'].values)
print(model_tinggi)


# In[193]:


prediction = model_tinggi.predict(df[df['Country'] == 'Japan']['Value'].values)


# In[194]:


fts_dates3 = df[df['Country'] == 'Japan']['Time'].values
data = pd.DataFrame({
    'date': fts_dates3,
    'actual': df[df['Country'] == 'Japan']['Value'].values,
    'forecast': prediction
})


# In[196]:


# Plot the data using Plotly
import plotly.graph_objects as go
fig = go.Figure()

# Add actual data
fig.add_trace(go.Scatter(
    x=data['date'], y=data['actual'], mode='lines', name='Actual'))

# Add forecast data
fig.add_trace(go.Scatter(
    x=data['date'], y=data['forecast'], mode='lines', name='Forecast'))

# Set layout
fig.update_layout(
    xaxis=dict(
        tickangle=45,
        tickfont=dict(size=12),
        tickformat='%Y-%m-%d'
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=20)
    ),
    legend=dict(
        x=1,
        y=0,
        xanchor='right',
        yanchor='bottom',
        font=dict(size=12)
    )
)

# Add vertical lines with alternating styles
for i in range(len(fts_dates3)):
    if i % 2 == 0:
        fig.add_shape(type='line', x0=fts_dates3[i], y0=0, x1=fts_dates3[i], y1=1, line=dict(
            color='black', width=1, dash='solid'))
    else:
        fig.add_shape(type='line', x0=fts_dates3[i], y0=0, x1=fts_dates3[i], y1=1, line=dict(
            color='black', width=1, dash='dash'))

fig.show()


# In[197]:


# show model performance
from pyFTS.benchmarks import Measures

print("RMSE : ", Measures.rmse(data['actual'], data['forecast']))
print("MAPE : ", Measures.mape(data['actual'], data['forecast']))


# ## Forecasting

# ## Slovenia

# In[144]:


slovenia = df[df['Country'] == 'Slovenia']['Value'].values
model_rendah = chen.ConventionalFTS(partitioner=fs_rendah)
model_rendah.fit(slovenia)


# In[145]:


prediction_slovenia = model_rendah.predict(slovenia)
actual_slovenia = slovenia
fts_dates2 = df[df['Country'] == 'Slovenia']['Time'].values


# In[153]:


from datetime import datetime 
month_year_str = 'March 2023'
month_year = datetime.strptime(month_year_str, '%B %Y')
tahun = month_year.year
bulan = month_year.month


# In[154]:


last_date_str = 'February 2010'
last_date = datetime.strptime(last_date_str, '%B %Y')
last_month = last_date.month


# In[179]:


# forecasting for 5 years
tahun_bulan_str = f"{bulan:02d}-{tahun}"

forecasting = model_rendah.forecast(slovenia, steps=60)
start_year = data_rendah['date'].iloc[-1]
last_year = int(last_date_str.split()[-1])
start_year = last_year + 1

forecasting_dates = pd.date_range(
    start=tahun_bulan_str, periods=60, freq='M').strftime("%B, %Y").tolist()

forecast_data = dict(zip(forecasting_dates, forecasting))

# show the line chart with plotly
fig = go.Figure()


# In[180]:


fig.add_trace(go.Scatter(
    x=forecasting_dates, y=forecasting, mode='lines', name='Forecasting'))


# ## Chile

# In[181]:


chile = df[df['Country'] == 'Chile']['Value'].values
model_sedang = chen.ConventionalFTS(partitioner=fs_sedang)
model_sedang.fit(chile)


# In[182]:


prediction_chile = model_sedang.predict(chile)
actual_chile = chile
fts_sedang = df[df['Country'] == 'Chile']['Time'].values


# In[183]:


last_date_str = 'February 2010'
last_date = datetime.strptime(last_date_str, '%B %Y')
last_month = last_date.month


# In[184]:


# forecasting for 5 years
tahun_bulan_str = f"{bulan:02d}-{tahun}"

forecasting = model_sedang.forecast(chile, steps=60)
start_year = data_sedang['date'].iloc[-1]
last_year = int(last_date_str.split()[-1])
start_year = last_year + 1

forecasting_dates = pd.date_range(
    start=tahun_bulan_str, periods=60, freq='M').strftime("%B, %Y").tolist()

forecast_data = dict(zip(forecasting_dates, forecasting))

# show the line chart with plotly
fig = go.Figure()


# In[185]:


fig.add_trace(go.Scatter(
    x=forecasting_dates, y=forecasting, mode='lines', name='Forecasting'))


# ## Japan

# In[198]:


df


# In[199]:


japan


# In[201]:


data_model = df[df['Country'] == 'Japan']['Value'].values
model_tinggi = chen.ConventionalFTS(partitioner=fs_tinggi)
model_tinggi.fit(data_model)


# In[202]:


prediction = model_tinggi.predict(data_model)
actual = data_model
fts_dates3 = df[df['Country'] == 'Japan']['Time'].values


# In[203]:


from datetime import datetime 
month_year_str = 'March 2023'
month_year = datetime.strptime(month_year_str, '%B %Y')
tahun = month_year.year
bulan = month_year.month


# In[204]:


last_date_str = 'February 2010'
last_date = datetime.strptime(last_date_str, '%B %Y')
last_month = last_date.month


# In[205]:


# forecasting for 5 years
tahun_bulan_str = f"{bulan:02d}-{tahun}"

forecasting = model_tinggi.forecast(data_model, steps=60)
start_year = data['date'].iloc[-1]
last_year = int(last_date_str.split()[-1])
start_year = last_year + 1

forecasting_dates = pd.date_range(
    start=tahun_bulan_str, periods=60, freq='M').strftime("%B, %Y").tolist()

forecast_data = dict(zip(forecasting_dates, forecasting))

# show the line chart with plotly
fig = go.Figure()


# In[206]:


fig.add_trace(go.Scatter(
    x=forecasting_dates, y=forecasting, mode='lines', name='Forecasting'))


# In[312]:


# show model performance
from pyFTS.benchmarks import Measures

print("RMSE : ", forecast_data)
print("MAPE : ", forecast_data)


# In[ ]:




