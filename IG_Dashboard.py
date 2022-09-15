# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 18:18:31 2022

@author: visha
"""

# Importing required Libraries
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import calendar

st.set_page_config(layout="wide")

# Load data
@st.cache(allow_output_mutation=True)

def load_data():
       df_followers = pd.read_excel("cleaned_followers_data.xlsx")
       df_followers['Date_Time'] = pd.to_datetime(df_followers['Date_Time'])
       df_followers = df_followers[~(df_followers['Date_Time'] >= '2022-09-01')]
       df_followers_asc = (df_followers.assign(Date=df_followers['Date_Time'].dt.date,
                            Time=df_followers['Date_Time'].dt.time)
                            .sort_values(['Date','Time'], ascending=[True, True]))
       df_followers_asc['Date'] = pd.to_datetime(df_followers_asc['Date'])
       df_followers_timeseries = df_followers_asc[['Date','Followers']]
       df_followers_timeseries = pd.pivot_table(df_followers_timeseries, index=['Date'], aggfunc='count').reset_index(level=0)
       df_followers_timeseries['Year'] = df_followers_timeseries['Date'].dt.year
       df_posts = pd.read_excel("cleaned_posts_data.xlsx")
       df_posts['Date_Time'] = pd.to_datetime(df_posts['Date_Time'])
       df_posts_dsc = (df_posts.assign(Date=df_posts['Date_Time'].dt.date,
                                   Time=df_posts['Date_Time'].dt.time)
                                   .sort_values(['Date','Time'], ascending =[False, False]))
       df_posts_dsc['Date'] = pd.to_datetime(df_posts_dsc['Date'])
       df_posts_dsc[['Profile Visits', 'Impressions', 
                     'Follows', 'Accounts reached','Shares']] = df_posts_dsc[['Profile Visits', 'Impressions', 'Follows', 
                                                                             'Accounts reached','Shares']].fillna(value=0).astype(int)
       df_posts_dsc[['Saves', 'Likes', 'Comments']] = df_posts_dsc[['Saves', 'Likes', 'Comments']].astype(int)
       df_posts_dsc['Post'] = df_posts_dsc.Post.fillna(' ')
       df_posts_dsc['Post'] = df_posts_dsc['Post'].replace(' ','No caption')
       df_reels = pd.read_excel("cleaned_reels_data.xlsx")
       df_reels['Date_Time'] = pd.to_datetime(df_reels['Date_Time'])
       df_reels = df_reels[~(df_reels['Date_Time'] >= '2022-09-01')]
       df_reels_dsc = (df_reels.assign(Date=df_reels['Date_Time'].dt.date,
                                   Time=df_reels['Date_Time'].dt.time)
                                   .sort_values(['Date','Time'], ascending =[False, False]))
       df_reels_dsc['Date'] = pd.to_datetime(df_reels_dsc['Date'])
       df_reels_dsc[['Instagram Shares', 'Instagram Saves',
                     'Facebook Play Count', 'Facebook Likes']] = df_reels_dsc[['Instagram Shares', 'Instagram Saves',
                                                               'Facebook Play Count', 'Facebook Likes']].fillna(value=0).astype(int)
       df_reels_dsc[['Duration',
       'Instagram and Facebook Plays', 'Instagram and Facebook Likes',
       'Accounts reached', 'Instagram Plays', 'Instagram Likes',
       'Instagram Comments']] = df_reels_dsc[['Duration',
                                                 'Instagram and Facebook Plays', 'Instagram and Facebook Likes',
                                                 'Accounts reached', 'Instagram Plays', 'Instagram Likes',
                                                 'Instagram Comments']].astype(int)
       return df_followers, df_followers_asc, df_followers_timeseries, df_posts, df_posts_dsc, df_reels, df_reels_dsc

# Create dataframes from the function
df_followers, df_followers_asc, df_followers_timeseries, df_posts, df_posts_dsc, df_reels, df_reels_dsc = load_data()
df_reels_dsc.head(20)
# Engineering data
## Posts data
df_posts_dsc_copy = df_posts_dsc.copy()
df_posts_8mo_date_metric = df_posts_dsc_copy['Date'].max()-pd.DateOffset(months=8)
df_posts_3mo_date_metric = df_posts_dsc_copy['Date'].max()-pd.DateOffset(months=3)
df_posts_jan_to_may_mo_agg = df_posts_dsc_copy[(df_posts_dsc_copy['Date'] >= df_posts_8mo_date_metric) & (df_posts_dsc_copy['Date'] <= df_posts_3mo_date_metric)].median()
numeric_cols = np.array(df_posts_dsc_copy.dtypes == 'int')
df_posts_dsc_copy.iloc[:,numeric_cols] = (df_posts_dsc_copy.iloc[:,numeric_cols] - df_posts_jan_to_may_mo_agg).div(df_posts_jan_to_may_mo_agg)

## Reels data
df_reels_dsc_copy = df_reels_dsc.copy()
df_reels_8mo_date_metric = df_reels_dsc_copy['Date'].max()-pd.DateOffset(months=8)
df_reels_3mo_date_metric = df_reels_dsc_copy['Date'].max()-pd.DateOffset(months=3)
df_reels_jan_to_may_mo_agg = df_reels_dsc_copy[(df_reels_dsc_copy['Date'] >= df_reels_8mo_date_metric) & (df_reels_dsc_copy['Date'] <= df_reels_3mo_date_metric)].median()
numeric_cols2 = np.array(df_reels_dsc_copy.dtypes == 'int')
df_reels_dsc_copy.iloc[:,numeric_cols2] = (df_reels_dsc_copy.iloc[:,numeric_cols2] - df_reels_jan_to_may_mo_agg).div(df_reels_jan_to_may_mo_agg)


# Exploring data
## Followers data
df_followers.isnull().values.any()
df_followers.isnull().sum()
df_followers.info()
df_followers_asc.info()
df_followers_timeseries.info()

## Posts data
df_posts.isnull().values.any()
df_posts.isnull().sum()
df_posts.info()
df_posts_dsc.info()
df_posts_dsc_copy.info()

## Reels data
df_reels.isnull().values.any()
df_reels.isnull().sum()
df_reels.info()
df_reels_dsc.info()
df_reels_dsc.describe()
df_reels_dsc_copy.info()

# Functions

def style_negative(v, props=''):
       try:
              return props if v < 0 else None
       except:
              pass

def style_positive(v, props=''):
       try:
              return props if v > 0 else None
       except:
              pass

df_posts_dsc_interactive_y_axis = df_posts_dsc.drop(['Post', 'Date_Time', 'Date', 'Time'], axis=1)
def interactive_plot1(df_posts_dsc_interactive_y_axis):
       y_axis_val1 = st.selectbox('Select Y-Axis attribute', options = df_posts_dsc_interactive_y_axis.columns)
       fig6 = px.line(df_posts_dsc, x = 'Date_Time', y = y_axis_val1)
       st.plotly_chart(fig6, use_container_width=True)

df_reels_dsc_rem_out = df_reels_dsc.drop(df_reels_dsc.index[[14]]) #Remove natural outlier to visualize better
df_reels_dsc_interactive_y_axis = df_reels_dsc_rem_out.drop(['Reel', 'Date_Time', 'Date', 'Time'], axis=1)
def interactive_plot2(df_reels_dsc_interactive_y_axis):
       y_axis_va2 = st.selectbox('Select Y-Axis attribute', options = df_reels_dsc_interactive_y_axis.columns)
       fig7 = px.line(df_reels_dsc_rem_out, x = 'Date_Time', y = y_axis_va2)
       st.plotly_chart(fig7, use_container_width=True)
       
# Building Dashboard
add_sidebar = st.sidebar.selectbox('Select Followers, Posts, or Reels insights:', ('Followers Insights','Posts Insights', 'Reels Insights'))


## Followers data
if add_sidebar == 'Followers Insights':
       st.title('Followers Insights')
       st.subheader('Total number of followers gained per year')
       #followers_sum = pd.pivot_table(df_followers_timeseries, index=['Year'], aggfunc='sum').reset_index() 
       followers_sum = df_followers_timeseries.groupby(['Year']).sum().reset_index()
       st.bar_chart(data=followers_sum, x='Year', y='Followers')
       
       st.subheader('Number of followers gained per day over the years')
       year = tuple(df_followers_timeseries['Date'].dt.year.unique())
       year_select = st.selectbox('Pick a year:', year)
       filtered = df_followers_timeseries[df_followers_timeseries['Date'] == year_select]
       sub_filter = df_followers_timeseries[df_followers_timeseries['Year'] == year_select]
       fig2 = px.line(sub_filter, x = 'Date', y = 'Followers')
       st.plotly_chart(fig2, use_container_width=True)              
       
       st.subheader('Cummulative number of followers gained over the years')
       df_followers_timeseries['Cummulative_Followers'] = df_followers_timeseries['Followers'].cumsum()
       fig3 = px.line(df_followers_timeseries, x = 'Date', y = 'Cummulative_Followers')
       st.plotly_chart(fig3, use_container_width=True)              

       
## Posts data
if add_sidebar == 'Posts Insights':
       df_posts_metrics = df_posts_dsc[['Date_Time','Profile Visits', 'Impressions', 'Follows', 'Accounts reached', 'Saves', 'Likes', 'Comments', 'Shares']]
       st.title('Posts Insights')
       st.subheader('Key Metrics')
       st.write('Value = Average value for each metrics from June to August 2022')
       st.write('Delta = Percentage growth calculated based on January to May 2022 average')
       df_posts_8mo_date_metric = df_posts_metrics['Date_Time'].max()-pd.DateOffset(months=8)
       df_posts_3mo_date_metric = df_posts_metrics['Date_Time'].max()-pd.DateOffset(months=3)
       df_posts_jan_to_may_mo_agg = df_posts_metrics[(df_posts_metrics['Date_Time'] >= df_posts_8mo_date_metric) & (df_posts_metrics['Date_Time'] <= df_posts_3mo_date_metric)].median()
       df_posts_3mo_agg = df_posts_metrics[df_posts_metrics['Date_Time'] >= df_posts_3mo_date_metric].median()
       
       col1, col2, col3, col4 = st.columns(4)
       columns = [col1, col2, col3, col4]
       count = 0
       
       for i in df_posts_3mo_agg.index:
              with columns[count]:
                     delta = (df_posts_3mo_agg[i] - df_posts_jan_to_may_mo_agg[i])/df_posts_jan_to_may_mo_agg[i]
                     st.metric(label = i, value = round(df_posts_3mo_agg[i],1), delta = "{:.2%}".format(delta))
                     count += 1
                     if count >= 4:
                            count = 0
       
       st.subheader('Percentage growth calculated based on January to May 2022 average for each post')
       df_posts_final = df_posts_dsc_copy.loc[:,['Date_Time','Post', 'Profile Visits', 'Impressions', 'Follows', 'Accounts reached', 'Saves', 'Likes', 'Comments', 'Shares']]
       #df_posts_final = df_posts_final[df_posts_final['Date_Time'] >= df_posts_3mo_date_metric] #If the dataframe that needs to be shown include on last 3 months data.
       df_posts_numeric_list = df_posts_final.median().index.tolist()
       df_to_pct = {}
       for i in df_posts_numeric_list:
              df_to_pct[i] = '{:.1%}'.format
       
       st.dataframe(df_posts_final.style.applymap(style_negative, props = 'color:red;').applymap(style_positive, props = 'color:green;').format(df_to_pct))
       
       df_number_of_posts = df_posts_dsc[['Date_Time','Post']]
       df_number_of_posts_posted = df_number_of_posts.groupby([df_number_of_posts['Date_Time'].dt.year.rename('year'), df_number_of_posts['Date_Time'].dt.month.rename('month')]).count().reset_index()
       df_number_of_posts_posted['month'] = df_number_of_posts_posted['month'].apply(lambda x: calendar.month_name[x])
       st.subheader('Number of posts uploaded in a month over the years')
       year1 = tuple(df_number_of_posts_posted['year'].unique())
       year_select1 = st.selectbox('Pick a year:', year1)
       filtered1 = df_number_of_posts_posted[df_number_of_posts_posted['month'] == year_select1]
       sub_filter1 = df_number_of_posts_posted[df_number_of_posts_posted['year'] == year_select1]
       fig4 = px.bar(sub_filter1, x = 'month', y = 'Post', text='Post')
       st.plotly_chart(fig4, use_container_width=True)
       
       st.subheader('Explore each attribute in a time series')
       interactive_plot1(df_posts_dsc_interactive_y_axis)
       
       
## Reels data
if add_sidebar == 'Reels Insights':
       df_reels_metrics = df_reels_dsc[['Date_Time', 'Accounts reached', 'Instagram Plays', 'Instagram Likes', 'Instagram Comments', 'Instagram Shares', 'Instagram Saves']]
       st.title('Reels Insights')
       st.subheader('Key Metrics')
       st.write('Value = Average value for each metric from June to August 2022')
       st.write('Delta = Percentage growth calculated based on January to May 2022 average')   
       df_reels_8mo_date_metric = df_reels_metrics['Date_Time'].max()-pd.DateOffset(months=8)
       df_reels_3mo_date_metric = df_reels_metrics['Date_Time'].max()-pd.DateOffset(months=3)
       df_reels_jan_to_may_mo_agg = df_reels_metrics[(df_reels_metrics['Date_Time'] >= df_reels_8mo_date_metric) & (df_reels_metrics['Date_Time'] <= df_reels_3mo_date_metric)].median()
       df_reels_3mo_agg = df_reels_metrics[df_reels_metrics['Date_Time'] >= df_reels_3mo_date_metric].median()
       
       col1, col2, col3, col4 = st.columns(4)
       columns = [col1, col2, col3, col4]
       count = 0
       
       for i in df_reels_3mo_agg.index:
              with columns[count]:
                     delta = (df_reels_3mo_agg[i] - df_reels_jan_to_may_mo_agg[i])/df_reels_jan_to_may_mo_agg[i]
                     st.metric(label = i, value = round(df_reels_3mo_agg[i],1), delta = "{:.2%}".format(delta))
                     count += 1
                     if count >= 4:
                            count = 0
       
       st.subheader('Percentage growth calculated based on January to May 2022 average for each post')
       df_reels_final = df_reels_dsc_copy.loc[:,['Date_Time', 'Reel', 'Accounts reached', 'Instagram Plays', 'Instagram Likes', 'Instagram Comments', 'Instagram Shares', 'Instagram Saves','Duration', 'Facebook Play Count']]
       #df_reels_final = df_reels_final[df_reels_final['Date_Time'] >= df_reels_3mo_date_metric] #If the dataframe that needs to be shown include on last 3 months data.
       df_reels_numeric_list = df_reels_final.median().index.tolist()
       df_to_pct = {}
       for i in df_reels_numeric_list:
              df_to_pct[i] = '{:.1%}'.format
       
       st.dataframe(df_reels_final.style.applymap(style_negative, props = 'color:red;').applymap(style_positive, props = 'color:green;').format(df_to_pct))
     
       df_number_of_reels = df_reels_dsc[['Date_Time','Reel']]
       df_number_of_reels_posted = df_number_of_reels.groupby([df_number_of_reels['Date_Time'].dt.year.rename('year'), df_number_of_reels['Date_Time'].dt.month.rename('month')]).count().reset_index()
       df_number_of_reels_posted['month'] = df_number_of_reels_posted['month'].apply(lambda x: calendar.month_name[x])
       st.subheader('Number of reels uploaded in a month over the years')
       year2 = tuple(df_number_of_reels_posted['year'].unique())
       year_select2 = st.selectbox('Pick a year:', year2)
       filtered2 = df_number_of_reels_posted[df_number_of_reels_posted['month'] == year_select2]
       sub_filter2 = df_number_of_reels_posted[df_number_of_reels_posted['year'] == year_select2]
       fig5 = px.bar(sub_filter2, x = 'month', y = 'Reel', text='Reel')
       st.plotly_chart(fig5, use_container_width=True)  
       
       st.subheader('Explore each attribute in a time series (A natural outlier has been removed - 2022-06-09 reel)')
       interactive_plot2(df_reels_dsc_interactive_y_axis)
       
         







