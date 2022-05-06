# Predicting Breast Cancer

### By: Sophie Rubenfeld, Mahak Bindal, Anna Chen


## Background
Breast cancer is the second most common cancer among women after melanomas (CDC). One in eight women (13%) 
will develop invasive breast cancer throughout their lifetime. In 2022, an estimated 287,850 new cases of metastatic (invasive) breast cancer are expected to be diagnosed, along with 51,400 new cases of non-invasive breast cancer (National Breast Cancer Org).

Typical breast cancer treatment primarily focuses on surgery, either a mastectomy (total removal of the breast) or a lumpectomy (partial removal of the breast). Surgery is supplemented in combination with chemotherapy, hormone therapy and/ or radiation (Mayo Clinic). Traditionally, total mastectomies have been the preferred surgical method, as it is seen as safer and has higher levels of remission. However, recent studies have shown that total mastectomy surgeries don't neccessarily have a better prognosis (survival rate) compared to lumpectomy combined with adjuvant chemotherapy, but may be more accessible to less affluent families due to less surveillance needed (Fisher et al.)

In terms of breast cancer risk factors, lifestyles such as high income, high socioeconomic status, and affluence may increase breast cancer incidence (Lehrer et al).
This is due to breast cancer risk factors like delayed childbirth, less breast-feeding, use of hormone supplements more common in affluent women. Affluent women are also more likely to have regulary check ups and access to mammograms, which detect many cancers that might not otherwise be diagnosed and at earlier stages. Genetic factors such as the BRCA 1 and 2 mutations can be known and detected through surveillance, genetic testing, and preventative surgery. Women in certain affluent ethnic groups—Ashkenazi Jews, Icelanders and the Dutch—are more likely to carry such genetic predispositions. Women with more income can afford better cancer care and survive longer than poorer women. For example, proximity to hospitals is important, especially for treatments that require repeated rounds. As such, we chose to focus on late-stage breast cancer (stage III and IV), as we hypothesized that women in poorer areas may be diagnosed later due to a lack of resources, and thus suffer worse outcomes. Through this, our goal is to build a machine learning model to predict a county’s breast cancer incidence rates based off of affluence factors.


## Research question:
Do county-level affluence measures affect Breast Cancer incidences (diagnoses) in the United States? If so, can we predict diagnoses using a machine learning model?


## Where we started...
To begin our inquiry, we started by identifying the sources of the data we wanted to use, and that would be most useful to us. From the NIH (National Institute of Health), we retrieved our primary data, detailing instances of Late-Stage Breast Cancer Incidences by US County averaged over 2014 to 2018.

To collect data on affluence, around this same timeframe, we then turned to the Economic Research Service, where we downloaded excel files (not CSVs, that came later). We retrieved 5 Excel files, providing us with:
* Population of each US County per the 2020 Census 
* Poverty Rates for every US County as of 2019 
* Unemployment and Median Household Income for each US County from 2019
* 5-year Averaged Education Levels by US County for 2015-2019, this included:
    * Percent of adults with less than a high school diploma
    * Percent of adults with a high school diploma only
    * Percent of adults completing some college or associate degree
    * Percent of adults with a bachelor's degree or higher

With this data on hand, we began our work with approximately 5,000 counties for which we had one or more pieces of data, though this number decreased during the data cleaning stage. 



## Data cleaning
After downloading our 5 data sets, we had to individually clean each and then combine it. For each of the 5 datasets we cut out the extras rows a the top of each dataset, as they held excess information or spacing. Then, the indices were only the row or column number, so we had to reassign the row indices (usually by county), and the columns by the desired variable. For some of the datasets, we either chose to drop specific columns in favor of others, or we decided to drop all of columns except one to simplify the final, combined dataset. We also dropped all the N/A values for each of the columns in each dataset. By doing this after eliminating unnecessary columns, we were able to preserve the largest number of counties in our dataset. The manner in which each state and county was represented varied among each dataset. Some of them used state abbreviations, others used the full name, and there was also a dataset that formatted in the information as "county, state". We had to normalize the way in which each dataset represented this information so it would be easier to join all 5 datasets in the final step. We found a mapping of the full state name to their abbreviations on github, and used that to map each county to their respective state. This also helped account for duplicate county names in different states. After that, we created a tuple, (county name, state abbreviation), which we ended up using to merge each dataset.

*As a note, even though the processes were very similar across all the data sets, each one had to be cleaned individually as they all had slightly different formatting. Even the 4 datasets from the Economic Research Service, despite originating from the same source, were not standardized. Thus, helper functions were not able to be used.*

## Compiling data set
After cleaning all 5 data sets, they had the desired columns without NAs and the correct indices for county, state abbreviation. We want to compile a dataset that has all the columns in each separate data set. To do this, the major concern is only keeping counties that have data across all the data sets. To do this, we made each dataset into a set. We first found the intersection of one data set compared to another and saved all those county names in a "compiled set". We then compared the 3rd dataset to the "compiled set," and saved the intersections of that comparison to the "compiled set". We did this for all the datasets (4 intersections). This produces a set that only has counties shared across all 5 data sets.

Next, we went through each dataset and dropped any rows (counties) that were not included in our compiled list of counties. We did this by appending a new column to each data frame. If the county was in the compiled list, then it was assigned the value 0, else it was assigned a N/A value. After this was done for all the data frames, we dropped the NAs, so each data set was left with only the counties shared across all 5. 

Now, to merge the data sets, we used the pandas ```merge``` function. When doing this, we specified only the columns we desired, not including helper columns such as the 0 column we used to drop NAs, or excess ```County, State``` columns that were used as indices.

The columns of our compiled dataset are below:

![](https://i.imgur.com/DclO4LM.png)

Here are the categories of each column:
* Breast cancer:
    * Age-Adjusted Incidence Rate - cases per 100,000	
    * Average Annual Count
    * Percent of Cases with Late-Stage Cancer (Stage III - IV)

* Population: 
    * Population (in each county) 2020

* Education:
    * Percent of adults with less than a high school diploma, 2015-19
    * Percent of adults with a high school diploma only, 2015-19
    * Percent of adults completing some college or associate degree, 2015-19
    * Percent of adults with a bachelor's degree or higher, 2015-19

* Poverty:
    * POVALL_2019 (overall poverty)
    * Unemployment_rate_2019
    * Median_Household_Income_2019

## Working with Cleaned Data
Once we had our cleaned data set on hand, we knew the goal: to create some variety of machine learning model (whether it be multivariate linear regression or a Multinomial Naive Bayes Model) that could predict instances of late-stage breast cancer in a given US county. To start, though, we wanted to make sure that these models would be able to work, so we utilized ```Pandas``` as well as the ```Seaborn``` package to create a correlation matrix and visualize it as a heat map (see below). In order to create this visualization, though, we first needed to create a helper function that converted every imported numerical value from a string to a float. Once this was completed, though, we were able to successfully utilize Pandas' ```.corr()```method to create the correlation matrix and subsequent heat map. 
![](https://i.imgur.com/qvPzEJP.png)

Seeing somewhat mixed results from our correlation heat map - notice that the correlation between population and poverty rates as well as between population and average annual count were both quite high while others were not as promising- we opted to utilize Multivariate Linear Regression to create our first model. We chose this model because a similar study from [Lehrer et al.](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6477537/) that we were inspired by had also seen promising results from this method. However, upon running the code to create the model, our predictions yielded an $R^2$ value of -0.0082. 



Having seen these less than promising results, we opted to attempt another form of predictive model: the Multinomial Naive Bayes model. Our goal with this next step was to start by creating a model that could predict (based off of affluence indicators) whether or not a county will see high or low instances of Breast Cancer. In this case, we used the ```Average Annual Counts``` variable from our NIH data to create a new column of our dataframe- a binary column indicating high or low instances. To determine the cutoff for high vs. low, we calculated both the median and mean of the ```Average Annual Counts``` data, ultimately using the median since the difference between the two (the median being 11 and the mean being closer to 30) indicated that there may be outliers influencing our data. With this, we used a helper function to create a binary column indicating that a county's annual count was below (0) or above (1) the median. Once we had our new binary column created, we created a copy of the dataframe with all of our data, dropping the columns for variables that we had gotten from our NIH data relating to breast cancer incidences, and utilized our new binary column as a target array. With our data now ready for the model, we trained 2 different models splitting the data 2 ways. First, we used the ```sklearn``` package's ```train_test_split``` function to train and test the first model. Then, we used the same package's ```KFold``` function to test and train our second model. Utilizing helper functions to calculate the F1 score and accuracy for each model, we saw accuracy scores around 80% for both testing and training sets as well as F1 scores in the same range. 
Because we initially faced troubles debugging the model, we also turned to utilizing more insightful measures to visualize and understand the data we did have. 


Our first step in this process was running a Principal Component Analysis (PCA). In running this, we first visualized the Explained Variance Ratio (see below) prior to creating a scatter plot of the first component as compared to the second (for both z-scored and non-z-scored data sets). The Explained Variance Ratio graph helped us understand that 99% of the variance in our data is explained by the first Principal Component. See the Explained Variance Ratio graphed below. 
 ![](https://i.imgur.com/GcVhFL1.png)
 
 
In graphing the first two principal components, we were able to see that the z-scored PCA helped us to better understand and visualize the direction of highest correlation. Though it can be seen slightly in the first graph, it is much more obviously in the second.
![](https://i.imgur.com/q2wlcXW.png)

Seeing these results, we also investigated what each Principal Component was composed of. As seen in the table below, Population and Poverty variables were given the highest weight in the first Principal Component, this is expected from what we saw in our correlation matrix, where poverty rates and population were most closely correlated with one another, and with Annual Average Counts. 

![](https://i.imgur.com/b3173l0.png)


### Data Visualization
After completing our PCA, we turned to visualizing the data that produced high correlation coefficients in the correlation heat map above. Based on the heat map, we saw that the Average Annual Count of breast cancer incidences and the Population of each county in 2020 had a correlation of 0.99. Similary, the case incidences and Poverty rates had a correlation of 0.97. We also decided to visualize the incidence *rates* because we were interested if there would be any trends between that and Population or Poverty.

*Histogram of Late Stage Breast Cancer Cases in US Counties:*
 ![](https://i.imgur.com/9HLQSBt.png)
 This histogram shows the distribution of late-stage breast cancer cases within US counties. The distribution is right skewed meaning the majority of counties have a smaller number of late-stage breast cancer incidences. The median is 11, but the mean is 34.12, so there are significant outliers in late-stage breast cancer incidences in a few counties.
 
*Scatterplots: Poverty and Population vs. Case Incidences (rates and counts):*
![](https://i.imgur.com/9XAtIxU.png)
These graphs compare the Age Adjusted Incidence Rates compared to the Population and the Poverty rates of each county. Although the visualizations for incidence rates versus Population and Poverty had similar shapes, we did not observe much correlation between the features.
![](https://i.imgur.com/DynVVB5.png)
This scatterplot on the right shows the correlation between counts of Breast Cancer Incidences and the poverty rates in each US county. The visualization shows some positive correlation between the two components, meaning counties with less incidences had smaller poverty rates, and counties with more incidences had higher poverty rates. There were also significant outliers in this visualization, with counties with very high poverty rates had just as high incidences of breast cancer. The left graph is a scatterplot of the Population vs. Case Incidence Counts. This also has a similar trend to the poverty rates vs case incidence counts in that the higher the population, the higher the number of Breast Cancer incidences.


## Discussion and Conclusion
From our benchmark paper, they found a significant relationship of income with breast cancer incidence in the 50 states in White women. The relationship of income to incidence was not significant in Black women and is of borderline significance in Hispanics. The study performed a multivariate linear regression, which indicated that the relationship with income of breast cancer incidence in White women was significant and independent of the relationship with alcohol consumption. This finding may have been partly due to higher rates of a genetic predisposition to breast cancer among White women. The study also found that income was significantly correlated with a 5-year relative survival in Whites with localized breast cancer (stage I and II).

Knowing this, we chose to do our study based off of counties, and to not separate by race. Since Lehrer et al. proved significant correlation between 5 year relative survival and localized breast cancer in whites, we chose to look at non-localized (lymph and metastasis) incidences, as we thought there would be better indications for the less affluent population. We did find that counties with less incidences of late-stage breast cancer had lower poverty rates, and counties with more incidences had higher poverty rates. 

Ultimately, we were able to complete our goal of creating a machine learning model to predict incidence of Breast Cancer diagnoses in US counties. Though our Linear Regression model was not as fruitful as we hoped it would be -- which could have been predicted by our correlation matrix-- we saw promising and interesting results from our Multinomial Naive Bayes model as well as our PCA and other visualizations.  Seeing that our model is successful, we could apply this model to early-stage Breast Cancer cases, and additional affluence indicators, similar to how our benchmark paper author's did. If we had more time, it would be interesting to create additional visualizations from the Multinomial Naive Bayes model. For example, by creating a target array that indicated high, medium, or low incidence counts, and seeing whether the model would predict high or low incidence rates for counties that, in reality, saw a medium level of incidences. It would also be interesting to train our model on the Age Adjusted Incidence rate, in addition to the Annual Average Counts which we were using. Ultimately, because of time constraints and initial debugging challenges, we were unable to carry out these further inquiries, but believe they would add additional insight. It could be helpful to include additional indicators of affluence, like race, in further research, just as the literature we examined did. In terms of the data itself, more time would also be valuable to isolate and examine counties with significant outliers in their Average Annual Counts or Incidence Rates, to determine possible causes. 


### Citations:
* Breastcancer.org 
* https://dx.doi.org/10.1111%2Ftbj.12630
* https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6477537/
* https://www.cdc.gov/cancer/breast/statistics/index.htm
* https://www.mayoclinic.org/diseases-conditions/breast-cancer/diagnosis-treatment/drc-20352475#:~:text=Most%20women%20undergo%20surgery%20for,before%20surgery%20in%20certain%20situations 
