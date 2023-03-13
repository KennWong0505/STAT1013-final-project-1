# STAT1013-final-project-1
---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="JLPf6D1q4LLv">

# STAT1013 Final Project Practical Assignment Part 1: Sharing Idea and Data

</div>

<div class="cell markdown" id="JhrhSbHA7D0D">

## 2022-2023 NBA Player Dataset Background

> 縮排段落

**Description**:

Dataset describing the 2022-2023 regular season NBA player stats per
game.

**Github**:
<https://raw.githubusercontent.com/KennWong0505/STAT1013-Final-Project/main/%5BUTF-8%5D%202022-2023%20NBA%20Player%20Stats%20-%20Regular.csv?token=GHSAT0AAAAAAB6ZSPURZJXLH43NXNHBUD2AY757TZQ>

**Sample size**: 553

**Feature documentation**:

| Feature | Class      | Shape | Dtype   |
|:--------|:-----------|:------|:--------|
| Player  | Tensor     |       | object  |
| Pos     | Tensor     |       | object  |
| Age     | Tensor     |       | int64   |
| Tm      | Tensor     |       | object  |
| G       | ClassLabel |       | int64   |
| GS      | Tensor     |       | int64   |
| MP      | Tensor     |       | float64 |
| FG      | Tensor     |       | float64 |
| FGA     | Tensor     |       | float64 |
| FG%     | Tensor     |       | float64 |
| 3P      | Tensor     |       | float64 |
| 3PA     | Tensor     |       | float64 |
| 3P%     | Tensor     |       | float64 |
| 2P      | Tensor     |       | float64 |
| 2PA     | Tensor     |       | float64 |
| eFG%    | Tensor     |       | float64 |
| FT      | Tensor     |       | float64 |
| FTA     | Tensor     |       | float64 |
| FT%     | Tensor     |       | float64 |
| ORB     | Tensor     |       | float64 |
| DRB     | Tensor     |       | float64 |
| TRB     | Tensor     |       | float64 |
| AST     | Tensor     |       | float64 |
| STL     | Tensor     |       | float64 |
| BLK     | Tensor     |       | float64 |
| TOV     | Tensor     |       | float64 |
| PF      | Tensor     |       | float64 |
| PTS     | Tensor     |       | float64 |

</div>

<div class="cell markdown" id="1lhgzF-yK40e">

## Hypothesis

-   Tell us what your idea is and why you have chosen to pursue this
    idea.

    -   I am interest in “Do NBA player scores more when their playing
        time in game excess 35 mins per game?”
    -   The reason I choose to pursue this idea is because when I
        watching the Box Score of the game, it is often to see the
        variation of extremely high score record and low score record. I
        may wonder know is there is a relationship between the score and
        the minutes per game.

-   What two groups you are comparing:

    -   **G1**: Minutes per game \> 35 mins;

    **G2**: Minutes per game \< 35 mins

-   What you will be measuring (i.e., what your response variable will
    be)

    -   `PTS` (points per game)

-   Is your response variable quantitative rather than categorical?

    -   Quantitative. The dtype of `PTS` is float64.

-   Make a prediction about what kind of difference you expect to see
    between your samples and WHY.

    -   I expect that G1 \> G2 since the more the time NBA players in
        game, the opportunity the player can participate in the
        offensive stage (chance to score increase).

-   Talk about how you will gather your data

    -   I will try to find it on internet as I am watching the box score
        of the NBA games nearly on a daily basis, I am pretty sure there
        will be some complete data on the internet.
    -   After doing so research, I found the data of NBA Player
        statistics on Kaggle:
        <https://www.kaggle.com/datasets/vivovinco/20222023-nba-player-stats-regular?resource=download>
    -   But unfortunately, the file is not \[UTF-8\]. I have to convert
        the file before read using colab.

-   If you had unlimited resources (time, money, staff, etc.) how would
    you collect your data?

    -   I will use the data of the past 30 years or perhaps more. With
        making use of more dataset to do the analysis over and over, and
        then we can take the expectation of mean of the analysis. With
        this kind of process, I am sure the result would be more
        accurate and trustworthy.

</div>

<div class="cell code" execution_count="12" id="KUPkg7By3mQo">

``` python
## Load the dataset from github
import pandas as pd

NBA = 'https://raw.githubusercontent.com/KennWong0505/STAT1013-Final-Project/main/%5BUTF-8%5D%202022-2023%20NBA%20Player%20Stats%20-%20Regular.csv?token=GHSAT0AAAAAAB6ZSPURZJXLH43NXNHBUD2AY757TZQ'
df = pd.read_csv(NBA, delimiter=";", encoding="latin-1", index_col=0)
```

</div>

<div class="cell code" execution_count="13"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:418}"
id="CcVhuUku-uBI" outputId="e598e51a-a741-4577-927c-c2ea789b1bc3">

``` python
df.head(5)
```

<div class="output execute_result" execution_count="13">

                     Player Pos  Age   Tm   G  GS    MP   FG   FGA    FG%  ...  \
    ï»¿Rk                                                                  ...   
    1      Precious Achiuwa   C   23  TOR  33   9  23.0  4.0   8.2  0.489  ...   
    2          Steven Adams   C   29  MEM  42  42  27.0  3.7   6.3  0.597  ...   
    3           Bam Adebayo   C   25  MIA  52  52  35.3  8.6  15.7  0.546  ...   
    4          Ochai Agbaji  SG   22  UTA  35   1  14.0  1.5   3.2  0.486  ...   
    5          Santi Aldama  PF   22  MEM  52  18  22.0  3.4   7.0  0.486  ...   

             FT%  ORB  DRB   TRB  AST  STL  BLK  TOV   PF   PTS  
    ï»¿Rk                                                        
    1      0.697  2.1  4.3   6.4  1.1  0.7  0.7  1.2  2.2  10.4  
    2      0.364  5.1  6.5  11.5  2.3  0.9  1.1  1.9  2.3   8.6  
    3      0.806  2.7  7.3  10.1  3.3  1.2  0.8  2.6  2.8  21.6  
    4      0.625  0.6  1.0   1.6  0.5  0.1  0.1  0.3  1.4   4.1  
    5      0.730  1.0  3.7   4.7  1.2  0.7  0.7  0.7  1.9   9.5  

    [5 rows x 29 columns]

</div>

</div>

<div class="cell code" execution_count="14"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="pMgGftsZAs5E" outputId="11643b9e-312a-4009-8210-8e6334511e16">

``` python
df.info()
```

<div class="output stream stdout">

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 553 entries, 1 to 505
    Data columns (total 29 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Player  553 non-null    object 
     1   Pos     553 non-null    object 
     2   Age     553 non-null    int64  
     3   Tm      553 non-null    object 
     4   G       553 non-null    int64  
     5   GS      553 non-null    int64  
     6   MP      553 non-null    float64
     7   FG      553 non-null    float64
     8   FGA     553 non-null    float64
     9   FG%     553 non-null    float64
     10  3P      553 non-null    float64
     11  3PA     553 non-null    float64
     12  3P%     553 non-null    float64
     13  2P      553 non-null    float64
     14  2PA     553 non-null    float64
     15  2P%     553 non-null    float64
     16  eFG%    553 non-null    float64
     17  FT      553 non-null    float64
     18  FTA     553 non-null    float64
     19  FT%     553 non-null    float64
     20  ORB     553 non-null    float64
     21  DRB     553 non-null    float64
     22  TRB     553 non-null    float64
     23  AST     553 non-null    float64
     24  STL     553 non-null    float64
     25  BLK     553 non-null    float64
     26  TOV     553 non-null    float64
     27  PF      553 non-null    float64
     28  PTS     553 non-null    float64
    dtypes: float64(23), int64(3), object(3)
    memory usage: 129.6+ KB

</div>

</div>

<div class="cell markdown" id="QqNzjeaLNZbf">

-   I want to compare the points player score `PTS` when the playing
    time in game `MP` is excess 35 mins or not greater than 35 mins.
    -   **G1**(PTS \| MP \> 35) vs **G2**(PTS \| MP \<= 35)

</div>

<div class="cell code" execution_count="15"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="k2hZj2rDGTqD" outputId="9705ab9c-e826-46df-d925-b00752961cfe">

``` python
## First 5 records of G1 (MP>35)
(df[df['MP'] > 35]['PTS']).head(5)
```

<div class="output execute_result" execution_count="15">

    ï»¿Rk
    3     21.6
    14    16.9
    21    22.9
    28    15.5
    57    17.3
    Name: PTS, dtype: float64

</div>

</div>

<div class="cell code" execution_count="16"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vmtQytvlPku3" outputId="9d5b76a4-fe15-49ca-b19c-6dae635ea959">

``` python
## First 5 records of G2 (MP<=35)
(df[df['MP'] <= 35]['PTS']).head(5)
```

<div class="output execute_result" execution_count="16">

    ï»¿Rk
    1    10.4
    2     8.6
    4     4.1
    5     9.5
    6     6.2
    Name: PTS, dtype: float64

</div>

</div>

<div class="cell code" execution_count="29" id="ypZIGw_SZU-m">

``` python
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

sns.set()
```

</div>

<div class="cell markdown" id="CnRKwiXcZiPI">

### The distribution of the Minutes in game `MP`, Points score `PTS` and also the points score distribution among minutes in game

</div>

<div class="cell code" execution_count="31"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:279}"
id="hi0u0-pDZ0PQ" outputId="1349c830-1eb2-4649-e651-d667131cbe5e">

``` python
sns.violinplot(data=df, x='MP')
plt.show()
```

<div class="output display_data">

![](520b981c26b4c2ccf6df363df8e1b8c09264f469.png)

</div>

</div>

<div class="cell code" execution_count="30"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:279}"
id="M7IlUi1pZql0" outputId="e19fcc80-2cdf-4c29-9558-f5bf8e7fa3b5">

``` python
sns.violinplot(data=df, x='PTS')
plt.show()
```

<div class="output display_data">

![](dd9cab3b010089350b2e3b56496ee51f69010d29.png)

</div>

</div>

<div class="cell code" execution_count="32"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:283}"
id="Rlx6cNELbB6O" outputId="469db0c6-33ca-4ce7-b2ef-65de458ff193">

``` python
# points score distribution among minutes in game
sns.scatterplot(data=df, x='MP', y='PTS')
plt.show()
```

<div class="output display_data">

![](b146a9a7ccbc7c2f7d6b4a9b7142b15bde439da5.png)

</div>

</div>

<div class="cell markdown" id="kFwQ3IzVaAst">

We can observe that most of the player have the 15 mins in game.

Meanwhile, most players with around 5 points per game.

More importantly, we can observe that from the third scatter graph that
the distribution is positively related, i.e. when `MP` increase, `PTS`
also increase.

</div>

<div class="cell markdown" id="BoSqgJKnXcFX">

### The Sample Size of Minutes in game excess 35 minutes and not excess 35 minutes

</div>

<div class="cell code" execution_count="27"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="OnUjrFBbXdcn" outputId="774fba45-6b37-493d-800c-c499f915707a">

``` python
print('Sample Size of Minutes in game excess than 35 minutes:')
print(len(df[df['MP'] > 35]))
print('Sample Size of Minutes in game not excess than 35 minutes:')
print(len(df[df['MP'] <= 35]))
```

<div class="output stream stdout">

    Sample Size of Minutes in game excess than 35 minutes:
    31
    Sample Size of Minutes in game not excess than 35 minutes:
    522

</div>

</div>

<div class="cell markdown" id="sLNJTSpbVkPc">

### Measure of the Central Tendency:

-   Sample Mean of `MP`
-   Sample Mean of `PTS`
-   Sample Mean of `PTS` when `MP` \> 35
-   Sample Mean of `PTS` when `MP` \<= 35

</div>

<div class="cell code" execution_count="17"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Z91XeYsLPxoJ" outputId="0e28c24a-5847-4fc4-9d9e-a90299fbd203">

``` python
## The statistics of the minutes in game and points scored of all players
print('Sample mean of Minutes in game of all players:')
print(df['MP'].mean())

print('---')

print('Sample mean of Points of all players:')
print(df['PTS'].mean())
```

<div class="output stream stdout">

    Sample mean of Minutes in game of all players:
    19.760397830018082
    ---
    Sample mean of Points of all players:
    9.100361663652803

</div>

</div>

<div class="cell code" execution_count="22"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="5FTaQhLbUOtm" outputId="14a10339-3873-4c7d-8849-1da2baa35156">

``` python
## Sample mean with the condition of MP (Minutes in game)
print('Sample mean of Points scored when the player have excess 35 mins in game:')
print(df[df['MP'] > 35]['PTS'].mean())
print('Sample mean of Points scored when the player have NOT excess 35 mins in game:')
print(df[df['MP'] <= 35]['PTS'].mean())
```

<div class="output stream stdout">

    Sample mean of Points scored when the player have excess 35 mins in game:
    24.187096774193545
    Sample mean of Points scored when the player have NOT excess 35 mins in game:
    8.2044061302682

</div>

</div>

<div class="cell markdown" id="mXaz4ZY8WMyU">

### Measure of Data Variability

-   Standard Deviation of `PTS`
-   Standard Deviation of `MP`
-   Standard Deviation of `PTS` when `MP`\>35
-   Standard Deviation of `PTS` when `MP`\<=35

</div>

<div class="cell code" execution_count="23"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="KAFmLh-DVOKH" outputId="91f20f39-7bcc-47bc-ce23-a30b18afd117">

``` python
print('Standard Deviation of Minutes in game of all players:')
print(df['MP'].std())

print('---')

print('Standard Deviation of Points of all players:')
print(df['PTS'].std())
```

<div class="output stream stdout">

    Standard Deviation of Minutes in game of all players:
    10.109079244658135
    ---
    Standard Deviation of Points of all players:
    7.023418374699309

</div>

</div>

<div class="cell code" execution_count="24"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="QUUUkcdNXJud" outputId="2466e185-ab96-4485-cdf0-69e842f14d55">

``` python
print('Standard Deviation of Points scored when the player have excess 35 mins in game:')
print(df[df['MP'] > 35]['PTS'].std())
print('Standard Deviation of Points scored when the player have NOT excess 35 mins in game:')
print(df[df['MP'] <= 35]['PTS'].std())
```

<div class="output stream stdout">

    Standard Deviation of Points scored when the player have excess 35 mins in game:
    4.635424607338855
    Standard Deviation of Points scored when the player have NOT excess 35 mins in game:
    6.0563159820373444

</div>

</div>

<div class="cell code" id="Iz3iPW6gXTUc">

``` python
```

</div>
