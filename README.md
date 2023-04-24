# Nanyang Technological University (NTU), SCSE, AI6103 Deep Learning and Application.

Project Assignment Submission for module AI6103, Deep Learning and Application.

Zhao Yuhan, zhao0431@e.ntu.edu.sg, Wang Qiyue, wang1901@e.ntu.edu.sg, Wang Yutian, wang1910@e.ntu.edu.sg, Eugene Ho, eho010@e.ntu.edu.sg, Koo Chia Wei, ckoo004@e.ntu.edu.sg

## Exploration on Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting 

In this project, we choose Informer as the research object. Informer is a prediction model based on the Transformer structure for long-term sequences. Informer uses methods such as multi-head self-attention mechanism, depthwise separable convolution, periodic attention mechanism, and stacked structure, which can enhance the robustness of the model and integrate Global and key local information, and consider different time scales, to achieve effective model-ing and forecasting of long sequences.
After carefully studying the paper and related work, we reproduced the entire work based on our own understand-ing, discussed the impact of different optimizers and loss functions on the model on the original data set, and on stock prices, S68.SI) as new data sets to explore Informer Generalization ability and existing problems.

## Model Training and Inference

1. Clone/download the entire file
2. Download the necessary package from requirements in requirements.txt
3. Open main_informer.py in any IDE
4. Change the --data and --data_path file to the data you want to run in arg.parser accordingly
5. Change the hyperparameters in the arg.parser accordingly
6. Run the "main_informer.py" in IDE

or 

1. Simply run in right environment with the requirements in requirements.txt installed
```bash
python main_informer.py --data WTH --data_path WTH.csv --optimizer adam --lossfunction = mse
```

## Data Source
-  Data Set 1: Weather Forecast
    - Obtained from [National Centers for Environmental Information](https://www.ncei.noaa.gov/data/local-climatological-data/)
    - Processed Data can refer to [dataset](dataset), labelled as [dataset](WTH.csv)
  
-  Data Set 2: Global Land and Temperature Average
    - Obtained from [Data.world, Global Climate Change Data](https://data.world/data-society/global-climate-change-data)
    - Processed Data can refer to [dataset](dataset), labelled as [dataset](GT.csv)

-  Data Set 3: Singapore Exchange Limited (S68.SI) 
    - Obtained from [Yahoo Finance, S68.SI](https://finance.yahoo.com/quote/S68.SI/history?p=S68.SI)
    - Processed Data can refer to [dataset](dataset), labelled as [dataset](S68_4.csv)

## Contributing

1. Clone the repository locally
1. Pull main branch to ensure your local main branch is up to date with the remote main branch
1. Create a new branch using `git checkout -b <branch-name>`
1. Merge the main branch to your working branch to keep your branch up to date with the latest changes `git merge main`
1. Once you have saved your changes, add -> commit -> push your changes to remote branch 
    ```
    # add all changes
    git add . 

    # commit your changes with a commit message
    git commit -m "<your commit message>"

    # push your changes to remote
    git push

    # you will be prompted to set a remote upstream if this is your first time pushing changes after creating the new branch
    git push --set-upstream origin <branch-name>
    ```
1. Create a Pull Request to merge your changes to the main branch. 
1. Repeat step 2 to 6 for subsequent contributions


## Installation & Usage

```
# initialise isolated python environment
python -m venv .venv

# initialise environment
source .venv/bin/activate # mac
.venv/Scripts/Activate # windows

# install dependencies
pip install -r requirements.txt


# freeze dependencies
pip freeze > requirements.txt
```

## Reference and Acknowlegdement

Cirstea Razvan-Gabriel [et al.] Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting--Full Version [Journal] // arXiv preprint arXiv. - 2022. - 2204.13767.
Feng Xiaoyu [et al.] SEFormer: Structure Embedding Transformer for 3D Object Detection [Journal] // arXiv preprint arXiv. - 2022. - 2209.01745.
Huber Peter J. Robust Estimation of a Location Parameter [Book Section] // Breakthroughs in Statistics: Methodology and Distribution / book auth. Kotz Samuel and Johnson Norman L.. - New York : Springer New York, NY, 1992.
Luo Liangchen [et al.] Adaptive Gradient Methods with Dynamic Bound of Learning Rate [Journal] // arXiv preprint arXiv. - 2019. - 1902.09843.
NCEI Index of /data/local-climatological-data [Online] // National Centers for Environmental Information. - January 5, 2023. - April 1, 2023. - https://www.ncei.noaa.gov/data/local-climatological-data/.
Society Data Global Climate Change Data [Online] // data.world. - 2015. - April 1, 2023. - Global Climate Change Data.
Yahoo Finance Singapore Exchange Limited (S68.SI) [Online] // Yahoo FInance. - 2023. - April 15, 2023. - https://finance.yahoo.com/quote/S68.SI/history?p=S68.SI.
Zeng Pengyu [et al.] Muformer: A long sequence time-series forecasting model based on modified multi-head attention [Journal] // Knowledge-Based Systems. - October 27, 2022. - 109584 : Vol. 254.

