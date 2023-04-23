# Nanyang Technological University (NTU), SCSE, AI6101 Deep Learning and Application.

Project Assignment Submission for module AI6101, Deep Learning and Application.

## Data Source
-  Data Set 1: WTH, Meteorological
    - Obtained from XXX
  

-  Data Set 2: Singapore Exchange Limited (S68.SI) 
    - Obtained from [Yahoo Finance, S68.SI](https://finance.yahoo.com/quote/S68.SI/history?p=S68.SI)
    - Processed Data can refer to [dataset](dataset)

## Exploration on Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting 

**Informer**

In this project, we choose Informer as the research object. Informer is a prediction model based on the Transformer structure for long-term sequences. Informer uses methods such as multi-head self-attention mechanism, depthwise separable convolution, periodic attention mechanism, and stacked structure, which can enhance the robustness of the model and integrate Global and key local information, and consider different time scales, to achieve effective model-ing and forecasting of long sequences.
After carefully studying the paper and related work, we reproduced the entire work based on our own understand-ing, discussed the impact of different optimizers and loss functions on the model on the original data set, and on stock prices, S68.SI) as new data sets to explore Informer Generalization ability and existing problems.

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

