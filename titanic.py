import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ind_var_vs_survived():
  """
  - 1st class only class more likely to survive. 2nd class about 50/50 but 3rd class
  had high fatality rate. This difference is not explained by fare.
  - More males on board but female fatality rate much lower.
  - S and Q disembark points have similar bad fatality rates. Also have lower fares.
  - Age may only matter past threshold of ~63
  - SibSp and Parch have worse fatality rate above ~2
  """
  ind_vars = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
  fig, axarr = plt.subplots(2, 4)
  axarr = axarr.ravel()
  for ax in axarr:
    try:
      next_var = ind_vars.pop(0)
    except IndexError:
      break
    else:
      if next_var in ["Age", "SibSp", "Parch", "Fare"]:
        sns.swarmplot(x="Survived", y=next_var, data=df, ax=ax, orient="v")
      else:
        sns.countplot(x=next_var, hue="Survived", data=df, ax=ax, orient="v")
  plt.tight_layout()
  plt.show()

def explore_fare():
  """
  - 1st class fare much higher but 2nd and 3rd class fares comparable in distribution.
  - In spread, 1st class mainly on high side and 2nd class mainly in middle. 3rd class
  max as high as 2nd class.
  - Lowest fares departed from Q. Highest from C.
  """
  fig, axs = plt.subplots(2, 1, sharex=True)
  sns.stripplot(x="Fare", y="Pclass", data=df, jitter=True, ax=axs[0])
  sns.boxplot(x="Fare", y="Embarked", data=df, ax=axs[1])
  plt.show()

def explore_family():
  """Not abundantly clear
  """
  fig, ax = plt.subplots(2, 2)
  survive_df = df[df.Survived==1]
  fatal_df = df[df.Survived==0]
  ax[0, 0].hist(x=fatal_df["SibSp"])
  ax[0, 0].set_title("Sibsp - Fatal")
  ax[0, 1].hist(x=fatal_df["Parch"])
  ax[0, 1].set_title("Parch - Fatal")
  ax[1, 0].hist(x=survive_df["SibSp"])
  ax[1, 0].set_title("Sibsp - Survive")
  ax[1, 1].hist(x=survive_df["Parch"])
  ax[1, 1].set_title("Parch - Survive")
  plt.show()
  

if __name__ == "__main__":
  df = pd.read_csv("titanic_train.csv")
  print(df.head())
  print(df.describe())
  ind_var_vs_survived()
  explore_fare()
  explore_family()
