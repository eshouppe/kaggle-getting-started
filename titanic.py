import math

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def explore_all_vars():
  """
  - 1st class only class more likely to survive. 2nd class about 50/50 but 3rd class
  had high fatality rate. This difference is not explained by fare.
  - More males on board but female fatality rate much lower.
  - S and Q disembark points have similar bad fatality rates. Also have lower fares.
  - Age may only matter past threshold of ~63
  - SibSp and Parch have worse fatality rate above ~2
  """
  ind_vars = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
  _, axarr = plt.subplots(2, 4)
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
  _, axs = plt.subplots(2, 1, sharex=True)
  sns.stripplot(x="Fare", y="Pclass", data=df, jitter=True, ax=axs[0])
  sns.boxplot(x="Fare", y="Embarked", data=df, ax=axs[1])
  plt.show()

def explore_family():
  """Not abundantly clear"""
  _, ax = plt.subplots(1, 2)
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

def preprocess_data():
  """Preprocess"""
  pass
  
def confusion_matrix_plot(y, yhat):
  """Print confusion matrix for model results"""
  tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
  idx = pd.MultiIndex.from_tuples([('Actual', 0), ('Actual', 1)])
  df = pd.DataFrame({('Predicted', 0):[tn, fn], ('Predicted', 1):[fp, tp]}, index=idx)
  print("Confusion Matrix")
  print(df)

def residuals_plot(x, y, yhat):
  """Plot residuals for each explanatory variable against
  the fitted values
  """
  df = pd.concat([x,y,yhat], axis=1)
  df["residual"] = df["y"] - df["yhat"]
  del df["y"]
  del df["yhat"]
  cols = list(df.columns).remove("residual")
  num_vars = len(cols)
  num_rows = math.ceil(num_vars / 2)

  _, axarr = plt.subplots(num_rows, 2)
  for idx, ax in enumerate(axarr.ravel()):
    sns.regplot(x=cols[idx], y="residual", data=df, fit_reg=False, label=cols[idx], ax=ax)
  
  plt.tight_layout()
  plt.show()

def logistic_regression(x_train, y_train, x_test, y_test):
  """Use l2 or l2 regularization. Choice to fit intercept.
  Try liblinear and sag solvers. Can preprocess all input
  to some scale.

  Logistic regressions assumes observations are independent,
  independent vars have little to no multicollinearity, and
  independent vars are linearly related to log odds.

  """
  param_combo = [
    {"penalty": 'l2', "solver": "sag", "intercept": True},
    {"penalty": 'l2', "solver": "sag", "intercept": False},
    {"penalty": 'l1', "solver": "saga", "intercept": True},
    {"penalty": 'l1', "solver": "saga", "intercept": False},
    {"penalty": 'l1', "solver": "liblinear", "intercept": True},
    {"penalty": 'l1', "solver": "liblinear", "intercept": False}
  ]
  for combo in param_combo:
    lr = LogisticRegression(penalty=combo["penalty"], fit_intercept=combo["intercept"],
                            random_state=42, solver=combo["solver"])
    lr.fit(x_train, y_train)
    yhat_test = lr.predict(x_test)

    print("Logistic Regression - {} - Intercept {}".format(combo["penalty"], combo["intercept"]))
    confusion_matrix(y_test, yhat_test)
    residuals_plot(x_test, y_test, yhat_test)
  
def cv():
  """Perform cross validation"""
  pass

if __name__ == "__main__":
  df = pd.read_csv("titanic_train.csv")
  # print(df.head())
  # print(df.describe())
  # explore_all_vars()
  # explore_fare()
  # explore_family()
