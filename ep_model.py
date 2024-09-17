import xgboost
import numpy as np

model = xgboost.Booster()
model.load_model('C:/Users/ctrask/Documents/NFL-RL/epa.model')

# NFLFastR's Expected Points (EP) model is an XGBoost Classifier
# It takes 9 variables as inputs and uses an encoding to map the string variables (and season) to boolean columns
# As a result, the actual XGBoost classifier model has 18 features
# ["half_seconds_remaining", "yardline_100", "home", "retractable", "dome", "outdoors", "ydstogo", "era0", "era1", "era2", "era3", "era4", "down1", "down2", "down3", "down4",
#   "posteam_timeouts_remaining", "defteam_timeouts_remaining"]

# Eras correspond to certain season ranges:
#   era0: Seasons 2001 and earlier
#   era1: Seasons (2001, 2005]
#   era2: Seasons (2005, 2013]
#   era3: Seasons (2013, 2017]
#   era4: Seasons after 2017
data = np.array([[1800, 90, 1, 0, 0, 0, 10, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3, 3]], dtype=object)
y = np.array([0.507])

dtest = xgboost.DMatrix(data, y)

probs = model.predict(dtest)
weights = np.array([7, -7, 3, -3, 2, -2, 0])

print(probs)
print(np.dot(probs,weights))