# The Logs of The Competition

## Nov 30, 2019

### Data Preprocessing

1. As for the **missing values**, we filled it with the **most frequent** values of each property.

2. Then we processed the **categorical columns** using the **One-Hot Encoding** technique.

### Model
We adopted the xgboost model and the setting of the parameters is presented 
in the following table.

| Parameters | Values |
| :--------: | :----: |
| n_estimators | 1000 |
| learning_rate | 0.05 |

### Result

Score: 14727.09201

Rank : 1297 (Top 8%)