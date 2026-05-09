<h1>OLD</h1>


Run the old_data_columns.py to get the data that is used for training the model

class definition of the model is within the onnx_save.ipynb.

Per Company transformers include the original .pth file and the evaluation score.

moves all old py files to old_ipynb files



--------------------------------------------------------------------------------------------------------------------------------

<h1>NEW</h1>

Training new models for company 

```
2110, 1080, 4030, 3092, 4348, 4190, 2222, 7030, 7010, 2050, 2060, 4321, 8100, 2120, 6010, 8250, 8300, 2200, 1060, 1010, 1321, 1180, 2285, 8200, 1050, 1030, 4081, 2286, 1150, 2320, 3080, 4340, 3090, 4342, 2283, 2284, 1323, 4009, 4003, 4334, 2020, 1140, 4084, 7020, 2370, 3030, 4261, 4012, 1020, 1830, 2150, 4260, 3020, 4333, 4001, 1835, 4165, 1320, 4007, 3040, 8280, 4347, 2382, 2223, 4163, 6070, 4262, 4083, 3010, 4164, 2270, 3003, 1834, 1120, 1303, 8170, 1214, 8010, 2081, 4162, 7204, 7202, 2281, 3002, 3050, 1833, 4143, 8030, 4250, 3005, 2230, 4280, 4020, 4004, 8210, 4142, 4263, 2280, 1212, 4002, 8012, 4150, 8160, 1831, 8070, 4051, 1322, 3004, 1182, 6017, 2080, 4161, 4031, 3060, 4071, 1302, 4322, 8020, 4264, 4016, 4072, 4300, 4090, 6001, 6004, 7040, 4191, 4332, 2381, 6014, 4017, 4015, 4324, 6015, 2310, 4336, 4006, 2300, 4338, 8230, 4082, 4144, 4200, 4014, 1111, 4291, 4339, 4013, 2282, 4325, 1304, 7203, 4018, 2084, 7200, 4100, 2070, 2287, 4331, 2190, 2250, 1211, 8313, 4193, 8180, 1183, 2290, 2240, 6020, 4192, 4345, 1301, 8270, 4180, 8240, 4210, 8040, 4130, 8310, 4290, 4344, 6016, 4050, 2082, 4040, 1210, 2083, 4080, 6013, 2010, 4170, 3007, 4240
```

Evaluation explaination 

MAE — Mean Absolute Error

MAE = average of |actual - predicted|

Example:

Actual price change   : +2.00%
Predicted price change: +0.30%
Error                 : 1.70%  ← this is your MAE

RMSE - Root Mean Absolute Error

RMSE = square root of average of (actual - predicted)²

Directional Accuracy

Dir Acc = % of times sign(predicted) == sign(actual)

Measures did the model get the direction right — up or down — regardless of magnitude.

Actual    : +2.00%  → direction = UP   (+)<br>
Predicted : +0.30%  → direction = UP   (+)  correct direction

Actual    : +2.00%  → direction = UP   (+)<br>
Predicted : -0.50%  → direction = DOWN (-)   wrong direction