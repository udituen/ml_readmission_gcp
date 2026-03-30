## UCI-diabetes-readmission-predictor
MLOps pipeline deployed on GCP for the task of predicting the readmission rate of diabetes patients

```
ml-readmission-gcp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schema.py
├── models/
│   └── xgb_model_v2.pkl
├── data/
│   └── diabetic_data.csv
├── train
│   ├── train.py        
├── Dockerfile
├── requirements.txt
└── .github/
    └── workflows/
        └── deploy.yml
├── README.md
```



### GCP Services in use

```
GCP Account
   └── ml-readmission-gcp
          ├── Cloud Run service
          ├── Images
          ├── Logs
          └── Permissions
```