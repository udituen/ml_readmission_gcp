## UCI-diabetes-readmission-predictor
MLOps pipeline deployed on GCP for the task of predicting the readmission rate of diabetes patients

```
ml-readmission-gcp/
│
├── app/
│   ├── main.py
│   ├── model_loader.py
│
├── training/
│   ├── train.py
│   ├── preprocess.py
│
├── models/
│   └── model.pkl
│
├── requirements.txt
├── Dockerfile
├── .github/workflows/deploy.yml
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