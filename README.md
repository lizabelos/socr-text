# SOCR TEXT

## Requirements

 - Pytorch 0.4 : ```conda install pytorch=0.4 -c pytorch```
 - OpenCV 3 : ```conda install opencv```
 - scikit-image :  ```conda install -c conda-forge scikit-image```
 - All in requirements.txt ```pip instal -r requirements.txt'```

## Compilation

```
python3 setup.py build_ext --inplace
```

## Training

```
python3 train.py --iamtrain [train_path] --iamtest [test_path]
```

## Evaluate

```
python3 evaluate.py path
```

## Dataset

This is the link to IAM Handwriting Database :

[IAM Handwriting Database](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

If you want to enable test during the training, you have to split yourself the dataset into a train part and a test part.