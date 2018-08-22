# SOCR TEXT

## Requirements

 - Pytorch 0.4 : ```conda install pytorch=0.4 -c pytorch```
 - OpenCV 3 : ```conda install opencv```
 - scikit-image :  ```conda install -c conda-forge scikit-image```
 - All in requirements.txt ```pip install -r requirements.txt'```

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

The ```lines``` folder, and the ```xml``` folder need to be placed into the same folder, for example name ```train```. So you will have a folder ```train/lines``` and ```train/xml```. This is the path to the train folder which need to be specified to the --iamtrain command line.

If you want to enable test during the training, you have to split yourself the dataset into a train part and a test part.