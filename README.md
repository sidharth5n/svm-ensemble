# svm-ensemble
Detecting objects in PASCAL VOC 2007 dataset using ensemble of exemplar SVMs. The codes used here are mostly from [exemplarsvm](https://github.com/quantombone/exemplarsvm) modified for only detecting objects and to be run on a single PC.

Compile all the files
```matlab
compile;
```

For training the model
```matlab
class = 'car';
data_dir = 'VOC2007/VOCdevkit';
dataset = 'VOC2007';
results_dir = 'results';
[models, M, models_name, params] = train_and_validate(class, data_dir, dataset, results_dir);
```

For testing the models
```matlab
test(models, M, models_name, params);
```

For more details refer T. Malisiewicz, A. Gupta and A. A. Efros, "Ensemble of exemplar-SVMs for object detection and beyond," ICCV, 2011.
