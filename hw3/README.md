# Software Requirements
- Keras, of course :)
- NumPy
- OpenCV
- TensorFlow, which I use as Keras backend
- matplotlib
- pickle
- scikit-learn

# Using My Pre-trained Model
```
bash test.sh <path_to_data_folder> trained_model <output_csv_filename>
```
Note that you just type *extacly* `trained_model`, i.e., *no need* to take care of my `_1` or `_2` postfix.

# Training
```
bash train.sh <path_to_data_folder> <model_name>
```
# Testing
```
bash test.sh <path_to_data_folder> <model_name> <output_csv_filename>
```
Given `<model_name>` in training phase, there will be 2 output models (`<model_name>_1` and `<model_name>_2`). Again, you just type *exactly* `<model_name>` (i.e., ignoring the `_1` and `_2` postfix).
