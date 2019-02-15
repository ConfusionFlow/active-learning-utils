# Script Usage
You can transform the logs from the `GRIS` format into the format required by `ConfusionFlow`.

The Python script takes 3 parameters:
- `filename` - path to file with logs
- `run_id` - run identifier
- `logdir` - path to output directory (default: ./logs)

Please make sure you use `Python3`. We also renamed `MNIST_GRIS.arff` folder to `MNIST_GRIS_logs`. The error messages from `pandas` can be ignored.

## Example command
```
pip install -r requirements.txt
python transform_logs.py --filename ./MNIST_GRIS_logs/accuracy/DenseAreasFirst_accuracy.json --run_id MNIST_GRIS_logs_DenseAreasFirst --logdir ./logs
python transform_logs.py --filename ./MNIST_GRIS_logs/accuracy/Greedy_accuracy.json --run_id MNIST_GRIS_logs_Greedy --logdir ./logs
python transform_logs.py --filename ./MNIST_GRIS_logs/accuracy/SmallestMargin_accuracy.json --run_id MNIST_GRIS_logs_SmallestMargin --logdir ./logs

confusionflow --logdir logs
```
