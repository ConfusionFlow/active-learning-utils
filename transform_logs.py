# coding: utf-8
import argparse
import json
import numpy as np
import pandas as pd

from confusionflow.logging import Run, Fold
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description="transformation of MNIST_GRIS_logs into ConfusionFlow format"
    )
    parser.add_argument(
        "--filename", type=str, required=True, help="path to file with logs"
    )
    parser.add_argument(
        "--run_id", type=str, required=True, help="identifier of run / experiment"
    ),
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="path to output directory (default: ./logs)",
    )

    args = parser.parse_args()

    with open(args.filename, "r") as f:
        content = f.read()
        results = json.loads(content)
        results_list = results[1]["testFVList"][1]

    results_list = list(map(lambda x: x[1], results_list))
    results_dataframe = pd.DataFrame(results_list)

    test_fold = Fold(None, "mnist-gris_test", "mnist-gris.yml")

    # we are required to specify the trainfoldId (simply using the id for the testfold for now)
    run = Run(runId=args.run_id, folds=[test_fold], trainfoldId="mnist-gris_test")

    for epoch in results_dataframe["iteration"].unique().tolist():
        results_dataframe_epoch = results_dataframe.loc[
            results_dataframe["iteration"] == epoch
        ]

        for fold, foldlog in zip(run.folds, run.foldlogs):
            assert fold.foldId == foldlog.foldId

            targetlabel = results_dataframe_epoch["targetLabel"]
            predictedlabel = results_dataframe_epoch["predictedLabel"]
            predictedlabel[pd.isna(predictedlabel)] = 0

            confmat = confusion_matrix(
                targetlabel.astype(np.int).tolist(),
                predictedlabel.astype(np.int).tolist(),
            )
            foldlog.add_epochdata(epochId=epoch, confmat=confmat.flatten().tolist())

    run.export(logdir=args.logdir)


if __name__ == "__main__":
    main()
