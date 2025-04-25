# src/metrics/base.py


class Metric:
    def compute(self, predictions, references):
        raise NotImplementedError("Must override compute method")
