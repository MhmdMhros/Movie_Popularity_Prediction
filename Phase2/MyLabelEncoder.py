from sklearn.preprocessing import LabelEncoder
import numpy as np


class MyLabelEncoder(LabelEncoder):
    def __init__(self, unseen_value=None):
        self.unseen_value = unseen_value
        super(MyLabelEncoder, self).__init__()

    def transform(self, y):
        if self.unseen_value is not None:
            # Map unseen labels to the specified value
            unseen_labels = set(y) - set(self.classes_)
            mapping = {label: self.unseen_value for label in unseen_labels}
            self.classes_ = np.concatenate([self.classes_, list(unseen_labels)])
            self.transform_map_ = {label: i for i, label in enumerate(self.classes_)}
            y_transformed = [self.transform_map_.get(label, self.unseen_value) for label in y]
        else:
            y_transformed = super(MyLabelEncoder, self).transform(y)
        return y_transformed
