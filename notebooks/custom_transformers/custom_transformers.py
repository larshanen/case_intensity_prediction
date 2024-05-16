from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer, FunctionTransformer, RobustScaler
from sklearn.feature_extraction import DictVectorizer
from functools import reduce
import numpy as np
import pandas as pd

class DFFunctionTransformer(TransformerMixin):
    # FunctionTransformer but for pandas DataFrames
    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt

class DFFeatureUnion(TransformerMixin):
    # FeatureUnion but for pandas DataFrames
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        # Assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion
    
class DFRobustScaler(TransformerMixin):
    # RobustScaler but for pandas DataFrames
    def __init__(self):
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        self.rs.fit(X)
        self.center_ = pd.Series(self.rs.center_, index=X.columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xrs = self.rs.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled

class ColumnExtractor(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        # Assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols
    
class DummyTransformer(TransformerMixin):
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # Assumes all columns of X are strings
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # Assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.feature_names_
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # Drop column indicating NaNs
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum

class BooleanTransformer(TransformerMixin):
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # Convert boolean columns to strings for DictVectorizer
        X = X.astype(str)
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # Convert boolean columns to strings for DictVectorizer
        X = X.astype(str)
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.feature_names_
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # Drop columns indicating NA values
        na_cols = [c for c in cols if '=<NA>' in c]
        Xdum = Xdum.drop(na_cols, axis=1)
        return Xdum
    
class MultiTransformer(TransformerMixin):
    def __init__(self, sep=', '):
        self.sep = sep
        self.mlbs = None

    def _col_transform(self, x, mlb):
        cols = [''.join([x.name, '=', c]) for c in mlb.classes_]
        xmlb = mlb.transform(x)
        xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
        return xdf

    def fit(self, X, y=None):
        Xsplit = X.apply(lambda x: x.str.split(self.sep))
        self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
        return self

    def transform(self, X):
        # Assumes X is a DataFrame
        Xsplit = X.apply(lambda x: x.str.split(self.sep))
        Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i])
                 for i, c in enumerate(X.columns)]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xmlbs)
        return Xunion
    
class CountTransformer(TransformerMixin):
    def __init__(self, sep=', '):
        self.sep = sep

    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        # Assumes X is a DataFrame
        Xlen = X.apply(lambda x: x.str.count(self.sep) + 1)
        Xlen = pd.DataFrame(Xlen, index=X.index, columns=X.columns)
        return Xlen

class DateTransformer(TransformerMixin):
    def fit(self, X, y=None):
        # Stateless transformer
        return self

    def transform(self, X):
        # Year as-is
        Xyear = X.apply(lambda x: pd.to_datetime(x).dt.year)

        # Month as a cyclical function of sin
        Xmonth = X.apply(lambda x: np.sin(2 * np.pi * pd.to_datetime(x).dt.month/12.0))

        # Day of month as a cyclical function of sin
        Xday_month = X.apply(lambda x: np.sin(2 * np.pi * pd.to_datetime(x).dt.day/pd.to_datetime(x).dt.daysinmonth))

        # Day of week as a cyclical function of sin
        Xday_week = X.apply(lambda x: np.sin(2 * np.pi * pd.to_datetime(x).dt.dayofweek/7.0))
        
        # Create dataframe with combination of all date features
        Xconcat = pd.concat([Xyear, Xmonth, Xday_month, Xday_week], axis=1)
        Xconcat.columns = ['Melding_jaar', 'Melding_maand', 'Melding_dag_maand', 'Melding_dag_week']
        return Xconcat
    
class AgeTransformer(TransformerMixin):
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xyear = X.Datum_melding.dt.year
        Xage = Xyear - X.Geboortejaar
        Xage = pd.DataFrame(Xage, index=X.index, columns=['Leeftijd'])
        return Xage
