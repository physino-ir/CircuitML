from circuitml import platforms
from circuitml.svm import is_svm, port_svm
from circuitml.rvm import is_rvm, port_rvm
from circuitml.sefr import is_sefr, port_sefr
from circuitml.decisiontreeclassifier import is_decisiontree, port_decisiontree
from circuitml.decisiontreeregressor import is_decisiontree_regressor, port_decisiontree_regressor
from circuitml.randomforestclassifier import is_randomforest, port_randomforest
from circuitml.randomforestregressor import is_randomforest_regressor, port_randomforest_regressor
from circuitml.logisticregression import is_logisticregression, port_logisticregression
from circuitml.gaussiannb import is_gaussiannb, port_gaussiannb
from circuitml.pca import is_pca, port_pca
from circuitml.principalfft import is_principalfft, port_principalfft
from circuitml.linear_regression import is_linear_regression, port_linear_regression
from circuitml.xgboost import is_xgboost, port_xgboost


def port(
        clf,
        classname=None,
        classmap=None,
        platform=platforms.ARDUINO,
        precision=None,
        **kwargs):
    """Port a classifier to plain C++"""
    assert platform in platforms.ALL, 'Unknown platform %s. Use one of %s' % (platform, ', '.join(platforms.ALL))

    if is_svm(clf):
        return port_svm(**locals())
    elif is_rvm(clf):
        return port_rvm(**locals())
    elif is_sefr(clf):
        return port_sefr(**locals())
    elif is_decisiontree(clf):
        return port_decisiontree(**locals())
    elif is_randomforest(clf):
        return port_randomforest(**locals())
    elif is_logisticregression(clf):
        return port_logisticregression(**locals())
    elif is_gaussiannb(clf):
        return port_gaussiannb(**locals())
    elif is_pca(clf):
        return port_pca(**locals())
    elif is_principalfft(clf):
        return port_principalfft(**locals(), **kwargs)
    elif is_linear_regression(clf):
        return port_linear_regression(**locals(), **kwargs)
    elif is_xgboost(clf):
        return port_xgboost(**locals(), **kwargs)
    elif is_decisiontree_regressor(clf):
        return port_decisiontree_regressor(**locals(), **kwargs)
    elif is_randomforest_regressor(clf):
        return port_randomforest_regressor(**locals(), **kwargs)
    raise TypeError('clf MUST be one of %s' % ', '.join(platforms.ALLOWED_CLASSIFIERS))
