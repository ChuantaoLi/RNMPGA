import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def _tree_params(opts):
    """决策树 / 随机森林共享参数 (从 opts 提取, None 值不传入)"""
    params = {}
    for key in ("criterion", "max_depth", "min_samples_split",
                "min_samples_leaf", "max_features", "random_state"):
        if key in opts and opts[key] is not None:
            params[key] = opts[key]
    return params


def build_classifier(opts):
    """根据 opts['classifier'] 构建分类器 (dt / rf / knn)

    分类器类型与参数统一在 config.yaml 的 classifier 段配置, 由 main.py
    合并进每个模型的参数; 算法内部 (eval_accuracy) 与本文件最终评估共用
    此函数, 保证"特征选择过程"与"最终评估"使用同一分类器。
    """
    ctype = opts.get('classifier', 'dt')
    if ctype == 'dt':                       # 决策树 (默认)
        return DecisionTreeClassifier(**_tree_params(opts))
    if ctype == 'rf':                       # 随机森林 (树数默认 20)
        params = _tree_params(opts)
        params['n_estimators'] = int(opts.get('n_estimators', 20))
        if opts.get('n_jobs') is not None:
            params['n_jobs'] = opts['n_jobs']
        return RandomForestClassifier(**params)
    if ctype == 'knn':                      # K近邻分类器
        params = dict(n_neighbors=int(opts.get('n_neighbors', 5)))
        for key in ("weights", "algorithm", "leaf_size", "p", "metric", "n_jobs"):
            if key in opts and opts[key] is not None:
                params[key] = opts[key]
        return KNeighborsClassifier(**params)
    raise ValueError(f"未知分类器类型: {ctype!r} (可选 dt/rf/knn)")


# error rate
def error_rate(xtrain, ytrain, x, opts):
    # parameters
    fold  = opts['fold']
    xt    = fold['xt']
    yt    = fold['yt']
    xv    = fold['xv']
    yv    = fold['yv']

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    xtrain  = xt[:, x == 1]
    ytrain  = yt.reshape(num_train)  # Solve bug
    xvalid  = xv[:, x == 1]
    yvalid  = yv.reshape(num_valid)  # Solve bug
    # Training
    mdl     = build_classifier(opts)
    mdl.fit(xtrain, ytrain)
    # Prediction
    ypred   = mdl.predict(xvalid)
    acc     = np.sum(yvalid == ypred) / num_valid
    error   = 1 - acc
    
    return error


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
    # Parameters (可通过 opts 配置, 默认 0.99 / 0.01)
    alpha    = opts.get('alpha', 0.99)
    beta     = opts.get('beta', 1 - alpha)
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error = error_rate(xtrain, ytrain, x, opts)
        # Objective function
        cost  = alpha * error + beta * (num_feat / max_feat)
        
    return cost
