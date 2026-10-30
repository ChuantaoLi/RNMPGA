"""
RNMPGA: Decomposition-Based Multi-Population Genetic Algorithm for
Feature Selection (一种用于特征选择的基于分解的多种群遗传算法)
=========================================================
算法全称: 一种用于特征选择的基于分解的多种群遗传算法
英文全称: A Decomposition-based Multi-population Genetic Algorithm
          for Feature Selection (RNMPGA)

设计要点 (V2.0.8 定稿):
  - 分解 (MOEA/D, Zhang & Li 2007): Das-Dennis 内部均匀权重
    w_k = (k+1)/(M+1) 将双目标问题分解为 M=7 个加权和标量化子问题:
    fit_k(x) = w_k·(1-Acc(x)) + (1-w_k)·|x|/D (高 w 攻精度, 低 w 攻稀疏);
  - 空集排除 (MOEA/D-FS, Nguyen et al. 2020): 锦标赛中空集 fit = +∞,
    永不成为亲本, 空集仅作为前沿稀疏端极值保留;
  - 多种群环拓扑: 7 个子种群排成环 (环 = 子问题邻域), 每 t_mig 代双向
    精英迁移 (门槛: 移民秩优于接收方最差个体) + 邻域锦标赛交配
    (mating restriction, 概率 p_trans);
  - 引导 (mRMR, Peng et al. 2005): 相关性短名单 + 贪心前向排序 ->
    指导向量 g ∈ [0,1]^D, 驱动初始化与变异;
  - 初始化 (平滑比例式): p_j = λ·(α+g_j)/(αD+S), α=1/8 为平滑伪计数
    (保证短名单外特征可被探索), λ 为期望初始特征数;
  - 均匀交叉 (相同位保持、相异位随机取 0/1) + 自适应指导变异
    (每后代期望翻转数 μ=1, 随代数衰减);
  - 环境选择: 子种群内 NSGA-II (非支配排序 + 拥挤距离);
  - 输出: 全部子种群个体并集的非支配前沿 (MOEA/D 标准输出)。

历史沿革 (Tchebycheff/参考点/加权和的取舍、初始化与变异的改进) 详见
IMPROVEMENT_LOG.md 与 Explanation/ 下对应文档。

接口与 Model 下其他算法一致:
  jfs(xtrain, ytrain, opts) -> {'sf', 'nf', 'c', 'pf', 'pop_stats'}
"""
import numpy as np

import os
from concurrent.futures import Future, ThreadPoolExecutor

from .functionHO import build_classifier


# ===========================================================================
# 零、基础工具函数 (自包含实现, 不依赖其他模型脚本)
# ===========================================================================

def mutual_information(feature, label):
    """互信息 MI(Fj; C), 完整复刻官方代码 MItest.m

    参数:
        feature: 特征列 (n,)
        label:   类标签列 (n,) (可为字符串, 如 UCI Sonar 的 'R'/'M')
    返回:
        标量互信息
    """
    n = len(feature)

    # 类标签可能为字符串 (本项目数据集标签未做数值化), 先映射为 0..K-1 数值
    label = np.asarray(label)
    if not np.issubdtype(label.dtype, np.number):
        uniq = np.unique(label)
        label = np.searchsorted(uniq, label)

    def quantize(v):
        v = np.asarray(v, dtype=float)
        vmin, vmax = v.min(), v.max()
        if vmax - vmin < 1e-12:
            return np.ones(n, dtype=int)  # 常值 -> 全部落入第 1 个量化区间
        # int16(x*255) 对非负值等价于向下取整, 与 MATLAB 行为一致
        return ((v - vmin) / (vmax - vmin) * 255.0).astype(np.int64) + 1

    a = quantize(feature)   # 特征量化值 1~256
    b = quantize(label)     # 标签量化值 1~256

    # 联合直方图 256×256
    hab = np.zeros((256, 256))
    np.add.at(hab, (a - 1, b - 1), 1)
    # 边缘直方图
    ha = np.bincount(a - 1, minlength=256)
    hb = np.bincount(b - 1, minlength=256)

    # 熵 (自然对数, 与 MATLAB log 一致)
    def entropy(counts):
        p = counts[counts > 0] / n
        return float(-(p * np.log(p)).sum())

    return entropy(ha) + entropy(hb) - entropy(hab.ravel())


def eval_accuracy(sel_idx, opts):
    """评估一个特征子集的分类准确率 (包装器方法, 论文式 (1) 中的 F2)

    实现选择: 使用项目统一入口的分类器 (build_classifier), 在训练集 fold['xt']
    上训练, 在验证集 fold['xv'] 上计算准确率。

    参数:
        sel_idx: 选中特征的索引数组 (可能为空)
        opts:    参数字典, 包含 'fold' (划分) 与分类器参数
    返回:
        准确率 (0~1 之间的浮点数)
    """
    fold = opts['fold']
    xt, yt, xv, yv = fold['xt'], fold['yt'], fold['xv'], fold['yv']
    if sel_idx.size == 0:
        # 未选中任何特征时准确率视为 0 (无法训练分类器)
        return 0.0
    mdl = build_classifier(opts)
    mdl.fit(xt[:, sel_idx], np.asarray(yt).ravel())
    y_pred = mdl.predict(xv[:, sel_idx])
    return float(np.mean(np.asarray(yv).ravel() == y_pred))


class EvalCache:
    """评估缓存 + 批量并行提交器 (纯工程优化, 与算法逻辑无关)

    准确率是 (子集, fold) 的确定性函数; 缓存按原始 D 维特征索引的升序
    元组作键, 同一子集只算一次。
    并行化保证逐位一致: 调用方先串行生成全部待评个体 (RNG 消耗顺序与
    串行实现完全相同), 再统一 batch() 批量提交; 评估结果与提交/完成
    顺序无关, 按序取回即为串行结果。
    线程安全: 并发 worker 对 dict 的 get/set 在 GIL 下原子; 同一键并发
    重复计算仅造成无害的重复 fit (结果一致, 后写覆盖)。
    评估函数可注入 (fn): 默认 eval_accuracy; fn 必须是 (子集, opts) 的
    确定性纯函数, 且调用方须遵守上面的"先串行生成、再统一 batch"约定。
    """
    def __init__(self, opts, fn=None):
        workers = int(opts.get('workers', 0))
        if workers <= 0:
            workers = min(int(os.cpu_count() or 1), 8)
        self._opts = opts
        self._fn = fn if fn is not None else eval_accuracy
        self._cache = {}
        self._exec = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def _key(self, sel):
        return tuple(int(v) for v in sel)   # 子集均按升序存储

    def _compute(self, key, sel):
        """worker: 计算准确率并写缓存 (并发重复计算同一子集无害)"""
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        acc = self._fn(sel, self._opts)
        self._cache[key] = acc
        return acc

    def __call__(self, sel, _opts=None):
        """同步单次评估 (与串行接口语义一致: 提交并等待结果)
        _opts 仅为兼容 eval_fn(sel, opts) 形式的调用点而存在, 被忽略,
        评估函数已在构造时绑定 opts。
        """
        return self.batch([sel])[0]

    def batch(self, sels):
        """批量评估: 缓存命中直接取; 未命中提交线程池, 按序返回结果"""
        futs = []
        for sel in sels:
            key = self._key(sel)
            cached = self._cache.get(key)
            if cached is not None:
                fut = Future()
                fut.set_result(cached)
            elif self._exec is None:
                fut = Future()
                fut.set_result(self._compute(key, sel))
            else:
                fut = self._exec.submit(self._compute, key, sel)
            futs.append(fut)
        return [f.result() for f in futs]

    def shutdown(self):
        if self._exec is not None:
            self._exec.shutdown(wait=True)


def _dominates(a, b):
    """判断目标向量 a 是否支配 b: 所有分量 <= 且至少一个分量严格 <。

    避免对 2 元素 numpy 数组走迭代器/归约 (np.all/np.any 等) 的 C 路径
    —— numpy 在某些 Windows 环境/版本上对极小数组的这些路径会随机触发
    原生崩溃。先一次性转成纯 Python float 再逐分量比较, 语义与逐元素
    比较完全等价。
    """
    try:
        a0, a1 = float(a[0]), float(a[1])
        b0, b1 = float(b[0]), float(b[1])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            f"目标向量必须包含两个数值分量: a={a!r}, b={b!r}"
        ) from exc
    return (a0 <= b0 and a1 <= b1) and (a0 < b0 or a1 < b1)


def fast_non_dominated_sort(obj):
    """快速非支配排序 (NSGA-II, Deb et al. 2002)

    所有目标均按最小化处理: 个体 i 支配 j 当且仅当
        obj[i] 每个分量都不大于 obj[j], 且至少一个分量严格小于。
    参数:
        obj: (N, n_obj) 目标矩阵 (每行一个个体, 每列一个目标, 均最小化)
    返回:
        fronts: 列表, 每个元素是位于同一非支配层级的个体索引数组
                (fronts[0] 即帕累托前沿 PF, fronts[-1] 即最差前沿 LF)
    """
    obj = np.asarray(obj, dtype=float)
    if obj.ndim != 2 or obj.shape[1] != 2:
        raise ValueError(f"非支配排序需要形状为 (N, 2) 的目标矩阵, 得到 {obj.shape}")
    N = obj.shape[0]
    rows = [(float(row[0]), float(row[1])) for row in obj]
    # dominated_count[i]: 支配个体 i 的个体数量
    dominated_count = np.zeros(N, dtype=int)
    # dom_list[i]: 被个体 i 支配的个体列表
    dom_list = [[] for _ in range(N)]

    # 两两比较, O(N^2)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if _dominates(rows[i], rows[j]):
                dom_list[i].append(j)
            elif _dominates(rows[j], rows[i]):
                dominated_count[i] += 1

    # 第一层: 不被任何个体支配的个体
    fronts = []
    current = [i for i in range(N) if dominated_count[i] == 0]
    while current:
        fronts.append(current)
        nxt = []
        for i in current:
            for j in dom_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def crowding_distance(obj, front):
    """拥挤距离 (NSGA-II)

    用于在同一非支配层级内衡量个体的稀疏程度: 距离越大说明个体周围越空旷,
    环境选择时优先保留, 以维持种群多样性。
    参数:
        obj:   (N, n_obj) 目标矩阵
        front: 某一层级的个体索引数组
    返回:
        与 front 等长的拥挤距离数组 (边界个体距离记为无穷大)
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)
    m = obj.shape[1]
    cd = np.zeros(n)
    for k in range(m):
        order = np.argsort(obj[front, k])
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        fmin = obj[front[order[0]], k]
        fmax = obj[front[order[-1]], k]
        if fmax - fmin < 1e-12:
            continue
        for t in range(1, n - 1):
            cd[order[t]] += (obj[front[order[t + 1]], k] - obj[front[order[t - 1]], k]) / (fmax - fmin)
    return cd


def rank_and_crowding(obj):
    """计算每个个体的 (非支配层级 rank, 拥挤距离)

    供锦标赛选择使用: 先比层级 (小者优), 层级相同比拥挤距离 (大者优)。
    """
    n = obj.shape[0]
    rank = np.zeros(n, dtype=int)
    cd = np.zeros(n)
    fronts = fast_non_dominated_sort(obj)
    for fi, f in enumerate(fronts):
        c = crowding_distance(obj, f)
        for k, i in enumerate(f):
            rank[i] = fi
            cd[i] = c[k]
    return rank, cd


# ===========================================================================
# 一、mRMR 单滤波指导向量 (Peng et al. 2005)
# ===========================================================================

def mrmr_guidance(xt, y, K=200):
    """mRMR 贪心前向排序 -> 指导向量 g ∈ [0,1]^D

    步骤:
      1. 相关性: 每个特征与类标签的互信息 r_j = I(f_j; c);
      2. 短名单: 相关性前 K 个特征 (控制成对互信息的计算规模,
         总开销 O(D·n + K²·n), 线性于特征数);
      3. mRMR 贪心前向: 依次选出
             f^(1) = argmax_j r_j
             f^(t) = argmax_{j∉S} [ r_j - (1/|S|)·Σ_{f'∈S} I(f_j; f') ]
      4. 指导向量: g_j = (K - pos_j)/K (pos_j 为 j 在 mRMR 排序中的
         位置, 越早入选权重越高), 未入短名单的特征为 0。
    """
    D = xt.shape[1]
    # 1) 相关性 (逐特征互信息)
    rel = np.array([mutual_information(xt[:, j], y) for j in range(D)])
    # 2) 短名单
    K = min(int(K), D)
    shortlist = np.argsort(-rel)[:K]          # 相关性降序的前 K 个
    sl_idx = {j: i for i, j in enumerate(shortlist)}   # 原特征号 -> 短名单位置
    # 3) 短名单内成对互信息矩阵 (K×K)
    M = np.zeros((K, K))
    for i in range(K):
        a = xt[:, shortlist[i]]
        for j in range(i + 1, K):
            v = mutual_information(a, xt[:, shortlist[j]])
            M[i, j] = M[j, i] = v
    # 4) mRMR 贪心前向
    first = int(np.argmax(rel[shortlist]))
    order = [first]
    red_sum = M[first].copy()                 # 累计冗余 Σ I(f_j; f')
    rem = [i for i in range(K) if i != first]
    while rem:
        best_i, best_s = rem[0], -np.inf
        for i in rem:
            # 准则: r_j - (1/|S|)·Σ I(f_j; f')
            s = rel[shortlist[i]] - red_sum[i] / len(order)
            if s > best_s:
                best_s = s
                best_i = i
        order.append(best_i)
        red_sum += M[best_i]
        rem.remove(best_i)
    # 5) 指导向量: 位置越靠前权重越高
    g = np.zeros(D)
    for pos, i in enumerate(order):
        g[shortlist[i]] = (K - pos) / K
    return g


# ===========================================================================
# 二、初始化与算子 (mRMR 指导贯穿)
# ===========================================================================

def guided_init(N, D, g, rng):
    """按 mRMR 指导向量初始化 (平滑比例式)

    p_j = λ·(α + g_j) / (α·D + S),  S = Σ_j g_j.

    逻辑: mRMR 得分 g_j 是对特征相关度的点估计, 对未入短名单的特征给出
    硬零过于武断; 给每个特征加平滑伪计数 α 得平滑相关度 α+g_j, 初始个体
    按该相关度比例分配 λ 个期望名额 (由 Σ_j p_j = λ 精确保证):
      - λ = max(2, min(K, round(D/200))): 期望初始特征数 (K 为短名单规模);
      - α = 1/8: 平滑伪计数, 保证短名单外特征 (g_j=0) 仍以正概率被探索。
    该式不引入任何比例式补丁或经验噪声参数 (详见 Explanation 初始化成因分析)。
    """
    g = np.asarray(g, dtype=float)
    S = float(g.sum())
    K = int(np.count_nonzero(g > 0))
    alpha = 1.0 / 8.0
    lam = max(2, min(K, int(round(D / 200.0))))
    denom = alpha * D + S
    p = (lam * (alpha + g) / denom) if denom > 0 else np.full(D, lam / D)
    p = np.clip(p, 0.0, 1.0)
    pop = rng.random((N, D)) < p
    none = pop.sum(axis=1) == 0
    if none.any():
        pop[none, rng.integers(0, D, size=int(none.sum()))] = True
    return pop


def evaluate_pop(pop, acc_of):
    """评估种群: 返回双目标矩阵 (|s|, 1-Acc) 与准确率数组

    整批提交给 acc_of (EvalCache) 而非逐个调用: 候选集在进入本函数前
    已由调用方生成完毕, 此处不再消耗随机数, 故批量并行与逐个串行
    评估的结果逐位一致 (见 EvalCache 文档)。
    """
    accs = np.asarray(acc_of.batch([np.where(v)[0] for v in pop]))
    obj = np.column_stack([pop.sum(axis=1), 1.0 - accs])
    return obj, accs


def uniform_crossover(p1, p2, rng):
    """均匀交叉 (算法唯一交叉算子)

    掩码 mask[i]=0 -> 子代1取父代1、子代2取父代2; =1 -> 反之。
    父代相同位两子代均保持原值; 相异位两子代各得一个互补的随机位
    (经典双后代均匀交叉, 每个相异位等概率取 0 或 1)。
    """
    mask = rng.integers(0, 2, size=p1.shape[0]).astype(bool)
    c1 = np.where(mask, p2, p1)
    c2 = np.where(mask, p1, p2)
    return c1, c2


def guided_mutate(pop, g, rng, mu=1.0):
    """mRMR 指导变异: 翻转概率随指导向量偏置

    p_flip = p0·(1-g_j) (x_j=1 时) 或 p0·g_j (x_j=0 时), 即高 g 特征
    更容易被开启、更难被关闭。按当前种群自适应:
        p0 = mu / Σ_j [ q_j·(1-g_j) + (1-q_j)·g_j ],  q_j = 种群中第 j 位
    为 1 的频率, 使得每个后代的期望翻转数恰为 mu (默认 1, 无经验常数)。
    """
    g = np.asarray(g, dtype=float)
    q = pop.mean(axis=0)
    denom = float((q * (1.0 - g) + (1.0 - q) * g).sum())
    p0 = mu / denom if denom > 0 else 0.0
    p_flip = np.where(pop, p0 * (1.0 - g), p0 * g)
    p_flip = np.clip(p_flip, 0.0, 1.0)
    return pop ^ (rng.random(pop.shape) < p_flip)


def tournament_offspring(pop, obj, w, N, pc, rng, D):
    """子问题 w 的加权和锦标赛 + 均匀交叉生成 N 个子代

    g^{ws}(x | w) = w·(1-Acc) + (1-w)·|x|/D  (理想点 z*=(0,0))
    空集排除 (MOEA/D-FS): 空集 (|s|=0) 的加权和值 = w 对小 w 子问题极小,
    会主导锦标赛导致退化; 令空集 fit = +inf, 永不成为亲本。
    """
    err = obj[:, 1]
    feat = obj[:, 0] / D
    fit = w * err + (1.0 - w) * feat
    fit[obj[:, 0] <= 0] = np.inf
    n = pop.shape[0]

    def pick():
        i, j = rng.choice(n, 2, replace=False)
        return i if fit[i] < fit[j] else j

    off = []
    while len(off) < N:
        p1 = pop[pick()].copy()
        p2 = pop[pick()].copy()
        if rng.random() < pc:
            c1, c2 = uniform_crossover(p1, p2, rng)
            off.append(c1)
            off.append(c2)
        else:
            off.append(p1)
            off.append(p2)
    return np.vstack(off[:N])


# ===========================================================================
# 三、环拓扑: 邻域精英迁移 + 邻域交配 (MOEA/D mating restriction)
# ===========================================================================

def ring_migrate(pops, objs, accs, n_elite, rng, acc_of):
    """双向环精英迁移 (接收门槛: 移民秩优于接收方最差个体)

    环上相邻子种群即 MOEA/D 中互为邻域的子问题; 迁移即邻域间
    的知识交换 (共享决策变量的耦合信息)。
    """
    M = len(pops)
    immigrants = [[] for _ in range(M)]
    elites = []
    for i in range(M):
        rank, cd = rank_and_crowding(objs[i])
        order = np.lexsort((-cd, rank))[:n_elite]
        elites.append((pops[i][order].copy(), objs[i][order].copy(),
                       accs[i][order].copy()))
    for i in range(M):
        for src in ((i - 1) % M, (i + 1) % M):
            for e, eo, ea in zip(*elites[src]):
                merged = np.vstack([objs[i], eo])
                rank_m, _ = rank_and_crowding(merged)
                rank_e = rank_m[-1]
                worst = int(np.argmax(rank_m[:-1]))
                if rank_e < rank_m[worst]:            # 适应度门槛
                    pops[i][worst] = e
                    objs[i][worst] = eo
                    accs[i][worst] = ea
                    # 保存 (掩码, 特征数, 准确率) 供邻域锦标赛使用
                    immigrants[i].append((e, int(e.sum()), float(ea)))
    return immigrants


def neighborhood_mating(pop, obj, accs, immigrants, ptrans, pc, rng, acc_of, N, D, w):
    """邻域交配 (MOEA/D mating restriction, 加权和适应度)

    对移民按本子问题的加权和标量化适应度做二元锦标赛,
    选出"对本子问题最有价值"的邻居解做均匀交叉 (移民的 acc 与
    |s| 已在来源子问题评估, 零额外分类器调用)。移民池不足 2 个时
    退化为唯一移民 (接收门槛拒绝多数精英是合法状态)。
    """
    if not immigrants:
        return pop, obj, accs
    im_masks = np.vstack([np.asarray(m[0], dtype=bool) for m in immigrants])
    im_n = np.array([m[1] for m in immigrants])
    im_acc = np.array([m[2] for m in immigrants])
    # 移民在本子问题 w 下的加权和适应度 (越小越好), 定义同锦标赛;
    # 空集同样排除 (见 tournament_offspring)
    im_fit = w * (1.0 - im_acc) + (1.0 - w) * (im_n / D)
    im_fit[im_n <= 0] = np.inf
    off = []
    for v in pop:
        if rng.random() < ptrans:
            if len(im_masks) == 1:
                base = im_masks[0]
            else:
                a, b = rng.choice(len(im_masks), 2, replace=False)
                base = im_masks[a if im_fit[a] <= im_fit[b] else b]
            c1, _ = uniform_crossover(v, base, rng)
            off.append(c1)
    if not off:
        return pop, obj, accs
    off = np.vstack(off)
    obj_o, acc_o = evaluate_pop(off, acc_of)
    obj_all = np.vstack([obj, obj_o])
    rank, cd = rank_and_crowding(obj_all)
    keep = np.lexsort((-cd, rank))[:N]
    pop2 = np.vstack([pop, off])
    return pop2[keep], obj_all[keep], np.concatenate([accs, acc_o])[keep]


# ===========================================================================
# 四、算法入口
# ===========================================================================

def jfs(xtrain, ytrain, opts):
    """RNMPGA 主流程: mRMR 分解式环多种群多目标遗传算法"""
    fold = opts['fold']
    xt, yt = fold['xt'], fold['yt']

    N = int(opts.get('N', 40))                # 每子种群规模
    M = int(opts.get('M', 7))                 # 子种群数 (环大小 = 子问题数)
    maxgen = int(opts.get('maxGen', 40))      # 最大代数
    pc = float(opts.get('pc', 0.9))           # 交叉概率
    t_mig = int(opts.get('t_mig', 5))         # 迁移间隔
    s_elite = int(opts.get('s_elite', 3))     # 每次迁移的精英数
    ptrans = float(opts.get('ptrans', 0.5))   # 邻域交配概率
    K = int(opts.get('mrmr_K', 200))          # mRMR 短名单规模
    seed = int(opts.get('seed', 42))
    rng = np.random.default_rng(seed)

    D = xtrain.shape[1]
    # 评估入口: 缓存 + 批量并行 (纯工程优化, 结果与串行逐位一致);
    # 并行度由 opts['workers'] 控制 (缺省 min(CPU 核数, 8))
    acc_of = EvalCache(opts)

    # ---------------- 1. mRMR 单滤波指导向量 ----------------
    g = mrmr_guidance(xt, yt, K=K)

    # ---------------- 2. 分解权重 (Das-Dennis 内部均匀点) ----------------
    # w_k = (k+1)/(M+1): 覆盖 (0,1) 开区间, 排除退化端点
    # (端点 w=1 -> 全特征解, w=0 -> 空集解, 均无信息量)
    weights = [(k + 1.0) / (M + 1.0) for k in range(M)]

    # ---------------- 3. 初始化 M 个子种群 ----------------
    pops, objs, accs = [], [], []
    for k in range(M):
        p = guided_init(N, D, g, rng)
        o, a = evaluate_pop(p, acc_of)
        pops.append(p)
        objs.append(o)
        accs.append(a)

    curve = np.zeros(maxgen)
    pop_curves = np.zeros((M, maxgen))        # 每子种群逐代最优精度 (实验分析用)
    immigrants = [[] for _ in range(M)]

    progress = opts.get('progress')

    # ---------------- 4. 演化主循环 ----------------
    for gen in range(maxgen):
        for k in range(M):
            # a. 子问题 k 的加权和锦标赛 + 均匀交叉
            off = tournament_offspring(pops[k], objs[k], weights[k],
                                       N, pc, rng, D)
            # b. mRMR 指导变异 (强度随代数衰减: 前期强引导后期均匀)
            decay = 1.0 - gen / maxgen
            off = guided_mutate(off, g, rng, mu=0.5 + 0.5 * decay)
            obj_o, acc_o = evaluate_pop(off, acc_of)
            # c. NSGA-II 环境选择 (子种群内保持双目标前沿)
            pops[k], objs[k], accs[k] = nsga2_select(
                pops[k], objs[k], accs[k], off, obj_o, acc_o, N)
            # d. 邻域交配 (邻域锦标赛亲本, MOEA/D mating restriction)
            if immigrants[k]:
                pops[k], objs[k], accs[k] = neighborhood_mating(
                    pops[k], objs[k], accs[k], immigrants[k],
                    ptrans, pc, rng, acc_of, N, D, weights[k])

        # e. 双向环精英迁移 (每 t_mig 代)
        if (gen + 1) % t_mig == 0:
            immigrants = ring_migrate(pops, objs, accs, s_elite, rng, acc_of)
        else:
            immigrants = [[] for _ in range(M)]

        pop_curves[:, gen] = [a.max() for a in accs]
        curve[gen] = float(pop_curves[:, gen].max())
        if progress is not None:
            progress(gen + 1)

    # ---------------- 5. 输出: 全部子问题解的非支配并集 ----------------
    pop_all = np.vstack(pops)
    obj_all = np.vstack(objs)
    acc_all = np.concatenate(accs)
    nd = fast_non_dominated_sort(obj_all)[0]
    final = pop_all[nd]
    final_obj = obj_all[nd]
    final_acc = acc_all[nd]
    # 按误差升序排列 (供主流程取最优)
    order = np.argsort(final_obj[:, 1])
    pf = [np.where(final[i])[0].astype(int) for i in order]
    best = int(order[np.argmin(final_obj[order, 1])])
    sf = np.where(final[best])[0].astype(int)

    # ---------------- 6. 每子种群统计 (供消融/敏感性分析与监控) ----------------
    sizes = np.array([len(o) for o in objs])
    offsets = np.cumsum(np.concatenate([[0], sizes]))
    pop_stats = []
    for k in range(M):
        o, a = objs[k], accs[k]
        ibest = int(np.argmin(o[:, 1]))          # 子种群内最小误差解
        iacc = int(np.argmax(a))                 # 子种群内最高精度解
        lo, hi = offsets[k], offsets[k + 1]
        n_nd = int(np.sum((nd >= lo) & (nd < hi)))
        pop_stats.append({
            'k': k,
            'w': weights[k],
            'best_acc': float(a[iacc]),
            'best_err': float(o[ibest, 1]),
            'feat_best_acc': int(o[iacc, 0]),
            'feat_best_err': int(o[ibest, 0]),
            'mean_feat': float(o[:, 0].mean()),
            'mean_err': float(o[:, 1].mean()),
            'n_empty': int(np.sum(o[:, 0] <= 0)),
            'nd_contrib': n_nd,
            'curve': [float(v) for v in pop_curves[k]],
        })

    return {'sf': sf, 'nf': int(sf.size), 'c': curve, 'pf': pf,
            'pop_stats': pop_stats}


def nsga2_select(pop, obj, accs, off, obj_o, acc_o, N):
    """父代 ∪ 子代的 NSGA-II 环境选择 (rank + crowding, 保留前 N)"""
    obj_all = np.vstack([obj, obj_o])
    rank, cd = rank_and_crowding(obj_all)
    keep = np.lexsort((-cd, rank))[:N]
    pop2 = np.vstack([pop, off])
    return pop2[keep], obj_all[keep], np.concatenate([accs, acc_o])[keep]
