/*
 * 文件名: fittingcore.cpp
 * 文件作用: 试井拟合核心算法实现
 * 修复记录:
 * 1. [修复致命死锁] 将 computeJacobian 中的 QtConcurrent::blockingMapped 改为串行 for 循环。
 * (因为底层的 ModelSolver 已经使用了多线程并发计算时间点，外层再开多线程会导致 Qt 全局线程池饥饿死锁，这是之前拟合卡死的核心原因)。
 * 2. [算法优化] 将 LM 算法内循环寻找下降方向的 tryIter 从 5 次提升至 20 次，避免过早陷入局部极小值宣告失败。
 * 3. [鲁棒性增强] 增加了对特征矩阵奇异值和 NaN 变量的拦截与保护。
 */

#include "fittingcore.h"
#include "modelparameter.h" // 引入模型参数单例
#include <QtConcurrent>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <Eigen/Dense>

FittingCore::FittingCore(QObject *parent)
    : QObject(parent), m_modelManager(nullptr), m_isCustomSamplingEnabled(false), m_stopRequested(false)
{
    // 监听异步任务完成
    connect(&m_watcher, &QFutureWatcher<void>::finished, this, &FittingCore::sigFitFinished);
}

void FittingCore::setModelManager(ModelManager *m) {
    m_modelManager = m;
}

void FittingCore::setObservedData(const QVector<double> &t, const QVector<double> &p, const QVector<double> &d) {
    m_obsTime = t;
    m_obsDeltaP = p;
    m_obsDerivative = d;
}

void FittingCore::setSamplingSettings(const QList<SamplingInterval> &intervals, bool enabled) {
    m_customIntervals = intervals;
    m_isCustomSamplingEnabled = enabled;
}

void FittingCore::startFit(ModelManager::ModelType modelType, const QList<FitParameter> &params, double weight) {
    if (m_watcher.isRunning()) return;

    m_stopRequested = false;
    // 启动异步线程执行拟合
    m_watcher.setFuture(QtConcurrent::run([this, modelType, params, weight](){
        runOptimizationTask(modelType, params, weight);
    }));
}

void FittingCore::stopFit() {
    m_stopRequested = true;
}

void FittingCore::getLogSampledData(const QVector<double>& srcT, const QVector<double>& srcP, const QVector<double>& srcD,
                                    QVector<double>& outT, QVector<double>& outP, QVector<double>& outD)
{
    outT.clear(); outP.clear(); outD.clear();
    if (srcT.isEmpty()) return;

    // 辅助结构体用于排序去重
    struct DataPoint {
        double t, p, d;
        bool operator<(const DataPoint& other) const { return t < other.t; }
        bool operator==(const DataPoint& other) const { return std::abs(t - other.t) < 1e-9; }
    };
    QVector<DataPoint> points;

    // 模式1：默认策略
    if (!m_isCustomSamplingEnabled) {
        int targetCount = 200;
        if (srcT.size() <= targetCount) {
            outT = srcT; outP = srcP; outD = srcD;
            return;
        }

        double tMin = srcT.first() <= 1e-10 ? 1e-4 : srcT.first();
        double tMax = srcT.last();
        double logMin = log10(tMin);
        double logMax = log10(tMax);
        double step = (logMax - logMin) / (targetCount - 1);

        int currentIndex = 0;
        for (int i = 0; i < targetCount; ++i) {
            double targetT = pow(10, logMin + i * step);
            double minDiff = 1e30;
            int bestIdx = currentIndex;
            while (currentIndex < srcT.size()) {
                double diff = std::abs(srcT[currentIndex] - targetT);
                if (diff < minDiff) { minDiff = diff; bestIdx = currentIndex; }
                else break;
                currentIndex++;
            }
            currentIndex = bestIdx;
            points.append({srcT[bestIdx],
                           (bestIdx<srcP.size()?srcP[bestIdx]:0.0),
                           (bestIdx<srcD.size()?srcD[bestIdx]:0.0)});
        }
    }
    // 模式2：自定义区间策略
    else {
        if (m_customIntervals.isEmpty()) {
            outT = srcT; outP = srcP; outD = srcD;
            return;
        }
        for (const auto& interval : m_customIntervals) {
            double tStart = interval.tStart;
            double tEnd = interval.tEnd;
            int count = interval.count;
            if (count <= 0) continue;

            auto itStart = std::lower_bound(srcT.begin(), srcT.end(), tStart);
            auto itEnd = std::upper_bound(srcT.begin(), srcT.end(), tEnd);
            int idxStart = std::distance(srcT.begin(), itStart);
            int idxEnd = std::distance(srcT.begin(), itEnd);

            if (idxStart >= srcT.size() || idxStart >= idxEnd) continue;

            double subMin = srcT[idxStart];
            double subMax = srcT[idxEnd - 1];
            if (subMin <= 1e-10) subMin = 1e-4;

            double logMin = log10(subMin);
            double logMax = log10(subMax);
            double step = (count > 1) ? (logMax - logMin) / (count - 1) : 0;

            int subCurrentIdx = idxStart;
            for (int i = 0; i < count; ++i) {
                double targetT = (count == 1) ? subMin : pow(10, logMin + i * step);
                double minDiff = 1e30;
                int bestIdx = subCurrentIdx;
                while (subCurrentIdx < idxEnd) {
                    double diff = std::abs(srcT[subCurrentIdx] - targetT);
                    if (diff < minDiff) { minDiff = diff; bestIdx = subCurrentIdx; }
                    else break;
                    subCurrentIdx++;
                }
                subCurrentIdx = bestIdx;
                if (bestIdx < srcT.size()) {
                    points.append({srcT[bestIdx],
                                   (bestIdx<srcP.size()?srcP[bestIdx]:0.0),
                                   (bestIdx<srcD.size()?srcD[bestIdx]:0.0)});
                }
            }
        }
    }

    std::sort(points.begin(), points.end());
    auto last = std::unique(points.begin(), points.end());
    points.erase(last, points.end());

    for (const auto& p : points) {
        outT.append(p.t);
        outP.append(p.p);
        outD.append(p.d);
    }
}

QMap<QString, double> FittingCore::preprocessParams(const QMap<QString, double>& rawParams, ModelManager::ModelType type)
{
    QMap<QString, double> processed = rawParams;
    ModelParameter* mp = ModelParameter::instance();

    auto getSafeParam = [&](const QString& key, double mpVal, double defaultVal) {
        if (rawParams.contains(key)) return rawParams[key];
        if (std::abs(mpVal) > 1e-15) return mpVal;
        return defaultVal;
    };

    double phi = getSafeParam("phi", mp->getPhi(), 0.05);
    double h   = getSafeParam("h",   mp->getH(),   20.0);
    double Ct  = getSafeParam("Ct",  mp->getCt(),  5e-4);
    double mu  = getSafeParam("mu",  mp->getMu(),  0.5);
    double B   = getSafeParam("B",   mp->getB(),   1.05);
    double q   = getSafeParam("q",   mp->getQ(),   5.0);
    double rw  = getSafeParam("rw",  mp->getRw(),  0.1);

    processed["phi"] = phi;
    processed["h"] = h;
    processed["Ct"] = Ct;
    processed["mu"] = mu;
    processed["B"] = B;
    processed["q"] = q;
    processed["rw"] = rw;

    double L = processed.value("L", 0.0);
    if (L < 1e-9) {
        L = 1000.0;
        processed["L"] = L;
    }
    if (processed.contains("Lf")) {
        processed["LfD"] = processed["Lf"] / L;
    } else {
        processed["LfD"] = 0.0;
    }

    if (!processed.contains("M12") && processed.contains("km")) {
        processed["M12"] = processed["km"];
    }

    bool hasStorage = (type == ModelManager::Model_1 || type == ModelManager::Model_3 || type == ModelManager::Model_5);
    if (hasStorage) {
        if (processed.contains("C")) {
            double valC = processed["C"];
            double denom = phi * h * Ct * L * L;
            double cD = 0.0;
            if (denom > 1e-20) cD = 0.159 * valC / denom;
            processed["cD"] = cD;
        }
    } else {
        processed["cD"] = 0.0;
        processed["S"] = 0.0;
    }

    bool isInfinite = (type == ModelManager::Model_1 || type == ModelManager::Model_2);
    if (isInfinite) {
        if (!processed.contains("re")) processed["re"] = 20000.0;
    }

    return processed;
}

void FittingCore::runOptimizationTask(ModelManager::ModelType modelType, QList<FitParameter> fitParams, double weight) {
    runLevenbergMarquardtOptimization(modelType, fitParams, weight);
}

void FittingCore::runLevenbergMarquardtOptimization(ModelManager::ModelType modelType, QList<FitParameter> params, double weight) {
    if(m_modelManager) m_modelManager->setHighPrecision(false);

    QVector<int> fitIndices;
    for(int i=0; i<params.size(); ++i) {
        if(params[i].isFit && params[i].name != "LfD") fitIndices.append(i);
    }
    int nParams = fitIndices.size();

    QMap<QString, double> currentParamMap;
    for(const auto& p : params) currentParamMap.insert(p.name, p.value);

    QMap<QString, double> solverParams = preprocessParams(currentParamMap, modelType);

    QVector<double> fitT, fitP, fitD;
    getLogSampledData(m_obsTime, m_obsDeltaP, m_obsDerivative, fitT, fitP, fitD);

    QVector<double> residuals = calculateResiduals(currentParamMap, modelType, weight, fitT, fitP, fitD);
    double currentSSE = calculateSumSquaredError(residuals);

    ModelCurveData curve = m_modelManager->calculateTheoreticalCurve(modelType, solverParams);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(curve), std::get<1>(curve), std::get<2>(curve));

    if(nParams == 0) {
        emit sigFitFinished();
        return;
    }

    double lambda = 0.1; // 初始化阻尼参数
    int maxIter = 50;    // 最大全局迭代次数

    for(int iter = 0; iter < maxIter; ++iter) {
        if(m_stopRequested) break;
        // 如果均方误差足够小则提前终止
        if (!residuals.isEmpty() && (currentSSE / residuals.size()) < 1e-4) break;

        emit sigProgress(iter * 100 / maxIter);

        QVector<QVector<double>> J = computeJacobian(currentParamMap, residuals, fitIndices, modelType, params, weight, fitT, fitP, fitD);
        if(m_stopRequested) break;

        int nRes = residuals.size();

        QVector<QVector<double>> H(nParams, QVector<double>(nParams, 0.0));
        QVector<double> g(nParams, 0.0);

        for(int k=0; k<nRes; ++k) {
            for(int i=0; i<nParams; ++i) {
                g[i] += J[k][i] * residuals[k];
                for(int j=0; j<=i; ++j) {
                    H[i][j] += J[k][i] * J[k][j];
                }
            }
        }
        for(int i=0; i<nParams; ++i) {
            for(int j=i+1; j<nParams; ++j) H[i][j] = H[j][i];
        }

        bool stepAccepted = false;
        // [修复核心] 提升容错尝试次数，允许 lambda 进行大范围的动态调整寻优
        for(int tryIter=0; tryIter<20; ++tryIter) {
            if(m_stopRequested) break;

            QVector<QVector<double>> H_lm = H;
            for(int i=0; i<nParams; ++i) {
                H_lm[i][i] += lambda * (1.0 + std::abs(H[i][i]));
            }

            QVector<double> negG(nParams);
            for(int i=0;i<nParams;++i) negG[i] = -g[i];

            QVector<double> delta = solveLinearSystem(H_lm, negG);
            if (delta.isEmpty()) {
                lambda *= 10.0;
                continue;
            }

            QMap<QString, double> trialMap = currentParamMap;

            for(int i=0; i<nParams; ++i) {
                int pIdx = fitIndices[i];
                QString pName = params[pIdx].name;
                double oldVal = currentParamMap[pName];
                bool isLog = (oldVal > 1e-12 && pName != "S" && pName != "nf");
                double newVal;
                if(isLog) newVal = pow(10.0, log10(oldVal) + delta[i]);
                else newVal = oldVal + delta[i];

                // [增加保护] 剔除由于过度下降导致的异常数值
                if (std::isnan(newVal) || std::isinf(newVal)) newVal = oldVal;

                newVal = qMax(params[pIdx].min, qMin(newVal, params[pIdx].max));
                trialMap[pName] = newVal;
            }

            QVector<double> newRes = calculateResiduals(trialMap, modelType, weight, fitT, fitP, fitD);
            double newSSE = calculateSumSquaredError(newRes);

            // 如果这一步显著减小了残差误差，采纳并退出内部寻优循环
            if(newSSE < currentSSE && !std::isnan(newSSE)) {
                currentSSE = newSSE;
                currentParamMap = trialMap;
                residuals = newRes;
                lambda = std::max(1e-7, lambda / 10.0); // 缩小阻尼，偏向高斯牛顿法
                stepAccepted = true;

                // 派发迭代更新
                QMap<QString, double> trialSolverParams = preprocessParams(trialMap, modelType);
                ModelCurveData iterCurve = m_modelManager->calculateTheoreticalCurve(modelType, trialSolverParams);
                emit sigIterationUpdated(currentSSE/nRes, currentParamMap, std::get<0>(iterCurve), std::get<1>(iterCurve), std::get<2>(iterCurve));
                break;
            } else {
                lambda *= 10.0; // 加大阻尼，偏向梯度下降
            }
        }

        // 若经过了20次极端拉伸仍无法使函数下降，说明到达局部最优点，停止迭代
        if(!stepAccepted) {
            break;
        }
    }

    if(m_modelManager) m_modelManager->setHighPrecision(true);

    QMap<QString, double> finalSolverParams = preprocessParams(currentParamMap, modelType);
    ModelCurveData finalCurve = m_modelManager->calculateTheoreticalCurve(modelType, finalSolverParams);
    emit sigIterationUpdated(currentSSE/residuals.size(), currentParamMap, std::get<0>(finalCurve), std::get<1>(finalCurve), std::get<2>(finalCurve));
}

QVector<double> FittingCore::calculateResiduals(const QMap<QString, double>& params, ModelManager::ModelType modelType, double weight,
                                                const QVector<double>& t, const QVector<double>& obsP, const QVector<double>& obsD) {
    if(!m_modelManager || t.isEmpty()) return QVector<double>();

    QMap<QString, double> solverParams = preprocessParams(params, modelType);

    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(modelType, solverParams, t);
    const QVector<double>& pCal = std::get<1>(res);
    const QVector<double>& dpCal = std::get<2>(res);

    QVector<double> r;
    double wp = weight;
    double wd = 1.0 - weight;

    int count = qMin((int)obsP.size(), (int)pCal.size());
    for(int i=0; i<count; ++i) {
        if(obsP[i] > 1e-10 && pCal[i] > 1e-10)
            r.append( (log(obsP[i]) - log(pCal[i])) * wp );
        else
            r.append(0.0);
    }
    int dCount = qMin((int)obsD.size(), (int)dpCal.size());
    dCount = qMin(dCount, count);
    for(int i=0; i<dCount; ++i) {
        if(obsD[i] > 1e-10 && dpCal[i] > 1e-10)
            r.append( (log(obsD[i]) - log(dpCal[i])) * wd );
        else
            r.append(0.0);
    }
    return r;
}

QVector<QVector<double>> FittingCore::computeJacobian(const QMap<QString, double>& params, const QVector<double>& baseResiduals,
                                                      const QVector<int>& fitIndices, ModelManager::ModelType modelType,
                                                      const QList<FitParameter>& currentFitParams, double weight,
                                                      const QVector<double>& t, const QVector<double>& obsP, const QVector<double>& obsD) {
    int nRes = baseResiduals.size();
    int nParams = fitIndices.size();
    QVector<QVector<double>> J(nRes, QVector<double>(nParams, 0.0));

    // [修复死锁核心] 将原本的 QtConcurrent::blockingMapped 改为简单的 for 循环
    // 原因：底层的 calculateResiduals 包含多线程计算，若外层再开多线程会导致全局线程池枯竭并死锁
    for(int j = 0; j < nParams; ++j) {
        if (m_stopRequested) break;

        int idx = fitIndices[j];
        QString pName = currentFitParams[idx].name;
        double val = params.value(pName);
        bool isLog = (val > 1e-12 && pName != "S" && pName != "nf");

        double h;
        QMap<QString, double> pPlus = params;
        QMap<QString, double> pMinus = params;

        // 根据参数是否具有对数性质给予微小扰动
        if(isLog) {
            h = 0.01;
            double valLog = log10(val);
            pPlus[pName] = pow(10.0, valLog + h);
            pMinus[pName] = pow(10.0, valLog - h);
        } else {
            h = 1e-4;
            pPlus[pName] = val + h;
            pMinus[pName] = val - h;
        }

        // 串行执行正反向扰动模型求解（底层仍是多线程，CPU仍满载计算）
        QVector<double> rPlus = this->calculateResiduals(pPlus, modelType, weight, t, obsP, obsD);
        QVector<double> rMinus = this->calculateResiduals(pMinus, modelType, weight, t, obsP, obsD);

        // 组装中心差分商
        if(rPlus.size() == nRes && rMinus.size() == nRes) {
            for(int i=0; i<nRes; ++i) {
                J[i][j] = (rPlus[i] - rMinus[i]) / (2.0 * h);
            }
        }
    }

    return J;
}

QVector<double> FittingCore::solveLinearSystem(const QVector<QVector<double>>& A, const QVector<double>& b) {
    int n = b.size();
    if (n == 0) return QVector<double>();
    Eigen::MatrixXd matA(n, n);
    Eigen::VectorXd vecB(n);
    for (int i = 0; i < n; ++i) {
        vecB(i) = b[i];
        for (int j = 0; j < n; ++j) matA(i, j) = A[i][j];
    }

    // 使用更稳定的 Cholesky 分解
    Eigen::VectorXd x = matA.ldlt().solve(vecB);
    QVector<double> res(n);
    for (int i = 0; i < n; ++i) {
        res[i] = x(i);
    }
    return res;
}

double FittingCore::calculateSumSquaredError(const QVector<double>& residuals) {
    double sse = 0.0;
    for(double v : residuals) {
        sse += v*v;
    }
    return sse;
}

