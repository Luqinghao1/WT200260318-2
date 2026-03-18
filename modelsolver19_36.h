/*
 * 文件名: modelsolver19_36.h
 * 文件作用与功能描述:
 * 1. 本代码文件为压裂水平井页岩型复合模型 Group 2 (Model 37-72) 的核心计算类头文件。
 * 2. 负责计算软件定义的另外 36 个以页岩型介质为主的试井模型理论图版数据。
 * 3. 涵盖三种储层组合的数学求解:
 * - 页岩型 + 页岩型   (Model 37-48 / 内部编号 1-12)
 * - 页岩型 + 均质     (Model 49-60 / 内部编号 13-24)
 * - 页岩型 + 双重孔隙 (Model 61-72 / 内部编号 25-36)
 * 4. 实现了用于描述页岩型介质的瞬态平板模型 (Transient Slab Matrix) 函数的 Laplace 形式。
 * 5. 基于与 MATLAB 一致的 nf 节点积分方程组算法，支持严格解析的 Laplace 域变井储耦合。
 */

#ifndef MODELSOLVER19_36_H
#define MODELSOLVER19_36_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver19_36
{
public:
    enum ModelType {
        Model_1 = 0, Model_2, Model_3, Model_4,
        Model_5, Model_6, Model_7, Model_8,
        Model_9, Model_10, Model_11, Model_12,
        Model_13, Model_14, Model_15, Model_16,
        Model_17, Model_18, Model_19, Model_20,
        Model_21, Model_22, Model_23, Model_24,
        Model_25, Model_26, Model_27, Model_28,
        Model_29, Model_30, Model_31, Model_32,
        Model_33, Model_34, Model_35, Model_36
    };

    /**
     * @brief 构造函数，初始化特定类型的求解器
     */
    explicit ModelSolver19_36(ModelType type);

    virtual ~ModelSolver19_36();

    /**
     * @brief 设置高精度计算模式
     */
    void setHighPrecision(bool high);

    /**
     * @brief 计算理论曲线的核心接口
     * @param params 模型参数集合 (物理约束同MATLAB要求)
     */
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());

    static QString getModelName(ModelType type, bool verbose = true);
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

private:
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    double flaplace_composite(double z, const QMap<QString, double>& p);

    /**
     * @brief 核心边界元方法求解函数，消除了微元切分，构建单条裂缝节点矩阵
     */
    double PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                         int n_fracs, ModelType type);

    // --- 介质函数 ---
    double calc_fs_dual(double u, double omega, double lambda);
    double calc_fs_shale(double u, double omega, double lambda);

    // --- 数学辅助 ---
    static double safe_bessel_i_scaled(int v, double x);
    static double safe_bessel_k_scaled(int v, double x);
    static double safe_bessel_k(int v, double x);

    double gauss15(std::function<double(double)> f, double a, double b);
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);

    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;
    bool m_highPrecision;
    QVector<double> m_stehfestCoeffs;
    int m_currentN;
};

#endif // MODELSOLVER19_36_H
