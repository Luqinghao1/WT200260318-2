/*
 * 文件名: modelsolver01-06.h
 * 文件作用与功能描述:
 * 1. 本代码文件为压裂水平井复合模型 Group 1 (Model 1-36) 的核心计算类头文件。
 * 2. 负责定义软件中前 36 个试井模型理论图版数据的计算结构和接口。
 * 3. 涵盖三种储层组合的数学求解体系:
 * - 夹层型 + 夹层型 (Model 1-12)
 * - 夹层型 + 均质   (Model 13-24)
 * - 径向复合 (均质+均质) (Model 25-36)
 * 4. 针对每种储层组合，定义了 3 种外边界(无限大、封闭、定压)和 4 种井筒储集类型(定井储、线源解、Fair、Hegeman)。
 * 5. 声明了基于 Stehfest 数值反演算法和离散裂缝网络(DFN)的面源积分方程核心求解方法。
 * 6. 包含了将变井储效应在 Laplace 域内进行强耦合运算的函数接口。
 */

#ifndef MODELSOLVER01_06_H
#define MODELSOLVER01_06_H

#include <QMap>
#include <QVector>
#include <QString>
#include <tuple>
#include <functional>
#include <QtConcurrent>

// 类型定义: <时间序列(t), 压力序列(Dp), 导数序列(Dp')>
using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelSolver01_06
{
public:
    // 模型类型枚举 (对应 modelsolver1.csv 中的 36 个模型)
    // 命名规则: Model_序号 (1-36)，保持数量和对应关系不变
    enum ModelType {
        // --- 第一组: 夹层型 + 夹层型 (1-12) ---
        // 无限大边界 (4种井储)
        Model_1 = 0, Model_2, Model_3, Model_4,
        // 封闭边界 (4种井储)
        Model_5, Model_6, Model_7, Model_8,
        // 定压边界 (4种井储)
        Model_9, Model_10, Model_11, Model_12,

        // --- 第二组: 夹层型 + 均质 (13-24) ---
        // 无限大边界
        Model_13, Model_14, Model_15, Model_16,
        // 封闭边界
        Model_17, Model_18, Model_19, Model_20,
        // 定压边界
        Model_21, Model_22, Model_23, Model_24,

        // --- 第三组: 径向复合(均质+均质) (25-36) ---
        // 无限大边界
        Model_25, Model_26, Model_27, Model_28,
        // 封闭边界
        Model_29, Model_30, Model_31, Model_32,
        // 定压边界
        Model_33, Model_34, Model_35, Model_36
    };

    /**
     * @brief 构造函数，初始化特定类型的求解器
     * @param type 模型类型 (Model_1 到 Model_36)
     */
    explicit ModelSolver01_06(ModelType type);

    /**
     * @brief 析构函数，释放相关资源
     */
    virtual ~ModelSolver01_06();

    /**
     * @brief 设置高精度计算模式
     * @param high true为高精度(Stehfest N=12)，false为低精度(N=8)
     */
    void setHighPrecision(bool high);

    /**
     * @brief 计算理论曲线的核心接口
     * @param params 模型参数集合 (包含 k, S, C, alpha, C_phi 等物理参数)
     * @param providedTime 指定的时间序列 (若为空则由内部自动生成对数时间分布)
     * @return 返回包含时间、压力、导数的元组
     */
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());

    /**
     * @brief 获取模型的中文名称和描述，用于界面显示
     * @param type 模型类型
     * @param verbose 是否包含详细描述(井储、边界、介质)，默认包含
     * @return 格式化好的中文字符串
     */
    static QString getModelName(ModelType type, bool verbose = true);

    /**
     * @brief 生成对数分布的时间步，满足早期至晚期的全阶段时间跨度
     * @param count 点数
     * @param startExp 起始时间数量级(指数)
     * @param endExp 终止时间数量级(指数)
     * @return 生成好的对数时间序列
     */
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

private:
    /**
     * @brief 驱动 Stehfest 反演算法，计算无因次压力和导数
     * @param tD 无因次时间序列
     * @param params 物理与几何参数
     * @param laplaceFunc 对应的 Laplace 空间求解函数句柄
     * @param outPD 传出参数：无因次压力
     * @param outDeriv 传出参数：无因次导数
     */
    void calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                             std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                             QVector<double>& outPD, QVector<double>& outDeriv);

    /**
     * @brief Laplace空间下的复合模型总控制与井储耦合计算函数
     * @param z Laplace 变量
     * @param p 物理与几何参数
     * @return 耦合了井筒效应后的总无因次井底压力
     */
    double flaplace_composite(double z, const QMap<QString, double>& p);

    /**
     * @brief 离散裂缝网络解析解核心方程 (完全同步MATLAB矩阵机制)
     * @param z Laplace 变量
     * @param fs1 内区介质函数
     * @param fs2 外区介质函数
     * @param M12 内外区流度比
     * @param LfD 无因次裂缝半长
     * @param rmD 无因次改造区(内区)半径
     * @param reD 无因次外区边界半径
     * @param n_fracs 裂缝总条数
     * @param type 所属的模型类型(用于判别边界条件等)
     * @return 纯地层线源基础解
     */
    double PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                         int n_fracs, ModelType type);

    // --- 数学辅助函数 ---

    /**
     * @brief 标度修正的 Bessel I 函数 (消去大实参时的指数爆炸)
     */
    static double safe_bessel_i_scaled(int v, double x);

    /**
     * @brief 标度修正的 Bessel K 函数 (配合指数修正以杜绝实参 > 700 时的溢出崩溃)
     */
    static double safe_bessel_k_scaled(int v, double x);

    /**
     * @brief 基础不带标度的Bessel K函数（托底保护）
     */
    static double safe_bessel_k(int v, double x);

    /**
     * @brief Gauss-Kronrod 15点求积公式，用于边界元的积分计算
     */
    double gauss15(std::function<double(double)> f, double a, double b);

    /**
     * @brief 自适应 Gauss 积分，提升奇异点和微小区间计算精度
     */
    double adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth);

    // --- Stehfest 反演算法辅助 ---
    double getStehfestCoeff(int i, int N);
    void precomputeStehfestCoeffs(int N);
    double factorial(int n);

private:
    ModelType m_type;                // 当前对象绑定的理论模型类型
    bool m_highPrecision;            // 是否采用高精度反演参数
    QVector<double> m_stehfestCoeffs;// 缓存的 Stehfest 反演常数项
    int m_currentN;                  // 当前 Stehfest 阶数
};

#endif // MODELSOLVER01_06_H
