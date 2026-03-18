/*
 * 文件名: pressurederivativecalculator.h
 * 文件作用: 压力导数计算器头文件
 * 功能描述:
 * 1. 定义了计算结果结构体 PressureDerivativeResult，兼容旧代码接口。
 * 2. 定义了计算配置结构体 PressureDerivativeConfig，包含试井类型和初始压力参数。
 * 3. 声明了计算核心类，支持自动计算压差和Bourdet导数。
 */

#ifndef PRESSUREDERIVATIVECALCULATOR_H
#define PRESSUREDERIVATIVECALCULATOR_H

#include <QObject>
#include <QString>
#include <QVector>
#include <QStandardItemModel>

// 压力导数计算结果结构
struct PressureDerivativeResult {
    bool success;
    QString errorMessage;

    // --- 新增字段：更详细的列索引信息 ---
    int deltaPColumnIndex;          // 压差列索引
    QString deltaPColumnName;       // 压差列名称
    int derivativeColumnIndex;      // 导数列索引
    QString derivativeColumnName;   // 导数列名称

    // --- 兼容字段：为了防止 pressurederivativecalculator1.cpp 报错而保留 ---
    int addedColumnIndex;           // 兼容旧代码，通常指向导数列
    QString columnName;             // 兼容旧代码，通常指向导数列名称

    int processedRows;

    PressureDerivativeResult() :
        success(false),
        deltaPColumnIndex(-1),
        derivativeColumnIndex(-1),
        addedColumnIndex(-1),
        processedRows(0) {}
};

// 压力导数计算配置
struct PressureDerivativeConfig {
    // 试井类型枚举
    enum TestType {
        Drawdown,   // 压力降落试井 (需要 Pi)
        Buildup     // 压力恢复试井 (基于 Pwf_shut_in)
    };

    int timeColumnIndex;      // 时间列索引
    int pressureColumnIndex;  // 压力列索引

    TestType testType;        // 试井类型
    double initialPressure;   // 地层初始压力 (仅降落试井使用)

    QString timeUnit;         // 时间单位 ("s", "min", "h")
    QString pressureUnit;     // 压力单位
    double lSpacing;          // L-Spacing平滑参数（对数周期，通常0.1-0.5）
    double timeOffset;        // 时间偏移量（用于处理t=0的情况）
    bool autoTimeOffset;      // 是否自动添加时间偏移

    PressureDerivativeConfig() :
        timeColumnIndex(-1),
        pressureColumnIndex(-1),
        testType(Drawdown),     // 默认降落试井
        initialPressure(0.0),
        timeUnit("h"),
        pressureUnit("MPa"),
        lSpacing(0.15),
        timeOffset(0.0001),
        autoTimeOffset(true) {}
};

/**
 * @brief 压力导数计算器类
 *
 * 功能：
 * 1. 根据试井类型（降落/恢复）将原始压力转换为压差 (Delta P)。
 * 2. 使用Bourdet算法计算压力导数。
 * 3. 将压差列和导数列写入数据模型。
 */
class PressureDerivativeCalculator : public QObject
{
    Q_OBJECT

public:
    explicit PressureDerivativeCalculator(QObject *parent = nullptr);
    ~PressureDerivativeCalculator();

    /**
     * @brief 计算压力导数（针对表格模型的封装）
     * @param model 数据模型
     * @param config 计算配置
     * @return 计算结果
     */
    PressureDerivativeResult calculatePressureDerivative(QStandardItemModel* model,
                                                         const PressureDerivativeConfig& config);

    /**
     * @brief 自动检测压力列和时间列
     * @param model 数据模型
     * @return 配置对象，包含检测到的列索引
     */
    PressureDerivativeConfig autoDetectColumns(QStandardItemModel* model);

    // =========================================================================
    // 静态核心算法接口 (Saphir 风格 Bourdet 导数)
    // =========================================================================

    /**
     * @brief 使用Bourdet导数算法计算导数 (统一入口)
     * @param timeData 时间数据 (t)
     * @param pressureDropData 压降数据 (Delta P)
     * @param lSpacing L-Spacing参数
     * @return 导数数据向量
     */
    static QVector<double> calculateBourdetDerivative(const QVector<double>& timeData,
                                                      const QVector<double>& pressureDropData,
                                                      double lSpacing);

signals:
    void progressUpdated(int progress, const QString& message);
    void calculationCompleted(const PressureDerivativeResult& result);

private:
    // 内部静态辅助函数
    static int findLeftPoint(const QVector<double>& timeData, int currentIndex, double lSpacing);
    static int findRightPoint(const QVector<double>& timeData, int currentIndex, double lSpacing);
    static double calculateDerivativeValue(double t1, double t2, double p1, double p2);

    int findPressureColumn(QStandardItemModel* model);
    int findTimeColumn(QStandardItemModel* model);
    double parseNumericValue(const QString& str);
    QString formatValue(double value, int precision = 6);
};

#endif // PRESSUREDERIVATIVECALCULATOR_H
