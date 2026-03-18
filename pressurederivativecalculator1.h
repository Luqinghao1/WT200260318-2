/*
 * pressurederivativecalculator1.h
 * 文件作用：高级压力导数计算器头文件
 * 功能描述：
 * 1. 继承或复用原有导数计算逻辑
 * 2. 新增平滑处理功能（类似Matlab smooth函数）
 * 3. 提供静态计算接口
 */

#ifndef PRESSUREDERIVATIVECALCULATOR1_H
#define PRESSUREDERIVATIVECALCULATOR1_H

#include <QObject>
#include <QVector>
#include "pressurederivativecalculator.h" // 引用原有计算器结构体定义

class PressureDerivativeCalculator1 : public QObject
{
    Q_OBJECT
public:
    explicit PressureDerivativeCalculator1(QObject *parent = nullptr);

    /**
     * @brief 计算平滑后的压力导数
     * @param model 数据模型
     * @param config 基础配置
     * @param smoothFactor 平滑因子（窗口大小，奇数）
     * @return 计算结果
     */
    PressureDerivativeResult calculateSmoothedDerivative(QStandardItemModel* model,
                                                         const PressureDerivativeConfig& config,
                                                         int smoothFactor);

    /**
     * @brief 移动平均平滑算法 (类似Matlab smooth)
     * @param data 原始数据
     * @param span 平滑窗口大小 (必须为正奇数，偶数会自动+1)
     * @return 平滑后的数据
     */
    static QVector<double> smoothData(const QVector<double>& data, int span);

signals:
    void progressUpdated(int progress, const QString& message);
    void calculationCompleted(const PressureDerivativeResult& result);

private:
    PressureDerivativeCalculator m_basicCalculator;
};

#endif // PRESSUREDERIVATIVECALCULATOR1_H
