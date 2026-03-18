/*
 * pressurederivativecalculator1.cpp
 * 文件作用：高级压力导数计算器实现文件
 * 功能描述：实现导数计算后的平滑处理逻辑
 */

#include "pressurederivativecalculator1.h"
#include <QtMath>
#include <QDebug>

PressureDerivativeCalculator1::PressureDerivativeCalculator1(QObject *parent)
    : QObject(parent)
{
}

PressureDerivativeResult PressureDerivativeCalculator1::calculateSmoothedDerivative(
    QStandardItemModel* model, const PressureDerivativeConfig& config, int smoothFactor)
{
    // 1. 先使用基础计算器计算标准的Bourdet导数
    // 注意：这里我们借用基础计算器的逻辑，但在写入模型前拦截数据进行平滑
    // 为了简化，我们手动执行提取数据、计算导数、平滑、写入的流程

    PressureDerivativeResult result;
    result.success = false;

    if (!model) {
        result.errorMessage = "数据模型为空";
        return result;
    }

    // 复用基础类的列检测和数据读取逻辑（此处简化为直接读取，实际项目中可提取基础类函数为public static）
    int rows = model->rowCount();
    QVector<double> timeData;
    QVector<double> pressureData;
    timeData.reserve(rows);
    pressureData.reserve(rows);

    for(int i=0; i<rows; ++i) {
        QStandardItem* tItem = model->item(i, config.timeColumnIndex);
        QStandardItem* pItem = model->item(i, config.pressureColumnIndex);
        if(tItem && pItem) {
            bool okT, okP;
            double t = tItem->text().toDouble(&okT);
            double p = pItem->text().toDouble(&okP);
            if(okT && okP) {
                timeData.append(t);
                pressureData.append(p);
            }
        }
    }

    if(timeData.isEmpty()) {
        result.errorMessage = "未能读取有效数据";
        return result;
    }

    // 处理时间偏移
    double offset = config.autoTimeOffset ? (timeData.first() <= 0 ? 0.0001 : 0.0) : config.timeOffset;
    QVector<double> adjustedTime;
    for(double t : timeData) adjustedTime.append(t + offset);

    // 计算压降
    double pInitial = pressureData.first();
    QVector<double> dp;
    for(double p : pressureData) dp.append(pInitial - p);

    // 计算Bourdet导数
    QVector<double> derivative = PressureDerivativeCalculator::calculateBourdetDerivative(adjustedTime, dp, config.lSpacing);

    // 2. 执行平滑处理
    QVector<double> smoothedDeriv = smoothData(derivative, smoothFactor);

    // 3. 写入数据模型
    int newCol = model->columnCount();
    model->insertColumn(newCol);
    QString header = QString("平滑导数(L=%1, S=%2)").arg(config.lSpacing).arg(smoothFactor);
    model->setHorizontalHeaderItem(newCol, new QStandardItem(header));

    for(int i=0; i<smoothedDeriv.size() && i<rows; ++i) {
        model->setItem(i, newCol, new QStandardItem(QString::number(smoothedDeriv[i], 'g', 6)));
    }

    result.success = true;
    result.addedColumnIndex = newCol;
    result.columnName = header;
    result.processedRows = smoothedDeriv.size();

    return result;
}

QVector<double> PressureDerivativeCalculator1::smoothData(const QVector<double>& data, int span)
{
    int n = data.size();
    if (n == 0) return QVector<double>();
    if (span <= 1) return data;

    // 确保span是奇数
    if (span % 2 == 0) span++;

    QVector<double> result(n);
    int halfSpan = (span - 1) / 2;

    for (int i = 0; i < n; ++i) {
        double sum = 0;
        int count = 0;

        // 简单的移动平均，边缘处窗口自动缩小（类似Matlab默认行为）
        int start = qMax(0, i - halfSpan);
        int end = qMin(n - 1, i + halfSpan);

        for (int j = start; j <= end; ++j) {
            sum += data[j];
            count++;
        }

        if (count > 0)
            result[i] = sum / count;
        else
            result[i] = data[i];
    }
    return result;
}
