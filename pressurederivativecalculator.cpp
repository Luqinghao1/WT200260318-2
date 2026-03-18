/*
 * 文件名: pressurederivativecalculator.cpp
 * 文件作用: 压力导数计算器实现
 * 功能描述:
 * 1. 实现了基于试井类型的压差计算逻辑 (降落: Pi-P, 恢复: P-Pwf)。
 * 2. 实现了 Bourdet 压力导数算法，并对边界端点进行了前向/后向抛物线拟合优化，彻底消除了端点“翘头/翘尾”现象。
 * 3. 自动匹配表格数据列，将计算生成的压差和导数写回数据模型。
 */

#include "pressurederivativecalculator.h"
#include <QStandardItem>
#include <QRegularExpression>
#include <QDebug>
#include <cmath>

PressureDerivativeCalculator::PressureDerivativeCalculator(QObject *parent)
    : QObject(parent)
{
}

PressureDerivativeCalculator::~PressureDerivativeCalculator()
{
}

PressureDerivativeResult PressureDerivativeCalculator::calculatePressureDerivative(
    QStandardItemModel* model, const PressureDerivativeConfig& config)
{
    PressureDerivativeResult result;
    result.success = false;
    result.deltaPColumnIndex = -1;
    result.derivativeColumnIndex = -1;
    result.addedColumnIndex = -1;
    result.processedRows = 0;

    if (!model) {
        result.errorMessage = "数据模型不存在";
        return result;
    }

    int rowCount = model->rowCount();
    if (rowCount < 3) {
        result.errorMessage = "数据行数不足（至少需要3行）";
        return result;
    }

    if (config.pressureColumnIndex < 0 || config.pressureColumnIndex >= model->columnCount()) {
        result.errorMessage = "压力列索引无效";
        return result;
    }

    if (config.timeColumnIndex < 0 || config.timeColumnIndex >= model->columnCount()) {
        result.errorMessage = "时间列索引无效";
        return result;
    }

    if (config.lSpacing <= 0) {
        result.errorMessage = "L-Spacing参数必须大于0";
        return result;
    }

    emit progressUpdated(10, "正在读取数据...");

    QVector<double> timeData;
    QVector<double> pressureData;
    timeData.reserve(rowCount);
    pressureData.reserve(rowCount);

    for (int row = 0; row < rowCount; ++row) {
        QStandardItem* timeItem = model->item(row, config.timeColumnIndex);
        QStandardItem* pressureItem = model->item(row, config.pressureColumnIndex);

        double timeValue = 0.0;
        double pressureValue = 0.0;

        if (timeItem) timeValue = parseNumericValue(timeItem->text());
        if (pressureItem) pressureValue = parseNumericValue(pressureItem->text());

        if (timeValue < 0) {
            result.errorMessage = QString("检测到无效时间值（行 %1），时间不能为负数").arg(row + 1);
            return result;
        }

        timeData.append(timeValue);
        pressureData.append(pressureValue);
    }

    double actualTimeOffset = 0.0;
    if (config.autoTimeOffset) {
        double minPositiveTime = -1;
        bool hasZeroTime = false;
        for (double t : timeData) {
            if (t <= 0) hasZeroTime = true;
            else {
                if (minPositiveTime < 0 || t < minPositiveTime) minPositiveTime = t;
            }
        }
        if (hasZeroTime) {
            if (minPositiveTime > 0) actualTimeOffset = minPositiveTime * 0.1;
            else actualTimeOffset = config.timeOffset;
        }
    } else {
        actualTimeOffset = config.timeOffset;
    }

    QVector<double> adjustedTimeData;
    adjustedTimeData.reserve(rowCount);
    for (double t : timeData) {
        adjustedTimeData.append(t + actualTimeOffset);
    }

    emit progressUpdated(30, "正在计算压差(Delta P)...");

    QVector<double> deltaPData;
    deltaPData.reserve(rowCount);

    if (config.testType == PressureDerivativeConfig::Drawdown) {
        double pi = config.initialPressure;
        for (double p : pressureData) {
            double dp = pi - p;
            deltaPData.append(std::abs(dp));
        }
    } else {
        double p_shut_in = pressureData.isEmpty() ? 0.0 : pressureData[0];
        for (double p : pressureData) {
            double dp = p - p_shut_in;
            deltaPData.append(std::abs(dp));
        }
    }

    emit progressUpdated(50, "正在计算Bourdet导数...");

    QVector<double> derivativeData = calculateBourdetDerivative(adjustedTimeData, deltaPData, config.lSpacing);

    if (derivativeData.size() != rowCount) {
        result.errorMessage = "导数计算结果数量不匹配";
        return result;
    }

    emit progressUpdated(80, "正在写入结果...");

    int deltaPColIdx = config.pressureColumnIndex + 1;
    model->insertColumn(deltaPColIdx);

    QString deltaPHeader = QString("压差(Delta P)\\%1").arg(config.pressureUnit);
    model->setHorizontalHeaderItem(deltaPColIdx, new QStandardItem(deltaPHeader));

    for (int row = 0; row < rowCount; ++row) {
        QString val = formatValue(deltaPData[row], 6);
        QStandardItem* item = new QStandardItem(val);
        item->setForeground(QBrush(QColor("darkgreen")));
        model->setItem(row, deltaPColIdx, item);
    }
    result.deltaPColumnIndex = deltaPColIdx;
    result.deltaPColumnName = deltaPHeader;

    int derivColIdx = deltaPColIdx + 1;
    model->insertColumn(derivColIdx);

    QString derivHeader = QString("压力导数\\%1").arg(config.pressureUnit);
    model->setHorizontalHeaderItem(derivColIdx, new QStandardItem(derivHeader));

    for (int row = 0; row < rowCount; ++row) {
        QString val = formatValue(derivativeData[row], 6);
        QStandardItem* item = new QStandardItem(val);
        item->setForeground(QBrush(QColor("#1565C0")));
        model->setItem(row, derivColIdx, item);
        result.processedRows++;
    }

    result.derivativeColumnIndex = derivColIdx;
    result.derivativeColumnName = derivHeader;
    result.addedColumnIndex = derivColIdx;
    result.columnName = derivHeader;

    emit progressUpdated(100, "计算完成");

    result.success = true;
    emit calculationCompleted(result);

    return result;
}

QVector<double> PressureDerivativeCalculator::calculateBourdetDerivative(
    const QVector<double>& timeData,
    const QVector<double>& pressureDropData,
    double lSpacing)
{
    QVector<double> derivativeData;
    int n = timeData.size();
    derivativeData.reserve(n);

    if (n == 0) return derivativeData;

    for (int i = 0; i < n; ++i) {
        double derivative = 0.0;
        double ti = timeData[i];
        double pi = pressureDropData[i];

        int leftIndex = findLeftPoint(timeData, i, lSpacing);
        int rightIndex = findRightPoint(timeData, i, lSpacing);

        // ====================================================================
        // 1. 内部点：Bourdet Standard 加权平均
        // ====================================================================
        if (leftIndex >= 0 && rightIndex >= 0) {
            double tj = timeData[leftIndex], pj = pressureDropData[leftIndex];
            double tk = timeData[rightIndex], pk = pressureDropData[rightIndex];

            double deltaXL = std::log(ti) - std::log(tj);
            double deltaXR = std::log(tk) - std::log(ti);

            double mL = calculateDerivativeValue(ti, tj, pi, pj);
            double mR = calculateDerivativeValue(tk, ti, pk, pi);

            if (deltaXL + deltaXR > 1e-12) {
                derivative = (mL * deltaXR + mR * deltaXL) / (deltaXL + deltaXR);
            } else {
                derivative = 0.0;
            }
        }
        // ====================================================================
        // 2. 左边界 (曲线早段)：向右构建三点前向抛物线，计算起点导数消除翘头
        // ====================================================================
        else if (leftIndex < 0 && rightIndex >= 0) {
            int j = rightIndex;
            int k = findRightPoint(timeData, j, lSpacing); // 继续找第3个点

            if (k >= 0) {
                double tj = timeData[j], pj = pressureDropData[j];
                double tk = timeData[k], pk = pressureDropData[k];

                double dx12 = std::log(tj) - std::log(ti);
                double dx13 = std::log(tk) - std::log(ti);

                double m12 = calculateDerivativeValue(tj, ti, pj, pi);
                double m23 = calculateDerivativeValue(tk, tj, pk, pj);

                // 前向抛物线插值导数公式
                if (dx13 > 1e-12) derivative = m12 - (dx12 / dx13) * (m23 - m12);
                else derivative = m12;
            } else {
                double tj = timeData[rightIndex], pj = pressureDropData[rightIndex];
                derivative = calculateDerivativeValue(tj, ti, pj, pi);
            }
        }
        // ====================================================================
        // 3. 右边界 (曲线尾段)：向左构建三点后向抛物线，计算终点导数消除翘尾
        // ====================================================================
        else if (leftIndex >= 0 && rightIndex < 0) {
            int j = leftIndex;
            int k = findLeftPoint(timeData, j, lSpacing); // 继续找第3个点

            if (k >= 0) {
                double tk = timeData[k], pk = pressureDropData[k]; // 最左侧点
                double tj = timeData[j], pj = pressureDropData[j]; // 中间点

                double dx23 = std::log(ti) - std::log(tj);
                double dx13 = std::log(ti) - std::log(tk);

                double m12 = calculateDerivativeValue(tj, tk, pj, pk);
                double m23 = calculateDerivativeValue(ti, tj, pi, pj);

                // 后向抛物线插值导数公式
                if (dx13 > 1e-12) derivative = m23 + (dx23 / dx13) * (m23 - m12);
                else derivative = m23;
            } else {
                double tj = timeData[leftIndex], pj = pressureDropData[leftIndex];
                derivative = calculateDerivativeValue(ti, tj, pi, pj);
            }
        }
        // ====================================================================
        // 4. 数据极稀疏或退化
        // ====================================================================
        else {
            if (i > 0 && i < n - 1) {
                double t_prev = timeData[i-1], p_prev = pressureDropData[i-1];
                double t_next = timeData[i+1], p_next = pressureDropData[i+1];
                double dxL = std::log(ti) - std::log(t_prev);
                double dxR = std::log(t_next) - std::log(ti);
                double mL = calculateDerivativeValue(ti, t_prev, pi, p_prev);
                double mR = calculateDerivativeValue(t_next, ti, p_next, pi);
                if (dxL + dxR > 1e-12) derivative = (mL * dxR + mR * dxL) / (dxL + dxR);
            } else if (i > 0) {
                derivative = calculateDerivativeValue(ti, timeData[i-1], pi, pressureDropData[i-1]);
            } else if (i < n - 1) {
                derivative = calculateDerivativeValue(timeData[i+1], ti, pressureDropData[i+1], pi);
            } else {
                derivative = 0.0;
            }
        }

        derivativeData.append(std::abs(derivative));
    }

    return derivativeData;
}

int PressureDerivativeCalculator::findLeftPoint(const QVector<double>& timeData, int currentIndex, double lSpacing)
{
    if (currentIndex <= 0 || timeData.isEmpty()) return -1;
    double ti = timeData[currentIndex];
    if (ti <= 0) return -1;
    double lnTi = std::log(ti);

    for (int j = currentIndex - 1; j >= 0; --j) {
        double tj = timeData[j];
        if (tj <= 0) continue;
        if ((lnTi - std::log(tj)) >= lSpacing) return j;
    }
    return -1;
}

int PressureDerivativeCalculator::findRightPoint(const QVector<double>& timeData, int currentIndex, double lSpacing)
{
    int n = timeData.size();
    if (currentIndex >= n - 1 || timeData.isEmpty()) return -1;
    double ti = timeData[currentIndex];
    if (ti <= 0) return -1;
    double lnTi = std::log(ti);

    for (int k = currentIndex + 1; k < n; ++k) {
        double tk = timeData[k];
        if (tk <= 0) continue;
        if ((std::log(tk) - lnTi) >= lSpacing) return k;
    }
    return -1;
}

double PressureDerivativeCalculator::calculateDerivativeValue(double t1, double t2, double p1, double p2)
{
    if (t1 <= 0 || t2 <= 0) return 0.0;
    double deltaLnT = std::log(t1) - std::log(t2);
    if (std::abs(deltaLnT) < 1e-10) return 0.0;
    return (p1 - p2) / deltaLnT;
}

PressureDerivativeConfig PressureDerivativeCalculator::autoDetectColumns(QStandardItemModel* model)
{
    PressureDerivativeConfig config;
    if (!model) return config;
    config.pressureColumnIndex = findPressureColumn(model);
    config.timeColumnIndex = findTimeColumn(model);
    return config;
}

int PressureDerivativeCalculator::findPressureColumn(QStandardItemModel* model)
{
    if (!model) return -1;
    QStringList pressureKeywords = {"压力", "pressure", "pres", "P\\", "压力\\"};
    for (int col = 0; col < model->columnCount(); ++col) {
        QStandardItem* headerItem = model->horizontalHeaderItem(col);
        if (headerItem) {
            QString headerText = headerItem->text();
            for (const QString& keyword : pressureKeywords) {
                if (headerText.contains(keyword, Qt::CaseInsensitive)) {
                    if (!headerText.contains("压降") && !headerText.contains("导数") && !headerText.contains("Delta")) {
                        return col;
                    }
                }
            }
        }
    }
    return -1;
}

int PressureDerivativeCalculator::findTimeColumn(QStandardItemModel* model)
{
    if (!model) return -1;
    QStringList timeKeywords = {"时间", "time", "t\\", "小时", "hour", "min", "sec"};
    for (int col = 0; col < model->columnCount(); ++col) {
        QStandardItem* headerItem = model->horizontalHeaderItem(col);
        if (headerItem) {
            QString headerText = headerItem->text();
            for (const QString& keyword : timeKeywords) {
                if (headerText.contains(keyword, Qt::CaseInsensitive)) {
                    return col;
                }
            }
        }
    }
    return -1;
}

double PressureDerivativeCalculator::parseNumericValue(const QString& str)
{
    if (str.isEmpty()) return 0.0;
    QString cleanStr = str.trimmed();
    bool ok;
    double value = cleanStr.toDouble(&ok);
    if (ok) return value;
    cleanStr.remove(QRegularExpression("[a-zA-Z%\\s]+$"));
    value = cleanStr.toDouble(&ok);
    return ok ? value : 0.0;
}

QString PressureDerivativeCalculator::formatValue(double value, int precision)
{
    if (std::isnan(value) || std::isinf(value)) return "0";
    return QString::number(value, 'g', precision);
}

