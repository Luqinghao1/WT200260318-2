/*
 * 文件名: fittingreport.h
 * 文件作用: 试井拟合报告生成器头文件
 * 功能描述:
 * 1. 定义报告生成所需的数据结构 FittingReportData。
 * 2. 声明 FittingReportGenerator 类，负责生成 HTML/Word 报告及关联的 CSV 数据表。
 */

#ifndef FITTINGREPORT_H
#define FITTINGREPORT_H

#include <QString>
#include <QVector>
#include <QList>
#include "fittingparameterchart.h"
#include "modelmanager.h"

// 报告所需的数据包
struct FittingReportData {
    QString wellName;                   // 井名
    ModelManager::ModelType modelType;  // 模型类型
    double mse;                         // 拟合误差 (MSE)

    // 观测数据
    QVector<double> t;
    QVector<double> p;
    QVector<double> d;

    // 参数列表
    QList<FitParameter> params;

    // 图表截图 (Base64编码)
    QString imgLogLog;
    QString imgSemiLog;
    QString imgCartesian;
};

class FittingReportGenerator
{
public:
    /**
     * @brief 生成报告
     * @param filePath 保存路径 (例如 "C:/.../report.doc")
     * @param data 报告数据结构体
     * @param errorMsg [输出] 如果失败，返回错误信息
     * @return 成功返回 true，失败返回 false
     */
    static bool generate(const QString& filePath, const FittingReportData& data, QString* errorMsg = nullptr);

private:
    // 内部辅助：生成 CSV 数据表
    static bool generateDataCSV(const QString& csvPath, const FittingReportData& data);

    // 内部辅助：生成 HTML 内容
    static QString buildHtmlContent(const FittingReportData& data, const QString& csvFileName);
};

#endif // FITTINGREPORT_H
