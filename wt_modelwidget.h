/*
 * wt_modelwidget.h
 * 文件作用: 压裂水平井复合页岩油模型界面类头文件 (View/Controller)
 * 功能描述:
 * 1. 管理用户界面，处理参数输入、按钮响应和图表展示。
 * 2. 包含两个可能的求解器指针 (m_solver1, m_solver2)，根据模型 ID 动态实例化。
 * 3. 继承自 QWidget，负责具体的业务交互逻辑。
 */

#ifndef WT_MODELWIDGET_H
#define WT_MODELWIDGET_H

#include <QWidget>
#include <QMap>
#include <QVector>
#include <QColor>
#include <tuple>
#include "chartwidget.h"
#include "modelsolver01-06.h"
#include "modelsolver19_36.h" // [新增]

namespace Ui {
class WT_ModelWidget;
}

class WT_ModelWidget : public QWidget
{
    Q_OBJECT

public:
    // [修改] 使用 int 作为通用的模型类型
    using ModelType = int;
    // 使用 Solver 中定义的曲线数据类型
    using ModelCurveData = ::ModelCurveData;

    explicit WT_ModelWidget(ModelType type, QWidget *parent = nullptr);
    ~WT_ModelWidget();

    // 设置高精度模式（转发给当前的 Solver）
    void setHighPrecision(bool high);

    // 直接调用求解器计算（供外部管理器使用，非 UI 交互）
    ModelCurveData calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime = QVector<double>());

    // 获取当前模型名称
    QString getModelName() const;

signals:
    // 计算完成信号，传递模型类型和参数
    void calculationCompleted(const QString& modelType, const QMap<QString, double>& params);

    // 请求模型选择界面的信号
    void requestModelSelection();

public slots:
    void onCalculateClicked();
    void onResetParameters();

    // [逻辑] 响应 L 或 Lf 变化，自动更新 LfD
    void onDependentParamsChanged();

    void onShowPointsToggled(bool checked);
    void onExportData();

private:
    void initUi();
    void initChart();
    void setupConnections();
    void runCalculation(); // UI 触发的计算流程封装

    // 辅助函数
    QVector<double> parseInput(const QString& text);
    void setInputText(QLineEdit* edit, double value);
    void plotCurve(const ModelCurveData& data, const QString& name, QColor color, bool isSensitivity);

private:
    Ui::WT_ModelWidget *ui;
    ModelType m_type;

    // [修改] 持有两个求解器指针，根据 m_type 决定实例化哪一个
    // m_solver1 处理 Model 1-18
    ModelSolver01_06* m_solver1;
    // m_solver2 处理 Model 19-36
    ModelSolver19_36* m_solver2;

    bool m_highPrecision;
    QList<QColor> m_colorList;

    // 缓存计算结果
    QVector<double> res_tD;
    QVector<double> res_pD;
    QVector<double> res_dpD;
};

#endif // WT_MODELWIDGET_H
