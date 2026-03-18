/*
 * 文件名: wt_plottingwidget.h
 * 文件作用: 图表分析主界面头文件
 * 功能描述:
 * 1. 管理试井分析曲线的创建、显示、修改和删除。
 * 2. CurveInfo 结构体支持双文件数据源（压力+产量）。
 * 3. 增加了视图状态保存功能，切换曲线时可保持上次的缩放和平移视图。
 * 4. [本次修改] 优化导出功能，支持中文表头，修正产量读取，增加导出后跳转文件的信号。
 */

#ifndef WT_PLOTTINGWIDGET_H
#define WT_PLOTTINGWIDGET_H

#include <QWidget>
#include <QStandardItemModel>
#include <QMap>
#include <QListWidgetItem>
#include "chartwidget.h"
#include "chartwindow.h"

// 曲线配置结构体
struct CurveInfo {
    QString name;
    QString legendName;
    QString sourceFileName;  // 主要数据源（对于Type1为压力数据）
    QString sourceFileName2; // [新增] 第二数据源（对于Type1为产量数据）

    int type; // 0: Simple, 1: Stacked (P+Q), 2: Derivative
    int xCol, yCol;
    QVector<double> xData, yData;

    QCPScatterStyle::ScatterShape pointShape = QCPScatterStyle::ssNone;
    QColor pointColor = Qt::black;
    Qt::PenStyle lineStyle = Qt::SolidLine;
    QColor lineColor = Qt::black;
    int lineWidth = 2; // [新增] 主曲线线宽

    // Type 1 Specific (Pressure + Rate)
    int x2Col, y2Col;
    QVector<double> x2Data, y2Data;
    QString prodLegendName;
    int prodGraphType; // 0: Step, 1: Scatter
    QColor prodColor;

    // [新增] 产量曲线独立样式
    QCPScatterStyle::ScatterShape prodPointShape = QCPScatterStyle::ssNone;
    QColor prodPointColor = Qt::blue;
    Qt::PenStyle prodLineStyle = Qt::SolidLine;
    QColor prodLineColor = Qt::blue;
    int prodLineWidth = 2;

    // Type 2 Specific (Pressure + Derivative)
    int testType;
    double initialPressure;
    double LSpacing;
    bool isSmooth;
    int smoothFactor;
    QVector<double> derivData;
    QCPScatterStyle::ScatterShape derivShape = QCPScatterStyle::ssNone;
    QColor derivPointColor = Qt::red;
    Qt::PenStyle derivLineStyle = Qt::SolidLine;
    QColor derivLineColor = Qt::red;
    int derivLineWidth = 2; // [新增] 导数曲线线宽

    QJsonObject toJson() const;
    static CurveInfo fromJson(const QJsonObject& json);
};

namespace Ui {
class WT_PlottingWidget;
}

class WT_PlottingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit WT_PlottingWidget(QWidget *parent = nullptr);
    ~WT_PlottingWidget();

    // 设置数据模型映射表
    void setDataModels(const QMap<QString, QStandardItemModel*>& models);

    // 设置项目文件夹路径 (已弃用，改用 ModelParameter)
    void setProjectFolderPath(const QString& path);

    void loadProjectData();
    void saveProjectData();

    // 清除所有图表
    void clearAllPlots();

    // 更新图表标题
    void updateChartTitle(const QString& title);

signals:
    // [新增] 信号：请求在数据界面打开导出的文件
    void viewExportedFile(const QString& filePath);

private slots:
    void on_btn_NewCurve_clicked();
    void on_btn_PressureRate_clicked();
    void on_btn_Derivative_clicked();

    void on_listWidget_Curves_itemDoubleClicked(QListWidgetItem *item);

    // 保存按钮槽函数
    void on_btn_Save_clicked();

    void on_btn_Manage_clicked();
    void on_btn_Delete_clicked();

    void onExportDataTriggered();
    void onGraphClicked(QCPAbstractPlottable *plottable, int dataIndex, QMouseEvent *event);

    // [新增] 处理图表数据被交互修改后的逻辑
    void onGraphDataModified(QCPGraph* graph);

    // [新增] 处理图表标题变更，同步更新列表和内部数据
    void onChartTitleChanged(const QString& newTitle);

    // [新增] 处理图表图例变更
    void onChartGraphsChanged();

private:
    Ui::WT_PlottingWidget *ui;

    // 存储所有已打开文件的数据模型
    QMap<QString, QStandardItemModel*> m_dataMap;

    // 默认模型 (Fallback)
    QStandardItemModel* m_defaultModel;

    QMap<QString, CurveInfo> m_curves;
    QString m_currentDisplayedCurve;

    QList<QWidget*> m_openedWindows;

    bool m_isSelectingForExport;
    int m_selectionStep;
    double m_exportStartIndex;
    double m_exportEndIndex;

    QCPGraph* m_graphPress;
    QCPGraph* m_graphProd;

    // [新增] 视图状态结构体，用于保存缩放和平移位置
    struct ViewState {
        bool saved = false; // 是否已保存
        // 单图模式 / 导数模式使用的坐标轴范围
        QCPRange xRange;
        QCPRange yRange;

        // 叠加模式 (Stacked) 使用的坐标轴范围
        QCPRange topXRange;
        QCPRange topYRange;
        QCPRange bottomXRange;
        QCPRange bottomYRange;
    };

    // [新增] 存储每条曲线的视图状态
    QMap<QString, ViewState> m_viewStates;

    // [新增] 保存和恢复视图状态的辅助函数
    void saveCurveViewState(const QString& name);
    void restoreCurveViewState(const QString& name);

    // [新增] 通用绘图入口：在指定组件上显示曲线
    void displayCurve(const CurveInfo& info, ChartWidget* widget);

    // [修改] 绘图辅助函数：支持传入目标 ChartWidget
    void addCurveToPlot(const CurveInfo& info, ChartWidget* widget);
    void drawStackedPlot(const CurveInfo& info, ChartWidget* widget);
    void drawDerivativePlot(const CurveInfo& info, ChartWidget* widget);

    void executeExport(bool fullRange, double start = 0, double end = 0);

    // [修改] 产量读取函数优化
    double getProductionValueFromGraph(double t, QCPGraph* graph);
    double getProductionValueAt(double t, const CurveInfo& info);

    QListWidgetItem* getCurrentSelectedItem();

    void applyDialogStyle(QWidget* dialog);
};

#endif // WT_PLOTTINGWIDGET_H
