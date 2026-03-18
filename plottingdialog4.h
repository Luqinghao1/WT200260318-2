/*
 * 文件名: plottingdialog4.h
 * 文件作用: 曲线管理与修改对话框头文件
 * 功能描述:
 * 1. 支持三种类型曲线的修改（通用、压力产量、压力导数）。
 * 2. 界面动态调整：根据曲线类型显示不同的数据设置、计算设置和样式设置布局。
 * 3. 修复了双栏模式下左侧（副本）控件无法选择、无内容的问题。
 * 4. 界面布局左右等宽，标签符合中文习惯。
 */

#ifndef PLOTTINGDIALOG4_H
#define PLOTTINGDIALOG4_H

#include <QDialog>
#include <QStandardItemModel>
#include <QColor>
#include <QMap>
#include <QComboBox>
#include "qcustomplot.h"

namespace Ui {
class PlottingDialog4;
}

// 用于在对话框和主窗口间传递数据的结构体
struct DialogCurveInfo {
    int type; // 0: 通用, 1: 压力产量, 2: 压力导数
    QString name;

    // Data 1 (Main / Pressure)
    QString sourceFileName;
    int xCol, yCol;

    // Data 2 (Production for Type 1)
    QString sourceFileName2;
    int x2Col, y2Col;

    // Calculation (Type 2)
    int testType; // 0: Drawdown, 1: Buildup
    double initialPressure;
    double LSpacing;
    bool isSmooth;
    int smoothFactor;

    // Style 1 (Main / Pressure / Delta P)
    QCPScatterStyle::ScatterShape pointShape;
    QColor pointColor;
    Qt::PenStyle lineStyle;
    QColor lineColor;
    int lineWidth;

    // Style 2 (Production / Derivative)
    int prodGraphType; // 0: Step, 1: Line, 2: Scatter
    QCPScatterStyle::ScatterShape style2PointShape;
    QColor style2PointColor;
    Qt::PenStyle style2LineStyle;
    QColor style2LineColor;
    int style2LineWidth;
};

class PlottingDialog4 : public QDialog
{
    Q_OBJECT

public:
    explicit PlottingDialog4(const QMap<QString, QStandardItemModel*>& models, QWidget *parent = nullptr);
    ~PlottingDialog4();

    // 初始化对话框数据和界面状态
    void initialize(const DialogCurveInfo& info);

    // 获取修改后的数据
    DialogCurveInfo getResult() const;

private slots:
    // 文件/列联动
    void onFile1Changed(int index);
    void onFile1DupChanged(int index); // 处理双栏模式下左侧文件切换
    void onFile2Changed(int index);

    // 产量绘图类型改变 (Type 1)
    void onProdTypeChanged(int index);

    // 试井类型改变 (Type 2)
    void onTestTypeChanged();
    void onSmoothToggled(bool checked);

private:
    Ui::PlottingDialog4 *ui;
    QMap<QString, QStandardItemModel*> m_dataMap;
    int m_currentType;

    // 辅助函数
    void setupStyleUI();
    void initColorComboBox(QComboBox* combo);
    QIcon createPointIcon(QCPScatterStyle::ScatterShape shape);
    QIcon createLineIcon(Qt::PenStyle style);

    // 刷新列下拉框：同时处理原始控件和副本控件
    void populateColumns(QString key, QComboBox* xCombo, QComboBox* yCombo,
                         QComboBox* xComboDup = nullptr, QComboBox* yComboDup = nullptr);
};

#endif // PLOTTINGDIALOG4_H
