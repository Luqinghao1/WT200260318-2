/*
 * 文件名: plottingdialog2.h
 * 文件作用: 压力-产量双坐标曲线配置对话框头文件
 * 功能描述:
 * 1. 左侧配置压力数据，右侧配置产量数据。
 * 2. 样式设置采用图标+中文预览。
 * 3. 默认名称为“压力产量分析+数字”。
 * 4. “显示数据来源”功能，若压力和产量来自不同文件，则用 & 连接文件名，并加括号。
 */

#ifndef PLOTTINGDIALOG2_H
#define PLOTTINGDIALOG2_H

#include <QDialog>
#include <QStandardItemModel>
#include <QColor>
#include <QMap>
#include <QComboBox>
#include "qcustomplot.h"

namespace Ui {
class PlottingDialog2;
}

class PlottingDialog2 : public QDialog
{
    Q_OBJECT

public:
    explicit PlottingDialog2(const QMap<QString, QStandardItemModel*>& models, QWidget *parent = nullptr);
    ~PlottingDialog2();

    // --- 获取曲线基础信息 ---
    QString getChartName() const;

    // --- 压力数据获取 ---
    QString getPressFileName() const;
    int getPressXCol() const;
    int getPressYCol() const;
    QString getPressLegend() const;

    // 压力样式
    QCPScatterStyle::ScatterShape getPressShape() const;
    QColor getPressPointColor() const;
    Qt::PenStyle getPressLineStyle() const;
    QColor getPressLineColor() const;
    int getPressLineWidth() const;

    // --- 产量数据获取 ---
    QString getProdFileName() const;
    int getProdXCol() const;
    int getProdYCol() const;
    QString getProdLegend() const;

    // 产量绘图类型: 0-阶梯图, 1-折线图, 2-散点图
    int getProdGraphType() const;

    // 产量样式
    QCPScatterStyle::ScatterShape getProdPointShape() const;
    QColor getProdPointColor() const;
    Qt::PenStyle getProdLineStyle() const;
    QColor getProdLineColor() const;
    int getProdLineWidth() const;

    // 是否在新建窗口显示
    bool isNewWindow() const;

private slots:
    // 文件选择改变
    void onPressFileChanged(int index);
    void onProdFileChanged(int index);

    // 产量绘图类型改变 (控制样式控件显隐)
    void onProdTypeChanged(int index);

    // 显示数据来源复选框改变
    void onShowSourceChanged(bool checked);

private:
    Ui::PlottingDialog2 *ui;
    QMap<QString, QStandardItemModel*> m_dataMap;
    QStandardItemModel* m_pressModel;
    QStandardItemModel* m_prodModel;

    static int s_chartCounter; // 用于实现“数字自小到大自动排序”
    QString m_lastSuffix;

    // --- 辅助函数 ---
    void populatePressColumns();
    void populateProdColumns();

    void setupStyleUI();
    void initColorComboBox(QComboBox* combo);

    // 更新曲线名称后缀
    void updateNameSuffix();

    QIcon createPointIcon(QCPScatterStyle::ScatterShape shape);
    QIcon createLineIcon(Qt::PenStyle style);
};

#endif // PLOTTINGDIALOG2_H
