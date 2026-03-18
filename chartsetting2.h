/*
 * chartsetting2.h
 * 文件作用：双坐标系图表设置对话框头文件
 * 功能描述：
 * 1. 采用 QTabWidget 分页：通用/压力/产量
 * 2. 详细的轴控制：可见性、标签、刻度、网格、子网格、范围
 */

#ifndef CHARTSETTING2_H
#define CHARTSETTING2_H

#include <QDialog>
#include "qcustomplot.h"

namespace Ui {
class ChartSetting2;
}

class ChartSetting2 : public QDialog
{
    Q_OBJECT

public:
    explicit ChartSetting2(QCustomPlot* plot, QCPAxisRect* top, QCPAxisRect* bottom, QCPTextElement* title, QWidget *parent = nullptr);
    ~ChartSetting2();

private slots:
    void on_buttonBox_accepted();

private:
    Ui::ChartSetting2 *ui;
    QCustomPlot* m_plot;
    QCPAxisRect* m_topRect;
    QCPAxisRect* m_bottomRect;
    QCPTextElement* m_title;

    void loadSettings();
    void applySettings();
};

#endif // CHARTSETTING2_H
