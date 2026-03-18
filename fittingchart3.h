/*
 * 文件名: fittingchart3.h
 * 文件作用: 标准笛卡尔坐标系图表类头文件 (Cartesian)
 * 功能描述:
 * 1. 专门用于显示历史拟合曲线 (History Plot)。
 * 2. X轴线性，Y轴线性。
 * 3. 基础交互功能。
 */

#ifndef FITTINGCHART3_H
#define FITTINGCHART3_H

#include <QWidget>
#include <QMenu>
#include <QCloseEvent>
#include "mousezoom.h"

namespace Ui {
class FittingChart3;
}

class FittingChart3 : public QWidget
{
    Q_OBJECT

public:
    explicit FittingChart3(QWidget *parent = nullptr);
    ~FittingChart3();

    void setTitle(const QString& title);
    MouseZoom* getPlot();
    void clearGraphs();

protected:
    void closeEvent(QCloseEvent *event) override;

signals:
    void exportDataTriggered();
    void titleChanged(const QString& newTitle);
    void graphsChanged();

private slots:
    void on_btnSavePic_clicked();
    void on_btnExportData_clicked();
    void on_btnSetting_clicked();
    void on_btnReset_clicked();

    void onLineStyleRequested(QCPItemLine* line);
    void onDeleteSelectedRequested();
    void onEditItemRequested(QCPAbstractItem* item);

private:
    void initUi();
    void initConnections();
    void setupAxisRect();
    void refreshTitleElement();

private:
    Ui::FittingChart3 *ui;
    MouseZoom* m_plot;
    QCPTextElement* m_titleElement;
};

#endif // FITTINGCHART3_H
