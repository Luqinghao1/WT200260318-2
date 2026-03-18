/*
 * 文件名: fittingchart2.h
 * 文件作用: 半对数坐标系图表类头文件 (Semi-Log)
 * 修改记录:
 * 1. 增加 sigLineMoved 信号。
 * 2. 增加 calculateAndEmitLineParams 辅助函数。
 */

#ifndef FITTINGCHART2_H
#define FITTINGCHART2_H

#include <QWidget>
#include <QMenu>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QCloseEvent>
#include "mousezoom.h"

namespace Ui {
class FittingChart2;
}

class FittingChart2 : public QWidget
{
    Q_OBJECT

public:
    explicit FittingChart2(QWidget *parent = nullptr);
    ~FittingChart2();

    void setTitle(const QString& title);
    MouseZoom* getPlot();
    void clearGraphs();

    // 添加直线
    void addStraightLine();

protected:
    void closeEvent(QCloseEvent *event) override;

signals:
    void exportDataTriggered();
    void titleChanged(const QString& newTitle);
    void graphsChanged();

    // [新增] 直线移动信号: k=斜率, b=截距(Pi)
    void sigLineMoved(double k, double b);

private slots:
    void on_btnSavePic_clicked();
    void on_btnExportData_clicked();
    void on_btnSetting_clicked();
    void on_btnReset_clicked();

    void onLineStyleRequested(QCPItemLine* line);
    void onDeleteSelectedRequested();
    void onEditItemRequested(QCPAbstractItem* item);

    // 鼠标事件
    void onPlotMousePress(QMouseEvent* event);
    void onPlotMouseMove(QMouseEvent* event);
    void onPlotMouseRelease(QMouseEvent* event);
    void onPlotMouseDoubleClick(QMouseEvent* event);

private:
    void initUi();
    void initConnections();
    void setupAxisRect();
    void refreshTitleElement();

    double distToSegment(const QPointF& p, const QPointF& s, const QPointF& e);
    // [新增] 计算并发送参数
    void calculateAndEmitLineParams();

private:
    Ui::FittingChart2 *ui;
    MouseZoom* m_plot;
    QCPTextElement* m_titleElement;

    // 交互状态
    enum InteractionMode {
        Mode_None,
        Mode_Dragging_Line,
        Mode_Dragging_Start,
        Mode_Dragging_End
    };

    InteractionMode m_interMode;
    QCPItemLine* m_activeLine;
    QPointF m_lastMousePos;
};

#endif // FITTINGCHART2_H
