/*
 * 文件名: fittingchart1.h
 * 文件作用: 双对数坐标系图表类头文件 (Log-Log)
 * 功能描述:
 * 1. 专门用于显示双对数曲线 (Log-Log Plot)。
 * 2. 集成 MouseZoom 交互绘图控件。
 * 3. 提供右键菜单功能：保存图片、导出数据、图表设置、复位。
 * 4. 提供特征线绘制功能 (斜率 k=1, 0.5 等)。
 * 5. 支持标注、箭头、线条的交互 (选中、拖拽、样式修改)。
 */

#ifndef FITTINGCHART1_H
#define FITTINGCHART1_H

#include <QWidget>
#include <QMenu>
#include <QMap>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QCloseEvent>
#include "mousezoom.h"

namespace Ui {
class FittingChart1;
}

// 标注结构体
struct ChartAnnotation1 {
    QCPItemText* textItem = nullptr;
    QCPItemLine* arrowItem = nullptr;
};

class FittingChart1 : public QWidget
{
    Q_OBJECT

public:
    explicit FittingChart1(QWidget *parent = nullptr);
    ~FittingChart1();

    // 设置图表标题
    void setTitle(const QString& title);
    // 获取绘图核心控件
    MouseZoom* getPlot();
    // 清除所有曲线
    void clearGraphs();

protected:
    void keyPressEvent(QKeyEvent *event) override;
    // 关闭事件确认
    void closeEvent(QCloseEvent *event) override;

signals:
    void exportDataTriggered();     // 导出数据信号
    void titleChanged(const QString& newTitle); // 标题改变信号
    void graphsChanged();           // 图表内容改变信号

private slots:
    // 右键菜单及工具栏功能槽函数
    void on_btnSavePic_clicked();
    void on_btnExportData_clicked();
    void on_btnSetting_clicked();
    void on_btnReset_clicked();

    // 交互操作槽函数
    void onAddAnnotationRequested(QCPItemLine* line);
    void onLineStyleRequested(QCPItemLine* line);
    void onDeleteSelectedRequested();
    void onEditItemRequested(QCPAbstractItem* item);

    // 鼠标事件槽函数
    void onPlotMousePress(QMouseEvent* event);
    void onPlotMouseMove(QMouseEvent* event);
    void onPlotMouseRelease(QMouseEvent* event);
    void onPlotMouseDoubleClick(QMouseEvent* event);

    // 绘制特征线
    void addCharacteristicLine(double slope);
    void addAnnotationToLine(QCPItemLine* line);
    void deleteSelectedItems();

private:
    void initUi();
    void initConnections();
    void setupAxisRect();

    // 辅助计算函数
    void calculateLinePoints(double slope, double centerX, double centerY,
                             double& x1, double& y1, double& x2, double& y2);
    double distToSegment(const QPointF& p, const QPointF& s, const QPointF& e);
    void constrainLinePoint(QCPItemLine* line, bool isMovingStart, double mouseX, double mouseY);
    void updateAnnotationArrow(QCPItemLine* line);
    void refreshTitleElement();

private:
    Ui::FittingChart1 *ui;
    MouseZoom* m_plot;
    QCPTextElement* m_titleElement;
    QMenu* m_lineMenu; // 特征线菜单

    QMap<QCPItemLine*, ChartAnnotation1> m_annotations;

    // 交互状态枚举
    enum InteractionMode {
        Mode_None,
        Mode_Dragging_Line,
        Mode_Dragging_Start,
        Mode_Dragging_End,
        Mode_Dragging_Text,
        Mode_Dragging_ArrowStart,
        Mode_Dragging_ArrowEnd
    };

    InteractionMode m_interMode;
    QCPItemLine* m_activeLine;
    QCPItemText* m_activeText;
    QCPItemLine* m_activeArrow;
    QPointF m_lastMousePos;
};

#endif // FITTINGCHART1_H
