/*
 * 文件名: fittingchart.h
 * 文件作用: 拟合绘图管理类头文件
 * 修改记录:
 * 1. [新增] 增加 sigManualPressureUpdated 信号，用于平移直线时通知外部更新参数(Pi)。
 */

#ifndef FITTINGCHART_H
#define FITTINGCHART_H

#include <QObject>
#include <QVector>
#include <QPointer>
#include <QAction>
#include <QPointF>
#include <QMap>
#include <QJsonObject>
#include "mousezoom.h"
#include "fittingdatadialog.h"
#include "fittingpressuredialog.h"
#include "styleselectordialog.h"

// 标注结构体
struct FittingChartAnnotation {
    QCPItemText* textItem = nullptr;
    QCPItemLine* arrowItem = nullptr;
};

class FittingChart : public QObject
{
    Q_OBJECT
public:
    explicit FittingChart(QObject *parent = nullptr);

    // 初始化图表指针
    void initializeCharts(MouseZoom* logLog, MouseZoom* semiLog, MouseZoom* cartesian);

    void setObservedData(const QVector<double>& t, const QVector<double>& deltaP,
                         const QVector<double>& deriv, const QVector<double>& rawP);

    void setSettings(const FittingDataSettings& settings);
    FittingDataSettings getSettings() const;

    // 绘制所有曲线
    void plotAll(const QVector<double>& t_model, const QVector<double>& p_model, const QVector<double>& d_model, bool isModelValid, bool autoScale = false);

    // 绘制抽样点
    void plotSampledPoints(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);
    void plotCartesian(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale); // 补全声明

    double getCalculatedInitialPressure() const;

    // 获取半对数拟合状态
    QJsonObject getManualPressureState() const;
    // 恢复半对数拟合状态
    void setManualPressureState(const QJsonObject& state);

signals:
    // [新增] 当手动拟合线发生变化(计算或平移)时发送此信号
    void sigManualPressureUpdated(double k, double b);

private slots:
    // 半对数图右键菜单
    void onSemiLogContextMenu(const QPoint &pos);

    // 交互槽函数
    void onShowPressureSolver();

    // 鼠标事件处理
    void onPlotMousePress(QMouseEvent *event);
    void onPlotMouseMove(QMouseEvent *event);
    void onPlotMouseRelease(QMouseEvent *event);

    // 弹窗交互
    void onPickStart();
    void onPickEnd();
    void onCalculatePressure();

    // 功能槽函数
    void onLineStyleRequested(QCPItemLine* line);
    void onAddAnnotationRequested(QCPItemLine* line);
    void onDeleteSelectedRequested();
    void onEditItemRequested(QCPAbstractItem* item);

private:
    MouseZoom* m_plotLogLog;
    MouseZoom* m_plotSemiLog;
    MouseZoom* m_plotCartesian;

    QVector<double> m_obsT;
    QVector<double> m_obsDeltaP;
    QVector<double> m_obsDeriv;
    QVector<double> m_obsRawP;

    FittingDataSettings m_settings;
    double m_calculatedPi;

    // --- 原始地层压力求解相关 ---
    QPointer<FittingPressureDialog> m_pressureDialog;
    bool m_isPickingStart;
    bool m_isPickingEnd;

    // 结果缓存
    bool m_hasManualPressure;
    double m_manualSlope;
    double m_manualIntercept;
    double m_manualStartX;
    double m_manualEndX;

    // 持久化存储文本框位置
    double m_manualTextX;
    double m_manualTextY;

    // 绘图项指针
    QPointer<QCPItemLine> m_manualFitLine;    // 拟合红线 (P*)
    QPointer<QCPItemLine> m_manualZeroLine;   // X=0 蓝线
    QPointer<QCPItemText> m_manualResultText; // 结果文本框

    // --- 交互状态管理 ---
    enum InteractionMode {
        Mode_None,
        Mode_Dragging_Line,        // 拖拽整条线 (平移)
        Mode_Dragging_Start,       // 拖拽线段起点
        Mode_Dragging_End,         // 拖拽线段终点
        Mode_Dragging_Text,        // 拖拽文本
        Mode_Dragging_ArrowStart,  // 拖拽箭头起点
        Mode_Dragging_ArrowEnd     // 拖拽箭头终点
    };

    InteractionMode m_interMode;
    QPointF m_lastMousePos;

    // 当前选中的活动项
    QCPItemLine* m_activeLine;
    QCPItemText* m_activeText;
    QCPItemLine* m_activeArrow;

    // 标注管理
    QMap<QCPItemLine*, FittingChartAnnotation> m_annotations;

    // 内部绘图函数
    void plotLogLog(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale);
    void plotSemiLog(const QVector<double>& tm, const QVector<double>& pm, const QVector<double>& dm, bool hasModel, bool autoScale);

    void drawPressureFitResult();
    void updateManualResultText();

    // 辅助函数
    double distToSegment(const QPointF& p, const QPointF& s, const QPointF& e);
    void constrainLinePoint(QCPItemLine* line, bool isMovingStart, double mouseX, double mouseY);
    void updateAnnotationArrow(QCPItemLine* line);
    void addAnnotationToLine(QCPItemLine* line);
};

#endif // FITTINGCHART_H
