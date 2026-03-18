/*
 * 文件名: datasamplingdialog.h
 * 文件作用: 数据滤波与取样设置对话框头文件
 * 功能描述:
 * 1. 提供用户界面以设置数据滤波和抽样参数，支持自定义选取X/Y轴。
 * 2. 界面配色优化：小标题加粗蓝色，应用预览(绿)、重置(蓝)、取消(红)，符合工程软件直觉。
 * 3. 抽样默认策略改为“对数抽样 (N=20)”，滤波默认为“无滤波”。
 * 4. 增加了 UI 状态联动：当选择无滤波时，“参考点数”输入框自动置灰不可用。
 * 5. 图例移至左上角，上下图数据点分别采用红色和深蓝色。
 */

#ifndef DATASAMPLINGDIALOG_H
#define DATASAMPLINGDIALOG_H

#include <QDialog>
#include <QVector>
#include <QStringList>
#include <QStandardItemModel>
#include "qcustomplot.h"

class QComboBox;
class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QPushButton;

class DataSamplingDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DataSamplingDialog(QStandardItemModel* model, QWidget *parent = nullptr);
    ~DataSamplingDialog();

    // 获取处理后的完整二维表格数据及表头
    QVector<QStringList> getProcessedTable() const;
    QStringList getHeaders() const;

private slots:
    void onPreviewClicked();
    void onResetClicked();
    void onDataColumnChanged();
    void onAxisScaleChanged();
    void onSplitBoxValueChanged();

    // 拖拽事件
    void onRawPlotMousePress(QMouseEvent *event);
    void onRawPlotMouseMove(QMouseEvent *event);
    void onRawPlotMouseRelease(QMouseEvent *event);

private:
    void initUI();
    void setupConnections();
    void applyStyle();

    void loadRawData();
    void updateSplitLines();
    void drawRawPlot();
    void drawProcessedPlot();
    void processData();

    void applyMedianFilter(QVector<double>& y, int windowSize);
    QVector<int> getLogSamplingIndices(const QVector<double>& x, int pointsPerDecade);
    QVector<int> getIntervalSamplingIndices(int totalCount, int interval);
    QVector<int> getTotalPointsSamplingIndices(int totalCount, int targetPoints);

    QStandardItemModel* m_model;
    QStringList m_headers;
    QVector<QStringList> m_rawTable;
    QVector<QStringList> m_processedTable;

    QVector<double> m_rawX, m_rawY;
    QVector<double> m_processedX, m_processedY;
    QString m_xName, m_yName;

    QComboBox* comboX;
    QComboBox* comboY;
    QCheckBox* chkLogX;
    QCheckBox* chkLogY;
    QDoubleSpinBox* spinMinY;
    QDoubleSpinBox* spinMaxY;
    QDoubleSpinBox* spinSplit1;
    QDoubleSpinBox* spinSplit2;

    struct StageUI {
        QComboBox* comboFilter;
        QSpinBox* spinFilterWin;
        QComboBox* comboSample;
        QSpinBox* spinSampleVal;
    };
    StageUI m_stages[3];

    QPushButton* btnPreview;
    QPushButton* btnReset;
    QPushButton* btnOk;
    QPushButton* btnCancel;

    QCustomPlot* customPlotRaw;
    QCustomPlot* customPlotProcessed;
    QCPItemStraightLine *vLine1, *vLine2;
    QCPItemStraightLine *vLine1_proc, *vLine2_proc;

    int m_dragLineIndex;
};

#endif // DATASAMPLINGDIALOG_H
