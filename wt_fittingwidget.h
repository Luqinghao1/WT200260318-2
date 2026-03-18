/*
 * 文件名: wt_fittingwidget.h
 * 修改说明: 删除了已移除按钮的槽函数声明，调整了变量顺序以消除警告。
 */

#ifndef WT_FITTINGWIDGET_H
#define WT_FITTINGWIDGET_H

#include <QWidget>
#include <QMap>
#include <QVector>
#include <QJsonObject>
#include <QMdiArea>
#include <QMdiSubWindow>
#include <QResizeEvent>
#include <QShowEvent>

#include "modelmanager.h"
#include "fittingparameterchart.h"
#include "fittingchart1.h"
#include "fittingchart2.h"
#include "fittingchart3.h"
#include "mousezoom.h"
#include "fittingcore.h"
#include "fittingsamplingdialog.h"
#include "fittingreport.h"
#include "fittingchart.h"

namespace Ui {
class FittingWidget;
}

class FittingWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FittingWidget(QWidget *parent = nullptr);
    ~FittingWidget();

    void setModelManager(ModelManager* m);
    void setProjectDataModels(const QMap<QString, QStandardItemModel*>& models);
    void setObservedData(const QVector<double>& t, const QVector<double>& deltaP, const QVector<double>& d);
    void setObservedData(const QVector<double>& t, const QVector<double>& deltaP,
                         const QVector<double>& d, const QVector<double>& rawP);

    void updateBasicParameters();
    QJsonObject getJsonState() const;
    void loadFittingState(const QJsonObject& root);
    QString getPlotImageBase64(MouseZoom* plot);

protected:
    void resizeEvent(QResizeEvent* event) override;
    void showEvent(QShowEvent* event) override;

signals:
    void sigRequestSave();

private slots:
    void on_btnLoadData_clicked();
    void on_btnSelectParams_clicked();
    // [已删除] void on_btnResetParams_clicked();  <-- 必须删除这行
    // [已删除] void on_btnUpdateLimits_clicked();  <-- 必须删除这行
    void on_btn_modelSelect_clicked();
    void onSliderWeightChanged(int value);
    void on_btnRunFit_clicked();
    void on_btnStop_clicked();
    void onFitFinished();
    void onOpenSamplingSettings();
    void on_btnExportData_clicked();
    void on_btnExportReport_clicked();
    void onExportCurveData();
    void on_btnImportModel_clicked();
    void on_btnSaveFit_clicked();

    void updateModelCurve(const QMap<QString, double>* explicitParams = nullptr, bool autoScale = false, bool calcError = true);

    void onIterationUpdate(double err, const QMap<QString,double>& p, const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve);
    void layoutCharts();

    // 响应半对数直线移动
    void onSemiLogLineMoved(double k, double b);

private:
    Ui::FittingWidget *ui;
    ModelManager* m_modelManager;
    FittingCore* m_core;
    FittingChart* m_chartManager;

    QMdiArea* m_mdiArea;
    FittingChart1* m_chartLogLog;
    FittingChart2* m_chartSemiLog;
    FittingChart3* m_chartCartesian;

    QMdiSubWindow* m_subWinLogLog;
    QMdiSubWindow* m_subWinSemiLog;
    QMdiSubWindow* m_subWinCartesian;

    MouseZoom* m_plotLogLog;
    MouseZoom* m_plotSemiLog;
    MouseZoom* m_plotCartesian;

    FittingParameterChart* m_paramChart;
    QMap<QString, QStandardItemModel*> m_dataMap;
    ModelManager::ModelType m_currentModelType;

    QVector<double> m_obsTime;
    QVector<double> m_obsDeltaP;
    QVector<double> m_obsDerivative;
    QVector<double> m_obsRawP;

    bool m_isFitting;
    bool m_isCustomSamplingEnabled;

    // [修改] 将新变量放在最后，或者在 cpp 初始化列表中调整顺序
    double m_userDefinedTimeMax;

    QList<SamplingInterval> m_customIntervals;

    void setupPlot();
    void initializeDefaultModel();
    QVector<double> parseSensitivityValues(const QString& text);
    void hideUnwantedParams();
    void loadProjectParams();
};

#endif // WT_FITTINGWIDGET_H
