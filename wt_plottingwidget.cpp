/*
 * 文件名: wt_plottingwidget.cpp
 * 文件作用: 图表分析主界面实现文件
 * 功能描述:
 * 1. 管理试井分析曲线的创建、显示、修改和删除。
 * 2. 实现了 PlottingDialog1/2/3/4 的交互逻辑。
 * 3. 实现了视图状态保存与恢复。
 * 4. [本次修改]
 * - 修复导出 CSV 时中文表头乱码的问题（添加 UTF-8 BOM）。
 * - 导出后发出的 viewExportedFile 信号将在 MainWindow 中处理跳转逻辑。
 */

#include "wt_plottingwidget.h"
#include "ui_wt_plottingwidget.h"
#include "plottingdialog1.h"
#include "plottingdialog2.h"
#include "plottingdialog3.h"
#include "plottingdialog4.h"
#include "chartwindow.h"
#include "modelparameter.h"
#include "chartsetting1.h"
#include "pressurederivativecalculator.h"
#include "pressurederivativecalculator1.h"
#include "xlsxdocument.h" //  QtXlsx 库

#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QtMath>
#include <QDebug>
#include <QSplitter>
#include <QStringConverter> // Qt6 编码支持

// ============================================================================
// 辅助函数与 CurveInfo 实现
// ============================================================================

QJsonArray vectorToJson(const QVector<double>& vec) {
    QJsonArray arr;
    for(double v : vec) arr.append(v);
    return arr;
}

QVector<double> jsonToVector(const QJsonArray& arr) {
    QVector<double> vec;
    for(const auto& val : arr) vec.append(val.toDouble());
    return vec;
}

QJsonObject CurveInfo::toJson() const {
    QJsonObject obj;
    obj["name"] = name;
    obj["legendName"] = legendName;
    obj["sourceFileName"] = sourceFileName;
    obj["sourceFileName2"] = sourceFileName2;
    obj["type"] = type;
    obj["xCol"] = xCol;
    obj["yCol"] = yCol;
    obj["xData"] = vectorToJson(xData);
    obj["yData"] = vectorToJson(yData);
    obj["pointShape"] = (int)pointShape;
    obj["pointColor"] = pointColor.name();
    obj["lineStyle"] = (int)lineStyle;
    obj["lineColor"] = lineColor.name();
    obj["lineWidth"] = lineWidth;

    if (type == 1) {
        obj["x2Col"] = x2Col;
        obj["y2Col"] = y2Col;
        obj["x2Data"] = vectorToJson(x2Data);
        obj["y2Data"] = vectorToJson(y2Data);
        obj["prodLegendName"] = prodLegendName;
        obj["prodGraphType"] = prodGraphType;
        obj["prodColor"] = prodColor.name();

        obj["prodPointShape"] = (int)prodPointShape;
        obj["prodPointColor"] = prodPointColor.name();
        obj["prodLineStyle"] = (int)prodLineStyle;
        obj["prodLineColor"] = prodLineColor.name();
        obj["prodLineWidth"] = prodLineWidth;
    }
    else if (type == 2) {
        obj["testType"] = testType;
        obj["initialPressure"] = initialPressure;
        obj["LSpacing"] = LSpacing;
        obj["isSmooth"] = isSmooth;
        obj["smoothFactor"] = smoothFactor;
        obj["derivData"] = vectorToJson(derivData);
        obj["derivShape"] = (int)derivShape;
        obj["derivPointColor"] = derivPointColor.name();
        obj["derivLineStyle"] = (int)derivLineStyle;
        obj["derivLineColor"] = derivLineColor.name();
        obj["derivLineWidth"] = derivLineWidth;
        obj["prodLegendName"] = prodLegendName;
    }
    return obj;
}

CurveInfo CurveInfo::fromJson(const QJsonObject& json) {
    CurveInfo info;
    info.name = json["name"].toString();
    info.legendName = json["legendName"].toString();
    info.sourceFileName = json["sourceFileName"].toString();
    info.sourceFileName2 = json["sourceFileName2"].toString();
    info.type = json["type"].toInt();
    info.xCol = json["xCol"].toInt(-1);
    info.yCol = json["yCol"].toInt(-1);

    info.xData = jsonToVector(json["xData"].toArray());
    info.yData = jsonToVector(json["yData"].toArray());

    info.pointShape = (QCPScatterStyle::ScatterShape)json["pointShape"].toInt();
    info.pointColor = QColor(json["pointColor"].toString());
    info.lineStyle = (Qt::PenStyle)json["lineStyle"].toInt();
    info.lineColor = QColor(json["lineColor"].toString());
    info.lineWidth = json["lineWidth"].toInt(2);

    if (info.type == 1) {
        info.x2Col = json["x2Col"].toInt(-1);
        info.y2Col = json["y2Col"].toInt(-1);
        info.x2Data = jsonToVector(json["x2Data"].toArray());
        info.y2Data = jsonToVector(json["y2Data"].toArray());
        info.prodLegendName = json["prodLegendName"].toString();
        info.prodGraphType = json["prodGraphType"].toInt();
        info.prodColor = QColor(json["prodColor"].toString());

        info.prodPointShape = (QCPScatterStyle::ScatterShape)json["prodPointShape"].toInt((int)QCPScatterStyle::ssNone);
        info.prodPointColor = QColor(json["prodPointColor"].toString(info.prodColor.name()));
        info.prodLineStyle = (Qt::PenStyle)json["prodLineStyle"].toInt((int)Qt::SolidLine);
        info.prodLineColor = QColor(json["prodLineColor"].toString(info.prodColor.name()));
        info.prodLineWidth = json["prodLineWidth"].toInt(2);
    } else if (info.type == 2) {
        info.testType = json["testType"].toInt(0);
        info.initialPressure = json["initialPressure"].toDouble(0.0);
        info.LSpacing = json["LSpacing"].toDouble();
        info.isSmooth = json["isSmooth"].toBool();
        info.smoothFactor = json["smoothFactor"].toInt();
        info.derivData = jsonToVector(json["derivData"].toArray());
        info.derivShape = (QCPScatterStyle::ScatterShape)json["derivShape"].toInt();
        info.derivPointColor = QColor(json["derivPointColor"].toString());
        info.derivLineStyle = (Qt::PenStyle)json["derivLineStyle"].toInt();
        info.derivLineColor = QColor(json["derivLineColor"].toString());
        info.derivLineWidth = json["derivLineWidth"].toInt(2);
        info.prodLegendName = json["prodLegendName"].toString();
    }
    return info;
}

// ============================================================================
// WT_PlottingWidget 主类实现
// ============================================================================

WT_PlottingWidget::WT_PlottingWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WT_PlottingWidget),
    m_defaultModel(nullptr),
    m_isSelectingForExport(false),
    m_selectionStep(0),
    m_exportStartIndex(0),
    m_exportEndIndex(0),
    m_graphPress(nullptr),
    m_graphProd(nullptr)
{
    ui->setupUi(this);

    QList<int> sizes;
    sizes << 200 << 800;
    ui->splitter->setSizes(sizes);
    ui->splitter->setCollapsible(0, false);

    connect(ui->customPlot, &ChartWidget::exportDataTriggered, this, &WT_PlottingWidget::onExportDataTriggered);
    connect(ui->customPlot->getPlot(), &QCustomPlot::plottableClick, this, &WT_PlottingWidget::onGraphClicked);

    // 连接数据修改信号
    connect(ui->customPlot, &ChartWidget::graphDataModified, this, &WT_PlottingWidget::onGraphDataModified);

    // 连接标题和图例修改信号
    connect(ui->customPlot, &ChartWidget::titleChanged, this, &WT_PlottingWidget::onChartTitleChanged);
    connect(ui->customPlot, &ChartWidget::graphsChanged, this, &WT_PlottingWidget::onChartGraphsChanged);

    ui->customPlot->setChartMode(ChartWidget::Mode_Single);
    ui->customPlot->setTitle("试井分析图表");
}

WT_PlottingWidget::~WT_PlottingWidget()
{
    qDeleteAll(m_openedWindows);
    delete ui;
}

// 处理图表标题变更，同步更新列表和内部数据
void WT_PlottingWidget::onChartTitleChanged(const QString& newTitle)
{
    if (m_currentDisplayedCurve.isEmpty()) return;
    if (newTitle == m_currentDisplayedCurve) return; // 没变

    // 检查名称冲突
    if (m_curves.contains(newTitle)) {
        QMessageBox::warning(this, "重命名失败", "该名称已存在，请使用其他名称。");
        // 恢复旧标题
        ui->customPlot->setTitle(m_currentDisplayedCurve);
        return;
    }

    // 更新 m_curves (取出旧的，修改名称，插入新的)
    CurveInfo info = m_curves.take(m_currentDisplayedCurve);
    info.name = newTitle;
    m_curves.insert(newTitle, info);

    // 同步更新视图状态Map的Key
    if (m_viewStates.contains(m_currentDisplayedCurve)) {
        ViewState state = m_viewStates.take(m_currentDisplayedCurve);
        m_viewStates.insert(newTitle, state);
    }

    // 更新 ListWidget
    QListWidgetItem* item = getCurrentSelectedItem();
    if (item && item->text() == m_currentDisplayedCurve) {
        item->setText(newTitle);
    } else {
        // 如果选中项不匹配，尝试查找
        auto items = ui->listWidget_Curves->findItems(m_currentDisplayedCurve, Qt::MatchExactly);
        if (!items.isEmpty()) items.first()->setText(newTitle);
    }

    m_currentDisplayedCurve = newTitle;
}

// 处理图表图例变更
void WT_PlottingWidget::onChartGraphsChanged()
{
    if (m_currentDisplayedCurve.isEmpty()) return;
    if (!m_curves.contains(m_currentDisplayedCurve)) return;

    CurveInfo& info = m_curves[m_currentDisplayedCurve];
    QCustomPlot* plot = ui->customPlot->getPlot();

    if (info.type == 1) { // Stacked: Pressure + Prod
        if (m_graphPress) info.legendName = m_graphPress->name();
        if (m_graphProd) info.prodLegendName = m_graphProd->name();
    }
    else if (info.type == 2) { // Derivative: DeltaP + Deriv
        if (plot->graphCount() > 0) info.legendName = plot->graph(0)->name();
        if (plot->graphCount() > 1) info.prodLegendName = plot->graph(1)->name();
    }
    else { // Single
        if (plot->graphCount() > 0) info.legendName = plot->graph(0)->name();
    }
}

void WT_PlottingWidget::setDataModels(const QMap<QString, QStandardItemModel*>& models) {
    m_dataMap = models;
    if (!m_dataMap.isEmpty()) {
        m_defaultModel = m_dataMap.first();
    } else {
        m_defaultModel = nullptr;
    }
}

void WT_PlottingWidget::setProjectFolderPath(const QString& path) { Q_UNUSED(path); }

void WT_PlottingWidget::updateChartTitle(const QString& title) {
    ui->customPlot->setTitle(title);
    if (m_curves.contains(m_currentDisplayedCurve)) {
        m_curves[m_currentDisplayedCurve].name = title;
        QListWidgetItem* item = getCurrentSelectedItem();
        if(item) item->setText(title);
    }
}

// [关键修改] 优化对话框样式，显式设置按钮为灰底黑字
void WT_PlottingWidget::applyDialogStyle(QWidget* dialog) {
    if(!dialog) return;
    QString qss = "QWidget { color: black; background-color: white; font-family: 'Microsoft YaHei'; }"
                  "QPushButton { "
                  "   background-color: #f0f0f0; "  // 浅灰背景
                  "   color: black; "               // 黑色文字
                  "   border: 1px solid #bfbfbf; "
                  "   border-radius: 3px; "
                  "   padding: 5px 15px; "
                  "   min-width: 60px; "
                  "}"
                  "QPushButton:hover { background-color: #e0e0e0; }"
                  "QPushButton:pressed { background-color: #d0d0d0; }";
    dialog->setStyleSheet(qss);
}

void WT_PlottingWidget::loadProjectData() {
    m_curves.clear();
    m_viewStates.clear();
    ui->listWidget_Curves->clear();
    ui->customPlot->clearGraphs();
    m_currentDisplayedCurve.clear();

    QJsonArray plots = ModelParameter::instance()->getPlottingData();
    if (plots.isEmpty()) return;

    for (const auto& val : plots) {
        CurveInfo info = CurveInfo::fromJson(val.toObject());
        m_curves.insert(info.name, info);
        ui->listWidget_Curves->addItem(info.name);
    }

    if (ui->listWidget_Curves->count() > 0) {
        on_listWidget_Curves_itemDoubleClicked(ui->listWidget_Curves->item(0));
    }
}

void WT_PlottingWidget::saveProjectData() {
    if (!ModelParameter::instance()->hasLoadedProject()) return;
    QJsonArray curvesArray;
    for(auto it = m_curves.begin(); it != m_curves.end(); ++it) {
        curvesArray.append(it.value().toJson());
    }
    ModelParameter::instance()->savePlottingData(curvesArray);
    QMessageBox::information(this, "保存", "绘图数据已保存。");
}

void WT_PlottingWidget::on_btn_Save_clicked() { saveProjectData(); }

void WT_PlottingWidget::clearAllPlots() {
    m_curves.clear();
    m_viewStates.clear();
    m_currentDisplayedCurve.clear();
    ui->listWidget_Curves->clear();
    ui->customPlot->clearGraphs();
    ui->customPlot->setTitle("试井分析图表");
    qDeleteAll(m_openedWindows);
    m_openedWindows.clear();
}

// 保存曲线视图状态
void WT_PlottingWidget::saveCurveViewState(const QString& name) {
    if (name.isEmpty()) return;

    ViewState state;
    state.saved = true;

    ChartWidget* widget = ui->customPlot;
    MouseZoom* plot = widget->getPlot();

    if (widget->getChartMode() == ChartWidget::Mode_Stacked) {
        if (widget->getTopRect()) {
            state.topXRange = widget->getTopRect()->axis(QCPAxis::atBottom)->range();
            state.topYRange = widget->getTopRect()->axis(QCPAxis::atLeft)->range();
        }
        if (widget->getBottomRect()) {
            state.bottomXRange = widget->getBottomRect()->axis(QCPAxis::atBottom)->range();
            state.bottomYRange = widget->getBottomRect()->axis(QCPAxis::atLeft)->range();
        }
    } else {
        state.xRange = plot->xAxis->range();
        state.yRange = plot->yAxis->range();
    }

    m_viewStates[name] = state;
}

// 恢复曲线视图状态
void WT_PlottingWidget::restoreCurveViewState(const QString& name) {
    if (!m_viewStates.contains(name)) return;

    ViewState state = m_viewStates[name];
    if (!state.saved) return;

    ChartWidget* widget = ui->customPlot;
    MouseZoom* plot = widget->getPlot();

    if (widget->getChartMode() == ChartWidget::Mode_Stacked) {
        if (widget->getTopRect()) {
            widget->getTopRect()->axis(QCPAxis::atBottom)->setRange(state.topXRange);
            widget->getTopRect()->axis(QCPAxis::atLeft)->setRange(state.topYRange);
        }
        if (widget->getBottomRect()) {
            widget->getBottomRect()->axis(QCPAxis::atBottom)->setRange(state.bottomXRange);
            widget->getBottomRect()->axis(QCPAxis::atLeft)->setRange(state.bottomYRange);
        }
    } else {
        plot->xAxis->setRange(state.xRange);
        plot->yAxis->setRange(state.yRange);
    }

    plot->replot();
}

// 列表双击：在主界面显示曲线
void WT_PlottingWidget::on_listWidget_Curves_itemDoubleClicked(QListWidgetItem *item) {
    QString name = item->text();
    if(!m_curves.contains(name)) return;

    if (!m_currentDisplayedCurve.isEmpty()) {
        saveCurveViewState(m_currentDisplayedCurve);
    }

    CurveInfo info = m_curves[name];
    m_currentDisplayedCurve = name;

    m_graphPress = nullptr;
    m_graphProd = nullptr;

    displayCurve(info, ui->customPlot);
    restoreCurveViewState(name);
}

// 通用绘图入口函数
void WT_PlottingWidget::displayCurve(const CurveInfo& info, ChartWidget* widget) {
    if (!widget) return;

    widget->clearGraphs();
    widget->setTitle(info.name);
    MouseZoom* plot = widget->getPlot();

    if (info.type == 1) { // Stacked (Pressure + Rate)
        widget->setChartMode(ChartWidget::Mode_Stacked);
        if (widget->getTopRect()) widget->getTopRect()->axis(QCPAxis::atLeft)->setLabel("Pressure");
        if (widget->getBottomRect()) {
            widget->getBottomRect()->axis(QCPAxis::atLeft)->setLabel("Production");
            widget->getBottomRect()->axis(QCPAxis::atBottom)->setLabel("Time");
        }
        drawStackedPlot(info, widget);
    }
    else if (info.type == 2) { // Derivative
        widget->setChartMode(ChartWidget::Mode_Single);
        plot->xAxis->setLabel("Time");
        plot->yAxis->setLabel("Pressure & Derivative");
        plot->xAxis->setScaleType(QCPAxis::stLogarithmic);
        plot->yAxis->setScaleType(QCPAxis::stLogarithmic);
        plot->xAxis->setTicker(QSharedPointer<QCPAxisTickerLog>(new QCPAxisTickerLog));
        plot->yAxis->setTicker(QSharedPointer<QCPAxisTickerLog>(new QCPAxisTickerLog));
        drawDerivativePlot(info, widget);
    }
    else { // Simple Curve
        widget->setChartMode(ChartWidget::Mode_Single);
        plot->xAxis->setScaleType(QCPAxis::stLinear);
        plot->yAxis->setScaleType(QCPAxis::stLinear);
        plot->xAxis->setTicker(QSharedPointer<QCPAxisTicker>(new QCPAxisTicker));
        plot->yAxis->setTicker(QSharedPointer<QCPAxisTicker>(new QCPAxisTicker));

        QStandardItemModel* model = m_defaultModel;
        if (!info.sourceFileName.isEmpty() && m_dataMap.contains(info.sourceFileName)) {
            model = m_dataMap.value(info.sourceFileName);
        }
        if(model) {
            if(info.xCol >=0) plot->xAxis->setLabel(model->headerData(info.xCol, Qt::Horizontal).toString());
            if(info.yCol >=0) plot->yAxis->setLabel(model->headerData(info.yCol, Qt::Horizontal).toString());
        }
        addCurveToPlot(info, widget);
    }
}

void WT_PlottingWidget::addCurveToPlot(const CurveInfo& info, ChartWidget* widget) {
    if (!widget) widget = ui->customPlot;
    MouseZoom* plot = widget->getPlot();

    QCPGraph* graph = plot->addGraph();
    graph->setName(info.legendName);
    graph->setData(info.xData, info.yData);
    graph->setScatterStyle(QCPScatterStyle(info.pointShape, info.pointColor, info.pointColor, 6));
    graph->setPen(QPen(info.lineColor, info.lineWidth, info.lineStyle));
    graph->setLineStyle(info.lineStyle == Qt::NoPen ? QCPGraph::lsNone : QCPGraph::lsLine);

    plot->rescaleAxes();
    plot->replot();
}

void WT_PlottingWidget::drawStackedPlot(const CurveInfo& info, ChartWidget* widget) {
    if (!widget) widget = ui->customPlot;

    widget->clearEventLines();

    QCPAxisRect* topRect = widget->getTopRect();
    QCPAxisRect* bottomRect = widget->getBottomRect();
    MouseZoom* plot = widget->getPlot();

    if (!topRect || !bottomRect) return;

    // 绘制压力曲线
    QCPGraph* gPress = plot->addGraph(topRect->axis(QCPAxis::atBottom), topRect->axis(QCPAxis::atLeft));
    gPress->setData(info.xData, info.yData);
    gPress->setName(info.legendName);
    gPress->setScatterStyle(QCPScatterStyle(info.pointShape, info.pointColor, info.pointColor, 6));
    gPress->setPen(QPen(info.lineColor, info.lineWidth, info.lineStyle));
    gPress->setLineStyle(info.lineStyle == Qt::NoPen ? QCPGraph::lsNone : QCPGraph::lsLine);

    // 绘制产量曲线
    QCPGraph* gProd = plot->addGraph(bottomRect->axis(QCPAxis::atBottom), bottomRect->axis(QCPAxis::atLeft));
    QVector<double> px, py;

    if(info.prodGraphType == 0) { // 阶梯图
        bool isAbsoluteTime = true;
        if (info.x2Data.size() > 1) {
            for (int i = 0; i < info.x2Data.size() - 1; ++i) {
                if (info.x2Data[i+1] <= info.x2Data[i]) {
                    isAbsoluteTime = false;
                    break;
                }
            }
        } else {
            isAbsoluteTime = false;
        }

        if (isAbsoluteTime && !info.x2Data.isEmpty()) {
            px = info.x2Data;
            py = info.y2Data;
        } else {
            double t_cum = 0;
            if(!info.x2Data.isEmpty() && !info.y2Data.isEmpty()) {
                px.append(0);
                py.append(info.y2Data[0]);
            }
            for(int i=0; i<info.x2Data.size(); ++i) {
                t_cum += info.x2Data[i];
                if(i+1 < info.y2Data.size()) {
                    px.append(t_cum);
                    py.append(info.y2Data[i+1]);
                }
                else if (i < info.y2Data.size()) {
                    px.append(t_cum);
                    py.append(info.y2Data[i]);
                }
            }
        }

        if (px.size() == py.size() && px.size() > 1) {
            for (int i = 0; i < px.size() - 1; ++i) {
                double valCurrent = py[i];
                double valNext = py[i+1];
                double timeNext = px[i+1];

                if (valCurrent > 1e-9 && valNext <= 1e-9) {
                    widget->addEventLine(timeNext, 0);
                }
                else if (valCurrent <= 1e-9 && valNext > 1e-9) {
                    widget->addEventLine(timeNext, 1);
                }
            }
        }

        gProd->setLineStyle(QCPGraph::lsStepLeft);
        gProd->setScatterStyle(QCPScatterStyle::ssNone);
        gProd->setBrush(QBrush(info.prodLineColor.lighter(170)));
        gProd->setPen(QPen(info.prodLineColor, info.prodLineWidth, info.prodLineStyle));

    } else { // 折线图或散点图
        px = info.x2Data;
        py = info.y2Data;

        gProd->setScatterStyle(QCPScatterStyle(info.prodPointShape, info.prodPointColor, info.prodPointColor, 6));
        gProd->setPen(QPen(info.prodLineColor, info.prodLineWidth, info.prodLineStyle));
        gProd->setBrush(Qt::NoBrush);

        if (info.prodGraphType == 1) {
            gProd->setLineStyle(QCPGraph::lsLine);
        } else {
            if (info.prodLineStyle != Qt::NoPen) gProd->setLineStyle(QCPGraph::lsLine);
            else gProd->setLineStyle(QCPGraph::lsNone);
        }
    }

    gProd->setData(px, py);
    gProd->setName(info.prodLegendName);

    gPress->rescaleAxes();
    gProd->rescaleAxes();
    plot->replot();

    if (widget == ui->customPlot) {
        m_graphPress = gPress;
        m_graphProd = gProd;
    }
}

void WT_PlottingWidget::drawDerivativePlot(const CurveInfo& info, ChartWidget* widget) {
    if (!widget) widget = ui->customPlot;
    MouseZoom* plot = widget->getPlot();

    QCPGraph* g1 = plot->addGraph();
    g1->setName(info.legendName);
    g1->setData(info.xData, info.yData);
    g1->setScatterStyle(QCPScatterStyle(info.pointShape, info.pointColor, info.pointColor, 6));
    g1->setPen(QPen(info.lineColor, info.lineWidth, info.lineStyle));
    g1->setLineStyle(info.lineStyle == Qt::NoPen ? QCPGraph::lsNone : QCPGraph::lsLine);

    QCPGraph* g2 = plot->addGraph();
    g2->setName(info.prodLegendName);
    g2->setData(info.xData, info.derivData);
    g2->setScatterStyle(QCPScatterStyle(info.derivShape, info.derivPointColor, info.derivPointColor, 6));
    g2->setPen(QPen(info.derivLineColor, info.derivLineWidth, info.derivLineStyle));
    g2->setLineStyle(info.derivLineStyle == Qt::NoPen ? QCPGraph::lsNone : QCPGraph::lsLine);

    plot->rescaleAxes();
    plot->replot();
}

void WT_PlottingWidget::onGraphDataModified(QCPGraph* graph) {
    if (!graph || m_currentDisplayedCurve.isEmpty()) return;
    if (!m_curves.contains(m_currentDisplayedCurve)) return;

    CurveInfo& info = m_curves[m_currentDisplayedCurve];

    if (info.type == 1) { // 压力产量图
        QVector<double> newX, newY;
        auto dataPtr = graph->data();
        for (auto it = dataPtr->begin(); it != dataPtr->end(); ++it) {
            newX.append(it->key);
            newY.append(it->value);
        }

        if (graph == m_graphPress) {
            info.xData = newX;
            info.yData = newY;
        }
        else if (graph == m_graphProd) {
            info.x2Data = newX;
            info.y2Data = newY;
        }
    }
}

// -----------------------------------------------------------------------------
// on_btn_Manage_clicked 修改曲线逻辑
// -----------------------------------------------------------------------------
void WT_PlottingWidget::on_btn_Manage_clicked() {
    QListWidgetItem* item = getCurrentSelectedItem();
    if(!item) return;
    QString name = item->text();
    if (!m_curves.contains(name)) return;
    CurveInfo& info = m_curves[name];

    DialogCurveInfo dlgInfo;
    dlgInfo.type = info.type;
    dlgInfo.name = info.name;
    dlgInfo.sourceFileName = info.sourceFileName;
    dlgInfo.xCol = info.xCol;
    dlgInfo.yCol = info.yCol;
    dlgInfo.pointShape = info.pointShape;
    dlgInfo.pointColor = info.pointColor;
    dlgInfo.lineStyle = info.lineStyle;
    dlgInfo.lineColor = info.lineColor;
    dlgInfo.lineWidth = info.lineWidth;

    if (info.type == 1) {
        dlgInfo.sourceFileName2 = info.sourceFileName2;
        dlgInfo.x2Col = info.x2Col;
        dlgInfo.y2Col = info.y2Col;
        dlgInfo.prodGraphType = info.prodGraphType;
        dlgInfo.style2PointShape = info.prodPointShape;
        dlgInfo.style2PointColor = info.prodPointColor;
        dlgInfo.style2LineStyle = info.prodLineStyle;
        dlgInfo.style2LineColor = info.prodLineColor;
        dlgInfo.style2LineWidth = info.prodLineWidth;
    } else if (info.type == 2) {
        dlgInfo.testType = info.testType;
        dlgInfo.initialPressure = info.initialPressure;
        dlgInfo.LSpacing = info.LSpacing;
        dlgInfo.isSmooth = info.isSmooth;
        dlgInfo.smoothFactor = info.smoothFactor;
        dlgInfo.style2PointShape = info.derivShape;
        dlgInfo.style2PointColor = info.derivPointColor;
        dlgInfo.style2LineStyle = info.derivLineStyle;
        dlgInfo.style2LineColor = info.derivLineColor;
        dlgInfo.style2LineWidth = info.derivLineWidth;
    }

    PlottingDialog4 dlg(m_dataMap, this);
    applyDialogStyle(&dlg);
    dlg.initialize(dlgInfo);

    if(dlg.exec() == QDialog::Accepted) {
        DialogCurveInfo result = dlg.getResult();

        bool nameChanged = (info.name != result.name);
        if (nameChanged) {
            m_curves.remove(name);
            m_viewStates.remove(name);

            info.name = result.name;
            item->setText(info.name);
            name = info.name;
            m_curves.insert(name, info);
            m_currentDisplayedCurve = name;
        }

        CurveInfo& currentInfo = m_curves[name];

        currentInfo.sourceFileName = result.sourceFileName;
        currentInfo.xCol = result.xCol;
        currentInfo.yCol = result.yCol;

        if (m_dataMap.contains(currentInfo.sourceFileName)) {
            QStandardItemModel* model = m_dataMap.value(currentInfo.sourceFileName);
            if (model && currentInfo.xCol >= 0 && currentInfo.xCol < model->columnCount() &&
                currentInfo.yCol >= 0 && currentInfo.yCol < model->columnCount()) {

                currentInfo.xData.clear();
                currentInfo.yData.clear();

                for(int i=0; i<model->rowCount(); ++i) {
                    QStandardItem* itemX = model->item(i, currentInfo.xCol);
                    QStandardItem* itemY = model->item(i, currentInfo.yCol);

                    if (itemX && itemY) {
                        double xVal = itemX->text().toDouble();
                        double yVal = itemY->text().toDouble();

                        if(currentInfo.type != 2) {
                            if (xVal > 1e-9 && yVal > 1e-9) {
                                currentInfo.xData.append(xVal);
                                currentInfo.yData.append(yVal);
                            }
                        } else {
                            if (xVal > 0) {
                                currentInfo.xData.append(xVal);
                                currentInfo.yData.append(yVal);
                            }
                        }
                    }
                }
            }
        }

        currentInfo.pointShape = result.pointShape;
        currentInfo.pointColor = result.pointColor;
        currentInfo.lineStyle = result.lineStyle;
        currentInfo.lineColor = result.lineColor;
        currentInfo.lineWidth = result.lineWidth;

        if (currentInfo.type == 1) {
            currentInfo.sourceFileName2 = result.sourceFileName2;
            currentInfo.x2Col = result.x2Col;
            currentInfo.y2Col = result.y2Col;

            if (m_dataMap.contains(currentInfo.sourceFileName2)) {
                QStandardItemModel* model = m_dataMap.value(currentInfo.sourceFileName2);
                if (model && currentInfo.x2Col >= 0 && currentInfo.x2Col < model->columnCount() &&
                    currentInfo.y2Col >= 0 && currentInfo.y2Col < model->columnCount()) {

                    currentInfo.x2Data.clear();
                    currentInfo.y2Data.clear();
                    for(int i=0; i<model->rowCount(); ++i) {
                        QStandardItem* itemX = model->item(i, currentInfo.x2Col);
                        QStandardItem* itemY = model->item(i, currentInfo.y2Col);
                        if (itemX && itemY) {
                            currentInfo.x2Data.append(itemX->text().toDouble());
                            currentInfo.y2Data.append(itemY->text().toDouble());
                        }
                    }
                }
            }

            currentInfo.prodGraphType = result.prodGraphType;
            currentInfo.prodPointShape = result.style2PointShape;
            currentInfo.prodPointColor = result.style2PointColor;
            currentInfo.prodLineStyle = result.style2LineStyle;
            currentInfo.prodLineColor = result.style2LineColor;
            currentInfo.prodLineWidth = result.style2LineWidth;
        }
        else if (currentInfo.type == 2) {
            currentInfo.testType = result.testType;
            currentInfo.initialPressure = result.initialPressure;
            currentInfo.LSpacing = result.LSpacing;
            currentInfo.isSmooth = result.isSmooth;
            currentInfo.smoothFactor = result.smoothFactor;

            currentInfo.derivShape = result.style2PointShape;
            currentInfo.derivPointColor = result.style2PointColor;
            currentInfo.derivLineStyle = result.style2LineStyle;
            currentInfo.derivLineColor = result.style2LineColor;
            currentInfo.derivLineWidth = result.style2LineWidth;

            QVector<double> rawT = currentInfo.xData;
            QVector<double> rawP = currentInfo.yData;
            currentInfo.xData.clear();
            currentInfo.yData.clear();

            double p_shutin = (rawP.isEmpty()) ? 0 : rawP[0];

            for(int i=0; i<rawT.size(); ++i) {
                if (i >= rawP.size()) break;
                double t = rawT[i];
                double p = rawP[i];
                double dp = (currentInfo.testType == 0) ? std::abs(currentInfo.initialPressure - p) : std::abs(p - p_shutin);
                if(t > 0 && dp > 0) {
                    currentInfo.xData.append(t);
                    currentInfo.yData.append(dp);
                }
            }

            QVector<double> derData = PressureDerivativeCalculator::calculateBourdetDerivative(currentInfo.xData, currentInfo.yData, currentInfo.LSpacing);
            if (currentInfo.isSmooth) derData = PressureDerivativeCalculator1::smoothData(derData, currentInfo.smoothFactor);
            currentInfo.derivData = derData;
        }

        if(m_currentDisplayedCurve == currentInfo.name) {
            displayCurve(currentInfo, ui->customPlot);
            m_viewStates.remove(currentInfo.name);
        }
    }
}

// -----------------------------------------------------------------------------
// on_btn_NewCurve_clicked: 新建通用曲线
// -----------------------------------------------------------------------------
void WT_PlottingWidget::on_btn_NewCurve_clicked() {
    if(m_dataMap.isEmpty()) return;
    PlottingDialog1 dlg(m_dataMap, this);
    applyDialogStyle(&dlg);
    if(dlg.exec() == QDialog::Accepted) {
        CurveInfo info;
        info.name = dlg.getCurveName();
        info.legendName = dlg.getLegendName();

        info.sourceFileName = dlg.getSelectedFileName();
        info.xCol = dlg.getXColumn();
        info.yCol = dlg.getYColumn();

        info.pointShape = dlg.getPointShape();
        info.pointColor = dlg.getPointColor();
        info.lineColor = dlg.getLineColor();
        info.lineStyle = dlg.getLineStyle();
        info.lineWidth = dlg.getLineWidth();

        info.type = 0;
        if (m_dataMap.contains(info.sourceFileName)) {
            QStandardItemModel* model = m_dataMap.value(info.sourceFileName);
            for(int i=0; i<model->rowCount(); ++i) {
                QStandardItem* itemX = model->item(i, info.xCol);
                QStandardItem* itemY = model->item(i, info.yCol);

                if (itemX && itemY) {
                    double xVal = itemX->text().toDouble();
                    double yVal = itemY->text().toDouble();
                    if (xVal > 1e-9 && yVal > 1e-9) {
                        info.xData.append(xVal);
                        info.yData.append(yVal);
                    }
                }
            }
        }
        m_curves.insert(info.name, info);
        ui->listWidget_Curves->addItem(info.name);

        if (dlg.isNewWindow()) {
            ChartWindow* cw = new ChartWindow(nullptr);
            cw->setAttribute(Qt::WA_DeleteOnClose);
            cw->setWindowTitle(info.name);
            cw->show();
            displayCurve(info, cw->getChartWidget());
            m_openedWindows.append(cw);
        } else {
            on_listWidget_Curves_itemDoubleClicked(ui->listWidget_Curves->item(ui->listWidget_Curves->count()-1));
        }
    }
}

// -----------------------------------------------------------------------------
// on_btn_PressureRate_clicked: 新建双坐标曲线
// -----------------------------------------------------------------------------
void WT_PlottingWidget::on_btn_PressureRate_clicked() {
    if(m_dataMap.isEmpty()) return;

    PlottingDialog2 dlg(m_dataMap, this);
    applyDialogStyle(&dlg);

    if(dlg.exec() == QDialog::Accepted) {
        CurveInfo info;
        info.name = dlg.getChartName();
        info.legendName = "压力";
        info.type = 1;

        info.sourceFileName = dlg.getPressFileName();
        info.sourceFileName2 = dlg.getProdFileName();

        info.xCol = dlg.getPressXCol();
        info.yCol = dlg.getPressYCol();
        info.x2Col = dlg.getProdXCol();
        info.y2Col = dlg.getProdYCol();

        if (m_dataMap.contains(info.sourceFileName)) {
            QStandardItemModel* modelP = m_dataMap.value(info.sourceFileName);
            for(int i=0; i<modelP->rowCount(); ++i) {
                QStandardItem* itemX = modelP->item(i, info.xCol);
                QStandardItem* itemY = modelP->item(i, info.yCol);
                if (itemX && itemY) {
                    info.xData.append(itemX->text().toDouble());
                    info.yData.append(itemY->text().toDouble());
                }
            }
        }

        if (m_dataMap.contains(info.sourceFileName2)) {
            QStandardItemModel* modelQ = m_dataMap.value(info.sourceFileName2);
            for(int i=0; i<modelQ->rowCount(); ++i) {
                QStandardItem* itemX = modelQ->item(i, info.x2Col);
                QStandardItem* itemY = modelQ->item(i, info.y2Col);
                if (itemX && itemY) {
                    info.x2Data.append(itemX->text().toDouble());
                    info.y2Data.append(itemY->text().toDouble());
                }
            }
        }

        info.pointShape = dlg.getPressShape();
        info.pointColor = dlg.getPressPointColor();
        info.lineStyle = dlg.getPressLineStyle();
        info.lineColor = dlg.getPressLineColor();
        info.lineWidth = dlg.getPressLineWidth();

        info.prodLegendName = "产量";

        info.prodGraphType = dlg.getProdGraphType();
        info.prodPointShape = dlg.getProdPointShape();
        info.prodPointColor = dlg.getProdPointColor();
        info.prodLineStyle = dlg.getProdLineStyle();
        info.prodLineColor = dlg.getProdLineColor();
        info.prodLineWidth = dlg.getProdLineWidth();
        info.prodColor = info.prodLineColor;

        m_curves.insert(info.name, info);
        ui->listWidget_Curves->addItem(info.name);

        if (dlg.isNewWindow()) {
            ChartWindow* cw = new ChartWindow(nullptr);
            cw->setAttribute(Qt::WA_DeleteOnClose);
            cw->setWindowTitle(info.name);
            cw->show();
            displayCurve(info, cw->getChartWidget());
            m_openedWindows.append(cw);
        } else {
            on_listWidget_Curves_itemDoubleClicked(ui->listWidget_Curves->item(ui->listWidget_Curves->count()-1));
        }
    }
}

// -----------------------------------------------------------------------------
// on_btn_Derivative_clicked: 新建导数分析曲线
// -----------------------------------------------------------------------------
void WT_PlottingWidget::on_btn_Derivative_clicked() {
    if(m_dataMap.isEmpty()) return;
    PlottingDialog3 dlg(m_dataMap, this);
    applyDialogStyle(&dlg);
    if(dlg.exec() == QDialog::Accepted) {
        CurveInfo info;
        info.name = dlg.getCurveName();
        info.legendName = "压差";
        info.type = 2;

        info.sourceFileName = dlg.getSelectedFileName();
        info.xCol = dlg.getTimeColumn();
        info.yCol = dlg.getPressureColumn();
        info.testType = (int)dlg.getTestType();
        info.initialPressure = dlg.getInitialPressure();
        info.LSpacing = dlg.getLSpacing();
        info.isSmooth = dlg.isSmoothEnabled();
        info.smoothFactor = dlg.getSmoothFactor();
        if (m_dataMap.contains(info.sourceFileName)) {
            QStandardItemModel* model = m_dataMap.value(info.sourceFileName);

            double p_shutin = 0;
            if (model->rowCount() > 0) {
                QStandardItem* item0 = model->item(0, info.yCol);
                if(item0) p_shutin = item0->text().toDouble();
            }

            for(int i=0; i<model->rowCount(); ++i) {
                QStandardItem* itemX = model->item(i, info.xCol);
                QStandardItem* itemY = model->item(i, info.yCol);

                if (itemX && itemY) {
                    double t = itemX->text().toDouble();
                    double p = itemY->text().toDouble();
                    double dp = (info.testType == 0) ? std::abs(info.initialPressure - p) : std::abs(p - p_shutin);
                    if(t > 0 && dp > 0) {
                        info.xData.append(t);
                        info.yData.append(dp);
                    }
                }
            }
        }
        QVector<double> derData = PressureDerivativeCalculator::calculateBourdetDerivative(info.xData, info.yData, info.LSpacing);
        if (info.isSmooth) derData = PressureDerivativeCalculator1::smoothData(derData, info.smoothFactor);
        info.derivData = derData;
        info.pointShape = dlg.getPressShape();
        info.pointColor = dlg.getPressPointColor();
        info.lineStyle = dlg.getPressLineStyle();
        info.lineColor = dlg.getPressLineColor();
        info.lineWidth = dlg.getPressLineWidth();

        info.derivShape = dlg.getDerivShape();
        info.derivPointColor = dlg.getDerivPointColor();
        info.derivLineStyle = dlg.getDerivLineStyle();
        info.derivLineColor = dlg.getDerivLineColor();
        info.derivLineWidth = dlg.getDerivLineWidth();

        info.prodLegendName = "压力导数";

        m_curves.insert(info.name, info);
        ui->listWidget_Curves->addItem(info.name);

        if (dlg.isNewWindow()) {
            ChartWindow* cw = new ChartWindow(nullptr);
            cw->setAttribute(Qt::WA_DeleteOnClose);
            cw->setWindowTitle(info.name);
            cw->show();
            displayCurve(info, cw->getChartWidget());
            m_openedWindows.append(cw);
        } else {
            on_listWidget_Curves_itemDoubleClicked(ui->listWidget_Curves->item(ui->listWidget_Curves->count()-1));
        }
    }
}

void WT_PlottingWidget::on_btn_Delete_clicked() {
    QListWidgetItem* item = getCurrentSelectedItem(); if(!item) return;
    QString name = item->text();
    if(QMessageBox::question(this, "确认删除", "确定要删除曲线 \"" + name + "\" 吗？") == QMessageBox::Yes) {
        m_curves.remove(name); delete item;
        m_viewStates.remove(name);
        if(m_currentDisplayedCurve == name) { ui->customPlot->clearGraphs(); m_currentDisplayedCurve.clear(); }
    }
}

// -----------------------------------------------------------------------------
// [修改] 优化导出功能的交互体验
// -----------------------------------------------------------------------------
void WT_PlottingWidget::onExportDataTriggered() {
    if(m_currentDisplayedCurve.isEmpty()) {
        QMessageBox::warning(this, "提示", "当前没有显示的曲线。");
        return;
    }
    QMessageBox msgBox(this);
    msgBox.setWindowTitle("导出数据");
    msgBox.setText("请选择导出范围：");
    msgBox.setIcon(QMessageBox::Question);
    QPushButton* btnAll = msgBox.addButton("全部数据", QMessageBox::ActionRole);
    QPushButton* btnPart = msgBox.addButton("部分数据", QMessageBox::ActionRole);
    msgBox.addButton("取消", QMessageBox::RejectRole);
    applyDialogStyle(&msgBox); // 应用统一的样式
    msgBox.exec();

    if(msgBox.clickedButton() == btnAll) {
        executeExport(true);
    }
    else if(msgBox.clickedButton() == btnPart) {
        m_isSelectingForExport = true;
        m_selectionStep = 1;
        ui->customPlot->getPlot()->setCursor(Qt::CrossCursor);
        QMessageBox::information(this, "提示", "请在曲线上点击起始点。");
    }
}

void WT_PlottingWidget::onGraphClicked(QCPAbstractPlottable *plottable, int dataIndex, QMouseEvent *event) {
    Q_UNUSED(event);
    if(!m_isSelectingForExport) return;
    QCPGraph* graph = qobject_cast<QCPGraph*>(plottable);
    if(!graph) return;
    double key = graph->dataMainKey(dataIndex);
    if(m_selectionStep == 1) {
        m_exportStartIndex = key;
        m_selectionStep = 2;
        QMessageBox::information(this, "提示", "请点击结束点。");
    } else {
        m_exportEndIndex = key;
        if(m_exportStartIndex > m_exportEndIndex) std::swap(m_exportStartIndex, m_exportEndIndex);
        m_isSelectingForExport = false;
        ui->customPlot->getPlot()->setCursor(Qt::ArrowCursor);
        executeExport(false, m_exportStartIndex, m_exportEndIndex);
    }
}

// 导出数据：使用 QtXlsx 导出真正的 Excel 文件，支持不同曲线的列定义
void WT_PlottingWidget::executeExport(bool fullRange, double start, double end) {
    if (m_currentDisplayedCurve.isEmpty() || !m_curves.contains(m_currentDisplayedCurve)) return;
    CurveInfo& info = m_curves[m_currentDisplayedCurve];

    // 1. 获取保存路径
    QString dir = ModelParameter::instance()->getProjectPath();
    if (dir.isEmpty()) dir = QDir::currentPath();

    // 默认为 xlsx
    QString defaultName = dir + "/" + info.name + ".xlsx";
    // 不再提供 .xls 选项
    QString filter = "Excel Files (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)";
    QString file = QFileDialog::getSaveFileName(this, "导出数据", defaultName, filter);
    if(file.isEmpty()) return;

    // 2. 准备数据容器
    QStringList headers;
    QVector<QStringList> dataRows;

    // 获取通用曲线的轴标签
    QString xLabel = "X轴";
    QString yLabel = "Y轴";
    if (ui->customPlot->getPlot()->xAxis) xLabel = ui->customPlot->getPlot()->xAxis->label();
    if (ui->customPlot->getPlot()->yAxis) yLabel = ui->customPlot->getPlot()->yAxis->label();
    if (xLabel.isEmpty()) xLabel = "X数据";
    if (yLabel.isEmpty()) yLabel = "Y数据";

    // =========================================================
    // 数据组装逻辑
    // =========================================================

    // --- Type 0: 通用曲线 ---
    if (info.type == 0) {
        if (fullRange) {
            // 全量: X轴名, Y轴名
            headers << xLabel << yLabel;
        } else {
            // 部分: X轴名, Y轴名, 原始+X轴名
            headers << xLabel << yLabel << ("原始" + xLabel);
        }

        for (int i = 0; i < info.xData.size(); ++i) {
            double t = info.xData[i];
            if (!fullRange && (t < start || t > end)) continue;

            QStringList row;
            double val = info.yData[i];

            if (fullRange) {
                row << QString::number(t) << QString::number(val);
            } else {
                row << QString::number(t - start) << QString::number(val) << QString::number(t);
            }
            dataRows.append(row);
        }
    }
    // --- Type 1: 压力产量分析 ---
    else if (info.type == 1) {
        // 始终保持: 时间, 压力, 产量, 原始时间
        headers << "时间" << "压力" << "产量" << "原始时间";

        for (int i = 0; i < info.xData.size(); ++i) {
            double t = info.xData[i];
            if (!fullRange && (t < start || t > end)) continue;

            double p = info.yData[i];
            double q = 0.0;
            // 获取产量值
            if (m_graphProd) {
                q = getProductionValueFromGraph(t, m_graphProd);
            } else if (i < info.y2Data.size()) {
                q = info.y2Data[i];
            }

            QStringList row;
            double timeExport = fullRange ? t : (t - start);

            row << QString::number(timeExport)
                << QString::number(p)
                << QString::number(q)
                << QString::number(t);
            dataRows.append(row);
        }
    }
    // --- Type 2: 压力导数分析 ---
    else if (info.type == 2) {
        if (fullRange) {
            // 全量: 时间, 压差, 压力导数
            headers << "时间" << "压差" << "压力导数";
        } else {
            // 部分: 时间, 压差, 原始时间
            headers << "时间" << "压差" << "原始时间";
        }

        for (int i = 0; i < info.xData.size(); ++i) {
            double t = info.xData[i];
            if (!fullRange && (t < start || t > end)) continue;

            double dp = info.yData[i];

            QStringList row;
            if (fullRange) {
                double deriv = (i < info.derivData.size()) ? info.derivData[i] : 0.0;
                row << QString::number(t) << QString::number(dp) << QString::number(deriv);
            } else {
                // 部分导出不包含导数值
                row << QString::number(t - start) << QString::number(dp) << QString::number(t);
            }
            dataRows.append(row);
        }
    }

    // =========================================================
    // 文件写入逻辑
    // =========================================================

    bool isXlsx = file.endsWith(".xlsx", Qt::CaseInsensitive);

    if (isXlsx) {
        // 使用 QtXlsx 库写入
        QXlsx::Document xlsx;

        // 1. 写表头 (第一行)
        for (int col = 0; col < headers.size(); ++col) {
            // QtXlsx 行列从 1 开始
            xlsx.write(1, col + 1, headers[col]);
        }

        // 2. 写数据
        for (int row = 0; row < dataRows.size(); ++row) {
            QStringList& lineData = dataRows[row];
            for (int col = 0; col < lineData.size(); ++col) {
                // 尝试转为 double 写入以保持数字格式，否则存为文本
                bool ok;
                double val = lineData[col].toDouble(&ok);
                if (ok) {
                    xlsx.write(row + 2, col + 1, val);
                } else {
                    xlsx.write(row + 2, col + 1, lineData[col]);
                }
            }
        }

        if (xlsx.saveAs(file)) {
            // Success
        } else {
            QMessageBox::warning(this, "错误", "保存 xlsx 文件失败，请检查文件是否被占用。");
            return;
        }

    } else {
        // CSV 或 TXT (保持 UTF-8 BOM)
        QFile f(file);
        if (!f.open(QIODevice::WriteOnly | QIODevice::Text)) {
            QMessageBox::warning(this, "错误", "无法打开文件进行写入。");
            return;
        }

        QTextStream out(&f);
        out.setGenerateByteOrderMark(true);
        out.setEncoding(QStringConverter::Utf8);

        QString sep = file.endsWith(".csv", Qt::CaseInsensitive) ? "," : "\t";

        // 写表头
        out << headers.join(sep) << "\n";

        // 写数据
        for (const QStringList& row : dataRows) {
            out << row.join(sep) << "\n";
        }
        f.close();
    }

    // 4. 导出后交互
    QMessageBox openMsg(this);
    openMsg.setWindowTitle("导出成功");
    openMsg.setText("数据导出完成。\n路径: " + file + "\n\n是否在数据界面打开导出的文件？");
    openMsg.setIcon(QMessageBox::Question);
    QPushButton* btnYes = openMsg.addButton("打开文件", QMessageBox::ActionRole);
    openMsg.addButton("关闭", QMessageBox::RejectRole);
    applyDialogStyle(&openMsg);
    openMsg.exec();

    if (openMsg.clickedButton() == btnYes) {
        emit viewExportedFile(file);
    }
}

double WT_PlottingWidget::getProductionValueFromGraph(double t, QCPGraph* graph) {
    if (!graph) return 0.0;

    if (graph->lineStyle() == QCPGraph::lsStepLeft) {
        auto data = graph->data();
        auto it = data->findBegin(t);
        if (it == data->end()) return 0.0;
        return it->value;
    }

    auto data = graph->data();
    auto it = data->findBegin(t);
    if (it == data->end()) return 0.0;
    if (qAbs(it->key - t) < 1e-9) return it->value;
    if (it == data->begin()) return it->value;
    auto prev = it;
    --prev;
    double t1 = prev->key;
    double v1 = prev->value;
    double t2 = it->key;
    double v2 = it->value;
    if (qAbs(t2 - t1) < 1e-9) return v1;
    return v1 + (t - t1) * (v2 - v1) / (t2 - t1);
}

double WT_PlottingWidget::getProductionValueAt(double t, const CurveInfo& info) { Q_UNUSED(t); return info.y2Data.isEmpty() ? 0 : info.y2Data.last(); }
QListWidgetItem* WT_PlottingWidget::getCurrentSelectedItem() { return ui->listWidget_Curves->currentItem(); }
