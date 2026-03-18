/*
 * 文件名: fittingmultiples.cpp
 * 文件作用: 多分析对比界面实现文件
 * 功能描述:
 * 1. 初始化界面，创建 ChartWidget，强制背景为白色。
 * 2. 创建并初始化统一的悬浮信息窗口。
 * 3. [修改] updateCharts 函数重构：
 * - 支持解析 JSON 中的实测数据 (observedData)。
 * - 根据 m_selections 中的配置，按需绘制实测压差、实测导数、理论压差、理论导数。
 * - 修复 rescaleAxes 逻辑，确保包含实测数据的显示范围。
 * 4. 绘制多条曲线，使用颜色区分不同分析。
 */

#include "fittingmultiples.h"
#include "ui_fittingmultiples.h"
#include "fittingparameterchart.h"
#include <QVBoxLayout>
#include <QHeaderView>
#include <QDebug>
#include <cmath>
#include <QJsonArray> // 需要引入数组解析

FittingMultiplesWidget::FittingMultiplesWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FittingMultiplesWidget),
    m_modelManager(nullptr),
    m_chartWidget(nullptr),
    m_plot(nullptr),
    m_infoDialog(nullptr),
    m_infoTabWidget(nullptr),
    m_tableModelInfo(nullptr),
    m_tableParams(nullptr),
    m_tableWeights(nullptr)
{
    ui->setupUi(this);

    // [修改] 2. 设置主界面背景为白色，遮挡可能露出的默认背景色
    this->setAttribute(Qt::WA_StyledBackground, true);
    this->setStyleSheet("background-color: white;");
    ui->plotContainer->setAttribute(Qt::WA_StyledBackground, true);
    ui->plotContainer->setStyleSheet("background-color: white;");

    // 1. 初始化图表
    m_chartWidget = new ChartWidget(this);
    m_chartWidget->setTitle("多分析对比 (Multiple Analysis Comparison)");
    ui->plotContainer->layout()->addWidget(m_chartWidget);
    m_plot = m_chartWidget->getPlot();

    setupPlot();

    // 2. 初始化统一的悬浮信息窗口
    initInfoDialog();
}

FittingMultiplesWidget::~FittingMultiplesWidget()
{
    if(m_infoDialog) delete m_infoDialog;
    delete ui;
}

void FittingMultiplesWidget::setModelManager(ModelManager *m)
{
    m_modelManager = m;
}

void FittingMultiplesWidget::setupPlot()
{
    if (!m_plot) return;

    m_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    m_plot->setBackground(Qt::white);
    m_plot->axisRect()->setBackground(Qt::white);

    // 设置对数坐标轴
    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    m_plot->xAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->xAxis->setTicker(logTicker);
    m_plot->yAxis->setScaleType(QCPAxis::stLogarithmic); m_plot->yAxis->setTicker(logTicker);

    m_plot->xAxis->setNumberFormat("eb"); m_plot->xAxis->setNumberPrecision(0);
    m_plot->yAxis->setNumberFormat("eb"); m_plot->yAxis->setNumberPrecision(0);

    QFont labelFont("Microsoft YaHei", 10, QFont::Bold);
    QFont tickFont("Microsoft YaHei", 9);
    m_plot->xAxis->setLabel("时间 Time (h)");
    m_plot->yAxis->setLabel("压差 & 导数 Delta P & Derivative (MPa)");
    m_plot->xAxis->setLabelFont(labelFont); m_plot->yAxis->setLabelFont(labelFont);
    m_plot->xAxis->setTickLabelFont(tickFont); m_plot->yAxis->setTickLabelFont(tickFont);

    // 开启网格
    m_plot->xAxis->grid()->setVisible(true); m_plot->yAxis->grid()->setVisible(true);
    m_plot->xAxis->grid()->setSubGridVisible(true); m_plot->yAxis->grid()->setSubGridVisible(true);

    m_plot->xAxis->setRange(1e-3, 1e3); m_plot->yAxis->setRange(1e-3, 1e2);

    m_plot->legend->setVisible(true);
    m_plot->legend->setFont(QFont("Microsoft YaHei", 9));
}

void FittingMultiplesWidget::initInfoDialog()
{
    m_infoDialog = new QDialog(this);
    m_infoDialog->setWindowTitle("对比分析详情");

    // 设置窗口标志
    m_infoDialog->setWindowFlags(Qt::Tool | Qt::WindowStaysOnTopHint | Qt::WindowTitleHint | Qt::CustomizeWindowHint);

    m_infoDialog->resize(650, 480);
    m_infoDialog->move(100, 100);

    // === 应用美化样式表 ===
    QString qss = R"(
        QDialog { background-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #dcdcdc; background: white; top: -1px; }
        QTabWidget::tab-bar { left: 5px; }
        QTabBar::tab { background: #f5f5f5; border: 1px solid #dcdcdc; padding: 6px 15px; min-width: 80px; color: #555; border-bottom-color: #dcdcdc; border-top-left-radius: 4px; border-top-right-radius: 4px; font-family: 'Microsoft YaHei'; font-size: 12px; }
        QTabBar::tab:selected { background: #ffffff; border-bottom-color: #ffffff; color: #0078d7; font-weight: bold; }
        QTabBar::tab:!selected { margin-top: 2px; }
        QTabBar::tab:hover { background-color: #eaf6fd; }
        QTableWidget { border: none; gridline-color: #f0f0f0; font-family: 'Microsoft YaHei'; font-size: 12px; selection-background-color: #e6f7ff; selection-color: #000000; outline: 0; }
        QHeaderView::section { background-color: #f9f9f9; padding: 6px; border: none; border-bottom: 1px solid #dcdcdc; border-right: 1px solid #dcdcdc; font-weight: bold; color: #444; font-family: 'Microsoft YaHei'; font-size: 12px; }
        QScrollBar:vertical { border: none; background: #f0f0f0; width: 10px; margin: 0px 0px 0px 0px; }
        QScrollBar::handle:vertical { background: #cdcdcd; min-height: 20px; border-radius: 5px; }
        QScrollBar::handle:vertical:hover { background: #a6a6a6; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
    )";
    m_infoDialog->setStyleSheet(qss);

    QVBoxLayout* layout = new QVBoxLayout(m_infoDialog);
    layout->setContentsMargins(10, 10, 10, 10);

    m_infoTabWidget = new QTabWidget(m_infoDialog);
    layout->addWidget(m_infoTabWidget);

    auto createTable = [&](const QString& tabName) -> QTableWidget* {
        QTableWidget* table = new QTableWidget();
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setAlternatingRowColors(true);
        table->setSelectionMode(QAbstractItemView::ContiguousSelection);
        table->setFocusPolicy(Qt::NoFocus);
        return table;
    };

    m_tableModelInfo = createTable("模型信息");
    m_tableWeights   = createTable("拟合权重");
    m_tableParams    = createTable("拟合参数");

    m_infoTabWidget->addTab(m_tableModelInfo, "模型信息");
    m_infoTabWidget->addTab(m_tableWeights, "拟合权重");
    m_infoTabWidget->addTab(m_tableParams, "拟合参数");
}

// [修改] 初始化函数，增加 selection 参数
void FittingMultiplesWidget::initialize(const QMap<QString, QJsonObject> &states, const QMap<QString, CurveSelection>& selections)
{
    m_states = states;
    m_selections = selections; // 保存选择
    updateCharts();
    updateWindowsData();
}

void FittingMultiplesWidget::updateCharts()
{
    if(!m_modelManager || !m_plot) return;

    m_plot->clearGraphs();

    int idx = 0;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) {
        QString name = it.key();
        QJsonObject state = it.value();
        QColor color = getColor(idx);

        // 获取该分析的显示选项，默认为全显示
        CurveSelection sel;
        if (m_selections.contains(name)) {
            sel = m_selections[name];
        }

        // ==========================================
        // 1. 绘制实测数据 (如果勾选)
        // ==========================================
        QVector<double> obsT, obsP, obsD;
        if (state.contains("observedData")) {
            QJsonObject obsObj = state["observedData"].toObject();
            QJsonArray tArr = obsObj["time"].toArray();
            QJsonArray pArr = obsObj["pressure"].toArray();
            QJsonArray dArr = obsObj["derivative"].toArray();

            for(auto v : tArr) obsT.append(v.toDouble());
            for(auto v : pArr) obsP.append(v.toDouble());
            for(auto v : dArr) obsD.append(v.toDouble());
        }

        if (sel.showObsP && !obsT.isEmpty() && !obsP.isEmpty()) {
            QCPGraph* gObsP = m_plot->addGraph();
            gObsP->setData(obsT, obsP);
            // 实测压差：空心圆，无连线
            gObsP->setLineStyle(QCPGraph::lsNone);
            gObsP->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, color, Qt::white, 6));
            gObsP->setName(name + " (实测 P)");
        }

        if (sel.showObsD && !obsT.isEmpty() && !obsD.isEmpty()) {
            QCPGraph* gObsD = m_plot->addGraph();
            gObsD->setData(obsT, obsD);
            // 实测导数：空心三角形，无连线
            gObsD->setLineStyle(QCPGraph::lsNone);
            gObsD->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, color, Qt::white, 6));
            gObsD->setName(name + " (实测 P')");
        }

        // ==========================================
        // 2. 绘制理论曲线 (如果勾选)
        // ==========================================
        if (sel.showTheoP || sel.showTheoD) {
            // 解析模型参数
            int typeInt = state["modelType"].toInt();
            ModelManager::ModelType type = (ModelManager::ModelType)typeInt;
            QMap<QString, double> paramMap;
            QJsonArray pArr = state["parameters"].toArray();
            for(auto v : pArr) {
                QJsonObject pObj = v.toObject();
                paramMap.insert(pObj["name"].toString(), pObj["value"].toDouble());
            }
            if(paramMap.contains("L") && paramMap.contains("Lf") && paramMap["L"] > 1e-9)
                paramMap["LfD"] = paramMap["Lf"] / paramMap["L"];
            else
                paramMap["LfD"] = 0.0;

            // 确定计算时间序列 (优先用实测时间，否则生成默认)
            QVector<double> tCalc = obsT;
            if (tCalc.isEmpty()) {
                for(double e = -4; e <= 4; e += 0.1) tCalc.append(pow(10, e));
            }

            ModelCurveData curves = m_modelManager->calculateTheoreticalCurve(type, paramMap, tCalc);
            QVector<double> vt = std::get<0>(curves);
            QVector<double> vp = std::get<1>(curves);
            QVector<double> vd = std::get<2>(curves);

            if (sel.showTheoP) {
                QCPGraph* gP = m_plot->addGraph();
                gP->setData(vt, vp);
                gP->setPen(QPen(color, 2, Qt::SolidLine));
                gP->setName(name + " (理论 P)");
            }

            if (sel.showTheoD) {
                QCPGraph* gD = m_plot->addGraph();
                gD->setData(vt, vd);
                gD->setPen(QPen(color, 2, Qt::DashLine));
                gD->setName(name + " (理论 P')");
            }
        }

        idx++;
    }

    // [修改] 自动缩放：考虑所有可见图表
    m_plot->rescaleAxes();

    // 强制下限保护，避免 Log 坐标轴报错
    if(m_plot->xAxis->range().lower <= 0) m_plot->xAxis->setRangeLower(1e-3);
    if(m_plot->yAxis->range().lower <= 0) m_plot->yAxis->setRangeLower(1e-3);

    m_plot->replot();
}

void FittingMultiplesWidget::updateWindowsData()
{
    // ... (updateWindowsData 代码保持原样，未做逻辑变更，故省略以节省篇幅，实际输出时请保留原代码) ...
    // 为确保完整性，此处建议保留原文件中的 updateWindowsData 实现，不做删减。
    // 以下为原代码逻辑复述：

    // 1. 更新【模型信息】表格
    m_tableModelInfo->clear();
    QStringList rowHeaders = {"模型名称", "井筒模型", "井模型", "储层模型", "边界条件"};
    m_tableModelInfo->setRowCount(rowHeaders.size());
    m_tableModelInfo->setVerticalHeaderLabels(rowHeaders);

    QStringList colHeaders;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) colHeaders << it.key();
    m_tableModelInfo->setColumnCount(colHeaders.size());
    m_tableModelInfo->setHorizontalHeaderLabels(colHeaders);

    int col = 0;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) {
        QJsonObject state = it.value();
        int typeInt = state["modelType"].toInt();
        QString modelName = state["modelName"].toString();

        QString wellbore, well, reservoir, boundary;
        well = "压裂水平井"; reservoir = "复合油藏";
        if (typeInt % 2 == 0) wellbore = "变井储"; else wellbore = "恒定井储";
        if (typeInt == 0 || typeInt == 1) boundary = "无穷大外边界";
        else if (typeInt == 2 || typeInt == 3) boundary = "封闭边界";
        else boundary = "定压边界";

        auto setItem = [&](int r, int c, const QString& txt) {
            QTableWidgetItem* item = new QTableWidgetItem(txt);
            item->setTextAlignment(Qt::AlignCenter);
            m_tableModelInfo->setItem(r, c, item);
        };
        setItem(0, col, modelName); setItem(1, col, wellbore);
        setItem(2, col, well); setItem(3, col, reservoir); setItem(4, col, boundary);
        col++;
    }
    m_tableModelInfo->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    m_tableModelInfo->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

    // 2. 更新【拟合权重】表格
    m_tableWeights->clear();
    m_tableWeights->setColumnCount(2);
    m_tableWeights->setHorizontalHeaderLabels({"分析名称", "压差权重 (%)"});
    m_tableWeights->setRowCount(m_states.size());
    m_tableWeights->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    int row = 0;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) {
        m_tableWeights->setItem(row, 0, new QTableWidgetItem(it.key()));
        m_tableWeights->setItem(row, 1, new QTableWidgetItem(QString::number(it.value()["fitWeightVal"].toInt())));
        row++;
    }

    // 3. 更新【拟合参数】表格
    m_tableParams->clear();
    QSet<QString> allKeys;
    QMap<QString, bool> isFitInAny;
    for(const auto& state : m_states) {
        QJsonArray pArr = state["parameters"].toArray();
        for(auto v : pArr) {
            QJsonObject obj = v.toObject();
            QString pName = obj["name"].toString();
            if (pName == "LfD") continue;
            allKeys.insert(pName);
            if (obj["isFit"].toBool()) isFitInAny[pName] = true;
        }
    }
    QStringList fittedParams, fixedParams;
    QStringList allKeysList = allKeys.values(); allKeysList.sort();
    for(const QString& key : allKeysList) {
        if(isFitInAny.value(key, false)) fittedParams.append(key); else fixedParams.append(key);
    }
    QStringList sortedParams = fittedParams + fixedParams;

    QStringList rowLabels;
    for(const QString& key : sortedParams) {
        QString chName, html, uni, unit;
        FittingParameterChart::getParamDisplayInfo(key, chName, html, uni, unit);
        if (chName.isEmpty()) chName = key;
        rowLabels.append(QString("%1 (%2)").arg(chName).arg(key));
    }
    m_tableParams->setRowCount(sortedParams.size());
    m_tableParams->setColumnCount(m_states.size());
    m_tableParams->setVerticalHeaderLabels(rowLabels);
    m_tableParams->verticalHeader()->setSectionResizeMode(QHeaderView::ResizeToContents);

    QStringList paramColHeaders;
    col = 0;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) {
        paramColHeaders << it.key();
        QJsonObject state = it.value();
        QMap<QString, QPair<double, bool>> infoMap;
        QJsonArray pArr = state["parameters"].toArray();
        for(auto v : pArr) {
            QJsonObject obj = v.toObject();
            infoMap.insert(obj["name"].toString(), qMakePair(obj["value"].toDouble(), obj["isFit"].toBool()));
        }
        for(int pRow = 0; pRow < sortedParams.size(); ++pRow) {
            QString pName = sortedParams[pRow];
            QTableWidgetItem* item = new QTableWidgetItem();
            item->setTextAlignment(Qt::AlignCenter);
            if(infoMap.contains(pName)) {
                auto pi = infoMap[pName];
                item->setText(QString::number(pi.first, 'g', 5));
                if(pi.second) {
                    item->setBackground(QColor(220, 255, 220));
                    item->setToolTip("参与拟合");
                    QFont f = item->font(); f.setBold(true); item->setFont(f);
                }
            } else item->setText("-");
            m_tableParams->setItem(pRow, col, item);
        }
        col++;
    }
    m_tableParams->setHorizontalHeaderLabels(paramColHeaders);
    m_tableParams->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
}

QColor FittingMultiplesWidget::getColor(int index)
{
    static QList<QColor> colors = {
        QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"), QColor("#9467bd"),
        QColor("#8c564b"), QColor("#e377c2"), QColor("#7f7f7f"), QColor("#bcbd22"), QColor("#17becf")
    };
    return colors[index % colors.size()];
}

void FittingMultiplesWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    if(m_infoDialog) {
        m_infoDialog->show();
        m_infoDialog->raise();
        m_infoDialog->activateWindow();
    }
}

void FittingMultiplesWidget::hideEvent(QHideEvent *event)
{
    QWidget::hideEvent(event);
    if(m_infoDialog) m_infoDialog->hide();
}

QJsonObject FittingMultiplesWidget::getJsonState() const
{
    QJsonObject root;
    root["type"] = "multiple";
    QJsonObject statesObj;
    for(auto it = m_states.begin(); it != m_states.end(); ++it) {
        statesObj[it.key()] = it.value();
    }
    root["subStates"] = statesObj;

    // [注意] 暂不保存 selections 状态到文件，如果需要可在此添加
    return root;
}

void FittingMultiplesWidget::loadState(const QJsonObject& state)
{
    if(state["type"].toString() == "multiple") {
        QJsonObject statesObj = state["subStates"].toObject();
        m_states.clear();
        for(auto it = statesObj.begin(); it != statesObj.end(); ++it) {
            m_states.insert(it.key(), it.value().toObject());
        }
        updateCharts();
        updateWindowsData();
    }
}
