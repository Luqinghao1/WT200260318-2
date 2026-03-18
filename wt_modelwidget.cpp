/*
 * 文件名: wt_modelwidget.cpp
 * 文件作用: 压裂水平井复合模型理论图版计算界面类实现
 * 修改记录:
 * 1. [修复] 默认参数初始化(onResetParameters)已完全强制设定为 MATLAB 测试代码中提供的真实物理常数。
 * 2. [修复] 渗透率单位明确为 mD (kf=50.0)，压敏系数默认恢复 0.02。
 */

#include "wt_modelwidget.h"
#include "ui_wt_modelwidget.h"
#include "modelmanager.h"
#include "modelparameter.h"

#include <QDebug>
#include <QMessageBox>
#include <QFileDialog>
#include <QTextStream>
#include <QDateTime>
#include <QCoreApplication>
#include <QSplitter>
#include <QLabel>
#include <QLineEdit>
#include <QGridLayout>
#include <cmath>

WT_ModelWidget::WT_ModelWidget(ModelType type, QWidget *parent)
    : QWidget(parent)
    , ui(new Ui::WT_ModelWidget)
    , m_type(type)
    , m_solver1(nullptr)
    , m_solver2(nullptr)
    , m_highPrecision(true)
{
    ui->setupUi(this);

    if (m_type >= 0 && m_type <= 35) {
        m_solver1 = new ModelSolver01_06((ModelSolver01_06::ModelType)m_type);
    }
    else if (m_type >= 36 && m_type <= 71) {
        m_solver2 = new ModelSolver19_36((ModelSolver19_36::ModelType)(m_type - 36));
    }

    m_colorList = { Qt::red, Qt::blue, QColor(0,180,0), Qt::magenta, QColor(255,140,0), Qt::cyan };

    QList<int> sizes;
    sizes << 240 << 960;
    ui->splitter->setSizes(sizes);
    ui->splitter->setCollapsible(0, false);

    ui->btnSelectModel->setText(getModelName());

    initUi();
    initChart();
    setupConnections();
    onResetParameters();
}

WT_ModelWidget::~WT_ModelWidget()
{
    if(m_solver1) delete m_solver1;
    if(m_solver2) delete m_solver2;
    delete ui;
}

QString WT_ModelWidget::getModelName() const {
    if (m_solver1) return ModelSolver01_06::getModelName((ModelSolver01_06::ModelType)m_type, false);
    if (m_solver2) return ModelSolver19_36::getModelName((ModelSolver19_36::ModelType)(m_type - 36), false);
    return "未知模型";
}

WT_ModelWidget::ModelCurveData WT_ModelWidget::calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    if (m_solver1) return m_solver1->calculateTheoreticalCurve(params, providedTime);
    if (m_solver2) return m_solver2->calculateTheoreticalCurve(params, providedTime);
    return ModelCurveData();
}

void WT_ModelWidget::setHighPrecision(bool high)
{
    m_highPrecision = high;
    if (m_solver1) m_solver1->setHighPrecision(high);
    if (m_solver2) m_solver2->setHighPrecision(high);
}

void WT_ModelWidget::initUi() {
    if(ui->label_km) ui->label_km->setText("流度比 M12");
    if(ui->label_L) ui->label_L->setText("水平井总长 L(m)");
    if(ui->label_rmD) ui->label_rmD->setText("复合半径 rm (m)");
    if(ui->label_reD) ui->label_reD->setText("外区半径 re (m)");
    if(ui->label_cD) ui->label_cD->setText("井筒储集 C (m³/MPa)");

    if(ui->eta12Edit) {
        ui->eta12Edit->setReadOnly(true);
        ui->eta12Edit->setStyleSheet("background-color: #f0f0f0; color: #808080;");
    }

    int groupIdx = m_type % 12;
    bool isInfinite = (groupIdx < 4);
    ui->label_reD->setVisible(!isInfinite);
    ui->reDEdit->setVisible(!isInfinite);

    int storageType = m_type % 4;

    bool hasBasicStorage = (storageType != 1);
    ui->label_cD->setVisible(hasBasicStorage);
    ui->cDEdit->setVisible(hasBasicStorage);
    ui->label_s->setVisible(hasBasicStorage);
    ui->sEdit->setVisible(hasBasicStorage);

    bool hasVarStorage = (storageType == 2 || storageType == 3);
    ui->label_alpha->setVisible(hasVarStorage);
    ui->alphaEdit->setVisible(hasVarStorage);
    ui->label_cphi->setVisible(hasVarStorage);
    ui->cPhiEdit->setVisible(hasVarStorage);

    bool hasInnerParams = false;
    bool hasOuterParams = false;

    if (m_type <= 35) {
        if (m_type <= 23) hasInnerParams = true;
        if (m_type <= 11) hasOuterParams = true;
    } else {
        hasInnerParams = true;
        int sub = m_type - 36;
        if (sub <= 11 || sub >= 24) hasOuterParams = true;
    }

    ui->label_omga1->setVisible(hasInnerParams);
    ui->omga1Edit->setVisible(hasInnerParams);
    ui->label_remda1->setVisible(hasInnerParams);
    ui->remda1Edit->setVisible(hasInnerParams);

    ui->label_omga2->setVisible(hasOuterParams);
    ui->omga2Edit->setVisible(hasOuterParams);
    ui->label_remda2->setVisible(hasOuterParams);
    ui->remda2Edit->setVisible(hasOuterParams);
}

void WT_ModelWidget::initChart() {
    MouseZoom* plot = ui->chartWidget->getPlot();
    plot->setBackground(Qt::white);
    plot->axisRect()->setBackground(Qt::white);

    QSharedPointer<QCPAxisTickerLog> logTicker(new QCPAxisTickerLog);
    plot->xAxis->setScaleType(QCPAxis::stLogarithmic); plot->xAxis->setTicker(logTicker);
    plot->yAxis->setScaleType(QCPAxis::stLogarithmic); plot->yAxis->setTicker(logTicker);
    plot->xAxis->setNumberFormat("eb"); plot->xAxis->setNumberPrecision(0);
    plot->yAxis->setNumberFormat("eb"); plot->yAxis->setNumberPrecision(0);

    QFont labelFont("Microsoft YaHei", 10, QFont::Bold);
    QFont tickFont("Microsoft YaHei", 9);
    plot->xAxis->setLabel("测试时间 t (h)");
    plot->yAxis->setLabel("压力差 \xce\x94p & 压力导数 d(\xce\x94p)/d(ln t) (MPa)");
    plot->xAxis->setLabelFont(labelFont); plot->yAxis->setLabelFont(labelFont);
    plot->xAxis->setTickLabelFont(tickFont); plot->yAxis->setTickLabelFont(tickFont);

    plot->xAxis2->setVisible(true); plot->yAxis2->setVisible(true);
    plot->xAxis2->setTickLabels(false); plot->yAxis2->setTickLabels(false);
    connect(plot->xAxis, SIGNAL(rangeChanged(QCPRange)), plot->xAxis2, SLOT(setRange(QCPRange)));
    connect(plot->yAxis, SIGNAL(rangeChanged(QCPRange)), plot->yAxis2, SLOT(setRange(QCPRange)));
    plot->xAxis2->setScaleType(QCPAxis::stLogarithmic); plot->yAxis2->setScaleType(QCPAxis::stLogarithmic);
    plot->xAxis2->setTicker(logTicker); plot->yAxis2->setTicker(logTicker);

    plot->xAxis->grid()->setVisible(true); plot->yAxis->grid()->setVisible(true);
    plot->xAxis->grid()->setSubGridVisible(true); plot->yAxis->grid()->setSubGridVisible(true);
    plot->xAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    plot->yAxis->grid()->setPen(QPen(QColor(220, 220, 220), 1, Qt::SolidLine));
    plot->xAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));
    plot->yAxis->grid()->setSubGridPen(QPen(QColor(240, 240, 240), 1, Qt::DotLine));

    plot->xAxis->setRange(1e-3, 1e4); plot->yAxis->setRange(1e-3, 1e2);

    plot->legend->setVisible(true);
    plot->legend->setFont(QFont("Microsoft YaHei", 9));
    plot->legend->setBrush(QBrush(QColor(255, 255, 255, 200)));

    ui->chartWidget->setTitle("多段压裂水平井有因次压力双对数响应曲线");
}

void WT_ModelWidget::setupConnections() {
    connect(ui->calculateButton, &QPushButton::clicked, this, &WT_ModelWidget::onCalculateClicked);
    connect(ui->resetButton, &QPushButton::clicked, this, &WT_ModelWidget::onResetParameters);
    connect(ui->chartWidget, &ChartWidget::exportDataTriggered, this, &WT_ModelWidget::onExportData);
    connect(ui->btnExportDataTab, &QPushButton::clicked, this, &WT_ModelWidget::onExportData);
    connect(ui->checkShowPoints, &QCheckBox::toggled, this, &WT_ModelWidget::onShowPointsToggled);
    connect(ui->btnSelectModel, &QPushButton::clicked, this, &WT_ModelWidget::requestModelSelection);

    if(ui->kmEdit && ui->eta12Edit) {
        connect(ui->kmEdit, &QLineEdit::textChanged, this, [this](const QString &text) {
            bool ok;
            double m12 = text.toDouble(&ok);
            if(ok && m12 > 0) {
                ui->eta12Edit->setText(QString::number(1.0 / m12, 'g', 6));
            }
        });
    }
}

QVector<double> WT_ModelWidget::parseInput(const QString& text) {
    QVector<double> values;
    QString cleanText = text;
    cleanText.replace("，", ",");
    QStringList parts = cleanText.split(",", Qt::SkipEmptyParts);
    for(const QString& part : parts) {
        bool ok;
        double v = part.trimmed().toDouble(&ok);
        if(ok) values.append(v);
    }
    if(values.isEmpty()) values.append(0.0);
    return values;
}

void WT_ModelWidget::setInputText(QLineEdit* edit, double value) {
    if(!edit) return;
    edit->setText(QString::number(value, 'g', 8));
}

void WT_ModelWidget::onResetParameters() {
    // 【强制修复】完全抛弃外部默认值，100% 同步为 MATLAB 代码里的默认参数组合
    setInputText(ui->kfEdit, 50.0); // [mD]
    setInputText(ui->kmEdit, 5.0);  // [M12]
    if(ui->eta12Edit) ui->eta12Edit->setText(QString::number(1.0 / 5.0, 'g', 6));

    setInputText(ui->LEdit, 1000.0); // 水平井总长 [m]
    setInputText(ui->LfEdit, 50.0);  // [m]
    setInputText(ui->rmDEdit, 1500.0); // [m]
    setInputText(ui->omga1Edit, 0.4);
    setInputText(ui->omga2Edit, 0.08);
    setInputText(ui->remda1Edit, 1e-3);
    setInputText(ui->remda2Edit, 1e-4);

    if (ui->reDEdit->isVisible()) setInputText(ui->reDEdit, 20000.0); // [m]
    if (ui->cDEdit->isVisible()) {
        setInputText(ui->cDEdit, 1e-4); // [m3/MPa]
        setInputText(ui->sEdit, 1.0);
    }

    // 物性基准参数
    setInputText(ui->phiEdit, 0.05);
    setInputText(ui->hEdit, 20.0);
    setInputText(ui->rwEdit, 0.1);
    setInputText(ui->muEdit, 0.5);
    setInputText(ui->BEdit, 1.2);
    setInputText(ui->CtEdit, 5e-4);
    setInputText(ui->qEdit, 50.0);
    setInputText(ui->nfEdit, 9.0);

    // 效应系数
    setInputText(ui->gamaDEdit, 0.02); // 压敏系数默认恢复 0.02
    setInputText(ui->alphaEdit, 0.1);
    setInputText(ui->cPhiEdit, 1e-4);

    // 渲染参数
    setInputText(ui->tEdit, 10000.0);
    setInputText(ui->pointsEdit, 100);
}

void WT_ModelWidget::onDependentParamsChanged() {}

void WT_ModelWidget::onShowPointsToggled(bool checked) {
    MouseZoom* plot = ui->chartWidget->getPlot();
    for(int i = 0; i < plot->graphCount(); ++i) {
        if (checked) plot->graph(i)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 5));
        else plot->graph(i)->setScatterStyle(QCPScatterStyle::ssNone);
    }
    plot->replot();
}

void WT_ModelWidget::onCalculateClicked() {
    ui->calculateButton->setEnabled(false);
    ui->calculateButton->setText("计算中...");
    QCoreApplication::processEvents();
    runCalculation();
    ui->calculateButton->setEnabled(true);
    ui->calculateButton->setText("开始计算");
}

void WT_ModelWidget::runCalculation() {
    MouseZoom* plot = ui->chartWidget->getPlot();
    plot->clearGraphs();

    QMap<QString, QVector<double>> rawParams;

    rawParams["phi"] = parseInput(ui->phiEdit->text());
    rawParams["h"] = parseInput(ui->hEdit->text());
    rawParams["rw"] = parseInput(ui->rwEdit->text());
    rawParams["mu"] = parseInput(ui->muEdit->text());
    rawParams["B"] = parseInput(ui->BEdit->text());
    rawParams["Ct"] = parseInput(ui->CtEdit->text());
    rawParams["q"] = parseInput(ui->qEdit->text());
    rawParams["t"] = parseInput(ui->tEdit->text());

    rawParams["kf"] = parseInput(ui->kfEdit->text());
    rawParams["M12"] = parseInput(ui->kmEdit->text());

    rawParams["L"] = parseInput(ui->LEdit->text());
    rawParams["Lf"] = parseInput(ui->LfEdit->text());
    rawParams["nf"] = parseInput(ui->nfEdit->text());
    rawParams["rm"] = parseInput(ui->rmDEdit->text());

    rawParams["omega1"] = parseInput(ui->omga1Edit->text());
    rawParams["omega2"] = parseInput(ui->omga2Edit->text());
    rawParams["lambda1"] = parseInput(ui->remda1Edit->text());
    rawParams["lambda2"] = parseInput(ui->remda2Edit->text());
    rawParams["gamaD"] = parseInput(ui->gamaDEdit->text());

    if (ui->alphaEdit->isVisible()) rawParams["alpha"] = parseInput(ui->alphaEdit->text());
    else rawParams["alpha"] = {0.1};

    if (ui->cPhiEdit->isVisible()) rawParams["C_phi"] = parseInput(ui->cPhiEdit->text());
    else rawParams["C_phi"] = {1e-4};

    if (ui->reDEdit->isVisible()) rawParams["re"] = parseInput(ui->reDEdit->text());
    else rawParams["re"] = {20000.0};

    if (ui->cDEdit->isVisible()) {
        rawParams["C"] = parseInput(ui->cDEdit->text());
        rawParams["S"] = parseInput(ui->sEdit->text());
    } else {
        rawParams["C"] = {0.0};
        rawParams["S"] = {0.0};
    }

    QString sensitivityKey = "";
    QVector<double> sensitivityValues;
    for(auto it = rawParams.begin(); it != rawParams.end(); ++it) {
        if(it.key() == "t") continue;
        if(it.value().size() > 1) {
            sensitivityKey = it.key();
            sensitivityValues = it.value();
            break;
        }
    }
    bool isSensitivity = !sensitivityKey.isEmpty();

    QMap<QString, double> baseParams;
    for(auto it = rawParams.begin(); it != rawParams.end(); ++it) {
        baseParams[it.key()] = it.value().isEmpty() ? 0.0 : it.value().first();
    }

    int nPoints = ui->pointsEdit->text().toInt();
    if(nPoints < 5) nPoints = 5;
    double maxTime = baseParams.value("t", 10000.0);
    QVector<double> t = ModelManager::generateLogTimeSteps(nPoints, -3.0, log10(maxTime));

    int iterations = isSensitivity ? sensitivityValues.size() : 1;
    iterations = qMin(iterations, (int)m_colorList.size());

    QString resultTextHeader = QString("计算完成 (%1)\n").arg(getModelName());
    if(isSensitivity) resultTextHeader += QString("敏感性参数: %1\n").arg(sensitivityKey);

    for(int i = 0; i < iterations; ++i) {
        QMap<QString, double> currentParams = baseParams;
        double val = 0;
        if (isSensitivity) {
            val = sensitivityValues[i];
            currentParams[sensitivityKey] = val;
        }

        ModelCurveData res = calculateTheoreticalCurve(currentParams, t);
        res_tD = std::get<0>(res);
        res_pD = std::get<1>(res);
        res_dpD = std::get<2>(res);

        QColor curveColor = isSensitivity ? m_colorList[i] : Qt::red;
        QString legendName = isSensitivity ? QString("%1 = %2").arg(sensitivityKey).arg(val) : "压力差 \xce\x94p";
        plotCurve(res, legendName, curveColor, isSensitivity);
    }

    QString resultText = resultTextHeader;
    resultText += "t (h)\t\tDp (MPa)\t\tdDp (MPa)\n";
    for(int i=0; i<res_pD.size(); ++i) {
        resultText += QString("%1\t%2\t%3\n").arg(res_tD[i],0,'e',4).arg(res_pD[i],0,'e',4).arg(res_dpD[i],0,'e',4);
    }
    ui->resultTextEdit->setText(resultText);

    ui->chartWidget->getPlot()->rescaleAxes();
    plot->replot();
    onShowPointsToggled(ui->checkShowPoints->isChecked());
    emit calculationCompleted(getModelName(), baseParams);
}

void WT_ModelWidget::plotCurve(const ModelCurveData& data, const QString& name, QColor color, bool isSensitivity) {
    MouseZoom* plot = ui->chartWidget->getPlot();
    const QVector<double>& t = std::get<0>(data);
    const QVector<double>& p = std::get<1>(data);
    const QVector<double>& d = std::get<2>(data);

    QCPGraph* graphP = plot->addGraph();
    graphP->setData(t, p);
    graphP->setPen(QPen(color, 2, Qt::SolidLine));

    QCPGraph* graphD = plot->addGraph();
    graphD->setData(t, d);

    if (isSensitivity) {
        graphD->setPen(QPen(color, 2, Qt::DashLine));
        graphP->setName(name);
        graphD->removeFromLegend();
    } else {
        graphP->setPen(QPen(color, 2, Qt::SolidLine));
        graphP->setName(name);
        graphD->setPen(QPen(Qt::blue, 2, Qt::SolidLine));
        graphD->setName("压力导数 d(\xce\x94p)/d(ln t)");
    }
}

void WT_ModelWidget::onExportData() {
    if (res_tD.isEmpty()) return;
    QString defaultDir = ModelParameter::instance()->getProjectPath();
    if(defaultDir.isEmpty()) defaultDir = ".";
    QString path = QFileDialog::getSaveFileName(this, "导出CSV数据", defaultDir + "/CalculatedData.csv", "CSV Files (*.csv)");
    if (path.isEmpty()) return;
    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << "t,Dp,dDp\n";
        for (int i = 0; i < res_tD.size(); ++i) {
            double dp = (i < res_dpD.size()) ? res_dpD[i] : 0.0;
            out << res_tD[i] << "," << res_pD[i] << "," << dp << "\n";
        }
        f.close();
        QMessageBox::information(this, "导出成功", "数据文件已保存");
    }
}

