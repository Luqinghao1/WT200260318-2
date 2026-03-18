/*
 * 文件名: datasamplingdialog.cpp
 * 文件作用: 数据滤波与取样设置对话框实现文件
 * 功能描述:
 * 1. 扁平化美化界面：操作按钮采用绿、蓝、红主题色，组标题加粗纯蓝。
 * 2. 优化交互细节：滤波模式与参数输入框实现禁用/启用联动。
 * 3. 优化图表展示：图例固定左上角，散点图颜色上下区分（红 vs 深蓝），对比更直观。
 * 4. 边界自适应优化：增加向 10 的次幂智能对齐与视觉留白算法。
 * 5. [优化显示] 对数坐标系的刻度标签采用常规数据格式（"g"），仅保留有效数字，更符合现场直观读图习惯。
 */

#include "datasamplingdialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QComboBox>
#include <QSpinBox>
#include <QPushButton>
#include <QMessageBox>
#include <QMouseEvent>
#include <algorithm>
#include <cmath>

DataSamplingDialog::DataSamplingDialog(QStandardItemModel* model, QWidget *parent)
    : QDialog(parent), m_model(model), m_dragLineIndex(0)
{
    this->setWindowTitle("数据滤波与多阶段抽样设置");
    this->resize(1300, 800);

    initUI();
    setupConnections();
    applyStyle();

    if (m_model && m_model->columnCount() > 0) {
        for(int i = 0; i < m_model->columnCount(); ++i) {
            QString name = m_model->headerData(i, Qt::Horizontal).toString();
            if(name.isEmpty()) name = QString("列 %1").arg(i+1);
            m_headers.append(name);
            comboX->addItem(name, i);
            comboY->addItem(name, i);
        }
        if (m_model->columnCount() >= 2) {
            comboX->setCurrentIndex(0);
            comboY->setCurrentIndex(1);
        }
        loadRawData();
        updateSplitLines();
    }
}

DataSamplingDialog::~DataSamplingDialog() {}

void DataSamplingDialog::initUI()
{
    QHBoxLayout* mainLayout = new QHBoxLayout(this);
    mainLayout->setContentsMargins(20, 20, 20, 20);
    mainLayout->setSpacing(20);

    // ================== 左侧：参数设置区 ==================
    QVBoxLayout* leftLayout = new QVBoxLayout();
    leftLayout->setSpacing(15);

    // 1. 数据源与坐标系
    QGroupBox* groupData = new QGroupBox("1. 数据源与坐标系");
    QVBoxLayout* vData = new QVBoxLayout(groupData);
    QHBoxLayout* hX = new QHBoxLayout();
    hX->addWidget(new QLabel("X 轴数据:"));
    comboX = new QComboBox(); chkLogX = new QCheckBox("X轴对数");
    hX->addWidget(comboX, 1); hX->addWidget(chkLogX);

    QHBoxLayout* hY = new QHBoxLayout();
    hY->addWidget(new QLabel("Y 轴数据:"));
    comboY = new QComboBox(); chkLogY = new QCheckBox("Y轴对数");
    hY->addWidget(comboY, 1); hY->addWidget(chkLogY);
    vData->addLayout(hX); vData->addLayout(hY);
    leftLayout->addWidget(groupData);

    // 2. 全局异常点
    QGroupBox* groupGlobal = new QGroupBox("2. 全局异常点剔除 (Y轴)");
    QVBoxLayout* vGlobal = new QVBoxLayout(groupGlobal);
    QHBoxLayout* hMin = new QHBoxLayout();
    hMin->addWidget(new QLabel("下限:"));
    spinMinY = new QDoubleSpinBox(); spinMinY->setRange(-1e9, 1e9); spinMinY->setValue(0.0);
    hMin->addWidget(spinMinY);

    QHBoxLayout* hMax = new QHBoxLayout();
    hMax->addWidget(new QLabel("上限:"));
    spinMaxY = new QDoubleSpinBox(); spinMaxY->setRange(-1e9, 1e9); spinMaxY->setValue(100.0);
    hMax->addWidget(spinMaxY);
    vGlobal->addLayout(hMin); vGlobal->addLayout(hMax);
    leftLayout->addWidget(groupGlobal);

    // 3. 区间划分
    QGroupBox* groupSplit = new QGroupBox("3. 区间划分 (支持图上拖拽调整)");
    QVBoxLayout* vSplit = new QVBoxLayout(groupSplit);
    QHBoxLayout* hS1 = new QHBoxLayout();
    hS1->addWidget(new QLabel("早期-中期边界(X):"));
    spinSplit1 = new QDoubleSpinBox(); spinSplit1->setRange(-1e9, 1e9); spinSplit1->setDecimals(4);
    hS1->addWidget(spinSplit1);
    QHBoxLayout* hS2 = new QHBoxLayout();
    hS2->addWidget(new QLabel("中期-晚期边界(X):"));
    spinSplit2 = new QDoubleSpinBox(); spinSplit2->setRange(-1e9, 1e9); spinSplit2->setDecimals(4);
    hS2->addWidget(spinSplit2);
    vSplit->addLayout(hS1); vSplit->addLayout(hS2);
    leftLayout->addWidget(groupSplit);

    // 4. 分阶段策略
    QGroupBox* groupStage = new QGroupBox("4. 分阶段处理策略");
    QHBoxLayout* hStage = new QHBoxLayout(groupStage);
    hStage->setSpacing(10);
    hStage->setContentsMargins(0, 15, 0, 0);

    QStringList stageNames = {"早期阶段", "中期阶段", "晚期阶段"};
    for(int i=0; i<3; ++i) {
        QWidget* box = new QWidget();
        QVBoxLayout* l = new QVBoxLayout(box);
        l->setContentsMargins(5, 5, 5, 5);
        l->setSpacing(8);

        QLabel* titleLabel = new QLabel(stageNames[i]);
        titleLabel->setStyleSheet("font-weight: bold; color: #555555;");
        l->addWidget(titleLabel, 0, Qt::AlignCenter);

        // 滤波设置
        m_stages[i].comboFilter = new QComboBox();
        m_stages[i].comboFilter->addItems({"无滤波", "中值滤波"});
        m_stages[i].comboFilter->setCurrentIndex(0); // 默认无滤波
        l->addWidget(new QLabel("滤波:")); l->addWidget(m_stages[i].comboFilter);

        QHBoxLayout* hl1 = new QHBoxLayout();
        hl1->addWidget(new QLabel("参考点数:"));
        m_stages[i].spinFilterWin = new QSpinBox();
        m_stages[i].spinFilterWin->setRange(3, 101); m_stages[i].spinFilterWin->setSingleStep(2);
        m_stages[i].spinFilterWin->setValue(5);
        m_stages[i].spinFilterWin->setEnabled(false); // 默认禁用
        hl1->addWidget(m_stages[i].spinFilterWin);
        l->addLayout(hl1);

        // 抽样设置
        m_stages[i].comboSample = new QComboBox();
        m_stages[i].comboSample->addItems({"对数抽样", "间隔抽样", "总点数"});
        m_stages[i].comboSample->setCurrentIndex(0); // 默认全部对数抽样
        l->addWidget(new QLabel("抽样:")); l->addWidget(m_stages[i].comboSample);

        QHBoxLayout* hl2 = new QHBoxLayout();
        hl2->addWidget(new QLabel("参数N:"));
        m_stages[i].spinSampleVal = new QSpinBox();
        m_stages[i].spinSampleVal->setRange(1, 100000);
        m_stages[i].spinSampleVal->setValue(20);
        hl2->addWidget(m_stages[i].spinSampleVal);
        l->addLayout(hl2);

        hStage->addWidget(box);

        connect(m_stages[i].comboFilter, QOverload<int>::of(&QComboBox::currentIndexChanged),
                this, [this, i](int index){
                    m_stages[i].spinFilterWin->setEnabled(index == 1);
                });
    }
    leftLayout->addWidget(groupStage);

    leftLayout->addStretch();
    QHBoxLayout* opLayout = new QHBoxLayout();
    btnPreview = new QPushButton("应用预览 >>");
    btnPreview->setObjectName("btnPreview");

    btnReset = new QPushButton("重置数据");
    btnReset->setObjectName("btnReset");

    opLayout->addWidget(btnPreview); opLayout->addWidget(btnReset);
    leftLayout->addLayout(opLayout);

    // ================== 右侧：双图表区 ==================
    QVBoxLayout* rightLayout = new QVBoxLayout();
    rightLayout->setSpacing(15);

    customPlotRaw = new QCustomPlot();
    customPlotRaw->plotLayout()->insertRow(0);
    customPlotRaw->plotLayout()->addElement(0, 0, new QCPTextElement(customPlotRaw, "抽样前 (原始数据与阶段划分)", QFont("Microsoft YaHei", 12, QFont::Bold)));
    customPlotRaw->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignLeft);

    customPlotProcessed = new QCustomPlot();
    customPlotProcessed->plotLayout()->insertRow(0);
    customPlotProcessed->plotLayout()->addElement(0, 0, new QCPTextElement(customPlotProcessed, "抽样后 (处理后数据)", QFont("Microsoft YaHei", 12, QFont::Bold)));
    customPlotProcessed->axisRect()->insetLayout()->setInsetAlignment(0, Qt::AlignTop | Qt::AlignLeft);

    rightLayout->addWidget(customPlotRaw, 1);
    rightLayout->addWidget(customPlotProcessed, 1);

    QHBoxLayout* bottomLayout = new QHBoxLayout();
    bottomLayout->addStretch();
    btnOk = new QPushButton("确定并生成新文件");
    btnOk->setObjectName("btnOk");
    btnOk->setDefault(true);

    btnCancel = new QPushButton("取消");
    btnCancel->setObjectName("btnCancel");

    bottomLayout->addWidget(btnOk); bottomLayout->addWidget(btnCancel);
    rightLayout->addLayout(bottomLayout);

    mainLayout->addLayout(leftLayout, 4);
    mainLayout->addLayout(rightLayout, 5);

    vLine1 = new QCPItemStraightLine(customPlotRaw);
    vLine2 = new QCPItemStraightLine(customPlotRaw);
    vLine1->setPen(QPen(QColor(24, 144, 255), 2, Qt::DashLine));
    vLine2->setPen(QPen(QColor(82, 196, 26), 2, Qt::DashLine));

    vLine1_proc = new QCPItemStraightLine(customPlotProcessed);
    vLine2_proc = new QCPItemStraightLine(customPlotProcessed);
    vLine1_proc->setPen(QPen(QColor(24, 144, 255), 2, Qt::DotLine));
    vLine2_proc->setPen(QPen(QColor(82, 196, 26), 2, Qt::DotLine));
}

void DataSamplingDialog::setupConnections()
{
    connect(comboX, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &DataSamplingDialog::onDataColumnChanged);
    connect(comboY, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &DataSamplingDialog::onDataColumnChanged);

    connect(chkLogX, &QCheckBox::stateChanged, this, &DataSamplingDialog::onAxisScaleChanged);
    connect(chkLogY, &QCheckBox::stateChanged, this, &DataSamplingDialog::onAxisScaleChanged);

    connect(spinSplit1, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &DataSamplingDialog::onSplitBoxValueChanged);
    connect(spinSplit2, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &DataSamplingDialog::onSplitBoxValueChanged);

    connect(btnPreview, &QPushButton::clicked, this, &DataSamplingDialog::onPreviewClicked);
    connect(btnReset, &QPushButton::clicked, this, &DataSamplingDialog::onResetClicked);
    connect(btnOk, &QPushButton::clicked, this, &QDialog::accept);
    connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);

    connect(customPlotRaw, &QCustomPlot::mousePress, this, &DataSamplingDialog::onRawPlotMousePress);
    connect(customPlotRaw, &QCustomPlot::mouseMove, this, &DataSamplingDialog::onRawPlotMouseMove);
    connect(customPlotRaw, &QCustomPlot::mouseRelease, this, &DataSamplingDialog::onRawPlotMouseRelease);
}

void DataSamplingDialog::applyStyle()
{
    QString qss = "QWidget { color: #333333; background-color: #FFFFFF; font-family: 'Microsoft YaHei'; font-size: 13px; }"
                  "QGroupBox { border: none; border-top: 1px solid #E5E5E5; margin-top: 15px; padding-top: 15px; }"
                  "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 0px; padding-top: 0px; font-weight: bold; font-size: 14px; color: #1890FF; }"

                  "QPushButton { background-color: #FFFFFF; color: #333333; border: 1px solid #D9D9D9; border-radius: 4px; padding: 6px 15px; min-width: 80px; }"
                  "QPushButton:hover { color: #1890FF; border-color: #1890FF; }"
                  "QPushButton:pressed { background-color: #F5F5F5; }"

                  "QPushButton#btnPreview { background-color: #52C41A; color: white; border: none; font-weight: bold; }"
                  "QPushButton#btnPreview:hover { background-color: #73D13D; }"

                  "QPushButton#btnReset { background-color: #1890FF; color: white; border: none; font-weight: bold; }"
                  "QPushButton#btnReset:hover { background-color: #40A9FF; }"

                  "QPushButton#btnOk { background-color: #1890FF; color: white; border: none; font-weight: bold; }"
                  "QPushButton#btnOk:hover { background-color: #40A9FF; }"

                  "QPushButton#btnCancel { background-color: #FF4D4F; color: white; border: none; font-weight: bold; }"
                  "QPushButton#btnCancel:hover { background-color: #FF7875; }"

                  "QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #D9D9D9; border-radius: 4px; padding: 2px 5px; min-height: 24px; }"
                  "QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #1890FF; }"
                  "QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { background-color: #F5F5F5; color: #BFBFBF; border-color: #D9D9D9; }";
    this->setStyleSheet(qss);
}

void DataSamplingDialog::loadRawData()
{
    if (!m_model) return;

    m_rawTable.clear(); m_rawX.clear(); m_rawY.clear();
    int colX = comboX->currentData().toInt();
    int colY = comboY->currentData().toInt();
    m_xName = comboX->currentText();
    m_yName = comboY->currentText();

    double maxY = -1e9;

    for(int r = 0; r < m_model->rowCount(); ++r) {
        QStringList rowData;
        for(int c = 0; c < m_model->columnCount(); ++c) {
            rowData.append(m_model->index(r, c).data().toString());
        }

        bool okX, okY;
        double x = m_model->index(r, colX).data().toDouble(&okX);
        double y = m_model->index(r, colY).data().toDouble(&okY);

        if(okX && okY) {
            m_rawTable.append(rowData);
            m_rawX.append(x);
            m_rawY.append(y);
            if (y > maxY) maxY = y;
        }
    }

    if (maxY > -1e8) {
        spinMaxY->blockSignals(true);
        spinMaxY->setValue(maxY);
        spinMaxY->blockSignals(false);
    }

    m_processedTable = m_rawTable;
    m_processedX = m_rawX;
    m_processedY = m_rawY;

    drawRawPlot();
    drawProcessedPlot();
}

void DataSamplingDialog::onDataColumnChanged() {
    loadRawData();
    updateSplitLines();
}

// 核心优化：不论是否对数坐标系，数字显示均采用“g”(自动剔除无用零，保留有效数字)格式
void DataSamplingDialog::onAxisScaleChanged() {
    // 切换 X 轴
    if (chkLogX->isChecked()) {
        customPlotRaw->xAxis->setScaleType(QCPAxis::stLogarithmic);
        QSharedPointer<QCPAxisTickerLog> logTickerX1(new QCPAxisTickerLog);
        customPlotRaw->xAxis->setTicker(logTickerX1);
        customPlotRaw->xAxis->setNumberFormat("g"); // 修改为常规数字显示
        customPlotRaw->xAxis->setNumberPrecision(6);

        customPlotProcessed->xAxis->setScaleType(QCPAxis::stLogarithmic);
        QSharedPointer<QCPAxisTickerLog> logTickerX2(new QCPAxisTickerLog);
        customPlotProcessed->xAxis->setTicker(logTickerX2);
        customPlotProcessed->xAxis->setNumberFormat("g"); // 修改为常规数字显示
        customPlotProcessed->xAxis->setNumberPrecision(6);
    } else {
        customPlotRaw->xAxis->setScaleType(QCPAxis::stLinear);
        QSharedPointer<QCPAxisTicker> linTickerX1(new QCPAxisTicker);
        customPlotRaw->xAxis->setTicker(linTickerX1);
        customPlotRaw->xAxis->setNumberFormat("g");
        customPlotRaw->xAxis->setNumberPrecision(6);

        customPlotProcessed->xAxis->setScaleType(QCPAxis::stLinear);
        QSharedPointer<QCPAxisTicker> linTickerX2(new QCPAxisTicker);
        customPlotProcessed->xAxis->setTicker(linTickerX2);
        customPlotProcessed->xAxis->setNumberFormat("g");
        customPlotProcessed->xAxis->setNumberPrecision(6);
    }

    // 切换 Y 轴
    if (chkLogY->isChecked()) {
        customPlotRaw->yAxis->setScaleType(QCPAxis::stLogarithmic);
        QSharedPointer<QCPAxisTickerLog> logTickerY1(new QCPAxisTickerLog);
        customPlotRaw->yAxis->setTicker(logTickerY1);
        customPlotRaw->yAxis->setNumberFormat("g"); // 修改为常规数字显示
        customPlotRaw->yAxis->setNumberPrecision(6);

        customPlotProcessed->yAxis->setScaleType(QCPAxis::stLogarithmic);
        QSharedPointer<QCPAxisTickerLog> logTickerY2(new QCPAxisTickerLog);
        customPlotProcessed->yAxis->setTicker(logTickerY2);
        customPlotProcessed->yAxis->setNumberFormat("g"); // 修改为常规数字显示
        customPlotProcessed->yAxis->setNumberPrecision(6);
    } else {
        customPlotRaw->yAxis->setScaleType(QCPAxis::stLinear);
        QSharedPointer<QCPAxisTicker> linTickerY1(new QCPAxisTicker);
        customPlotRaw->yAxis->setTicker(linTickerY1);
        customPlotRaw->yAxis->setNumberFormat("g");
        customPlotRaw->yAxis->setNumberPrecision(6);

        customPlotProcessed->yAxis->setScaleType(QCPAxis::stLinear);
        QSharedPointer<QCPAxisTicker> linTickerY2(new QCPAxisTicker);
        customPlotProcessed->yAxis->setTicker(linTickerY2);
        customPlotProcessed->yAxis->setNumberFormat("g");
        customPlotProcessed->yAxis->setNumberPrecision(6);
    }

    drawRawPlot();
    drawProcessedPlot();
}

void DataSamplingDialog::updateSplitLines() {
    if(m_rawX.isEmpty()) return;
    double minX = *std::min_element(m_rawX.begin(), m_rawX.end());
    double maxX = *std::max_element(m_rawX.begin(), m_rawX.end());
    if (minX == maxX) maxX += 1.0;

    double s1 = minX + (maxX - minX) * 0.1;
    double s2 = minX + (maxX - minX) * 0.5;

    spinSplit1->blockSignals(true); spinSplit2->blockSignals(true);
    spinSplit1->setValue(s1);       spinSplit2->setValue(s2);
    spinSplit1->blockSignals(false);spinSplit2->blockSignals(false);

    onSplitBoxValueChanged();
}

void DataSamplingDialog::onSplitBoxValueChanged() {
    double s1 = spinSplit1->value();
    double s2 = spinSplit2->value();
    if (s1 > s2) { s1 = s2 - 1e-4; spinSplit1->setValue(s1); }

    vLine1->point1->setCoords(s1, 0); vLine1->point2->setCoords(s1, 1);
    vLine2->point1->setCoords(s2, 0); vLine2->point2->setCoords(s2, 1);

    vLine1_proc->point1->setCoords(s1, 0); vLine1_proc->point2->setCoords(s1, 1);
    vLine2_proc->point1->setCoords(s2, 0); vLine2_proc->point2->setCoords(s2, 1);

    customPlotRaw->replot();
    customPlotProcessed->replot();
}

void DataSamplingDialog::onRawPlotMousePress(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        double x = customPlotRaw->xAxis->pixelToCoord(event->pos().x());
        double s1 = vLine1->point1->key();
        double s2 = vLine2->point1->key();
        double range = customPlotRaw->xAxis->range().size();
        double tol = range * 0.05;

        if (std::abs(x - s1) < tol && std::abs(x - s1) <= std::abs(x - s2)) m_dragLineIndex = 1;
        else if (std::abs(x - s2) < tol) m_dragLineIndex = 2;
        else m_dragLineIndex = 0;
    }
}

void DataSamplingDialog::onRawPlotMouseMove(QMouseEvent *event) {
    if (m_dragLineIndex != 0) {
        double x = customPlotRaw->xAxis->pixelToCoord(event->pos().x());
        if (m_dragLineIndex == 1) {
            double s2 = spinSplit2->value();
            if (x >= s2) x = s2 - 1e-4;
            spinSplit1->setValue(x);
        } else if (m_dragLineIndex == 2) {
            double s1 = spinSplit1->value();
            if (x <= s1) x = s1 + 1e-4;
            spinSplit2->setValue(x);
        }
    }
}

void DataSamplingDialog::onRawPlotMouseRelease(QMouseEvent *) { m_dragLineIndex = 0; }

void adjustLogAxisRange(QCPAxis* axis, const QVector<double>& data) {
    double minPos = 1e-3, maxPos = 1e-3;
    bool found = false;
    for (double v : data) {
        if (v > 0) {
            if (!found) { minPos = v; maxPos = v; found = true; }
            else { if (v < minPos) minPos = v; if (v > maxPos) maxPos = v; }
        }
    }

    if (found && maxPos > minPos) {
        double minLog = std::log10(minPos);
        double maxLog = std::log10(maxPos);

        double lowerLog = std::floor(minLog);
        double upperLog = std::ceil(maxLog);

        if (minLog - lowerLog < 0.15) lowerLog -= 1.0;
        if (upperLog - maxLog < 0.15) upperLog += 1.0;

        if (upperLog <= lowerLog + 1.0) {
            lowerLog -= 1.0;
            upperLog += 1.0;
        }

        axis->setRange(std::pow(10, lowerLog), std::pow(10, upperLog));
    } else if (found && maxPos == minPos) {
        double logV = std::floor(std::log10(minPos));
        axis->setRange(std::pow(10, logV - 1), std::pow(10, logV + 1));
    } else {
        axis->setRange(1e-3, 1e3);
    }
}

void DataSamplingDialog::drawRawPlot() {
    customPlotRaw->clearGraphs();
    customPlotRaw->addGraph();
    customPlotRaw->graph(0)->setData(m_rawX, m_rawY);

    QPen redPen(Qt::red);
    redPen.setWidthF(1.0);
    customPlotRaw->graph(0)->setPen(redPen);
    customPlotRaw->graph(0)->setLineStyle(QCPGraph::lsLine);
    customPlotRaw->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 3));
    customPlotRaw->graph(0)->setName(QString("原始数据 (共 %1 点)").arg(m_rawX.size()));

    customPlotRaw->xAxis->setLabel(m_xName);
    customPlotRaw->yAxis->setLabel(m_yName);

    if (chkLogX->isChecked()) {
        adjustLogAxisRange(customPlotRaw->xAxis, m_rawX);
    } else {
        customPlotRaw->xAxis->rescale();
    }

    if (chkLogY->isChecked()) {
        adjustLogAxisRange(customPlotRaw->yAxis, m_rawY);
    } else {
        customPlotRaw->yAxis->rescale();
    }

    customPlotRaw->legend->setVisible(true);
    customPlotRaw->replot();
}

void DataSamplingDialog::drawProcessedPlot() {
    customPlotProcessed->clearGraphs();
    customPlotProcessed->addGraph();
    customPlotProcessed->graph(0)->setData(m_processedX, m_processedY);

    QPen darkBluePen(Qt::darkBlue);
    darkBluePen.setWidthF(1.5);
    customPlotProcessed->graph(0)->setPen(darkBluePen);
    customPlotProcessed->graph(0)->setLineStyle(QCPGraph::lsLine);
    customPlotProcessed->graph(0)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, 4));
    customPlotProcessed->graph(0)->setName(QString("处理后保留数据 (共 %1 点)").arg(m_processedX.size()));

    customPlotProcessed->xAxis->setLabel(m_xName);
    customPlotProcessed->yAxis->setLabel(m_yName);

    if (chkLogX->isChecked()) {
        adjustLogAxisRange(customPlotProcessed->xAxis, m_processedX);
    } else {
        customPlotProcessed->xAxis->rescale();
    }

    if (chkLogY->isChecked()) {
        adjustLogAxisRange(customPlotProcessed->yAxis, m_processedY);
    } else {
        customPlotProcessed->yAxis->rescale();
    }

    customPlotProcessed->legend->setVisible(true);
    customPlotProcessed->replot();
}

void DataSamplingDialog::onPreviewClicked() {
    processData();
    drawProcessedPlot();
}

void DataSamplingDialog::onResetClicked() {
    m_processedTable = m_rawTable;
    m_processedX = m_rawX; m_processedY = m_rawY;
    drawProcessedPlot();
}

void DataSamplingDialog::processData()
{
    if (m_rawTable.isEmpty()) return;

    double s1 = spinSplit1->value();
    double s2 = spinSplit2->value();
    double minY = spinMinY->value();
    double maxY = spinMaxY->value();

    int colY = comboY->currentData().toInt();

    QVector<QStringList> tb[3];
    QVector<double> tx[3], ty[3];

    for (int i = 0; i < m_rawTable.size(); ++i) {
        double x = m_rawX[i];
        double y = m_rawY[i];
        if (y < minY || y > maxY) continue;

        int st = (x < s1) ? 0 : ((x < s2) ? 1 : 2);
        tb[st].append(m_rawTable[i]);
        tx[st].append(x);
        ty[st].append(y);
    }

    m_processedTable.clear(); m_processedX.clear(); m_processedY.clear();

    for (int i = 0; i < 3; ++i) {
        if (tx[i].isEmpty()) continue;

        if (m_stages[i].comboFilter->currentIndex() == 1) {
            applyMedianFilter(ty[i], m_stages[i].spinFilterWin->value());
            for(int j = 0; j < ty[i].size(); ++j) {
                tb[i][j][colY] = QString::number(ty[i][j], 'f', 6);
            }
        }

        int sType = m_stages[i].comboSample->currentIndex();
        int sVal = m_stages[i].spinSampleVal->value();
        QVector<int> indices;

        if (sType == 0) indices = getLogSamplingIndices(tx[i], sVal);
        else if (sType == 1) indices = getIntervalSamplingIndices(tx[i].size(), sVal);
        else if (sType == 2) indices = getTotalPointsSamplingIndices(tx[i].size(), sVal);

        for (int idx : indices) {
            m_processedTable.append(tb[i][idx]);
            m_processedX.append(tx[i][idx]);
            m_processedY.append(ty[i][idx]);
        }
    }
}

void DataSamplingDialog::applyMedianFilter(QVector<double>& y, int windowSize) {
    if (y.size() < windowSize) return;
    int halfWin = windowSize / 2;
    QVector<double> y_filtered = y;
    for (int i = halfWin; i < y.size() - halfWin; ++i) {
        QVector<double> winData;
        for (int j = -halfWin; j <= halfWin; ++j) winData.append(y[i + j]);
        std::sort(winData.begin(), winData.end());
        y_filtered[i] = winData[winData.size() / 2];
    }
    y = y_filtered;
}

QVector<int> DataSamplingDialog::getLogSamplingIndices(const QVector<double>& x, int pointsPerDecade) {
    QVector<int> indices;
    if (x.isEmpty() || pointsPerDecade <= 0) return indices;
    indices.append(0);

    double xMin = x.first() > 0 ? x.first() : 1e-4;
    double logStep = 1.0 / pointsPerDecade;
    double nextTargetLog = std::log10(xMin) + logStep;

    for (int i = 1; i < x.size() - 1; ++i) {
        if (x[i] > 0 && std::log10(x[i]) >= nextTargetLog) {
            indices.append(i);
            nextTargetLog += logStep;
        }
    }
    if (indices.last() != x.size() - 1) indices.append(x.size() - 1);
    return indices;
}

QVector<int> DataSamplingDialog::getIntervalSamplingIndices(int totalCount, int interval) {
    QVector<int> indices;
    if (interval <= 1) { for(int i=0; i<totalCount; ++i) indices.append(i); return indices; }
    for (int i = 0; i < totalCount; i += interval) indices.append(i);
    if (indices.last() != totalCount - 1) indices.append(totalCount - 1);
    return indices;
}

QVector<int> DataSamplingDialog::getTotalPointsSamplingIndices(int totalCount, int targetPoints) {
    QVector<int> indices;
    if (totalCount <= targetPoints || targetPoints <= 2) {
        for(int i=0; i<totalCount; ++i) indices.append(i);
        return indices;
    }
    double step = (double)(totalCount - 1) / (targetPoints - 1);
    for (int i = 0; i < targetPoints; ++i) {
        int idx = qRound(i * step);
        if (idx >= totalCount) idx = totalCount - 1;
        indices.append(idx);
    }
    return indices;
}

QVector<QStringList> DataSamplingDialog::getProcessedTable() const { return m_processedTable; }
QStringList DataSamplingDialog::getHeaders() const { return m_headers; }
