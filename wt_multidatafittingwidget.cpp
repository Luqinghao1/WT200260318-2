/**
 * @file wt_multidatafittingwidget.cpp
 * @brief 多数据试井拟合分析核心界面实现文件
 *
 * 本代码文件的作用与功能：
 * 1. 提供多数据试井拟合的具体管理逻辑及图表可视化显示。
 * 2. 负责提取并规整不同观测数据的文件名、权重及显隐状态。
 * 3. 包含智能多数据融合拟合算法（如时间网格插值和加权平均），实现实时动态图表逼近。
 * 4. 统一调用与单数据拟合一致的物理底层参数，保证理论计算的精确性与模块间一致性。
 */

#include "wt_multidatafittingwidget.h"
#include "ui_wt_multidatafittingwidget.h"
#include "fittingdatadialog.h"
#include "fittingsamplingdialog.h"
#include "modelselect.h"
#include "paramselectdialog.h"
#include "pressurederivativecalculator.h"
#include "pressurederivativecalculator1.h"
#include "modelparameter.h"

#include <QMessageBox>
#include <QDebug>
#include <cmath>
#include <algorithm>
#include <QHeaderView>
#include <QFileDialog>
#include <QFile>
#include <QFileInfo>
#include <QTextStream>
#include <QColorDialog>
#include <QMenu>
#include <QCheckBox>
#include <QHBoxLayout>
#include <QApplication>

/**
 * @brief 构造函数：初始化核心组件、UI布局以及绑定系统内部核心信号槽
 * @param parent 父级窗口指针
 */
WT_MultidataFittingWidget::WT_MultidataFittingWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WT_MultidataFittingWidget),
    m_modelManager(nullptr),
    m_core(new FittingCore(this)),
    m_chartManager(new FittingChart(this)),
    m_mdiArea(nullptr),
    m_chartLogLog(nullptr), m_chartSemiLog(nullptr), m_chartCartesian(nullptr),
    m_currentModelType(ModelManager::Model_1),
    m_isFitting(false),
    m_isUpdatingTable(false),
    m_userDefinedTimeMax(-1.0)
{
    ui->setupUi(this);

    // 清理可能存在的旧布局并设置新布局
    if (ui->plotContainer->layout()) {
        QLayoutItem* item;
        while ((item = ui->plotContainer->layout()->takeAt(0)) != nullptr) {
            delete item->widget(); delete item;
        }
        delete ui->plotContainer->layout();
    }
    QVBoxLayout* containerLayout = new QVBoxLayout(ui->plotContainer);
    containerLayout->setContentsMargins(0,0,0,0);
    containerLayout->setSpacing(0);

    // 初始化 MDI 多文档窗口区域
    m_mdiArea = new QMdiArea(this);
    m_mdiArea->setViewMode(QMdiArea::SubWindowView);
    m_mdiArea->setBackground(QBrush(QColor(240, 240, 240)));
    containerLayout->addWidget(m_mdiArea);

    // 实例化三种主要图表窗口
    m_chartLogLog = new FittingChart1(this);
    m_chartSemiLog = new FittingChart2(this);
    m_chartCartesian = new FittingChart3(this);

    m_plotLogLog = m_chartLogLog->getPlot();
    m_plotSemiLog = m_chartSemiLog->getPlot();
    m_plotCartesian = m_chartCartesian->getPlot();

    m_chartLogLog->setTitle("双对数曲线 (Log-Log)");
    m_chartSemiLog->setTitle("半对数曲线 (Semi-Log)");
    m_chartCartesian->setTitle("历史拟合曲线 (History Plot)");

    m_subWinLogLog = m_mdiArea->addSubWindow(m_chartLogLog);
    m_subWinSemiLog = m_mdiArea->addSubWindow(m_chartSemiLog);
    m_subWinCartesian = m_mdiArea->addSubWindow(m_chartCartesian);

    ui->splitter->setSizes(QList<int>() << 350 << 1000);
    ui->splitter->setCollapsible(0, false);

    // 绑定参数表控件
    m_paramChart = new FittingParameterChart(ui->tableParams, this);

    setupPlot();
    m_chartManager->initializeCharts(m_plotLogLog, m_plotSemiLog, m_plotCartesian);

    // 初始化已加载数据组展示表格的格式
    ui->tableDataGroups->setColumnCount(5);
    ui->tableDataGroups->setHorizontalHeaderLabels({"数据名称", "颜色", "压差", "压力导数", "权重"});
    ui->tableDataGroups->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    ui->tableDataGroups->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    ui->tableDataGroups->horizontalHeader()->setSectionResizeMode(2, QHeaderView::ResizeToContents);
    ui->tableDataGroups->horizontalHeader()->setSectionResizeMode(3, QHeaderView::ResizeToContents);
    ui->tableDataGroups->horizontalHeader()->setSectionResizeMode(4, QHeaderView::ResizeToContents);
    ui->tableDataGroups->setFocusPolicy(Qt::NoFocus);
    ui->tableDataGroups->setContextMenuPolicy(Qt::CustomContextMenu);

    // 连接各种信号槽机制
    connect(ui->tableDataGroups, &QTableWidget::cellChanged, this, &WT_MultidataFittingWidget::onDataWeightChanged);
    connect(ui->tableDataGroups, &QTableWidget::cellClicked, this, &WT_MultidataFittingWidget::onTableCellClicked);
    connect(ui->tableDataGroups, &QTableWidget::customContextMenuRequested, this, &WT_MultidataFittingWidget::showContextMenu);

    connect(m_paramChart, &FittingParameterChart::parameterChangedByWheel, this, &WT_MultidataFittingWidget::onParameterChangedByWheel);
    connect(ui->tableParams, &QTableWidget::itemChanged, this, &WT_MultidataFittingWidget::onParameterTableItemChanged);

    connect(m_core, &FittingCore::sigIterationUpdated, this, &WT_MultidataFittingWidget::onIterationUpdate, Qt::QueuedConnection);
    connect(m_core, &FittingCore::sigProgress, ui->progressBar, &QProgressBar::setValue);
    connect(m_core, &FittingCore::sigFitFinished, this, &WT_MultidataFittingWidget::onFitFinished);

    connect(m_chartLogLog, &FittingChart1::exportDataTriggered, this, &WT_MultidataFittingWidget::onExportCurveData);
    connect(m_chartSemiLog, &FittingChart2::exportDataTriggered, this, &WT_MultidataFittingWidget::onExportCurveData);
    connect(m_chartCartesian, &FittingChart3::exportDataTriggered, this, &WT_MultidataFittingWidget::onExportCurveData);
    connect(m_chartManager, &FittingChart::sigManualPressureUpdated, this, &WT_MultidataFittingWidget::onSemiLogLineMoved);
}

/**
 * @brief 析构函数：释放 UI 及其相关内存资源
 */
WT_MultidataFittingWidget::~WT_MultidataFittingWidget()
{
    delete ui;
}

/**
 * @brief 槽函数：保存当前多数据拟合的进度状态
 */
void WT_MultidataFittingWidget::on_btnSave_clicked()
{
    if (m_dataGroups.isEmpty()) {
        QMessageBox::information(this, "提示", "当前没有分析数据可保存。");
        return;
    }
    m_paramChart->updateParamsFromTable();
    loadProjectParams();
    QMessageBox::information(this, "保存成功", "当前多数据分析与拟合参数状态已保存。");
}

/**
 * @brief 设置底层使用的试井模型管理器
 * @param m 模型管理器指针
 */
void WT_MultidataFittingWidget::setModelManager(ModelManager *m)
{
    m_modelManager = m;
    m_paramChart->setModelManager(m);
    if (m_core) m_core->setModelManager(m);
    initializeDefaultModel();
}

/**
 * @brief 导入并设置供用户选择的项目观测数据模型
 * @param models 数据映射表
 */
void WT_MultidataFittingWidget::setProjectDataModels(const QMap<QString, QStandardItemModel *> &models)
{
    m_dataMap = models;
}

/**
 * @brief 初始化图表交互属性（设置允许拖拽、缩放及选择）
 */
void WT_MultidataFittingWidget::setupPlot() {
    m_plotLogLog->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    m_plotSemiLog->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
    m_plotCartesian->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
}

/**
 * @brief 初始化加载默认的试井模型并填充参数表格
 */
void WT_MultidataFittingWidget::initializeDefaultModel()
{
    if(!m_modelManager) return;

    m_isUpdatingTable = true;
    m_currentModelType = ModelManager::Model_1;

    // 统一样式：按钮仅显示名称
    ui->btn_modelSelect->setText(ModelManager::getModelTypeName(m_currentModelType));

    m_paramChart->resetParams(m_currentModelType, true);
    loadProjectParams();
    hideUnwantedParams();
    m_isUpdatingTable = false;
}

/**
 * @brief 调整窗口大小事件响应
 * @param event 尺寸变化事件指针
 */
void WT_MultidataFittingWidget::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    layoutCharts();
}

/**
 * @brief 窗口显示事件响应
 * @param event 显示事件指针
 */
void WT_MultidataFittingWidget::showEvent(QShowEvent *event)
{
    QWidget::showEvent(event);
    layoutCharts();
}

/**
 * @brief MDI 子窗口布局计算函数，使得图表左一右二排列分布
 */
void WT_MultidataFittingWidget::layoutCharts()
{
    if (!m_mdiArea || !m_subWinLogLog || !m_subWinSemiLog || !m_subWinCartesian) return;
    QRect rect = m_mdiArea->contentsRect();
    int w = rect.width(), h = rect.height();
    if (w <= 0 || h <= 0) return;
    m_subWinLogLog->setGeometry(0, 0, w / 2, h);
    m_subWinCartesian->setGeometry(w / 2, 0, w - (w / 2), h / 2);
    m_subWinSemiLog->setGeometry(w / 2, h / 2, w - (w / 2), h - (h / 2));
}

/**
 * @brief 槽函数：响应参数表格中通过鼠标滚轮修改参数数值的事件，实时刷新曲线
 */
void WT_MultidataFittingWidget::onParameterChangedByWheel()
{
    if (m_isUpdatingTable || m_isFitting) return;
    m_paramChart->updateParamsFromTable();
    updateModelCurve(nullptr, false, false);
}

/**
 * @brief 槽函数：响应参数表格内文本手动修改完成后的事件，重新绘制曲线
 */
void WT_MultidataFittingWidget::onParameterTableItemChanged(QTableWidgetItem *item)
{
    if (m_isUpdatingTable || m_isFitting || !item) return;
    if (item->column() >= 2 && item->column() <= 4) {
        m_paramChart->updateParamsFromTable();
        updateModelCurve(nullptr, false, false);
    }
}

/**
 * @brief 槽函数：弹出选择对话框，解析并添加新的观测数据组到拟合空间
 */
void WT_MultidataFittingWidget::on_btnAddData_clicked()
{
    if (m_dataMap.isEmpty()) {
        QMessageBox::warning(this, "提示", "未检测到项目数据源，请确保是从已加载项目的状态进入。");
    }

    FittingDataDialog dlg(m_dataMap, this);
    if (dlg.exec() != QDialog::Accepted) return;

    FittingDataSettings settings = dlg.getSettings();
    QStandardItemModel* sourceModel = dlg.getPreviewModel();

    if (!sourceModel || sourceModel->rowCount() == 0) return;

    QVector<double> rawTime, rawPressureData, finalDeriv;
    int skip = settings.skipRows;
    for (int i = skip; i < sourceModel->rowCount(); ++i) {
        QStandardItem* itemT = sourceModel->item(i, settings.timeColIndex);
        QStandardItem* itemP = sourceModel->item(i, settings.pressureColIndex);
        if (itemT && itemP) {
            double t = itemT->text().toDouble();
            double p = itemP->text().toDouble();
            if (t > 0) {
                rawTime.append(t);
                rawPressureData.append(p);
                if (settings.derivColIndex >= 0) {
                    QStandardItem* itemD = sourceModel->item(i, settings.derivColIndex);
                    finalDeriv.append(itemD ? itemD->text().toDouble() : 0.0);
                }
            }
        }
    }

    QVector<double> finalDeltaP;
    double p_shutin = rawPressureData.first();
    for (double p : rawPressureData) {
        finalDeltaP.append(settings.testType == Test_Drawdown ? std::abs(settings.initialPressure - p) : std::abs(p - p_shutin));
    }

    if (settings.derivColIndex == -1) {
        finalDeriv = PressureDerivativeCalculator::calculateBourdetDerivative(rawTime, finalDeltaP, settings.lSpacing);
    }
    if (settings.enableSmoothing) {
        finalDeriv = PressureDerivativeCalculator1::smoothData(finalDeriv, settings.smoothingSpan);
    }

    QString defaultName;
    // 【修改点】只保留纯净的文件名作为数据组名称，去除 .txt、.csv 等后缀，满足 UI 清洁度要求
    if (settings.isFromProject) defaultName = QFileInfo(settings.projectFileName).completeBaseName();
    else defaultName = QFileInfo(settings.filePath).completeBaseName();

    if (defaultName.isEmpty()) defaultName = QString("数据组 %1").arg(m_dataGroups.size() + 1);

    DataGroup group;
    group.groupName = defaultName;
    group.color = getColor(m_dataGroups.size());
    group.weight = 1.0;
    group.showDeltaP = true;
    group.showDerivative = true;
    group.time = rawTime;
    group.deltaP = finalDeltaP;
    group.derivative = finalDeriv;
    group.origTime = rawTime;
    group.origDeltaP = finalDeltaP;
    group.origDerivative = finalDeriv;
    group.rawP = rawPressureData;

    m_dataGroups.append(group);

    // 平分各组权重
    for (int i = 0; i < m_dataGroups.size(); ++i) m_dataGroups[i].weight = 1.0 / m_dataGroups.size();

    refreshDataTable();
    updateModelCurve(nullptr, true, false);
}

/**
 * @brief 根据内部维护的数据组列表重绘下方的可视化数据管理表格
 */
void WT_MultidataFittingWidget::refreshDataTable()
{
    ui->tableDataGroups->blockSignals(true);
    ui->tableDataGroups->clearContents();
    ui->tableDataGroups->setRowCount(m_dataGroups.size());

    for(int i = 0; i < m_dataGroups.size(); ++i) {
        const DataGroup& g = m_dataGroups[i];

        QTableWidgetItem* nameItem = new QTableWidgetItem(g.groupName);
        ui->tableDataGroups->setItem(i, 0, nameItem);

        QTableWidgetItem* colorItem = new QTableWidgetItem();
        colorItem->setBackground(QBrush(g.color));
        colorItem->setFlags(colorItem->flags() & ~Qt::ItemIsEditable);
        ui->tableDataGroups->setItem(i, 1, colorItem);

        QWidget* pWidgetP = new QWidget();
        pWidgetP->setStyleSheet("background: transparent; border: none;");
        QHBoxLayout* pLayoutP = new QHBoxLayout(pWidgetP);
        pLayoutP->setContentsMargins(0,0,0,0);
        QCheckBox* cbDeltaP = new QCheckBox();
        cbDeltaP->setStyleSheet("background: transparent; border: none;");
        cbDeltaP->setChecked(g.showDeltaP);
        pLayoutP->addWidget(cbDeltaP);
        pLayoutP->setAlignment(Qt::AlignCenter);
        ui->tableDataGroups->setCellWidget(i, 2, pWidgetP);
        connect(cbDeltaP, &QCheckBox::checkStateChanged, this, [this, cbDeltaP](Qt::CheckState state) {
            for(int r = 0; r < ui->tableDataGroups->rowCount(); ++r) {
                QWidget* w = ui->tableDataGroups->cellWidget(r, 2);
                if (w && w->isAncestorOf(cbDeltaP)) {
                    m_dataGroups[r].showDeltaP = (state == Qt::Checked);
                    updateModelCurve(nullptr, false, false);
                    break;
                }
            }
        });

        QWidget* pWidgetD = new QWidget();
        pWidgetD->setStyleSheet("background: transparent; border: none;");
        QHBoxLayout* pLayoutD = new QHBoxLayout(pWidgetD);
        pLayoutD->setContentsMargins(0,0,0,0);
        QCheckBox* cbDeriv = new QCheckBox();
        cbDeriv->setStyleSheet("background: transparent; border: none;");
        cbDeriv->setChecked(g.showDerivative);
        pLayoutD->addWidget(cbDeriv);
        pLayoutD->setAlignment(Qt::AlignCenter);
        ui->tableDataGroups->setCellWidget(i, 3, pWidgetD);
        connect(cbDeriv, &QCheckBox::checkStateChanged, this, [this, cbDeriv](Qt::CheckState state) {
            for(int r = 0; r < ui->tableDataGroups->rowCount(); ++r) {
                QWidget* w = ui->tableDataGroups->cellWidget(r, 3);
                if (w && w->isAncestorOf(cbDeriv)) {
                    m_dataGroups[r].showDerivative = (state == Qt::Checked);
                    updateModelCurve(nullptr, false, false);
                    break;
                }
            }
        });

        QTableWidgetItem* weightItem = new QTableWidgetItem(QString::number(g.weight, 'f', 2));
        ui->tableDataGroups->setItem(i, 4, weightItem);
    }
    ui->tableDataGroups->blockSignals(false);
}

/**
 * @brief 槽函数：处理数据表点击事件，若点击颜色列则弹窗调用调色板
 * @param row 表格行
 * @param col 表格列
 */
void WT_MultidataFittingWidget::onTableCellClicked(int row, int col)
{
    if (col == 1) {
        QColor newColor = QColorDialog::getColor(m_dataGroups[row].color, this, "选择数据曲线颜色");
        if (newColor.isValid()) {
            m_dataGroups[row].color = newColor;
            ui->tableDataGroups->item(row, col)->setBackground(QBrush(newColor));
            updateModelCurve(nullptr, false, false);
        }
    }
}

/**
 * @brief 槽函数：处理用户在表格中修改数据权重或组名称
 * @param row 表格行
 * @param col 表格列
 */
void WT_MultidataFittingWidget::onDataWeightChanged(int row, int col)
{
    if (col == 0) {
        m_dataGroups[row].groupName = ui->tableDataGroups->item(row, col)->text();
    }
    else if (col == 4) {
        bool ok;
        double newWeight = ui->tableDataGroups->item(row, col)->text().toDouble(&ok);
        if(ok && newWeight >= 0 && newWeight <= 1.0) {
            m_dataGroups[row].weight = newWeight;
        } else {
            ui->tableDataGroups->item(row, col)->setText(QString::number(m_dataGroups[row].weight, 'f', 2));
            QMessageBox::warning(this, "权重输入错误", "权重必须在0到1之间！");
        }
    }
}

/**
 * @brief 槽函数：唤出数据管理表格的右键菜单，提供删除指定行数据的操作功能
 */
void WT_MultidataFittingWidget::showContextMenu(const QPoint &pos)
{
    int row = ui->tableDataGroups->rowAt(pos.y());
    if (row >= 0 && row < m_dataGroups.size()) {
        QMenu menu(this);
        menu.setStyleSheet("QMenu { background-color: white; color: #333333; border: 1px solid gray; } QMenu::item:selected { background-color: #e0e0e0; }");
        QAction* delAct = menu.addAction("删除本行数据");
        if (menu.exec(ui->tableDataGroups->viewport()->mapToGlobal(pos)) == delAct) {
            if (QMessageBox::question(this, "确认删除", QString("确定要移除【%1】吗？").arg(m_dataGroups[row].groupName), QMessageBox::Yes | QMessageBox::No) == QMessageBox::Yes) {
                m_dataGroups.removeAt(row);
                if (!m_dataGroups.isEmpty()) {
                    for(auto& g : m_dataGroups) g.weight = 1.0 / m_dataGroups.size();
                }
                refreshDataTable();
                updateModelCurve(nullptr, true, false);
            }
        }
    }
}

/**
 * @brief 验证数据列表中各组的拟合权重总和是否为 1.0
 * @return 若总和校验通过返回true，否则提醒用户并返回false
 */
bool WT_MultidataFittingWidget::validateDataWeights()
{
    double sum = 0.0;
    for(const auto& g : m_dataGroups) sum += g.weight;
    if(std::abs(sum - 1.0) > 1e-4) {
        QMessageBox::warning(this, "权重校验失败", QString("当前各组数据权重之和为 %1，必须恰好等于 1.0！").arg(sum));
        return false;
    }
    return true;
}

/**
 * @brief 槽函数：弹出模型列表框供用户选择指定的试井解释理论模型
 */
void WT_MultidataFittingWidget::on_btn_modelSelect_clicked()
{
    ModelSelect dlg(this);
    dlg.setCurrentModelCode(QString("modelwidget%1").arg((int)m_currentModelType + 1));
    if (dlg.exec() == QDialog::Accepted) {
        QString code = dlg.getSelectedModelCode();
        code.remove("modelwidget");

        m_isUpdatingTable = true;
        m_currentModelType = (ModelManager::ModelType)(code.toInt() - 1);
        m_paramChart->switchModel(m_currentModelType);

        ui->btn_modelSelect->setText(ModelManager::getModelTypeName(m_currentModelType));

        loadProjectParams();
        hideUnwantedParams();
        m_isUpdatingTable = false;

        updateModelCurve(nullptr, true);
    }
}

/**
 * @brief 槽函数：弹出参数设置弹窗，配置试井解释中待拟合物理参数及时间计算范围
 */
void WT_MultidataFittingWidget::on_btnSelectParams_clicked()
{
    m_paramChart->updateParamsFromTable();
    double currentTime = m_userDefinedTimeMax > 0 ? m_userDefinedTimeMax : 10000.0;
    ParamSelectDialog dlg(m_paramChart->getParameters(), m_currentModelType, currentTime, this);
    if(dlg.exec() == QDialog::Accepted) {
        auto params = dlg.getUpdatedParams();
        for(auto& p : params) if(p.name == "LfD") p.isFit = false;

        m_isUpdatingTable = true;
        m_paramChart->setParameters(params);
        m_userDefinedTimeMax = dlg.getFittingTime();
        hideUnwantedParams();
        m_isUpdatingTable = false;

        updateModelCurve(nullptr, false);
    }
}

/**
 * @brief 槽函数：弹出多数据特有的抽样设置界面，配置重采样及平滑条件
 */
void WT_MultidataFittingWidget::on_btnSampling_clicked()
{
    if (m_dataGroups.isEmpty()) return;

    double globalMinT = 1e9, globalMaxT = -1e9;
    for (const auto& g : m_dataGroups) {
        if (g.origTime.isEmpty()) continue;
        if (g.origTime.first() < globalMinT) globalMinT = g.origTime.first();
        if (g.origTime.last() > globalMaxT) globalMaxT = g.origTime.last();
    }
    if (globalMinT > globalMaxT || globalMinT <= 0) { globalMinT = 0.01; globalMaxT = 1000.0; }

    QList<SamplingInterval> intervals;
    SamplingSettingsDialog dlg(intervals, true, globalMinT, globalMaxT, this);
    if(dlg.exec() == QDialog::Accepted) {
        if (!dlg.isCustomSamplingEnabled()) {
            for (auto& g : m_dataGroups) {
                g.time = g.origTime;
                g.deltaP = g.origDeltaP;
                g.derivative = g.origDerivative;
            }
            updateModelCurve(nullptr, true, false);
            return;
        }

        intervals = dlg.getIntervals();
        QVector<double> masterTimeGrid;

        for (const auto& interval : intervals) {
            if (interval.count <= 0 || interval.tStart >= interval.tEnd || interval.tStart <= 0) continue;
            if (interval.count == 1) {
                masterTimeGrid.append(interval.tStart);
            } else {
                double logStart = std::log10(interval.tStart);
                double logEnd = std::log10(interval.tEnd);
                double step = (logEnd - logStart) / (interval.count - 1);
                for (int i = 0; i < interval.count; ++i) {
                    masterTimeGrid.append(std::pow(10, logStart + i * step));
                }
            }
        }

        std::sort(masterTimeGrid.begin(), masterTimeGrid.end());
        auto last = std::unique(masterTimeGrid.begin(), masterTimeGrid.end(), [](double a, double b) {
            return std::abs(a - b) < 1e-6;
        });
        masterTimeGrid.erase(last, masterTimeGrid.end());

        if (masterTimeGrid.isEmpty()) return;

        for (auto& g : m_dataGroups) {
            if (g.origTime.isEmpty()) continue;
            g.time = masterTimeGrid;
            g.deltaP = interpolate(g.origTime, g.origDeltaP, masterTimeGrid);
            g.derivative = interpolate(g.origTime, g.origDerivative, masterTimeGrid);
        }
        updateModelCurve(nullptr, true, false);
    }
}

/**
 * @brief 槽函数：拟合融合模式改变（均值法/主从法），触发重新计算显示
 */
void WT_MultidataFittingWidget::on_comboFitMode_currentIndexChanged(int index)
{
    Q_UNUSED(index);
    updateModelCurve(nullptr, true, false);
}

/**
 * @brief 工具函数：对一维序列进行线性插值，将原时间轴数据映射至目标时间轴
 * @param srcX 原数据的时间序列
 * @param srcY 原数据的Y轴数值序列
 * @param targetX 期望映射到的目标时间序列
 * @return 映射后的Y轴数值序列
 */
QVector<double> WT_MultidataFittingWidget::interpolate(const QVector<double>& srcX, const QVector<double>& srcY, const QVector<double>& targetX)
{
    QVector<double> result;
    int n = srcX.size();
    if(n == 0) return result;
    for(double x : targetX) {
        if(x <= srcX.first()) { result.append(srcY.first()); continue; }
        if(x >= srcX.last()) { result.append(srcY.last()); continue; }
        int l = 0, r = n - 1;
        while(l < r - 1) {
            int mid = l + (r - l) / 2;
            if(srcX[mid] < x) l = mid;
            else r = mid;
        }
        double weight = (x - srcX[l]) / (srcX[r] - srcX[l]);
        result.append(srcY[l] + weight * (srcY[r] - srcY[l]));
    }
    return result;
}

/**
 * @brief 工具函数：生成覆盖目前所有测试数据的统一时间对数网格
 */
QVector<double> WT_MultidataFittingWidget::generateCommonTimeGrid()
{
    QVector<double> commonT;
    double minT = 1e9, maxT = -1e9;
    for(const auto& g : m_dataGroups) {
        if(g.time.isEmpty()) continue;
        if(g.time.first() < minT) minT = g.time.first();
        if(g.time.last() > maxT) maxT = g.time.last();
    }
    if(minT > maxT) return commonT;

    int N = 300;
    double logMin = std::log10(minT);
    double logMax = std::log10(maxT);
    double step = (logMax - logMin) / (N - 1);
    for(int i = 0; i < N; ++i) {
        commonT.append(std::pow(10, logMin + i * step));
    }
    return commonT;
}

/**
 * @brief 槽函数：启动后台的核心迭代拟合引擎计算最优参数
 */
void WT_MultidataFittingWidget::on_btnRunFit_clicked()
{
    if(m_isFitting) return;
    if(m_dataGroups.isEmpty()) {
        QMessageBox::warning(this, "错误", "请先导入至少一组观测数据才能进行计算。");
        return;
    }
    if(!validateDataWeights()) return;

    m_paramChart->updateParamsFromTable();
    m_isFitting = true;
    ui->btnRunFit->setEnabled(false);

    int mode = ui->comboFitMode->currentIndex();
    QVector<double> fitT, fitP, fitD;

    if (mode == 0) { // 方法一：纯粹的加权平均融合
        QVector<double> unionT;
        for(const auto& g : m_dataGroups) {
            if(g.weight > 0 && !g.time.isEmpty()) unionT.append(g.time);
        }
        std::sort(unionT.begin(), unionT.end());
        auto last = std::unique(unionT.begin(), unionT.end(), [](double a, double b) { return std::abs(a - b) < 1e-6; });
        unionT.erase(last, unionT.end());

        for (double t : unionT) {
            double sumP = 0.0, sumW = 0.0;
            for (const auto& g : m_dataGroups) {
                if (g.time.isEmpty() || g.weight <= 0) continue;
                if (t >= g.time.first() && t <= g.time.last()) {
                    QVector<double> tArr = {t};
                    double valP = interpolate(g.time, g.deltaP, tArr).first();
                    sumP += valP * g.weight;
                    sumW += g.weight;
                }
            }
            if (sumW > 0) {
                fitT.append(t);
                fitP.append(sumP / sumW);
                fitD.append(0.0);
            }
        }
    }
    else { // 方法二：指定权重最大者作为主曲线基准
        int masterIdx = -1;
        double maxW = -1.0;
        for(int i=0; i<m_dataGroups.size(); ++i) {
            if(m_dataGroups[i].weight > maxW && !m_dataGroups[i].time.isEmpty()) {
                maxW = m_dataGroups[i].weight;
                masterIdx = i;
            }
        }
        if (masterIdx >= 0) {
            const auto& masterG = m_dataGroups[masterIdx];
            for (int j = 0; j < masterG.time.size(); ++j) {
                double t = masterG.time[j];
                double sumP = masterG.deltaP[j] * masterG.weight;
                double sumW = masterG.weight;

                for (int i = 0; i < m_dataGroups.size(); ++i) {
                    if (i == masterIdx) continue;
                    const auto& slaveG = m_dataGroups[i];
                    if (slaveG.time.isEmpty() || slaveG.weight <= 0) continue;
                    if (t >= slaveG.time.first() && t <= slaveG.time.last()) {
                        QVector<double> tArr = {t};
                        double valP = interpolate(slaveG.time, slaveG.deltaP, tArr).first();
                        sumP += valP * slaveG.weight;
                        sumW += slaveG.weight;
                    }
                }
                fitT.append(t);
                fitP.append(sumP / sumW);
                fitD.append(0.0);
            }
        }
    }

    m_core->setObservedData(fitT, fitP, fitD);
    m_core->startFit(m_currentModelType, m_paramChart->getParameters(), 1.0);
}

/**
 * @brief 槽函数：强制中断拟合计算引擎
 */
void WT_MultidataFittingWidget::on_btnStop_clicked()
{
    if(m_core) m_core->stopFit();
}

/**
 * @brief 槽函数：点击按钮强制将表格中的最新参数输入并更新理论曲线绘制
 */
void WT_MultidataFittingWidget::on_btnImportModel_clicked()
{
    m_paramChart->updateParamsFromTable();
    updateModelCurve(nullptr, false, false);
}

/**
 * @brief 核心重绘函数：生成对应的模型理论曲线并绘制，将底层参数规范对齐至拟合模块级别
 * @param explicitParams 明确指定替代的临时参数字典
 * @param autoScale 标识本次重绘是否需要进行画面缩放调整
 * @param calcError 是否同时触发底层的计算误差
 */
void WT_MultidataFittingWidget::updateModelCurve(const QMap<QString, double>* explicitParams, bool autoScale, bool calcError)
{
    if(!m_modelManager) return;

    m_plotLogLog->clearGraphs();
    m_plotSemiLog->clearGraphs();
    m_plotCartesian->clearGraphs();

    // 绘制用户已加载的各组观测曲线
    for(int i = 0; i < m_dataGroups.size(); ++i) {
        const auto& g = m_dataGroups[i];
        if(g.origTime.isEmpty()) continue;

        bool isSampled = (g.time.size() < g.origTime.size() && g.time.size() > 0);

        if (g.showDeltaP) {
            QCPGraph* graphP_orig = m_plotLogLog->addGraph();
            graphP_orig->setData(g.origTime, g.origDeltaP);
            graphP_orig->setLineStyle(QCPGraph::lsNone);
            graphP_orig->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, g.color, Qt::white, 5));
            graphP_orig->setName(g.groupName + " (原始压差)"); // 此处因为完整提取了文件名，使得图例中不含多余的后缀

            if (isSampled) {
                QCPGraph* graphP_samp = m_plotLogLog->addGraph();
                graphP_samp->setData(g.time, g.deltaP);
                graphP_samp->setLineStyle(QCPGraph::lsNone);
                graphP_samp->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, g.color, g.color, 5));
                graphP_samp->setName(g.groupName + " (抽样压差)");
            }
        }

        if (g.showDerivative) {
            QCPGraph* graphD_orig = m_plotLogLog->addGraph();
            graphD_orig->setData(g.origTime, g.origDerivative);
            graphD_orig->setLineStyle(QCPGraph::lsNone);
            graphD_orig->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, g.color, Qt::white, 5));
            graphD_orig->setName(g.groupName + " (原始导数)");

            if (isSampled) {
                QCPGraph* graphD_samp = m_plotLogLog->addGraph();
                graphD_samp->setData(g.time, g.derivative);
                graphD_samp->setLineStyle(QCPGraph::lsNone);
                graphD_samp->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssTriangle, g.color, g.color, 5));
                graphD_samp->setName(g.groupName + " (抽样导数)");
            }
        }
    }

    QMap<QString, double> rawParams;
    if (explicitParams) rawParams = *explicitParams;
    else {
        for(const auto& p : m_paramChart->getParameters()) rawParams.insert(p.name, p.value);
    }

    QMap<QString, double> solverParams = FittingCore::preprocessParams(rawParams, m_currentModelType);
    QVector<double> targetT = generateCommonTimeGrid();
    if(targetT.isEmpty()) targetT = ModelManager::generateLogTimeSteps(300, -4, 4);

    ModelCurveData res = m_modelManager->calculateTheoreticalCurve(m_currentModelType, solverParams, targetT);
    QVector<double> tCurve = std::get<0>(res), pCurve = std::get<1>(res), dCurve = std::get<2>(res);

    // 【修改点】规范理论模型的画笔名称与颜色，使多数据分析界面和单数据拟合分析的画风完全对齐
    QCPGraph* gModelP = m_plotLogLog->addGraph();
    gModelP->setData(tCurve, pCurve);
    gModelP->setPen(QPen(Qt::red, 2, Qt::SolidLine)); // 理论压差设为红色实线
    gModelP->setName("理论压差");

    QCPGraph* gModelD = m_plotLogLog->addGraph();
    gModelD->setData(tCurve, dCurve);
    gModelD->setPen(QPen(Qt::blue, 2, Qt::SolidLine)); // 理论导数设为蓝色实线
    gModelD->setName("理论导数");

    if(autoScale) m_plotLogLog->rescaleAxes();

    m_plotLogLog->replot();
    m_plotSemiLog->replot();
    m_plotCartesian->replot();
}

/**
 * @brief 槽函数：接收并绘制算法传回的最新一步迭代计算数据
 * @param err 迭代均方误差
 * @param p 迭代产出的最新模型参数对象
 * @param t 离散时间轴数据
 * @param p_curve 迭代生成的压差理论数组
 * @param d_curve 迭代生成的导数理论数组
 */
void WT_MultidataFittingWidget::onIterationUpdate(double err, const QMap<QString,double>& p, const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve)
{
    ui->label_Error->setText(QString("综合误差(MSE): %1").arg(err, 0, 'e', 3));

    m_isUpdatingTable = true;
    ui->tableParams->blockSignals(true);
    for(int i=0; i<ui->tableParams->rowCount(); ++i) {
        QString key = ui->tableParams->item(i, 1)->data(Qt::UserRole).toString();
        if(p.contains(key)) {
            QTableWidgetItem* valItem = ui->tableParams->item(i, 2);
            if(valItem) {
                valItem->setText(QString::number(p[key], 'g', 5));
            }
        }
    }
    ui->tableParams->blockSignals(false);
    m_isUpdatingTable = false;

    updateModelCurve(&p, false, false);
    QApplication::processEvents(QEventLoop::ExcludeUserInputEvents);
}

/**
 * @brief 槽函数：接收后台引擎结束迭代事件，并使界面按钮重新激活
 */
void WT_MultidataFittingWidget::onFitFinished()
{
    m_isFitting = false;
    ui->btnRunFit->setEnabled(true);
    QMessageBox::information(this, "完成", "多数据拟合迭代计算已正常结束。");
}

/**
 * @brief 槽函数：导出并落地多数据分析环境下的多重拟合网格与数据到外部 CSV 表格中
 */
void WT_MultidataFittingWidget::onExportCurveData()
{
    QString defaultDir = ModelParameter::instance()->getProjectPath();
    if(defaultDir.isEmpty()) defaultDir = ".";
    QString path = QFileDialog::getSaveFileName(this, "导出多数据拟合曲线", defaultDir + "/MultiFittingCurves.csv", "CSV Files (*.csv)");
    if (path.isEmpty()) return;

    QFile f(path);
    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&f);
        out << "DataGroup,Time,DeltaP,Derivative\n";

        for(const auto& g : m_dataGroups) {
            for(int i = 0; i < g.time.size(); ++i) {
                out << g.groupName << "," << g.time[i] << "," << g.deltaP[i] << ",";
                if(i < g.derivative.size()) out << g.derivative[i] << "\n";
                else out << "\n";
            }
        }

        out << "TheoreticalModel,Time,DeltaP,Derivative\n";
        QMap<QString, double> rawParams;
        for(const auto& p : m_paramChart->getParameters()) rawParams.insert(p.name, p.value);
        QMap<QString, double> solverParams = FittingCore::preprocessParams(rawParams, m_currentModelType);
        QVector<double> targetT = generateCommonTimeGrid();
        if(targetT.isEmpty()) targetT = ModelManager::generateLogTimeSteps(300, -4, 4);

        ModelCurveData res = m_modelManager->calculateTheoreticalCurve(m_currentModelType, solverParams, targetT);
        QVector<double> tCurve = std::get<0>(res), pCurve = std::get<1>(res), dCurve = std::get<2>(res);
        for(int i = 0; i < tCurve.size(); ++i) {
            out << "Model," << tCurve[i] << "," << pCurve[i] << "," << dCurve[i] << "\n";
        }

        f.close();
        QMessageBox::information(this, "导出成功", "多数据拟合曲线数据已保存。");
    }
}

/**
 * @brief 槽函数：半对数截距手工标定更新（联动地层压力参数联动调整）
 * @param k 斜率
 * @param b 截距
 */
void WT_MultidataFittingWidget::onSemiLogLineMoved(double k, double b)
{
    Q_UNUSED(k);
    QList<FitParameter> params = m_paramChart->getParameters();
    bool updated = false;
    for(auto& p : params) {
        if(p.name == "Pi" || p.name == "p*") {
            p.value = b;
            updated = true;
            break;
        }
    }
    if(updated) {
        m_isUpdatingTable = true;
        m_paramChart->setParameters(params);
        m_isUpdatingTable = false;
        updateModelCurve(nullptr, false, false);
    }
}

/**
 * @brief 隐藏表内非独立拟合的附属计算因子（如 LfD 修正系数）
 */
void WT_MultidataFittingWidget::hideUnwantedParams()
{
    for(int i = 0; i < ui->tableParams->rowCount(); ++i) {
        QTableWidgetItem* item = ui->tableParams->item(i, 1);
        if(item && item->data(Qt::UserRole).toString() == "LfD") ui->tableParams->setRowHidden(i, true);
    }
}

/**
 * @brief 读取单例中的全局测井/物性参数，补充传递以保证双侧界面拟合核心数据一致
 */
void WT_MultidataFittingWidget::loadProjectParams()
{
    ModelParameter* mp = ModelParameter::instance();
    QList<FitParameter> params = m_paramChart->getParameters();
    bool changed = false;
    // 【修改点】补齐了原先遗漏的粘度、压缩系数、体积系数和产量这4个物理参数，使得底层计算器拿到的预参数与拟合界面一致
    for(auto& p : params) {
        if(p.name == "phi") { p.value = mp->getPhi(); changed = true; }
        else if(p.name == "h")  { p.value = mp->getH();  changed = true; }
        else if(p.name == "rw") { p.value = mp->getRw(); changed = true; }
        else if(p.name == "mu") { p.value = mp->getMu(); changed = true; }
        else if(p.name == "Ct") { p.value = mp->getCt(); changed = true; }
        else if(p.name == "B")  { p.value = mp->getB();  changed = true; }
        else if(p.name == "q")  { p.value = mp->getQ();  changed = true; }
    }
    if(changed) {
        m_isUpdatingTable = true;
        m_paramChart->setParameters(params);
        m_isUpdatingTable = false;
    }
}

/**
 * @brief 获取内部预设颜色序列，用以区分多个数据对象
 * @param index 色谱游标
 * @return 对应的QColor对象
 */
QColor WT_MultidataFittingWidget::getColor(int index) {
    static QList<QColor> colors = { QColor("#1f77b4"), QColor("#ff7f0e"), QColor("#2ca02c"), QColor("#d62728"), QColor("#9467bd") };
    return colors[index % colors.size()];
}
